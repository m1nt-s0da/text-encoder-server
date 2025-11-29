import torch
from torch.nn.utils.rnn import pad_sequence
import asyncio
from contextlib import asynccontextmanager
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel
from env import LABSE_MODEL_NAME, BATCH_SIZE, EMBEDDINGS_DIR
from typing import cast
from logging import getLogger
from dataclasses import dataclass
import os
from safetensors.torch import save_file, load_file
from threading import Thread, Lock

__all__ = ["start_encoder_loop", "encode_sentences", "EncodeSentenceReult"]

logger = getLogger("uvicorn." + __name__)


def _load_encoder_sync() -> BertModel:
    model = AutoModel.from_pretrained(LABSE_MODEL_NAME)
    model.eval()
    return cast(BertModel, model)


def _save_data_sync(
    hash: str,
    attention_mask: torch.Tensor,
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
):
    target_path = os.path.join(EMBEDDINGS_DIR, hash[:2], f"{hash[2:]}.safetensors")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    save_file(
        {
            "attention_mask": attention_mask,
            "pooler_output": pooler_output,
            "last_hidden_state": last_hidden_state,
        },
        target_path,
    )


async def _save_data(
    hash: str,
    attention_mask: torch.Tensor,
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        _save_data_sync,
        hash,
        attention_mask,
        pooler_output,
        last_hidden_state,
    )


async def _encoder_loop():
    logger.info("Starting to load encoder...")
    model = _load_encoder_sync()
    assert _ready_loop is not None
    assert _ready is not None
    _ready_loop.call_soon_threadsafe(_ready.set_result, True)
    logger.info("Encoder loaded successfully.")
    while True:
        if len(_encode_queue) == 0:
            await asyncio.sleep(0.1)
            continue
        with _encode_queue_lock:
            sentences = list(_encode_queue.values())
        sentences.sort(key=lambda x: len(x.input_ids), reverse=False)
        sentences = sentences[:BATCH_SIZE]
        input_ids = [torch.tensor(s.input_ids, dtype=torch.long) for s in sentences]
        attention_masks = [
            torch.tensor(s.attention_mask, dtype=torch.long) for s in sentences
        ]
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks_padded = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids_padded, attention_mask=attention_masks_padded
            )

        coroutines = [
            _save_data(
                sentences[i].hash,
                attention_masks_padded[i],
                outputs.pooler_output[i],
                outputs.last_hidden_state[i][: len(input_ids[i])],
            )
            for i in range(len(sentences))
        ]
        await asyncio.gather(*coroutines)

        for i in range(len(sentences)):
            result = EncodeSentenceReult(
                attention_mask=attention_masks_padded[i][: len(input_ids[i])],
                pooler_output=outputs.pooler_output[i],
                last_hidden_state=outputs.last_hidden_state[i][: len(input_ids[i])],
                lengths=torch.tensor(len(input_ids[i]), dtype=torch.long),
            )
            for loop, future in _encode_callbacks[sentences[i].hash]:
                loop.call_soon_threadsafe(future.set_result, result)
            with _encode_queue_lock:
                del _encode_callbacks[sentences[i].hash]
                del _encode_queue[sentences[i].hash]
        logger.info(f"Encoded and saved {len(sentences)} sentences.")


@dataclass
class EncodeSentenceReult:
    attention_mask: torch.Tensor
    pooler_output: torch.Tensor
    last_hidden_state: torch.Tensor
    lengths: torch.Tensor


@dataclass
class EncodeSentenceRequest:
    hash: str
    input_ids: list[int]
    attention_mask: list[int]


_encode_queue_lock = Lock()
_encode_queue: dict[str, EncodeSentenceRequest] = {}
_encode_callbacks: dict[
    str, list[tuple[asyncio.AbstractEventLoop, asyncio.Future[EncodeSentenceReult]]]
] = {}


async def encode_sentence(
    hash: str, input_ids: list[int], attention_mask: list[int]
) -> EncodeSentenceReult:
    target_path = os.path.join(EMBEDDINGS_DIR, hash[:2], f"{hash[2:]}.safetensors")
    if os.path.exists(target_path):
        data = load_file(target_path, device="cpu")
        return EncodeSentenceReult(
            attention_mask=data["attention_mask"],
            pooler_output=data["pooler_output"],
            last_hidden_state=data["last_hidden_state"],
            lengths=torch.tensor(len(input_ids), dtype=torch.long),
        )

    loop = asyncio.get_event_loop()
    sentence = EncodeSentenceRequest(
        hash=hash, input_ids=input_ids, attention_mask=attention_mask
    )
    future = loop.create_future()
    with _encode_queue_lock:
        if sentence.hash not in _encode_callbacks:
            _encode_callbacks[sentence.hash] = []
        _encode_callbacks[sentence.hash].append((loop, future))
        _encode_queue[sentence.hash] = sentence
    return await future


async def encode_sentences(
    hashes: list[str], ids_list: list[list[int]], mask_list: list[list[int]]
) -> EncodeSentenceReult:
    assert len(hashes) == len(ids_list) == len(mask_list)
    count = len(hashes)
    tasks = [
        encode_sentence(hashes[i], ids_list[i], mask_list[i]) for i in range(count)
    ]
    result = await asyncio.gather(*tasks)
    return EncodeSentenceReult(
        attention_mask=pad_sequence(
            [r.attention_mask for r in result], batch_first=True, padding_value=0
        ),
        pooler_output=torch.stack([r.pooler_output for r in result]),
        last_hidden_state=(
            pad_sequence(
                [r.last_hidden_state for r in result], batch_first=True, padding_value=0
            )
        ),
        lengths=torch.tensor([r.lengths.item() for r in result], dtype=torch.long),
    )


_encoder_thread: Thread | None = None
_ready: asyncio.Future | None = None
_ready_loop: asyncio.AbstractEventLoop | None = None


@asynccontextmanager
async def start_encoder_loop():
    global _encoder_thread, _ready_loop, _ready
    _ready_loop = asyncio.get_event_loop()
    _ready = _ready_loop.create_future()

    if _encoder_thread is not None:
        raise RuntimeError("Encoder thread already running.")

    _encoder_thread = Thread(target=asyncio.run, args=(_encoder_loop(),), daemon=True)
    _encoder_thread.start()

    try:
        yield
    finally:
        if _encoder_thread is not None:
            _encoder_thread.join(timeout=1.0)
            _encoder_thread = None


async def check_ready():
    assert _ready is not None
    await _ready
