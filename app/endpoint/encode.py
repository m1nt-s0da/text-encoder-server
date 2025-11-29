from ..app import app
from dataclasses import dataclass
import hashlib
from ..tokenizer import get_tokenizer
from ..encoder import encode_sentences
from safetensors.torch import save
from fastapi import Response


@dataclass
class EncodeRequest:
    sentences: list[str]
    include_last_hidden_state: bool = False


@app.post("/encode")
async def encode_endpoint(request_data: EncodeRequest):
    hashes = []
    for sentence in request_data.sentences:
        hash_object = hashlib.sha256(sentence.encode())
        hash_hex = hash_object.hexdigest()
        hashes.append(hash_hex)

    tokenizer = get_tokenizer()
    inputs = tokenizer(request_data.sentences, return_length=True)
    output = await encode_sentences(hashes, inputs.input_ids, inputs.attention_mask)
    result = {
        "attention_mask": output.attention_mask,
        "pooler_output": output.pooler_output,
        "lengths": output.lengths,
    }
    if request_data.include_last_hidden_state:
        result["last_hidden_state"] = output.last_hidden_state
    return Response(content=save(result), media_type="application/octet-stream")
