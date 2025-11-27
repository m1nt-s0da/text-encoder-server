import asyncio
from contextlib import asynccontextmanager
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from env import LABSE_MODEL_NAME
from typing import cast
from logging import getLogger

__all__ = ["start_tokenizer_loop", "get_tokenizer"]

logger = getLogger("uvicorn." + __name__)

_tokenizer: BertTokenizerFast | None = None


def get_tokenizer() -> BertTokenizerFast:
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded yet.")
    return _tokenizer


@asynccontextmanager
async def start_tokenizer_loop():
    global _tokenizer
    _tokenizer = cast(
        BertTokenizerFast, AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    )
    logger.info("Tokenizer loaded.")

    try:
        yield
    finally:
        _tokenizer = None
