from fastapi import FastAPI
from contextlib import asynccontextmanager
from .encoder import start_encoder_loop
from .tokenizer import start_tokenizer_loop

__all__ = ["lifespan"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with start_encoder_loop(), start_tokenizer_loop():
        yield
