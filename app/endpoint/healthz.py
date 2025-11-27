from ..app import app
from ..tokenizer import get_tokenizer


@app.get("/healthz")
async def health_check():
    get_tokenizer()
    return {"status": "ok"}
