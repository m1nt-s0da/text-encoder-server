from ..app import app
from ..tokenizer import get_tokenizer
from ..encoder import check_ready


@app.get("/healthz")
async def health_check():
    await check_ready()
    get_tokenizer()
    return {"status": "ok"}
