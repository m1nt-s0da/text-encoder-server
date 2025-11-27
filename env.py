import os

LABSE_MODEL_NAME = os.getenv("LABSE_MODEL_NAME", "sentence-transformers/LaBSE")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "./embeddings")
