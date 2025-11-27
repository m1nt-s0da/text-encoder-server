import os
from transformers import AutoTokenizer
from transformers import AutoModel

LABSE_MODEL_NAME = os.getenv("LABSE_MODEL_NAME", "sentence-transformers/LaBSE")

if __name__ == "__main__":
    AutoTokenizer.from_pretrained(LABSE_MODEL_NAME)
    AutoModel.from_pretrained(LABSE_MODEL_NAME)
    print("Model and tokenizer downloaded successfully.")
