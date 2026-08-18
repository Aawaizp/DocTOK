import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"  

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 8
TEMPERATURE = 0.35

CHROMA_PATH = "chroma_db"
UPLOAD_DIR = "uploaded_pdfs"

