# config.py
import os

BASE_URL = "james-linux.local:11434"
# MODEL_NAME = "qwen2.5"
# MODEL_NAME = "qwen2.5:0.5b"
# MODEL_NAME = "phi4"
MODEL_NAME = "llama3.2"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
