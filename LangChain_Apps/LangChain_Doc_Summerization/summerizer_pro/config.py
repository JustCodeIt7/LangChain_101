# config.py
import os

BASE_URL = "james-linux.local:11434"
MODEL_NAME = "llama3.2"
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
