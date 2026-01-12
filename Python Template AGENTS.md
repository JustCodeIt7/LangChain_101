# AGENTS.md

## Project overview


## Preferences and dependencies

1. Use Python 3.12 or later
2. Install dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Load environment variables from a `.env` file for API keys and configurations.

4. Use Ollama as the local LLM backend.

```Python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
llm = ChatOllama(model='llama3.2', base_url=OLLAMA_BASE_URL)
embedding = OllamaEmbeddings(model='nomic-embed-text', base_url=OLLAMA_BASE_URL)

```

## Project structure

## Key files and their purposes

## Build and test commands

## Code style guidelines
