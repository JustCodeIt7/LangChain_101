# LangChain_101 AI Agent Instructions

## Core Context
This repository is an educational collection of LangChain tutorials and modular AI applications. It balances local development using Ollama with cloud-based inference (OpenAI, Groq, OpenRouter).

## Tech Stack
- **Framework**: LangChain 0.3.x (Modularized imports like `langchain_core`, `langchain_ollama`, `langchain_openai`).
- **Environment**: Python 3.11+, managed via `uv`.
- **LLM Providers**:
  - **Local**: `ChatOllama` (common models: `llama3.2`, `phi4`, `qwen2.5:0.5b`).
  - **Cloud**: `ChatOpenAI`, `ChatGroq`, and OpenRouter (`openai_api_base="https://openrouter.ai/api/v1"`).
- **Utility**: `rich` for formatting, `python-dotenv` for configuration.

## Coding Patterns & Conventions
- **Interactive Development**: Use `# %%` cell markers in `.py` files to enable Jupyter-like execution in VS Code.
- **LCEL First**: Use LangChain Expression Language (LCEL) for chains.
  ```python
  from langchain_core.output_parsers import StrOutputParser
  chain = prompt | llm | StrOutputParser()
  ```
- **Terminal UX**: Use `from rich import print` for all user-facing terminal output and debugging.
- **Secure Credentials**: 
  - Standard check: `if "API_KEY" not in os.environ: os.environ["API_KEY"] = getpass.getpass(...)`.
  - Prefer `load_dotenv()` at the start of scripts.
- **Reliable Paths**: In application entry points (e.g., `main.py`), ensure the working directory is set to the file's directory:
  ```python
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  ```

## Project Structure
- **Tutorials**: Numbered directories (`01-Models/`, `02-Data-Connections/`) follow a sequential learning path.
- **Apps**: `LangChain_Apps/` contains production-style modular code (separate `config.py`, `summarizer.py`, etc.).
- **Cache/Persistence**: Vector stores typically reside in `02-Data-Connections/chroma/` or `faiss/`.

## Development Workflows
- **Dependency Management**: Always run `uv sync` when adding new packages to `pyproject.toml`.
- **Script Execution**: Most `.py` files are designed to be run as standalone demos or via VS Code "Run Cell".
