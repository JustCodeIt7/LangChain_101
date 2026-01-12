#%%
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Load environment variables from .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
llm = ChatOllama(model='gpt-oss:20b', base_url=OLLAMA_BASE_URL)

#%%
agent = create_agent(
    model="ollama:qwen3:4b",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path=os.getcwd(),
            # use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)

#%%
# Agent can now use glob_search and grep_search tools
result = agent.invoke({
    "messages": [HumanMessage("Find all Python files in folder and list them")]
})
# The agent will use:
# 1. glob_search(pattern="**/*.py") to find Python files
# 2. grep_search(pattern="async def", include="*.py") to find async functions
#%%

print(result)
# %%
print(result['messages'][-1].content)
# %%
