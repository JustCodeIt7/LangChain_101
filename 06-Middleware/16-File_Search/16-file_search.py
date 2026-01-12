#%%
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool

# set current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
print(f"OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")
llm = ChatOllama(model='gpt-oss:20b', base_url=OLLAMA_BASE_URL)

#%% ############# File Search Middleware Agent #########################
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path=os.getcwd(),
            use_ripgrep=True,
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
# %% ##################### Advanced Usage #########################

# create file edit tool
@tool
def edit_file_tool(file_path: str, new_content: str) -> str:
    """Edit the contents of a file."""
    with open(file_path, 'w') as f:
        f.write(new_content)
    return f"File {file_path} updated."



# create agent with file search and edit tools
agent = create_agent(
    model=llm,
    tools=[edit_file_tool],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path=os.getcwd(),
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)

#%%
# Agent can now use glob_search, grep_search, and edit_file_tool
result2 = agent.invoke({
    "messages": [HumanMessage("find test.md and replace its contents with 'Hello, LangChain!'")]
})
# %%
print(result2)
# %%
print(result2['messages'][-1].content)
# %%

