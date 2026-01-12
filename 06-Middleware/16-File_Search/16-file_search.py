#%%
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.tools import tool

############## Environment Setup ###############

# Ensure the script runs relative to its own location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Initialize the local LLM via Ollama
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
llm = ChatOllama(model='gpt-oss:20b', base_url=OLLAMA_BASE_URL)

#%% 
################# File Search Middleware Agent ###########

# Initialize agent with filesystem middleware to enable native file searching
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path=os.getcwd(),      # Restrict search scope to current directory
            use_ripgrep=True,           # Performance optimization for pattern matching
            max_file_size_mb=10,        # Prevent processing overly large files
        ),
    ],
)

#%%
# Agent can now use glob_search and grep_search tools
# Execute search request using natural language
result = agent.invoke({
    "messages": [HumanMessage("Find all Python files in folder and list them")]
})
# The agent will use:
# 1. glob_search(pattern="**/*.py") to find Python files
# 2. grep_search(pattern="async def", include="*.py") to find async functions
#%%

# Output the raw result object
print(result)
# %%
# Extract and print the final response text from the message history
print(result['messages'][-1].content)

# %% 
################## Advanced Usage #####################

# Define a custom tool to allow the agent to modify local files
@tool
def edit_file_tool(file_path: str, new_content: str) -> str:
    """Edit the contents of a file."""
    # Overwrite the specified file with the provided content
    with open(file_path, 'w') as f:
        f.write(new_content)
    return f"File {file_path} updated."

# Re-initialize the agent with both search capabilities and write access
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
# Chain search and write operations in a single request
result2 = agent.invoke({
    "messages": [HumanMessage("find test.md and replace its contents with 'Hello, LangChain!'")]
})

# %%
# Display execution metadata for the file edit task
print(result2)

# %%
# Display the agent's confirmation of the file edit
print(result2['messages'][-1].content)

# %%
# Display the previous search result metadata
print(result)

# %%
# Display the previous search result content
print(result['messages'][-1].content)

# %% 
################# Demo Results #######################

# Display initial file listing results
print(result)
# %%
print(result['messages'][-1].content)

#%% 
# Display results of the file modification task
print(result2)
# %%
print(result2['messages'][-1].content)
# %%
