# %%
####################### Environment Setup #######################

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
import subprocess
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
    DockerExecutionPolicy,
    RedactionRule,
    CodexSandboxExecutionPolicy,
)
from langchain.tools import tool

# set current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
load_dotenv()

# Initialize the local LLM via Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = ChatOllama(model="gpt-oss:20b", base_url=OLLAMA_BASE_URL)

# %%
######################## Custom Tooling #########################


@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


# %%
#################### Host-Based Shell Agent #####################

agent = create_agent(
    model=llm,
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="./",
            execution_policy=HostExecutionPolicy(),  # Execute commands on the host OS
        ),
    ],
)

# %%
# Request a directory listing followed by a simulated web search
result = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "Run 'ls -la' in the shell and search for the contents of the directory."
            )
        ],
    }
)
print(result)
# %%
# Output the final response message content
print(result["messages"][-1].content)
# %%
# Note: Complex multi-line file creation can cause JSON parsing errors with local LLMs
# Better to use simpler commands or create files separately
result = agent.invoke(
    {
        "messages": [
            HumanMessage("Create a file called hello.txt with the text 'Hello World'")
        ],
    }
)
print(result["messages"][-1].content)
