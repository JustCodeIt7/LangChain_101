#%%
####################### Environment Setup #######################

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
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
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
llm = ChatOllama(model='gpt-oss:20b', base_url=OLLAMA_BASE_URL)

#%%
######################## Custom Tooling #########################

@tool
def search_tool(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

#%%
#################### Host-Based Shell Agent #####################

# Basic shell tool with host execution
# Configure the agent to interact with the local filesystem directly
agent = create_agent(
    model=llm,
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="./",
            execution_policy=HostExecutionPolicy(), # Execute commands on the host OS
        ),
    ],
)

#%%
# Request a directory listing followed by a simulated web search
result = agent.invoke({
    "messages": [HumanMessage("Run 'ls -la' in the shell and search for the contents of the directory.")],
})
print(result)
# %%
# Output the final response message content
print(result['messages'][-1].content)

#%%
################### Docker-Isolated Shell Agent ##################

# Docker isolation with startup commands
# Create a sandboxed environment to prevent host machine interference
agent_docker = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="./",
            # Prepare the container environment before execution
            # startup_commands=["pip install requests", "export PYTHONPATH=./"],
            execution_policy=DockerExecutionPolicy(
                image="python:3.12-slim",
                command_timeout=30.0,           # Terminate long-running processes
            ),
        ),
    ],
)

#%%
# Execute a shell command inside the isolated container
result_docker = agent_docker.invoke({
    "messages": [HumanMessage("Run 'python --version' in the shell.")],
})
print(result_docker)
# %%
print(result_docker['messages'][-1].content)

#%%
#################### Redacted Output Agent ######################

# With output redaction (applied post execution)
# Scrub sensitive patterns from the command output before the LLM sees it
agent_redacted = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="./",
            redaction_rules=[
                # Identify and mask OpenAI-style API keys
                RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
            ],
        ),
    ],
)

#%%
# Test the redaction logic by forcing a sensitive string output
result_redacted = agent_redacted.invoke({
    "messages": [HumanMessage("Echo an API key like 'sk-1234567890abcdef1234567890abcdef'")],
})
print(result_redacted)
# %%
print(result_redacted['messages'][-1].content)

# %%

################ Codex Sandbox Shell Agent ######################
# Using Codex Sandbox for execution
# Leverage OpenAI Codex to safely execute shell commands in a simulated environment
agent_codex = create_agent(
    model='gpt-4o',
    middleware=[
        ShellToolMiddleware(
            workspace_root="./",
            execution_policy=CodexSandboxExecutionPolicy(
                platform="macos",  # Specify the target OS environment
            ),
        ),
    ],
)
#%%
# Execute a shell command in the Codex sandbox
result_codex = agent_codex.invoke({
    "messages": [HumanMessage("Run 'uname -a' in the shell.")],
})
print(result_codex)
# %%
print(result_codex['messages'][-1].content)
# %%