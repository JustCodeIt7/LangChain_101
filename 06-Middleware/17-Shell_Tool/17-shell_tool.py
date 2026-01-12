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

# ShellToolMiddleware Parameters:
# workspace_root (str | Path | None):
#   Base directory for the shell session. If omitted, a temporary directory is 
#   created when the agent starts and removed when it ends.
#
# startup_commands (tuple[str, ...] | list[str] | str | None):
#   Optional commands executed sequentially after the session starts
#
# shutdown_commands (tuple[str, ...] | list[str] | str | None):
#   Optional commands executed before the session shuts down
#
# execution_policy (BaseExecutionPolicy | None):
#   Execution policy controlling timeouts, output limits, and resource configuration. Options:
#   - HostExecutionPolicy: Full host access (default); best for trusted environments 
#     where the agent already runs inside a container or VM
#   - DockerExecutionPolicy: Launches a separate Docker container for each agent run, 
#     providing harder isolation
#   - CodexSandboxExecutionPolicy: Reuses the Codex CLI sandbox for additional 
#     syscall/filesystem restrictions
#
# redaction_rules (tuple[RedactionRule, ...] | list[RedactionRule] | None):
#   Optional redaction rules to sanitize command output before returning it to the model.
#   Redaction rules are applied post execution and do not prevent exfiltration of secrets 
#   or sensitive data when using HostExecutionPolicy.
#
# tool_description (str | None):
#   Optional override for the registered shell tool description
#
# shell_command (Sequence[str] | str | None):
#   Optional shell executable (string) or argument sequence used to launch the persistent 
#   session. Defaults to /bin/bash.
#
# env (Mapping[str, Any] | None):
#   Optional environment variables to supply to the shell session. Values are coerced 
#   to strings before command execution.

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
# Note: Complex multi-line file creation can cause JSON parsing errors with local LLMs
# Better to use simpler commands or create files separately
result = agent.invoke({
    "messages": [HumanMessage("Create a file called hello.txt with the text 'Hello World'")],
})
print(result['messages'][-1].content)
#%%
################### Docker-Isolated Shell Agent ##################

# Docker isolation with startup commands
# Create a sandboxed environment to prevent host machine interference
# REQUIREMENTS:
# 1. Docker Desktop must be installed and running
# 2. Docker daemon must be accessible (check with: docker ps)
# 3. The specified image must be available or pullable (python:3.12-slim)
#
# KNOWN ISSUE: DockerExecutionPolicy may fail with "Shell session is not running"
# This is a known limitation with LangChain's Docker middleware implementation.
# The session initialization may fail due to timing issues or Docker socket configuration.
#
# WORKAROUND: Use HostExecutionPolicy with manual Docker commands if needed,
# or test with GPT-4 which has better Docker middleware support.

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
                session_timeout=60.0,           # Allow more time for session startup
            ),
        ),
    ],
)

#%%
# Execute a shell command inside the isolated container
# Note: This may fail with "Shell session is not running" due to Docker middleware limitations
result_docker = agent_docker.invoke({
    "messages": [HumanMessage("Run 'python --version' in the shell.")],
})
   

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