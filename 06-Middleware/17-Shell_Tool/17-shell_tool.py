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


# %%
################### Secure Environment Agent ####################
# This example demonstrates: workspace_root, startup_commands, redaction_rules, env

# Ensure the restricted directory exists for the example
restricted_dir = "./restricted_data"
if not os.path.exists(restricted_dir):
    os.makedirs(restricted_dir)

secure_agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root=restricted_dir,
            startup_commands=["echo 'Secure Session Initialized'", "touch session.log"],
            redaction_rules=[
                RedactionRule(
                    pattern=r"API_KEY=[\w-]+", replacement="API_KEY=[REDACTED]"
                )
            ],
            env={"MODE": "secure", "API_KEY": "sk-secret-12345"},
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

secure_result = secure_agent.invoke(
    {
        "messages": [
            HumanMessage("Print the API_KEY environment variable and list files.")
        ]
    }
)
print("\n--- Secure Agent Output ---")
print(secure_result["messages"][-1].content)


# %%
################ Custom Maintenance Shell Agent #################
# This example demonstrates: tool_description, shell_command, shutdown_commands

maintenance_agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            tool_description="System maintenance shell. Use for cleanup and system checks.",
            shell_command="/bin/zsh",  # Explicitly use zsh
            shutdown_commands=["rm -rf ./temp_logs", "echo 'Cleanup Complete'"],
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

maintenance_result = maintenance_agent.invoke(
    {"messages": [HumanMessage("Check the current shell version and disk usage.")]}
)
print("\n--- Maintenance Agent Output ---")
print(maintenance_result["messages"][-1].content)

# %%
