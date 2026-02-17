# %%
# MASTERING SHELL TOOL MIDDLEWARE - TUTORIAL CODE
#####################################################################
# This script demonstrates 3 key configurations for ShellToolMiddleware:
# 1. Basic Host Access
# 2. Secure/Restricted Environment (Redaction & Env Vars)
# 3. Custom Maintenance Shell (Zsh & Cleanup)

import os
from dotenv import load_dotenv
from rich import print
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
    RedactionRule,
)

# 1. Setup Environment
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Initialize Local LLM
llm = ChatOllama(
    model="gpt-oss:20b",
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

# %%
# EXAMPLE 1: BASIC HOST SHELL AGENT
#####################################################################
# The simplest configuration. Grants the agent access to the host
# system to run commands.

print("--- 1. Running Basic Host Agent ---")

basic_agent = create_agent(
    model=llm,
    tools=[],  # We can add other tools here if needed
    middleware=[
        ShellToolMiddleware(
            # workspace_root: Directory where the shell session starts
            workspace_root="./",
            # execution_policy: Controls HOW commands are run (Host vs Docker)
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

result_1 = basic_agent.invoke(
    {
        "messages": [
            HumanMessage(
                "Create a file 'hello.txt' with text 'Hello World', then list files."
            )
        ]
    }
)
print(result_1)
print(result_1["messages"][-1].content)


# %%
# EXAMPLE 2: SECURE / RESTRICTED AGENT
#####################################################################
# Demonstrates how to:
# - Set environment variables
# - Run startup commands
# - Redact sensitive information (PII)

print("\n--- 2. Running Secure Agent ---")

# Create a restricted directory for this example
restricted_dir = "./restricted_data"
if not os.path.exists(restricted_dir):
    os.makedirs(restricted_dir)

secure_agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root=restricted_dir,
            # env: Inject secrets or config into the shell session
            env={"MODE": "restricted", "API_KEY": "sk-secret-12345"},
            # startup_commands: Run these immediately when the session starts
            startup_commands=["echo 'Secure Session Started'", "touch session.lock"],
            # redaction_rules: Regex patterns to hide sensitive output
            redaction_rules=[
                RedactionRule(pii_type="custom_api_key", detector=r"API_KEY=[\w-]+")
            ],
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

result_2 = secure_agent.invoke(
    {"messages": [HumanMessage("Check the API_KEY env var and list files.")]}
)
print(result_2)
print(result_2["messages"][-1].content)


# %%
# EXAMPLE 3: CUSTOM MAINTENANCE SHELL
#####################################################################
# Demonstrates how to:
# - Use a specific shell (zsh vs bash)
# - Provide a custom tool description for the LLM
# - Run shutdown commands for cleanup

print("\n--- 3. Running Maintenance Agent ---")

maintenance_agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        ShellToolMiddleware(
            # tool_description: Helps the LLM understand when/how to use this shell
            tool_description="System maintenance shell for cleanup and updates.",
            # shell_command: Explicitly use zsh (or any other shell executable)
            shell_command="/bin/zsh",
            # shutdown_commands: Run these when the agent session closes
            shutdown_commands=[
                "rm -rf ./temp_logs",
                "echo 'Cleanup Complete'",
            ],
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

result_3 = maintenance_agent.invoke(
    {"messages": [HumanMessage("Check shell version and disk usage.")]}
)
print(result_3)
print(result_3["messages"][-1].content)

# %%
