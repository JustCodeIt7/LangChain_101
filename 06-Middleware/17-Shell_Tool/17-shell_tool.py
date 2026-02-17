"""ShellToolMiddleware config reference (under 150 lines)."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
    RedactionRule,
    ShellToolMiddleware,
)
from langchain.messages import HumanMessage
from langchain_ollama import ChatOllama
from rich import print

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)

COMMON_MW_ARGS = dict(
    workspace_root="./",
    startup_commands=["echo 'session started'", "pwd"],
    shutdown_commands="echo 'session ended'",
    redaction_rules=[
        RedactionRule(pii_type="api_key", detector=r"sk-[A-Za-z0-9]{32}")
    ],
    tool_description="Run shell commands in a controlled session.",
    shell_command=["/bin/bash", "-l", "-i"],
    env={"APP_ENV": "demo", "MAX_ROWS": 25},
)

host_middleware = ShellToolMiddleware(
    **COMMON_MW_ARGS,
    tool_name="shell_host",
    execution_policy=HostExecutionPolicy(
        command_timeout=20,
        startup_timeout=20,
        termination_timeout=5,
        max_output_lines=80,
        max_output_bytes=50_000,
        cpu_time_seconds=2,
        memory_bytes=256 * 1024 * 1024,
        create_process_group=True,
    ),
)

docker_middleware = ShellToolMiddleware(
    **COMMON_MW_ARGS,
    tool_name="shell_docker",
    execution_policy=DockerExecutionPolicy(
        binary="docker",
        image="python:3.12-alpine3.19",
        remove_container_on_exit=True,
        network_enabled=False,
        extra_run_args=["--pids-limit", "128"],
        memory_bytes=256 * 1024 * 1024,
        cpu_time_seconds=2,
        cpus="1.0",
        read_only_rootfs=False,
        user="1000:1000",
        command_timeout=20,
        startup_timeout=30,
        termination_timeout=5,
        max_output_lines=80,
        max_output_bytes=50_000,
    ),
)

codex_middleware = ShellToolMiddleware(
    **COMMON_MW_ARGS,
    tool_name="shell_codex",
    execution_policy=CodexSandboxExecutionPolicy(
        binary="codex",
        platform="macos",
        config_overrides={"sandbox_mode": "workspace-write"},
        command_timeout=20,
        startup_timeout=20,
        termination_timeout=5,
        max_output_lines=80,
        max_output_bytes=50_000,
    ),
)

# Pick one middleware for runtime (swap as needed).
agent = create_agent(model=llm, tools=[], middleware=[host_middleware])

if __name__ == "__main__":
    print("[bold]Available middleware configs:[/bold] host, docker, codex")
    result = agent.invoke(
        {"messages": [HumanMessage("Run `echo $APP_ENV && uname -s` using the shell tool.")]}
    )
    print(result["messages"][-1].content)
# %%
