import os
import subprocess
import tempfile

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool


@tool(parse_docstring=True)
def create_file(filename: str, content: str) -> str:
    """Create a new file with the given content.

    Args:
        filename: Name of the file to create.
        content: Content to write to the file.

    Returns:
        Confirmation message.
    """
    # Create in a temp directory for demo purposes
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, filename)
    with open(file_path, "w") as f:
        f.write(content)
    return f"Created {file_path} successfully"


@tool(parse_docstring=True)
def run_command(command: str) -> str:
    """Run a shell command and return the output.

    Args:
        command: Shell command to execute.

    Returns:
        Command output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=10,
            cwd=tempfile.gettempdir(),
        )
        if result.returncode == 0:
            return f"Command succeeded:\n{result.stdout}"
        return f"Command failed (exit code {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds"
    except Exception as e:
        return f"Error running command: {e}"


agent = create_agent(
    model="openai:gpt-4o",
    tools=[create_file, run_command],
    system_prompt="You are a software development assistant.",
    middleware=[TodoListMiddleware()],
)
