#%%
import os
import subprocess
import tempfile

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.tools import tool
from rich import print
from langchain.messages import HumanMessage, SystemMessage

# %% ############################ Tool Definitions #############################

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
    temp_dir = os.getcwd()
    file_path = os.path.join(temp_dir, filename)
    # Ensure file persistence in the current working directory
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
    # Execute system commands safely with timeout constraints
    try:
        result = subprocess.run(
            command,
            shell=True,  # noqa: S602
            capture_output=True,
            text=True,
            timeout=10, # Prevent long-running processes from hanging the agent
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            return f"Command succeeded:\n{result.stdout}"
        return f"Command failed (exit code {result.returncode}):\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds"
    except Exception as e:
        return f"Error running command: {e}"


# %% ################################ Agent Configuration #######################

# Initialize the agent with specialized middleware for state management
agent = create_agent(
    model="openai:gpt-4.1-nano",
    tools=[create_file, run_command],
    system_prompt="You are a software development assistant.",
    middleware=[TodoListMiddleware()], # Enable automatic todo list tracking
)

# Start a new conversation thread to generate the initial task list
response = agent.invoke(
    {
        "messages": [
            HumanMessage(
                "Create a todo list for preparing a presentation: research topic, create slides, and practice delivery. save to todo.md"
            )
        ]
    },
    config={"configurable": {"thread_id": "presentation-001"}}, # Use thread_id for state persistence
)

# %% ################################ Output Results ################################

print(response)
print("\nTodos:", response["todos"])

# %% ########################## Task Execution Loop #########################

# loop through todos and have the llm complete
for i, todo in enumerate(response["todos"]):
    # Process only unfinished tasks
    if todo["status"] == "pending":
        task_message = f"Complete the following task from the todo list: {todo['content']}"
        
        # Request the agent to execute the specific sub-task
        result = agent.invoke(
            {
                "messages": [HumanMessage(task_message)]
            },
            config={"configurable": {"thread_id": "presentation-001"}}, # Maintain context within the same thread
        )
        
        # Display progress and the agent's work for each step
        print("\n" + "=" * 60)
        print(f"STEP {i+1}: Completing Todo")
        print(f"Completed Task: {todo['content']}")
        print("=" * 60)
        print(result["messages"][-1].content) # Print the final response from the agent logic
    
# %%
