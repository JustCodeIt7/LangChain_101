# %%
"""
Human-in-the-Loop Middleware Tutorial
======================================
Three practical examples showing how to use HumanInTheLoopMiddleware
to require human approval for sensitive operations.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from rich import print
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
model = ChatOllama(model="gpt-oss:20b", temperature=0.1, base_url=base_url)

# %% ############# Approval Logic ###########
print("\n=== APPROVAL LOGIC DEFINITION ===\n")

def get_human_approval(action_name: str, details: dict) -> str:
    """
    Simple interactive approval function.
    Returns: 'approve' or 'reject'
    """
    print("\n" + "=" * 50)
    print("🔔 HUMAN APPROVAL REQUIRED")
    print("=" * 50)
    print(f"Action: {action_name}")
    print("Details: ")
    print(details)
    print("=" * 50)

    # Poll for user input until a valid decision is made
    while True:
        decision = input("Enter 'approve'/'1' or 'reject'/'0': ").lower().strip()
        if decision in ["approve", "1"]:
            return "approve"
        elif decision in ["reject", "0"]:
            return "reject"
        print("Invalid input. Please enter 'approve'/'1' or 'reject'/'0'.")


def check_for_interrupt(state, tool_name):
    # Process the interruption if the agent state indicates a pause
    if state.next:
        print("\n⚠️  Agent interrupted! Requires human approval.")

        # Request user decision for the specific tool execution
        decision = get_human_approval(tool_name, state.tasks[0])

        if decision == "approve":
            print("\n✓ Approved! Continuing...")
            return True
        else:
            print("\n✗ Rejected!")
            return False
    return True

# %% ################# Email Management ####################
print("\n=== EXAMPLE 1: Email Management ===\n")


def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email {email_id}: 'Meeting tomorrow at 3pm. Can you confirm?'"


def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email (requires human approval)."""
    return f"✓ Email sent to {recipient}: '{subject}'"


# Initialize agent with middleware to intercept high-stakes tools
email_agent = create_agent(
    model=model,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),  # Enable state persistence for interruptions
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,  # Pause execution for this tool
                "read_email_tool": False,  # Allow automatic execution
            }
        ),
    ],
)

print("\nStarting email agent...")
# Configure runnable with a unique thread ID for state tracking
config = RunnableConfig(configurable={"thread_id": "email_1"})

# Invoke the agent to read an email and send a reply
result = email_agent.invoke(
    {
        "messages": [
            HumanMessage("Read email 12345 and send a reply confirming the meeting")
        ]
    },
    config=config,
)

# Retrieve current agent state to evaluate if an interrupt occurred
state = email_agent.get_state(config)
check_for_interrupt(state, "send_email_tool")


# %% ################# Content Publishing #####################
print("\n=== EXAMPLE 2: Content Publishing ===\n")


def draft_content_tool(topic: str, length: int = 100) -> str:
    """Generate draft content."""
    return f"Draft: '{topic} is important because...' ({length} words)"


def publish_content_tool(content: str, platform: str) -> str:
    """Publish content (requires human review)."""
    return f"✓ Published to {platform}: {content[:50]}..."


def schedule_post_tool(content: str, time: str) -> str:
    """Schedule a post (requires human review)."""
    return f"✓ Scheduled for {time}: {content[:50]}..."


# Configure complex approval workflows with specific allowed decisions
content_agent = create_agent(
    model=model,
    tools=[draft_content_tool, publish_content_tool, schedule_post_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "publish_content_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "schedule_post_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "draft_content_tool": False,  # Mark as safe for automation
            }
        ),
    ],
)
# Configure runnable with a unique thread ID for state tracking
config = RunnableConfig(configurable={"thread_id": "content_1"})

result = content_agent.invoke(
    {"messages": [HumanMessage("Draft a post about AI safety and publish it on Twitter at 3pm") ]}, config=config
)
print(result)


# Inspect the state to trigger human review process
state = content_agent.get_state(config)
# print(state)
check_for_interrupt(state, "schedule_post_tool")

# %%
