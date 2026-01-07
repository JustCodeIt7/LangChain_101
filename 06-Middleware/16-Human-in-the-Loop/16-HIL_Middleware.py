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
# Load environment variables from .env file
load_dotenv()
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

model = ChatOllama(model="gpt-oss:20b", temperature=0.1, base_url=base_url)

# %%
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

    while True:
        decision = input("Enter 'approve'/'1' or 'reject'/'0': ").lower().strip()
        if decision in ["approve", "1"]:
            return "approve"
        elif decision in ["reject", "0"]:
            return "reject"
        print("Invalid input. Please enter 'approve'/'1' or 'reject'/'0'.")


# %%
# =============================================================================
# Example 1: Email Management with Selective Interrupts
# =============================================================================
print("\n=== EXAMPLE 1: Email Management ===\n")

def read_email_tool(email_id: str) -> str:
    """Read an email by its ID."""
    return f"Email {email_id}: 'Meeting tomorrow at 3pm. Can you confirm?'"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Send an email (requires human approval)."""
    return f"✓ Email sent to {recipient}: '{subject}'"

email_agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": True,  # Require approval for sending
                "read_email_tool": False,  # No approval needed for reading
            }
        ),
    ],
).with_config(RunnableConfig(configurable={"thread_id": "email_1"}))

# Initial invocation
print("\nStarting email agent...")
config = RunnableConfig(configurable={"thread_id": "email_1"})
result = email_agent.invoke({"messages": [("user", "Read email 12345 and send a reply confirming the meeting")]})

# Check if interrupted (waiting for approval)
state = email_agent.get_state(config)
if state.next:  # Agent is interrupted
    print("\n⚠️  Agent interrupted! Requires human approval.")

    # Get human decision
    decision = get_human_approval("send_email_tool", state.tasks[0])

    if decision == "approve":
        print("\n✓ Approved! Continuing...")
        result = email_agent.invoke(None)  # Continue execution
        print("\nFinal Result:", result)
    else:
        print("\n✗ Rejected!")
else:
    print("\nFinal Result:", result)

# %%
# =============================================================================
# Example 2: Database Operations with Critical Action Protection
# =============================================================================
print("\n=== EXAMPLE 2: Database Operations ===\n")

def query_database_tool(query: str) -> str:
    """Execute a SELECT query."""
    return "Query result: [user1, user2, user3] (3 rows)"

def delete_records_tool(table: str, condition: str) -> str:
    """Deleting records for the database (requires human approval)."""
    return f"✓ Deleted records from {table} where {condition}"

def update_records_tool(table: str, updates: str) -> str:
    """Updating records for the database (requires human approval)."""
    return f"✓ Updated {table}: {updates}"

db_agent = create_agent(
    model="gpt-5-nano",
    tools=[query_database_tool, delete_records_tool, update_records_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_records_tool": True,  # Always require approval
                "update_records_tool": True,  # Always require approval
                "query_database_tool": False,  # Safe, no approval needed
            }
        ),
    ],
).with_config(RunnableConfig(configurable={"thread_id": "db_1"}))

# Initial invocation
print("\nStarting database agent...")
# config = RunnableConfig(configurable={"thread_id": "db_1"})
# Try one of these prompts to see different behaviors:
# 1. Triggers delete_records_tool:
result = db_agent.invoke({
    "messages": [("user", "Delete all records from users table where last_login is older than 2 years")]
})
# 2. Triggers update_records_tool:
# result = db_agent.invoke({"messages": [("user", "Update the users table and set status to 'inactive' for inactive accounts")]})
# 3. No interruption (just queries):
# result = db_agent.invoke({"messages": [("user", "Find all inactive users in the database")]})

# Check if interrupted
state = db_agent.get_state(config)
if state.next:
    print("\n⚠️  Agent interrupted! Requires human approval.")

    # Detect which tool caused the interruption
    task_info = state.tasks[0] if state.tasks else {}
    print(f"\nDebug - Task info: {task_info}")

    decision = get_human_approval("database_operation", task_info)

    if decision == "approve":
        print("\n✓ Approved! Continuing...")
        result = db_agent.invoke(None)
        print("\nFinal Result:", result)
    else:
        print("\n✗ Rejected!")
else:
    print("\nFinal Result:", result)
    print("\n⚠️ Note: Agent completed without interruption. This means it either:")
    print("  - Didn't call any interrupt_on=True tools (update/delete)")
    print("  - Or only used safe operations (query)")

# %%
# %%
# =============================================================================
# Example 2: Content Publishing with Review Workflow
# =============================================================================
print("\n=== EXAMPLE 3: Content Publishing ===\n")


def draft_content_tool(topic: str, length: int = 100) -> str:
    """Generate draft content."""
    return f"Draft: '{topic} is important because...' ({length} words)"


def publish_content_tool(content: str, platform: str) -> str:
    """Publish content (requires human review)."""
    return f"✓ Published to {platform}: {content[:50]}..."


def schedule_post_tool(content: str, time: str) -> str:
    """Schedule a post (requires human review)."""
    return f"✓ Scheduled for {time}: {content[:50]}..."


content_agent = create_agent(
    model="gpt-4o",
    tools=[draft_content_tool, publish_content_tool, schedule_post_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "publish_content_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "schedule_post_tool": {"allowed_decisions": ["approve", "edit", "reject"]},
                "draft_content_tool": False,  # Drafting is safe
            }
        ),
    ],
).with_config(RunnableConfig(configurable={"thread_id": "content_1"}))

result = content_agent.invoke({"messages": [("user", "Write a post about AI safety and publish it")]})
print(result)

# %%
