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
    """Delete records (requires human approval)."""
    return f"✓ Deleted records from {table} where {condition}"

def update_records_tool(table: str, updates: str) -> str:
    """Update records (requires human approval)."""
    return f"✓ Updated {table}: {updates}"

db_agent = create_agent(
    model="gpt-4o",
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
config = RunnableConfig(configurable={"thread_id": "db_1"})
result = db_agent.invoke({"messages": [("user", "Find inactive users and delete accounts older than 2 years")]})

# Check if interrupted
state = db_agent.get_state(config)
if state.next:
    print("\n⚠️  Agent interrupted! Requires human approval.")

    decision = get_human_approval("delete_records_tool", state.tasks[0])

    if decision == "approve":
        print("\n✓ Approved! Continuing...")
        result = db_agent.invoke(None)
        print("\nFinal Result:", result)
    else:
        print("\n✗ Rejected!")
else:
    print("\nFinal Result:", result)
