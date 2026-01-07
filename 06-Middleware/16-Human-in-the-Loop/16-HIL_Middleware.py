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

result = email_agent.invoke({
    "messages": [("user", "Read email 12345 and draft a confirmation reply")]
})
print(result)

# %%
# =============================================================================
# Example 2: Database Operations with Critical Action Protection
# =============================================================================
print("\n=== EXAMPLE 2: Database Operations ===\n")

def query_database_tool(query: str) -> str:
    """Execute a SELECT query."""
    return f"Query result: [user1, user2, user3] (3 rows)"

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

result = db_agent.invoke({
    "messages": [("user", "Find inactive users and delete accounts older than 2 years")]
})
print(result)

# %%
# =============================================================================
# Example 3: Content Publishing with Review Workflow
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
                "publish_content_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "schedule_post_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "draft_content_tool": False,  # Drafting is safe
            }
        ),
    ],
).with_config(RunnableConfig(configurable={"thread_id": "content_1"}))

result = content_agent.invoke({
    "messages": [("user", "Write a post about AI safety and publish it")]
})
print(result)

# %%
"""
Key Takeaways:
--------------
1. Set interrupt_on=True for tools that need human approval
2. Set interrupt_on=False for safe, read-only operations
3. Use allowed_decisions to specify approval options
4. Checkpointer is required for middleware to work
5. Each agent needs a unique thread_id for state management
"""
# %%