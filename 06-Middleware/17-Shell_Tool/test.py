import os

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HostExecutionPolicy,
    HumanInTheLoopMiddleware,
    ShellToolMiddleware,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from rich import print

shell_middleware = ShellToolMiddleware(
    workspace_root=os.getcwd(),
    env=os.environ,  # danger
    execution_policy=HostExecutionPolicy()
)

hil_middleware = HumanInTheLoopMiddleware(interrupt_on={"shell": True})

checkpointer = InMemorySaver()

agent = create_agent(
    "openai:gpt-4.1-mini",
    middleware=[shell_middleware, hil_middleware],
    checkpointer=checkpointer,
)

input_message = {"role": "user", "content": "run `which python`"}

config = {"configurable": {"thread_id": "1"}}

result = agent.invoke(
    {"messages": [input_message]},
    config=config,
    durability="exit",
)

for message in result["messages"]:
    message.pretty_print()

print("\n--- interrupted ---\n")
for request in result["__interrupt__"][0].value["action_requests"]:
    print(request["description"])

next_result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    durability="exit",
)

for message in next_result["messages"]:
    message.pretty_print()



