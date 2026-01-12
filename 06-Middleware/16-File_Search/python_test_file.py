#%%
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage
from dotenv import load_dotenv
from rich import print
import os
load_dotenv()
# print working directory
print(f"Working directory: {os.getcwd()}")

#%%
agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path=os.getcwd(),
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)

#%%

# Agent can now use glob_search and grep_search tools
result = agent.invoke({
    "messages": [HumanMessage("Find all Python files containing 'async def'")]
})
# The agent will use:
# 1. glob_search(pattern="**/*.py") to find Python files
# 2. grep_search(pattern="async def", include="*.py") to find async functions
#%%

print(result)
# %%
