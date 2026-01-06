#%%
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from rich import print
from dotenv import load_dotenv
import os
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.invoke("Obama's first name?")
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = ChatOllama(model="tinyllama")

model = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# %%
# Single condition: trigger if tokens >= 4000
agent = create_agent(
    tools=[search],
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)

# %%
# run agent1 with some messages to see summarization in action
result = agent.invoke({"messages": [HumanMessage("Tell Me a story.")]})
print(result)


# %%
# Multiple conditions: trigger if number of tokens >= 3000 OR messages >= 6 keep last 20 messages
agent2 = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=[
                ("tokens", 3000),
                ("messages", 6),
            ],
            keep=("messages", 20),
        ),
    ],
)
#%%
# Using agent2 with some messages to see summarization in action
result = agent2.invoke({"messages": [HumanMessage("Tell Me a story.")]})
print(result)

# %%
# Using fractional limits for trigger and keep conditions
agent3 = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)
# %%
result = agent3.invoke({"messages": [HumanMessage("Tell Me a story.")]})
print(result)
# %%
