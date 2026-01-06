# %%
################################ Environment Setup ################################

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage
from rich import print
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM and UI components
# Ensure you have ollama running: `ollama run tinyllama`
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print(f"Using Ollama URL: {OLLAMA_URL}")
model = ChatOllama(model="llama3.2", base_url=OLLAMA_URL, max_input_tokens=2048)

# Model profile information is required for fractional token limits
# Standard models may not have this attribute set by default.


console = Console()

# Define a sequence of queries to fill the context window
conversation_messages = [
    HumanMessage("Tell me about the history of Ancient Rome in 3 sentences."),
    HumanMessage("Now tell me about Ancient Greece in 3 sentences."),
    HumanMessage("What about Ancient Egypt? 3 sentences please."),
    HumanMessage("Tell me about the Maya civilization in 3 sentences."),
    HumanMessage("Finally, tell me about Ancient China in 3 sentences."),
]

# %%
################################ Example 1: WITHOUT Summarization ################################

console.print(Panel.fit("Example 1: Agent WITHOUT Summarization Middleware", style="bold red"))

# Initialize a standard agent without any history management
agent_no_summary = create_agent(model=model)

messages = []

# Iterate through the history to demonstrate how context length increases linearly
for msg in conversation_messages:
    messages.append(msg)
    result = agent_no_summary.invoke({"messages": messages})
    messages = result["messages"]  # Update history with the full raw exchange

console.print(f"\n[yellow]Total messages in context:[/yellow] {len(messages)}")
console.print(f"[yellow]Context keeps growing without summarization![/yellow]\n")

# %%
################################ Example 2: Simple Summarization ################################

console.print(Panel.fit("Example 2: Simple Message Count Trigger", style="bold green"))

# Attach middleware to prune history once it reaches a specific length
agent_with_summary = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 6),  # Compress history after 6 messages are reached
            keep=("messages", 3),  # Retain only the 3 most recent interactions
        ),
    ],
)

messages = []
for i, msg in enumerate(conversation_messages, 1):
    messages.append(msg)
    result = agent_with_summary.invoke({"messages": messages})
    messages = result["messages"]

    console.print(f"\n[cyan]After Turn {i}:[/cyan]")
    console.print(f"  Messages in context: {len(messages)}")

    # Detect if the middleware injected a summary message into the history
    if any("summary" in str(m.content).lower() for m in messages):
        console.print("  [green]✓ Summarization triggered![/green]")

console.print(f"\n[yellow]Final message count:[/yellow] {len(messages)}\n")

# %%
################################ Example 3: Multiple Conditions ################################

console.print(Panel.fit("Example 3: Multiple Conditions (Tokens OR Messages)", style="bold purple"))

# This agent triggers if EITHER:
# 1. Token count exceeds 200 (set low for demo purposes)
# 2. Message count reaches 10
agent_multi_condition = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=[
                ("tokens", 200),  # Trigger if tokens > 1000
                ("messages", 10),  # OR if messages >= 10
            ],
            keep=("messages", 3),  # Keep last 3 messages
        ),
    ],
)

messages = []
console.print("[yellow]Starting conversation with Multi-Condition Agent...[/yellow]")

for i, msg in enumerate(conversation_messages, 1):
    messages.append(msg)
    result = agent_multi_condition.invoke({"messages": messages})
    messages = result["messages"]

    # Calculate rough token count (approximation for display)
    estimated_tokens = sum(len(str(m.content)) / 4 for m in messages)

    console.print(f"\n[cyan]After Turn {i}:[/cyan]")
    console.print(f"  Messages: {len(messages)}")
    console.print(f"  Est. Tokens: ~{int(estimated_tokens)}")

    if any("summary" in str(m.content).lower() for m in messages):
        console.print("  [green]✓ Summarization triggered![/green]")
        # Check why it likely triggered
        if len(messages) < 10:
            console.print("  [dim](Likely triggered by Token limit)[/dim]")
        else:
            console.print("  [dim](Likely triggered by Message count)[/dim]")

console.print(f"\n[purple]Final message count:[/purple] {len(messages)}\n")

# %%
################################ Example 4: Fractional Limits ################################

console.print(Panel.fit("Example 4: Fractional Limits (Percentage of Context)", style="bold cyan"))
# need to set if using ollama
model.profile = {"max_input_tokens": 2048}
# This agent triggers based on context window usage
# Note: tinyllama has a context window of 2048.
# We use 0.05 (5%) here just so you can see it trigger quickly in this demo.
agent_fractional = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            # Trigger when 5% of the context window is filled
            trigger=("fraction", 0.8),
            # When condensing, aim to reduce it to 2% of context
            keep=("fraction", 0.3),
        ),
    ],
)

messages = []
console.print("[yellow]Starting conversation with Fractional Agent...[/yellow]")

for i, msg in enumerate(conversation_messages, 1):
    messages.append(msg)
    result = agent_fractional.invoke({"messages": messages})
    messages = result["messages"]

    console.print(f"\n[cyan]After Turn {i}:[/cyan]")
    console.print(f"  Messages: {len(messages)}")

    if any("summary" in str(m.content).lower() for m in messages):
        console.print("  [green]✓ Summarization triggered by context fraction![/green]")

console.print(f"\n[magenta]Final message count:[/magenta] {len(messages)}")
console.print("[green]Demonstration Complete![/green]\n")

# %%
