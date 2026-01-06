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

# Set fallback values for the local Ollama instance
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the local LLM with a defined token limit
model = ChatOllama(model="llama3.2", base_url=OLLAMA_URL)

console = Console()

# Create a sample dataset of prompts to fill the conversation history
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

# Instantiate a baseline agent to demonstrate default history growth
agent_no_summary = create_agent(model=model)

messages = []

# Process prompts sequentially to track how memory expands
for msg in conversation_messages:
    messages.append(msg)
    # Perform the inference and retrieve updated history
    result = agent_no_summary.invoke({"messages": messages})
    messages = result["messages"]  # Update history with the full raw exchange

console.print(f"\n[yellow]Total messages in context:[/yellow] {len(messages)}")
console.print(f"[yellow]Context keeps growing without summarization![/yellow]\n")

# %%
################################ Example 2: Simple Summarization ################################

console.print(Panel.fit("Example 2: Simple Message Count Trigger", style="bold green"))

# Wrap the agent with middleware to prune history based on message count
agent_with_summary = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 6),  # Initiate compression after the 6th message
            keep=("messages", 3),  # Preserve only the 3 most recent interactions
        ),
    ],
)
# %%
messages = []
# Simulate a turn-based conversation to observe the pruning logic
for i, msg in enumerate(conversation_messages, 1):
    messages.append(msg)
    result = agent_with_summary.invoke({"messages": messages})
    messages = result["messages"]

    console.print(f"\n[cyan]After Turn {i}:[/cyan]")
    console.print(f"  Messages in context: {len(messages)}")

    # Check for the presence of a summary record in the message list
    if any("summary" in str(m.content).lower() for m in messages):
        console.print("  [green]✓ Summarization triggered![/green]")

console.print(f"\n[yellow]Final message count:[/yellow] {len(messages)}\n")

# %%
################################ Example 3: Multiple Conditions ################################

console.print(Panel.fit("Example 3: Multiple Conditions (Tokens OR Messages)", style="bold purple"))

# Configure an agent with hybrid triggers for more robust memory management
agent_multi_condition = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=[
                ("tokens", 200),  # Compress if the estimated token count exceeds 200
                ("messages", 10),  # Compress if the message count hits 10
            ],
            keep=("messages", 3),  # Retain the most recent window of context
        ),
    ],
)

# %%
messages = []
console.print("[yellow]Starting conversation with Multi-Condition Agent...[/yellow]")

for i, msg in enumerate(conversation_messages, 1):
    messages.append(msg)
    result = agent_multi_condition.invoke({"messages": messages})
    messages = result["messages"]

    # Calculate an approximate token count based on string length
    estimated_tokens = sum(len(str(m.content)) / 4 for m in messages)

    console.print(f"\n[cyan]After Turn {i}:[/cyan]")
    console.print(f"  Messages: {len(messages)}")
    console.print(f"  Est. Tokens: ~{int(estimated_tokens)}")

    # Identify which condition likely caused the history to collapse
    if any("summary" in str(m.content).lower() for m in messages):
        console.print("  [green]✓ Summarization triggered![/green]")
        if len(messages) < 10:
            console.print("  [dim](Likely triggered by Token limit)[/dim]")
        else:
            console.print("  [dim](Likely triggered by Message count)[/dim]")

console.print(f"\n[purple]Final message count:[/purple] {len(messages)}\n")

# %%
################## Example 4: Fractional Limits ##################

console.print(Panel.fit("Example 4: Fractional Limits (Percentage of Context)", style="bold cyan"))

# Manually define model profile for middleware token calculations
# Note: llama3.2 has a context window of 2048 tokens
model.profile = {"max_input_tokens": 2048}

# Use fractional triggers to manage memory relative to the model's capacity
# Note: We use low percentages (5% and 2%) here so you can see it trigger quickly in this demo.
# With agent overhead (system prompts, scaffolding), even small conversations fill the context.
agent_fractional = create_agent(
    model=model,
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("fraction", 0.05),  # Trigger when 5% of context is consumed (~102 tokens)
            keep=("fraction", 0.02),  # Reduce usage down to 2% during compression (~41 tokens)
        ),
    ],
)

# %%
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
