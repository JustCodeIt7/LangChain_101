# Quick test to see actual token counts
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
model = ChatOllama(model="llama3.2", base_url=OLLAMA_URL)

# Simulate what happens during agent execution
test_messages = [
    HumanMessage("Tell me about the history of Ancient Rome in 3 sentences."),
    AIMessage("Ancient Rome was a powerful civilization that dominated the Mediterranean region for nearly 1,000 years, from its founding in 753 BCE to the fall of the Western Roman Empire in 476 CE. The Romans created a vast empire through military conquest and developed advanced engineering, law, and governance systems. Their cultural and architectural achievements, including the Colosseum and aqueducts, continue to influence modern society.")
]

tokens = model.get_num_tokens_from_messages(test_messages)
print(f"Tokens for 1 exchange (Q+A): {tokens}")
print(f"2048 * 0.8 = {2048 * 0.8} tokens (trigger threshold)")
print(f"Would trigger: {tokens >= 2048 * 0.8}")
print(f"\nThis means with 0.8 fraction, we'd need {2048 * 0.8 / 2:.1f} exchanges to trigger")
print(f"But agent overhead (system prompts, scaffolding) likely pushes us over immediately!")
