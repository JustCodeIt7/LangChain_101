# LangChain Expression Language (LCEL) Tutorial
# For YouTube Educational Content
# %%
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Import LangChain components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama

# %%
# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOllama(model="llama3.2",temperature=0.3)

# %%
# ###############################################################
# SECTION 1: Basic LCEL Chain
# ###############################################################
print("-" * 50,'\n',"DEMO 1: Basic LCEL Chain",'\n',"-" * 50)

# Create a simple prompt template
prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}.")

# Create a basic chain using the | operator (pipe)
basic_chain = prompt | llm | StrOutputParser()

# Run the chain
result = basic_chain.invoke({"topic": "programming"})
print(result)
print("\n")


# %% 
# ###############################################################
# SECTION 2: Parallel Processing with LCEL
# ###############################################################

print("-" * 50,'\n',"DEMO 2: Parallel Processing with LCEL",'\n',"-" * 50)

# Create two different prompts
pros_prompt = ChatPromptTemplate.from_template(
    "List 3 pros of {technology} technology."
)
cons_prompt = ChatPromptTemplate.from_template(
    "List 3 cons of {technology} technology."
)

# Create chains for each prompt
pros_chain = pros_prompt | llm | StrOutputParser()
cons_chain = cons_prompt | llm | StrOutputParser()

# Create a parallel chain
parallel_chain = RunnableParallel(pros=pros_chain, cons=cons_chain)

# Run the parallel chain
parallel_result = parallel_chain.invoke({"technology": "blockchain"})
print("PROS:")
print(parallel_result["pros"])
print("\nCONS:")
print(parallel_result["cons"])
print("\n")

# %%
# ###############################################################
# SECTION 3: Advanced Chain with Data Transformation
# ###############################################################
print("-" * 50,'\n',"DEMO 3: Advanced Chain with Data Transformation",'\n',"-" * 50)

# Define a function to process input
def preprocess_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process the input data before sending to the LLM."""
    # Add a timestamp, capitalize the query, etc.
    processed_data = input_data.copy()
    processed_data["query"] = input_data["query"].upper()
    processed_data["word_count"] = len(input_data["query"].split())
    return processed_data


# Define a function to process output
def postprocess_output(output: str) -> Dict[str, Any]:
    """Process the output from the LLM."""
    lines = output.strip().split("\n")
    return {
        "summary": lines[0] if lines else "",
        "details": lines[1:],
        "response_length": len(output),
    }


# Create an advanced prompt
advanced_prompt = ChatPromptTemplate.from_template(
    """Answer the following query: {query}

    The query has {word_count} words.
    Provide a detailed explanation.
    """
)

# Create an advanced chain with preprocessing and postprocessing
advanced_chain = (
    RunnableLambda(preprocess_input)
    | advanced_prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(postprocess_output)
)

# Run the advanced chain
advanced_result = advanced_chain.invoke(
    {"query": "how does LCEL improve LLM applications"}
)
print(f"Summary: {advanced_result['summary']}")
print("Details:")
for detail in advanced_result["details"]:
    if detail.strip():
        print(f"- {detail.strip()}")
print(f"Response Length: {advanced_result['response_length']} characters")
print("\n")

# %%
# ###############################################################
# SECTION 4: RAG Pattern with LCEL
# ###############################################################
print("-" * 50,'\n',"DEMO 4: RAG Pattern with LCEL (Simulated)",'\n',"-" * 50)


# Simulate a retriever function
def simulate_retrieval(query: str) -> List[str]:
    """Simulate document retrieval based on query."""
    # In a real application, this would query a vector database
    documents = [
        "LCEL allows for declarative chain construction in LangChain.",
        "LangChain Expression Language simplifies building complex LLM applications.",
        "LCEL supports streaming, async operations, and parallel execution.",
    ]
    return documents


# Create a RAG prompt
rag_prompt = ChatPromptTemplate.from_template(
    """Answer the question based on the following context:

    Context:
    {context}

    Question: {question}
    """
)


# Create a RAG chain
def format_docs(docs):
    return "\n".join(docs)


rag_chain = (
    RunnableParallel(
        {
            "context": RunnableLambda(simulate_retrieval) | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
    )
    | rag_prompt
    | llm
    | StrOutputParser()
)

# %%
# Run the RAG chain
rag_result = rag_chain.invoke("What is LCEL and why is it useful?")
print(rag_result)

# %%