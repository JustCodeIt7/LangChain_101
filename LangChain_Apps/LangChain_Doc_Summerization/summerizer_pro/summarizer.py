# summarizer.py
from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain


def initialize_llm(base_url: str, model_name: str, temperature: float = 0.0):
    print(
        f"Initializing LLM with base_url={base_url}, model_name={model_name}, temperature={temperature}"
    )
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
    )


def summarize_documents(llm, docs: list, chain_type: str = "map_reduce") -> str:
    print(f"Summarizing {len(docs)} documents using chain type {chain_type}")
    summarize_chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
    summary_result = summarize_chain.invoke(docs)
    print("Summarization complete")
    return summary_result.get("output_text", "").strip()
