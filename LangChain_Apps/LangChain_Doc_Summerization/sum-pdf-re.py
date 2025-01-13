import os
import re
from typing import Dict
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate


def initialize_llm(base_url: str, model_name: str, max_tokens: int) -> ChatOllama:
    """Initialize the language model."""
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def load_pdf_as_documents(pdf_path: str, chunk_size: int, chunk_overlap: int) -> list:
    """Load and split a PDF into smaller text chunks."""
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    documents = []
    for page in document_pages:
        page_text = re.sub(r"\s+", " ", page.page_content.strip())
        documents.extend(
            [Document(page_content=chunk) for chunk in text_splitter.split_text(page_text)]
        )
    return documents


def create_summarization_chain(llm) -> load_summarize_chain:
    """Create a summarization chain."""
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert summarizer. Carefully read the following text:
        {text}
        Generate a thorough, extended summary highlighting all essential details.
        """,
    )
    return load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=summary_prompt, combine_prompt=summary_prompt
    )


def summarize_documents(chain, documents: list) -> str:
    """Generate a detailed summary from the documents."""
    result = chain.invoke(documents)
    return result["output_text"].strip()


def generate_key_points(llm, summary: str) -> str:
    """Generate key points from the summary."""
    prompt = HumanMessage(
        content=(
            f"Given the following summary, provide a comprehensive list of key insights:\n\n"
            f"Summary:\n{summary}\n\nKey Points:"
        )
    )
    response = llm.invoke([prompt])
    return response.content.strip()


def save_to_markdown(data: Dict[str, str], output_path: str) -> None:
    """Save the summary and key points to a markdown file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Document Summary\n\n")
            f.write("## Detailed Summary\n\n")
            f.write(f"{data['detailed_summary']}\n\n")
            f.write("## Key Points\n\n")
            f.write(f"{data['key_points']}\n")
    except Exception as e:
        raise IOError(f"Error saving to {output_path}: {e}")


def summarize_pdf(
    pdf_path: str,
    base_url="james-linux.local:11434",
    openai_api_key: str = None,
    model_name: str = "phi4",
    chunk_size: int = 4000,
    chunk_overlap: int = 400,
    max_tokens: int = 32768,
) -> Dict[str, str]:
    """Main function to summarize a PDF and extract key points."""
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Set it in the environment or pass it explicitly."
        )

    llm = initialize_llm(base_url, model_name, max_tokens)
    documents = load_pdf_as_documents(pdf_path, chunk_size, chunk_overlap)
    summarize_chain = create_summarization_chain(llm)

    detailed_summary = summarize_documents(summarize_chain, documents)
    key_points = generate_key_points(llm, detailed_summary)

    return {"detailed_summary": detailed_summary, "key_points": key_points}


if __name__ == "__main__":
    import time

    PDF_PATH = "./docs/somatosensory.pdf"
    MODEL_NAME = "llama3.2"
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 200
    MAX_TOKENS = 64000

    start_time = time.time()

    try:
        summary_data = summarize_pdf(
            pdf_path=PDF_PATH,
            model_name=MODEL_NAME,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            max_tokens=MAX_TOKENS,
        )
        output_path = PDF_PATH.replace(".pdf", "_summary.md")
        save_to_markdown(summary_data, output_path)
        print(f"Summary saved to: {output_path}")
    except Exception as e:
        print(f"Error: {e}")

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
