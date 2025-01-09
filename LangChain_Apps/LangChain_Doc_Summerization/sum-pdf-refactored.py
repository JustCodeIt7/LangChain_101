# config.py

import os

BASE_URL = "james-linux.local:11434"
MODEL_NAME = "phi4"
CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# pdf_loader.py

import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document


def load_and_preprocess_pdf(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()

    docs = []
    for page in document_pages:
        page_text = page.page_content
        page_text = re.sub(r"\s+", " ", page_text.strip())
        docs.append(Document(page_content=page_text))

    return docs


# text_splitter.py

from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_docs = []
    for doc in docs:
        for chunk in text_splitter.split_text(doc.page_content):
            split_docs.append(Document(page_content=chunk))

    return split_docs


# summarizer.py

from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain


def initialize_llm(base_url: str, model_name: str, temperature: float = 0.0):
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
    )


def summarize_documents(llm, docs: list, chain_type: str = "map_reduce") -> str:
    summarize_chain = load_summarize_chain(llm, chain_type=chain_type, verbose=False)
    summary_result = summarize_chain.invoke(docs)
    return summary_result.get("output_text", "").strip()


# key_points_extractor.py

from langchain.schema import HumanMessage


def extract_key_points(llm, summary: str) -> str:
    key_points_prompt = (
        "Given the following summary, extract the most important points as bullet points. "
        "Ensure each point is concise and captures essential information:\n\n"
        f"Summary:\n{summary}\n\n"
        "Now produce a clear, concise list of key points:"
    )

    key_points_message = [HumanMessage(content=key_points_prompt)]
    key_points_response = llm.invoke(key_points_message)
    return key_points_response.content.strip()


# output_handler.py


def save_summary(summary_data: dict, output_path: str) -> None:
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Document Summary\n\n")
            f.write("## Detailed Summary\n\n")
            f.write(f"{summary_data['detailed_summary']}\n\n")
            f.write("## Key Points\n\n")
            f.write(f"{summary_data['key_points']}\n")
    except Exception as e:
        print(f"Error saving summary to {output_path}: {e}")
        raise


# main.py

import os
from config import (
    BASE_URL,
    MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
)
from pdf_loader import load_and_preprocess_pdf
from text_splitter import split_documents
from summarizer import initialize_llm, summarize_documents
from key_points_extractor import extract_key_points
from output_handler import save_summary


def summarize_pdf(
    pdf_path: str,
    base_url: str = BASE_URL,
    model_name: str = MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Set it in your environment variables or pass it in."
        )

    llm = initialize_llm(base_url=base_url, model_name=model_name)
    raw_docs = load_and_preprocess_pdf(pdf_path)
    split_docs = split_documents(raw_docs, chunk_size, chunk_overlap)
    detailed_summary = summarize_documents(llm, split_docs)
    key_points = extract_key_points(llm, detailed_summary)

    return {"detailed_summary": detailed_summary, "key_points": key_points}


def main():
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print(f"Current working directory: {os.getcwd()}")

        PDF_PATH = "./docs/dep.pdf"
        summary_data = summarize_pdf(pdf_path=PDF_PATH)

        output_path = PDF_PATH.replace(".pdf", "_summary.md")
        save_summary(summary_data, output_path)

        print(f"\nSummary saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    main()
