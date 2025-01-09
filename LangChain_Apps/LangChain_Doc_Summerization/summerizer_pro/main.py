# main.py
import os
from tracemalloc import start
from turtle import st

from attr import s
from sympy import im
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
    print(f"Starting PDF summarization for {pdf_path}")
    if not OPENAI_API_KEY:
        raise ValueError(
            "OpenAI API key not found. Set it in your environment variables or pass it in."
        )

    llm = initialize_llm(base_url=base_url, model_name=model_name)
    raw_docs = load_and_preprocess_pdf(pdf_path)
    split_docs = split_documents(raw_docs, chunk_size, chunk_overlap)
    detailed_summary = summarize_documents(llm, split_docs)
    key_points = extract_key_points(llm, detailed_summary)

    print("PDF summarization complete")
    return {"detailed_summary": detailed_summary, "key_points": key_points}


def main():
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print(f"Current working directory: {os.getcwd()}")

        PDF_PATH = "./docs/dep.pdf"
        print(f"Starting summarization for {PDF_PATH}")
        summary_data = summarize_pdf(pdf_path=PDF_PATH)

        output_path = PDF_PATH.replace(".pdf", "_summary.md")
        save_summary(summary_data, output_path)

        print(f"\nSummary saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    print(f"Time taken: {time.time() - start}")
    print(MODEL_NAME)
