import os
import re
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document


def summarize_pdf(
    pdf_path: str,
    openai_api_key: str = None,
    model_name: str = "gpt-3.5-turbo",
    chunk_size: int = 2000,
    chunk_overlap: int = 200,
):
    """
    Summarize and extract key points from a large PDF file using LangChain.

    :param pdf_path: Path to the PDF file to summarize.
    :param openai_api_key: Your OpenAI API key.
    :param model_name: The name of the OpenAI model to use (e.g., "gpt-3.5-turbo").
    :param chunk_size: Maximum number of tokens per chunk for text splitting.
    :param chunk_overlap: Number of overlapping tokens between chunks.
    :return: A dictionary containing 'detailed_summary' and 'key_points' strings.
    """

    # 1. Ensure we have an API key either passed in or found in environment variable
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Set it in your environment variables or pass it in."
        )

    # 2. Initialize the LLM
    llm = OpenAI(
        openai_api_key=openai_api_key,
        model_name=model_name,
        temperature=0.0,  # For factual summarization (less creative drift)
    )

    # 3. Load the PDF
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()

    # 4. Split PDF text into smaller chunks
    #    Here, we treat each 'Document' from the loader as a text chunk,
    #    but you can further chunk them if needed for especially large PDFs.
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Flatten the text from all pages into smaller chunks
    docs = []
    for page in document_pages:
        page_text = page.page_content
        # Clean up text if needed
        page_text = re.sub(r"\s+", " ", page_text.strip())

        for chunk in text_splitter.split_text(page_text):
            docs.append(Document(page_content=chunk))

    # 5. Create a summarization chain
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)

    # 6. Run the summarization on the chunked documents
    summary_result = summarize_chain.run(docs)

    # 7. Optionally, we can create a bullet-point list of key insights
    #    by prompting the model again or reusing the summary.
    key_points_prompt = (
        "Given the following summary, extract the most important points as bullet points. "
        "Ensure each point is concise and captures essential information:\n\n"
        f"Summary:\n{summary_result}\n\n"
        "Now produce a clear, concise list of key points:"
    )
    key_points = llm(key_points_prompt)

    return {"detailed_summary": summary_result.strip(), "key_points": key_points.strip()}


if __name__ == "__main__":
    # Example usage
    PDF_PATH = "large_document.pdf"  # Change to your PDF path
    summary_data = summarize_pdf(pdf_path=PDF_PATH)

    print("\n===== DETAILED SUMMARY =====\n")
    print(summary_data["detailed_summary"])
    print("\n===== KEY POINTS =====\n")
    print(summary_data["key_points"])
