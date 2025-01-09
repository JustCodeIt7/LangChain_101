import os
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage


def summarize_pdf(
    pdf_path: str,
    base_url="james-linux.local:11434",
    openai_api_key: str = None,
    model_name: str = "phi4",
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
):
    """
    Summarize and extract key points from a large PDF file using LangChain.

    :param pdf_path: Path to the PDF file to summarize.
    :param openai_api_key: Your OpenAI API key.
    :param model_name: The name of the Ollama model to use (e.g., "llama3.2").
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
    llm = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.0,  # For factual summarization (less creative drift)
    )

    # 3. Load the PDF
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()

    # 4. Split PDF text into smaller chunks
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
    # try:
    summary_result = summarize_chain.invoke(docs)
    print("Summary Result:", summary_result)  # Debugging statement
    # summary_result = summary_result.content
    summary_result = summary_result["output_text"]

    # 7. Optionally, we can create a bullet-point list of key insights
    key_points_prompt = (
        "Given the following summary, extract the most important points as bullet points. "
        "Ensure each point is concise and captures essential information:\n\n"
        f"Summary:\n{summary_result}\n\n"
        "Now produce a clear, concise list of key points:"
    )

    # Ensure the prompt is wrapped in a HumanMessage
    key_points_message = [HumanMessage(content=key_points_prompt)]
    print("Key Points Message:", key_points_message)  # Debugging statement
    key_points_response = llm.invoke(key_points_message)
    print("Key Points Response:", key_points_response)  # Debugging statement
    # if hasattr(key_points_response[0], "content"):
    # key_points = key_points_response[0].content.strip()
    key_points = key_points_response.content

    return {"detailed_summary": summary_result.strip(), "key_points": key_points}


def save_summary(summary_data: dict, output_path: str) -> None:
    """Save summary data to a markdown file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            # Write title
            f.write("# Document Summary\n\n")

            # Write detailed summary section
            f.write("## Detailed Summary\n\n")
            f.write(f"{summary_data['detailed_summary']}\n\n")

            # Write key points section
            f.write("## Key Points\n\n")
            f.write(f"{summary_data['key_points']}\n")

    except Exception as e:
        print(f"Error saving summary to {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print(f"Current working directory: {os.getcwd()}")
    # set working directory to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working directory: {os.getcwd()}")
    PDF_PATH = "./docs/dep.pdf"
    try:
        summary_data = summarize_pdf(pdf_path=PDF_PATH)

        # Generate output filename from input PDF
        output_path = PDF_PATH.replace(".pdf", "_summary.md")
        save_summary(summary_data, output_path)

        print(f"\nSummary saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
