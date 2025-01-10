import os
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage


def generate_detailed_notes(
    pdf_path: str,
    base_url="james-linux.local:11434",
    openai_api_key: str = None,
    model_name: str = "qwen2.5:0.5b",
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
):
    """
    Generate detailed notes from a large PDF file using LangChain.

    :param pdf_path: Path to the PDF file to process.
    :param openai_api_key: Your OpenAI API key.
    :param model_name: The name of the Ollama model to use (e.g., "phi4").
    :param chunk_size: Maximum number of tokens per chunk for text splitting.
    :param chunk_overlap: Number of overlapping tokens between chunks.
    :return: A string containing the detailed notes.
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
        temperature=0.0,  # For factual extraction (less creative drift)
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
        # print every 10 pages to check progress
        if page.page_number % 10 == 0:
            print(f"Processing page {page.page_number}")

        for chunk in text_splitter.split_text(page_text):
            docs.append(Document(page_content=chunk))

    # 5. Generate detailed notes for each chunk
    detailed_notes = []
    for doc in docs:
        detailed_notes_prompt = (
            f"Take the following text and write detailed notes that include all key arguments, "
            f"examples, and supporting data:\n\n{doc.page_content}"
        )
        response = llm.invoke([HumanMessage(content=detailed_notes_prompt)])
        detailed_notes.append(response.content)

    # 6. Organize the notes
    organized_prompt = (
        "Combine and organize the following detailed notes into a structured format with headings "
        "and subheadings:\n\n"
        f"{' '.join(detailed_notes)}"
    )
    organized_response = llm.invoke([HumanMessage(content=organized_prompt)])
    final_notes = organized_response.content

    return final_notes


def save_detailed_notes(notes_data: str, output_path: str) -> None:
    """Save detailed notes to a markdown file."""
    print(f"Saving detailed notes to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Detailed Notes\n\n")
            f.write(notes_data)
    except Exception as e:
        print(f"Error saving detailed notes to {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    print(f"Current working directory: {os.getcwd()}")
    # Set working directory to file location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Current working directory: {os.getcwd()}")
    PDF_PATH = "./docs/dep.pdf"
    try:
        detailed_notes = generate_detailed_notes(pdf_path=PDF_PATH)

        # Generate output filename from input PDF
        output_path = PDF_PATH.replace(".pdf", "_detailed_notes.md")
        save_detailed_notes(detailed_notes, output_path)

        print(f"\nDetailed notes saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
