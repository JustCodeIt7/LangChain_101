import os
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import HumanMessage
from typing import Dict, List


def create_detailed_analysis_prompt(text: str) -> str:
    """Create a detailed analysis prompt for the LLM."""
    return f"""Analyze the following text in great detail. Please provide:
    1. A comprehensive summary of the main ideas and arguments
    2. Detailed analysis of key concepts and their relationships
    3. Important definitions and terminology used
    4. Notable examples or case studies mentioned
    5. Any methodologies or frameworks discussed
    6. Critical insights and implications
    7. Potential applications or practical relevance
    8. Connections to related topics or fields

    Text to analyze: {text}

    Please structure your response in detailed sections covering each of these aspects."""


def extract_section_insights(text: str, section_name: str, llm) -> str:
    """Extract detailed insights for a specific section of the document."""
    print(f"Extracting insights for section: {section_name}")
    prompt = f"""Analyze the following text specifically focusing on {section_name}. 
    Provide detailed notes including:
    - Main points and arguments
    - Supporting evidence
    - Technical details
    - Practical implications
    - Critical analysis

    Text: {text}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def summarize_pdf(
    pdf_path: str,
    base_url="james-linux.local:11434",
    openai_api_key: str = None,
    model_name: str = "qwen2.5:0.5b",
    chunk_size: int = 3000,
    chunk_overlap: int = 300,
) -> Dict:
    """Enhanced PDF summarization with detailed analysis and section-by-section breakdown."""
    print(f"Starting summarization for PDF: {pdf_path}")

    # Initialize LLM and load PDF (previous initialization code remains the same)
    print("Initializing LLM...")
    llm = ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=0.1,  # Slightly increased for more detailed analysis
    )

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()
    print(f"Loaded {len(document_pages)} pages from PDF.")

    # Enhanced text splitting with section detection
    print("Splitting text into sections...")
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Process documents in sections
    sections: List[Dict] = []
    current_section = ""
    section_chunks = []

    for page in document_pages:
        page_text = page.page_content
        page_text = re.sub(r"\s+", " ", page_text.strip())

        # Attempt to identify section breaks (customize based on your documents)
        potential_sections = re.split(r"\n(?=[A-Z][A-Z\s]{2,}:?)|(?=\d+\.\s+[A-Z])", page_text)

        for section in potential_sections:
            # print progress every 10 sections
            if len(sections) % 10 == 0:
                print(f"Processed {len(sections)} sections...")

            if section.strip():
                section_chunks.append(Document(page_content=section))

                # If we have enough content for a section analysis
                if len(section_chunks) >= 3:
                    section_text = " ".join([doc.page_content for doc in section_chunks])
                    print(f"Analyzing section with {len(section_chunks)} chunks...")
                    sections.append(
                        {
                            "content": section_text,
                            "analysis": extract_section_insights(section_text, "this section", llm),
                        }
                    )
                    section_chunks = []

    # Generate comprehensive summary
    print("Generating comprehensive summary...")
    summarize_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=False)
    summary_result = summarize_chain.invoke(section_chunks)
    print("Comprehensive summary generated.")

    # Generate detailed analysis
    print("Generating detailed analysis...")
    detailed_analysis_prompt = create_detailed_analysis_prompt(summary_result["output_text"])
    detailed_analysis = llm.invoke([HumanMessage(content=detailed_analysis_prompt)])
    print("Detailed analysis generated.")

    # Extract key concepts and terminology
    print("Extracting key concepts and terminology...")
    terminology_prompt = f"""Based on the text, identify and explain:
    1. Key technical terms and their definitions
    2. Important concepts and their relationships
    3. Methodologies or frameworks mentioned
    4. Notable references or citations

    Text: {summary_result["output_text"]}"""

    terminology_analysis = llm.invoke([HumanMessage(content=terminology_prompt)])
    print("Key concepts and terminology extracted.")

    return {
        "executive_summary": summary_result["output_text"].strip(),
        "detailed_analysis": detailed_analysis.content,
        "terminology_and_concepts": terminology_analysis.content,
        "section_analyses": sections,
    }


def save_enhanced_summary(summary_data: Dict, output_path: str) -> None:
    """Save enhanced summary data to a markdown file with detailed sections."""
    print(f"Saving enhanced summary to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Comprehensive Document Analysis\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"{summary_data['executive_summary']}\n\n")

            # Detailed Analysis
            f.write("## Detailed Analysis\n\n")
            f.write(f"{summary_data['detailed_analysis']}\n\n")

            # Terminology and Concepts
            f.write("## Key Terminology and Concepts\n\n")
            f.write(f"{summary_data['terminology_and_concepts']}\n\n")

            # Section-by-Section Analysis
            f.write("## Section-by-Section Analysis\n\n")
            for i, section in enumerate(summary_data["section_analyses"], 1):
                f.write(f"### Section {i}\n\n")
                f.write(f"**Content Overview:**\n{section['content'][:200]}...\n\n")
                f.write(f"**Detailed Analysis:**\n{section['analysis']}\n\n")

    except Exception as e:
        print(f"Error saving enhanced summary to {output_path}: {e}")
        raise


if __name__ == "__main__":
    # Example usage remains the same
    print(f"Current working directory: {os.getcwd()}")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    PDF_PATH = "./docs/somatosensory.pdf"

    try:
        summary_data = summarize_pdf(pdf_path=PDF_PATH)
        output_path = PDF_PATH.replace(".pdf", "_enhanced_summary.md")
        save_enhanced_summary(summary_data, output_path)
        print(f"\nEnhanced summary saved to: {output_path}")
    except Exception as e:
        print(f"An error occurred in the main execution: {e}")
