import os
from typing import List, Dict
from pathlib import Path
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate


class PDFAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the PDF Analyzer with OpenAI API key."""
        self.llm = OpenAI(temperature=0, api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, chunk_overlap=200, length_function=len
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and split the PDF into manageable chunks."""
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return self.text_splitter.split_documents(pages)

    def generate_summary(self, docs: List[Document]) -> str:
        """Generate an executive summary of the entire document."""
        summary_prompt = PromptTemplate(
            template="""
            Create a detailed executive summary of the following text. Focus on:
            1. Main themes and key findings
            2. Critical conclusions
            3. Important recommendations
            
            Text: {text}
            
            DETAILED SUMMARY:
            """,
            input_variables=["text"],
        )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=summary_prompt,
            combine_prompt=summary_prompt,
        )

        return chain.run(docs)

    def extract_key_points(self, docs: List[Document]) -> List[str]:
        """Extract key points from each section of the document."""
        key_points_prompt = PromptTemplate(
            template="""
            Extract the most important key points from the following text.
            Present them in a clear, bullet-point format.
            
            Text: {text}
            
            KEY POINTS:
            """,
            input_variables=["text"],
        )

        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=key_points_prompt,
            combine_prompt=key_points_prompt,
        )

        return chain.run(docs)

    def create_structured_notes(self, pdf_path: str) -> Dict:
        """Create structured notes from the PDF document."""
        # Load and process the PDF
        docs = self.load_pdf(pdf_path)

        # Generate various analyses
        summary = self.generate_summary(docs)
        key_points = self.extract_key_points(docs)

        # Create structured output
        structured_notes = {
            "metadata": {
                "filename": Path(pdf_path).name,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_pages": len(docs),
            },
            "executive_summary": summary,
            "key_points": key_points,
        }

        return structured_notes

    def save_notes(self, notes: Dict, output_path: str):
        """Save the structured notes to a markdown file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Document Analysis Report\n\n")

            # Metadata
            f.write("## Document Metadata\n")
            f.write(f"- Filename: {notes['metadata']['filename']}\n")
            f.write(f"- Analysis Date: {notes['metadata']['analysis_date']}\n")
            f.write(f"- Total Pages: {notes['metadata']['total_pages']}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n")
            f.write(notes["executive_summary"] + "\n\n")

            # Key Points
            f.write("## Key Points\n")
            f.write(notes["key_points"] + "\n")


def main():
    # Initialize the analyzer with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    analyzer = PDFAnalyzer(api_key)

    # Specify your PDF file path
    pdf_path = "path/to/your/document.pdf"
    output_path = "document_analysis.md"

    # Process the PDF and generate notes
    notes = analyzer.create_structured_notes(pdf_path)

    # Save the notes to a markdown file
    analyzer.save_notes(notes, output_path)
    print(f"Analysis completed and saved to {output_path}")


if __name__ == "__main__":
    main()
