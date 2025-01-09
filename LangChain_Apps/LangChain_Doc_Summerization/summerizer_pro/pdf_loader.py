# pdf_loader.py
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document


def load_and_preprocess_pdf(pdf_path: str) -> list:
    print(f"Loading PDF from path: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    document_pages = loader.load()
    print(f"Loaded {len(document_pages)} pages from PDF")

    docs = []
    for page in document_pages:
        page_text = page.page_content
        page_text = re.sub(r"\s+", " ", page_text.strip())
        docs.append(Document(page_content=page_text))
        print(f"Processed page: {page_text[:50]}...")  # Log first 50 characters of the page text

    print(f"Total processed documents: {len(docs)}")
    return docs
