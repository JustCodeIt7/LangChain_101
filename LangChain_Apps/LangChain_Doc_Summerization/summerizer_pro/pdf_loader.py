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
