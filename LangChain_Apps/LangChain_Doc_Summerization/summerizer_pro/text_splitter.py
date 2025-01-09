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
