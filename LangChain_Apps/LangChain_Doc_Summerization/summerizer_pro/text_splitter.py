# text_splitter.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document


def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    print(
        f"Splitting {len(docs)} documents with chunk size {chunk_size} and overlap {chunk_overlap}"
    )
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_docs = []
    for doc in docs:
        # print(f"Splitting document: {doc}")
        for chunk in text_splitter.split_text(doc.page_content):
            split_docs.append(Document(page_content=chunk))
            print(f"Created chunk: {chunk[:10]}...")  # Log first 50 characters of the chunk

    print(f"Total chunks created: {len(split_docs)}")
    return split_docs
