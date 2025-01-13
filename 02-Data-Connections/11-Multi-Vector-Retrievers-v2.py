# %%
import os
import sys
import uuid

from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import MultiVectorRetriever, SearchType
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Helper Functions
def set_working_directory():
    """Set the working directory to the script's directory."""
    current_file_path = os.getcwd()
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)
    sys.path.append(current_dir)
    print(f"Working directory set to: {current_dir}")


def load_documents(file_paths):
    """Load documents from a list of file paths."""
    loaders = [TextLoader(path) for path in file_paths]
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def split_documents(docs, chunk_size):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    return text_splitter.split_documents(docs)


def create_retriever(vectorstore, byte_store, id_key):
    """Create a MultiVectorRetriever instance."""
    return MultiVectorRetriever(vectorstore=vectorstore, byte_store=byte_store, id_key=id_key)


def add_documents_to_retriever(retriever, docs, chunk_size):
    """Add documents to the retriever after splitting into smaller chunks."""
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    sub_docs = []

    for i, doc in enumerate(docs):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata["doc_id"] = _id
        sub_docs.extend(_sub_docs)

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, docs)))
    return doc_ids  # Return doc_ids for external use


# %%
# Set up the environment
# set_working_directory()
# %%
# Load and split documents
file_paths = ["data/langchain.md", "data/langchain2.md"]
docs = load_documents(file_paths)
docs = split_documents(docs, chunk_size=10000)

# Initialize vectorstore and retriever
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OllamaEmbeddings(model="snowflake-arctic-embed:33m"),
)
store = InMemoryByteStore()
retriever = create_retriever(vectorstore, store, id_key="doc_id")
# %%
# Add documents to the retriever
doc_ids = add_documents_to_retriever(retriever, docs, chunk_size=400)
# %%
# Test similarity search
similar_doc = retriever.vectorstore.similarity_search("LangChain")[0]
print(similar_doc)

# %%
# Set up LLM and summarization pipeline
llm = ChatOllama(model="llama3.2:1b")
# %%
chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)
# %%
# Generate summaries
summaries = chain.batch(docs, {"max_concurrency": 5})
# %%
# Reinitialize vectorstore for summaries
summary_vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
summary_retriever = create_retriever(summary_vectorstore, store, id_key="doc_id")
# %%
# Add summaries to the retriever
summary_docs = [
    Document(page_content=s, metadata={"doc_id": doc_ids[i]}) for i, s in enumerate(summaries)
]
summary_retriever.vectorstore.add_documents(summary_docs)
summary_retriever.docstore.mset(list(zip(doc_ids, docs)))
# %%
# Test retrieval of summaries
retrieved_summary_docs = summary_retriever.invoke("LangChain")
print(retrieved_summary_docs[0])

# %%
