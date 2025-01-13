# %%
# # Multi Query Retrievers
import os
import sys

# Set the working directory to the directory of the current file
current_file_path = os.getcwd()
current_dir = os.path.dirname(current_file_path)  # Extract the directory from the path
os.chdir(current_dir)  # Change the working directory to the file's directory

# Optional: Add the directory to the system path (useful for module imports)
sys.path.append(current_dir)

# Verify the working directory (for debugging purposes)
print(f"Working directory set to: {current_dir}")


# %%
from langchain.storage import InMemoryByteStore

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from langchain.retrievers.multi_vector import MultiVectorRetriever

loaders = [
    TextLoader("data/langchain.md"),
    TextLoader("data/langchain2.md"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OllamaEmbeddings(model="snowflake-arctic-embed:33m"),
)

# %%
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

# %%
# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

# %%
retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


# %%
retriever.vectorstore.similarity_search("LangChain")[0]

# %%
len(retriever.invoke("LangChain")[0].page_content)


# %%
from langchain.retrievers.multi_vector import SearchType

retriever.search_type = SearchType.mmr

len(retriever.invoke("LangChain")[0].page_content)


# %%
from langchain_ollama import ChatOllama

# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="llama3.2:1b")


# %%
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)


# %%
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

# %%
summaries = chain.batch(docs, {"max_concurrency": 5})


# %%
# The vectorstore to use to index the child chunks
vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)
doc_ids = [str(uuid.uuid4()) for _ in docs]

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


# %%
sub_docs = retriever.vectorstore.similarity_search("LangChain")

print(sub_docs[0])

# %%
retrieved_docs = retriever.invoke("LangChain")

len(retrieved_docs[0].page_content)


# %%
