# %% [markdown]
# # Multi Query Retrievers
# This tutorial demonstrates how to build and use multi-query retrievers with LangChain.
# We'll start by loading documents, splitting them into chunks, and using embeddings
# to enable similarity search and retrieval.

# %%
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
import uuid

# %% [markdown]
# ## Step 1: Load Documents
# Here, we load the documents using `TextLoader`. You can replace the file paths
# with your own document paths.

loaders = [
    TextLoader("data/langchain.md"),
    TextLoader("data/langchain2.md"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# %% [markdown]
# ## Step 2: Split Documents into Chunks
# Using `RecursiveCharacterTextSplitter`, we split documents into smaller chunks
# for more efficient processing and retrieval.

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(docs)

# %% [markdown]
# ## Step 3: Initialize the Vectorstore
# The vectorstore indexes the document chunks for similarity search. We use the `Chroma`
# library with embeddings from Ollama.

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=OllamaEmbeddings(model='snowflake-arctic-embed:33m')
)

# %% [markdown]
# ## Step 4: Create Smaller Chunks
# For fine-grained retrieval, we split the documents further into smaller chunks
# and associate metadata for linking with original documents.

from langchain.retrievers.multi_vector import MultiVectorRetriever

# Initialize storage for parent documents
store = InMemoryByteStore()
id_key = "doc_id"

# Create the retriever
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]

# Split into smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

# %% [markdown]
# ## Step 5: Add Documents to Vectorstore
# Add the smaller chunks and their metadata to the vectorstore for similarity search.

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# %% [markdown]
# ## Step 6: Perform a Similarity Search
# Use the retriever to find documents similar to the query "LangChain".

print(retriever.vectorstore.similarity_search("LangChain")[0])

# %% [markdown]
# ## Step 7: Multi-Modal Retrieval
# Modify the search type to use Maximal Marginal Relevance (MMR) for diverse results.

from langchain.retrievers.multi_vector import SearchType

retriever.search_type = SearchType.mmr
retriever.invoke("LangChain")

# %% [markdown]
# ## Step 8: Summarize Documents
# Associate summaries with documents using a language model.

import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Initialize LLM
llm = ChatOllama(model='llama3.2:1b')

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chain = (
    {"doc": lambda x: x.page_content}
    | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
    | llm
    | StrOutputParser()
)

# Summarize documents in parallel
summaries = chain.batch(docs, {"max_concurrency": 5})
print(summaries)

# %% [markdown]
# ## Step 9: Add Summaries to Vectorstore
# Store the summaries in the vectorstore for enhanced retrieval capabilities.

vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
store = InMemoryByteStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

# %% [markdown]
# ## Step 10: Retrieve Summaries
# Perform a similarity search on the summaries to get the most relevant results.

sub_docs = retriever.vectorstore.similarity_search("LangChain")
print(sub_docs[0])

retrieved_docs = retriever.invoke("LangChain")
len(retrieved_docs[0].page_content)

# %%
