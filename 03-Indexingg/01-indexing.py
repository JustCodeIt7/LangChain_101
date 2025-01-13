# %% [markdown]
"""
# Chroma Index Demo

Convert the script into a Jupyter Notebook.
"""

# %% [markdown]
"""
### Step 1: Initialize a vector store (Chroma) and set up embeddings (OllamaEmbeddings)
- Chroma is used as the vector store to store and retrieve embeddings.
- OllamaEmbeddings generates embeddings for the documents using the specified model.
- The collection_name is the namespace for the vector store.
"""

# %%
from langchain.indexes import SQLRecordManager, index
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

collection_name = "test_index"
embedding = OllamaEmbeddings(model="snowflake-arctic-embed:33m")
vectorstore = Chroma(collection_name=collection_name, embedding_function=embedding)

# %% [markdown]
"""
### Step 2: Initialize a record manager
- The SQLRecordManager keeps track of document writes into the vector store.
- The namespace uniquely identifies the record manager for this vector store.
- The record manager uses a SQLite database to store metadata and hashes for documents.
"""

# %%
namespace = f"chroma/{collection_name}"
record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")

# %% [markdown]
"""
### Step 3: Create schema for the record manager
- The schema is required to store document metadata and hashes in the SQLite database.
- This step ensures the record manager is ready to track document writes.
"""

# %%
record_manager.create_schema()

# %% [markdown]
"""
### Step 4: Define some test documents
Notes:
- Each document has `page_content` (the text content) and `metadata` (e.g., source information).
- The `source` metadata is crucial for tracking the origin of documents and enabling cleanup modes.
"""

# %%
doc1 = Document(page_content="kitty", metadata={"source": "kitty.txt"})
doc2 = Document(page_content="doggy", metadata={"source": "doggy.txt"})

# %% [markdown]
"""
### Step 5: Index documents using the "None" deletion mode
Notes:
- The "None" cleanup mode does not delete any existing documents in the vector store.
- It ensures that duplicate content is not re-indexed, saving time and resources.
- This mode is useful when you want to manually handle cleanup of old content.
"""

# %%
print("Indexing with 'None' deletion mode:")
result = index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup=None,
    source_id_key="source",
)
print(result)  # Output: {'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}

# %% [markdown]
"""
### Step 6: Index documents using the "incremental" deletion mode
- The "incremental" cleanup mode deletes old versions of documents if their content has changed.
- It continuously cleans up as new documents are indexed, minimizing the time old versions exist.
- This mode is ideal for updating documents while keeping the vector store clean.
"""

# %%
print("\nIndexing with 'incremental' deletion mode:")
result = index(
    [doc1, doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print(result)  # Output: {'num_added': 2, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}

# %% [markdown]
"""
### Step 7: Mutate a document and re-index using "incremental" mode
- When a document's content changes, the new version is indexed, and the old version is deleted.
- This ensures that the vector store always contains the latest version of each document.
"""

# %%
changed_doc2 = Document(page_content="puppy", metadata={"source": "doggy.txt"})
print("\nRe-indexing mutated document with 'incremental' deletion mode:")
result = index(
    [changed_doc2],
    record_manager,
    vectorstore,
    cleanup="incremental",
    source_id_key="source",
)
print(result)  # Output: {'num_added': 1, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 1}

# %% [markdown]
"""
### Step 8: Perform a similarity search
- The similarity search retrieves documents most similar to the query based on embeddings.
- The `k` parameter specifies the number of results to return.
- This demonstrates how indexed documents can be retrieved for a given query.
"""

# %%
print("\nPerforming similarity search:")
search_results = vectorstore.similarity_search("dog", k=5)
for doc in search_results:
    print(doc)
# %%
