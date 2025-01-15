# Multi Vector Retrieval (MVR) Notes

## Key Concepts

1. **Vectorstore:** Efficient indexing and searching of text using embeddings.
2. **Embeddings:** Numerical representations of text used for similarity comparison.
3. **Retrievers:** Facilitate querying and retrieving relevant document chunks.
4. **Summarization:** Condensing large chunks into concise summaries for faster insights.
5. **MMR:** Ensures diverse retrieval results for broader coverage.

## **Step 1: Load Documents**

- **Purpose:** Import raw text documents into the system.
- **Code:**

  ```python
  loaders = [
      TextLoader("data/langchain.md"),
      TextLoader("data/langchain2.md"),
  ]
  ```

  - `TextLoader` reads and loads text documents from specified file paths.
  - Documents are appended to a list `docs` for further processing.

---

## **Step 2: Split Documents into Large Chunks**

- **Purpose:** Chunk documents for better indexing and processing.
- **Code:**

  ```python
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
  docs = text_splitter.split_documents(docs)
  ```

  - **Why Split?** Chunking prevents memory overload and ensures more granular search/retrieval.
  - `RecursiveCharacterTextSplitter` splits documents into manageable pieces, each about 2000 characters long.

---

## **Step 3: Initialize the Vectorstore**

- **Purpose:** Create a **vectorstore** to index document chunks using embeddings for similarity search.
- **Code:**

  ```python
  vectorstore = Chroma(
      collection_name="full_documents",
      embedding_function=OllamaEmbeddings(model='snowflake-arctic-embed:33m')
  )
  ```

  - **Vectorstore:** Stores document embeddings (numerical representations of text).
  - **Embedding Function:** Converts text into embeddings using the `OllamaEmbeddings` model.

---

## **Step 4: Create Smaller Chunks**

- **Purpose:** Split documents further into finer-grained chunks and associate metadata.
- **Code:**

  ```python
  child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
  ```

  - Documents are split into smaller chunks of ~400 characters each.
  - Metadata (`doc_id`) is added to associate chunks with parent documents.

---

## **Step 5: Add Documents to Vectorstore**

- **Purpose:** Add the smaller chunks to the vectorstore for efficient retrieval.
- **Code:**

  ```python
  retriever.vectorstore.add_documents(sub_docs)
  retriever.docstore.mset(list(zip(doc_ids, docs)))
  ```

  - Document chunks are indexed in the vectorstore along with their metadata for quick retrieval.

---

## **Step 6: Perform a Similarity Search**

- **Purpose:** Query the vectorstore for documents similar to a given query.
- **Code:**

  ```python
  retriever.vectorstore.similarity_search("LangChain")[0:2]
  ```

  - The query `"LangChain"` retrieves the most relevant chunks using similarity search.

---

## **Step 7: Multi-Modal Retrieval**

- **Purpose:** Use a diversity-enhanced retrieval method like **Maximal Marginal Relevance (MMR)**.
- **Code:**

  ```python
  retriever.search_type = SearchType.mmr
  retriever.invoke("LangChain")[0:2]
  ```

  - MMR balances relevance with diversity to avoid redundancy in retrieved results.

---

## **Step 8: Summarize Documents**

- **Purpose:** Use a language model to summarize document content.
- **Code:**

  ```python
  chain = (
      {"doc": lambda x: x.page_content}
      | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
      | llm
      | StrOutputParser()
  )
  summaries = chain.batch(docs, {"max_concurrency": 5})
  ```

  - **Prompt Template:** Provides a summarization prompt for the model.
  - Summaries are generated in parallel for efficiency.

---

## **Step 9: Add Summaries to Vectorstore**

- **Purpose:** Store summaries in a separate vectorstore for enhanced retrieval.
- **Code:**

  ```python
  vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())
  retriever.vectorstore.add_documents(summary_docs)
  ```

  - Summaries are stored with metadata for efficient retrieval during queries.

---

## **Step 10: Retrieve Summaries**

- **Purpose:** Query the summaries vectorstore to get concise and relevant results.
- **Code:**

  ```python
  sub_docs = retriever.vectorstore.similarity_search("LangChain")
  retrieved_docs = retriever.invoke("LangChain")
  ```

  - Perform similarity search and retrieval on the summaries for quick insights.

