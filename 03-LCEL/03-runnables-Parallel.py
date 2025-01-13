#%%
import timeit
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama , OllamaEmbeddings
from rich import print
import timeit
# %%

model = ChatOllama(model="llama3.2:1b")
embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:33m")
# Invoke the model with a sample input to test its functionality
print(model.invoke("harrison worked at kensho"))
#%%

# Create a vector store using FAISS and Ollama's embeddings from a list of texts
vectorstore = FAISS.from_texts(["harrison worked at kensho"], embedding=embeddings)

# Convert the vector store into a retriever to fetch relevant documents based on queries
retriever = vectorstore.as_retriever()

# Define a template for the chat prompt that includes context and question placeholders
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
#%%

# Create a chat prompt template from the defined template string
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
#%%

# Print the retriever object to verify its setup
print(retriever)

# Create a retrieval chain that combines the retriever, prompt template, model, and output parser
retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
)
#%%

# Invoke the retrieval chain with a sample question to test its functionality
print(retrieval_chain.invoke("where did harrison work?"))


#%%
# Using itemgetter as shorthand
print('Using itemgetter as shorthand')
# %%
from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=embeddings
)
retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer in the following language: {language}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "language": itemgetter("language"),
    }
    | prompt
    | model
    | StrOutputParser()
)

print(chain.invoke({"question": "where did harrison work", "language": "italian"}))


# %%
# Parallelize steps
print('Parallelize steps')
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


joke_chain = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
poem_chain = (
    ChatPromptTemplate.from_template("write a 2-line poem about {topic}") | model
)

map_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

print(map_chain.invoke({"topic": "bear"}))
# %%
print(timeit.timeit('map_chain.invoke({"topic": "bear"})', globals=globals(), number=1))
# %%
print(timeit.timeit('map_chain.invoke({"topic": "bear"})', globals=globals(), number=1))

# %%
print(timeit.timeit('map_chain.invoke({"topic": "bear"})', globals=globals(), number=1))
# %%
