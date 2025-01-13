#%%

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama
from rich import print as pp

# Define a prompt for generating book recommendations
prompt = ChatPromptTemplate.from_template(
    "Suggest some books for the genre {genre}"
)

# Use a different model
model = ChatOllama(model="llama3.2")

#%%

output_parser = StrOutputParser()

from langchain.chains import LLMChain

# Create a chain for book recommendations
chain = LLMChain(
    prompt=prompt,
    llm=model,
    output_parser=output_parser
)

#%%
# Run the chain
out = chain.run(genre="Science Fiction")
print(out)

#%%
# Using LCEL syntax
lcel_chain = prompt | model | output_parser

# Run the chain
out = lcel_chain.invoke({"genre": "Science Fiction"})
print(out)

#%%

# Example of a custom Runnable for mathematical operations
class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            return other(self.func(*args, **kwargs))
        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def subtract_three(x):
    return x - 3

def divide_by_two(x):
    return x / 2

subtract_three = Runnable(subtract_three)
divide_by_two = Runnable(divide_by_two)

chain = subtract_three | divide_by_two
print(chain(10))  # Output: 3.5

#%%

from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Create an embedding model
embedding = OllamaEmbeddings(model="snowflake-arctic-embed:33m")

# Create two vector stores with different texts
vecstore_a = InMemoryVectorStore.from_texts(
    ["Python is a popular programming language", "Python is used for web development"],
    embedding=embedding
)
vecstore_b = InMemoryVectorStore.from_texts(
    ["Python is great for data science", "Python is known for its simplicity"],
    embedding=embedding
)

#%%
"""### Creating retrievers"""

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

retriever_a = vecstore_a.as_retriever()
retriever_b = vecstore_b.as_retriever()

prompt_str = """Answer the following questions about Python:

Context: {context}

Question: {question}

Answer: """

prompt = ChatPromptTemplate.from_template(prompt_str)

retriever = RunnableParallel(
    {'context': retriever_a, 'question': RunnablePassthrough()},
)

chain = retriever | prompt | model | output_parser

out = chain.invoke("What is Python used for?")
print(out)

#%%

prompt_str = """Answer the question below using the context:

Context:
{context_a}
{context_b}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

retrieval = RunnableParallel(
    {
        "context_a": retriever_a, "context_b": retriever_b,
        "question": RunnablePassthrough()
    }
)

chain = retrieval | prompt | model | output_parser

out = chain.invoke("Why is Python popular?")
print(out)

#%%
# Example of mathematical operations using RunnableLambda
def calc_cube(x):
    return x ** 3

def calc_reciprocal(x):
    return 1 / x

from langchain_core.runnables import RunnableLambda

# Wrap the functions with RunnableLambda
calc_cube = RunnableLambda(calc_cube)
calc_reciprocal = RunnableLambda(calc_reciprocal)

chain = calc_cube | calc_reciprocal
print(chain.invoke(2))  # Output: 0.125

#%%

prompt_str = "Tell me a fun fact about {animal}"
prompt = ChatPromptTemplate.from_template(prompt_str)

chain = prompt | model | output_parser
print(chain.invoke({"animal": "Dolphins"}))

#%%

# Example of extracting specific information from the output
def extract_fun_fact(x):
    if "\n\n" in x:
        return "\n".join(x.split("\n\n")[1:])
    else:
        return x

get_fun_fact = RunnableLambda(extract_fun_fact)

chain = prompt | model | output_parser | get_fun_fact
print(chain.invoke({"animal": "Dolphins"}))