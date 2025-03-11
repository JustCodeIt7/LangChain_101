# %%
import inspect

import tiktoken
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationKGMemory,
    ConversationSummaryMemory,
)
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from rich import print as pp

# %%

llm = ChatOllama(model="llama3.2:1b")


# %%
def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.invoke(query)

        print(f"Spent a total of {cb.total_tokens} tokens")
    return result


# %%


# %%
conversation = ConversationChain(
    llm=llm,
)
# conversation = RunnableWithMessageHistory(
#     llm=llm,
# )

pp(conversation.prompt.template)

# %%

print(inspect.getsource(conversation._call), inspect.getsource(conversation.apply))
# %%

print(inspect.getsource(LLMChain._call), inspect.getsource(LLMChain.apply))
# %%
conversation_buf = ConversationChain(llm=llm, memory=ConversationBufferMemory())
pp(conversation_buf.invoke("Hello AI, how are you today?"))

# %%

count_tokens(
    conversation_buf,
    "I'm interested in learning about the advancements in AI technology.",
)

count_tokens(conversation_buf, "Can you explain how AI can be used in healthcare?")

count_tokens(
    conversation_buf, "What are some challenges faced by AI in real-world applications?"
)

count_tokens(conversation_buf, "Can you remind me what we discussed about AI earlier?")
# %%
print(conversation_buf.memory.buffer)

# %%

conversation_sum = ConversationChain(llm=llm, memory=ConversationSummaryMemory(llm=llm))
# %%

print(conversation_sum.memory.prompt.template)
# %%

count_tokens(conversation_sum, "Hello AI, how are you today?")

count_tokens(
    conversation_sum,
    "I'm interested in learning about the advancements in AI technology.",
)

count_tokens(conversation_sum, "Can you explain how AI can be used in healthcare?")

count_tokens(
    conversation_sum, "What are some challenges faced by AI in real-world applications?"
)

count_tokens(conversation_sum, "Can you remind me what we discussed about AI earlier?")

print(conversation_sum.memory.buffer)
# %%


# initialize tokenizer
tokenizer = tiktoken.encoding_for_model("text-davinci-003")

# show number of tokens for the memory used by each memory type
print(
    f"Buffer memory conversation length: {len(tokenizer.encode(conversation_buf.memory.buffer))}\n"
    f"Summary memory conversation length: {len(tokenizer.encode(conversation_sum.memory.buffer))}"
)
# %%


conversation_bufw = ConversationChain(
    llm=llm, memory=ConversationBufferWindowMemory(k=1)
)

count_tokens(conversation_bufw, "Hello AI, how are you today?")

count_tokens(
    conversation_bufw,
    "I'm interested in learning about the advancements in AI technology.",
)

count_tokens(conversation_bufw, "Can you explain how AI can be used in healthcare?")

count_tokens(
    conversation_bufw,
    "What are some challenges faced by AI in real-world applications?",
)

count_tokens(conversation_bufw, "Can you remind me what we discussed about AI earlier?")
# %%


bufw_history = conversation_bufw.memory.load_memory_variables(inputs=[])["history"]

print(bufw_history)
# %%


print(
    f"Buffer memory conversation length: {len(tokenizer.encode(conversation_buf.memory.buffer))}\n"
    f"Summary memory conversation length: {len(tokenizer.encode(conversation_sum.memory.buffer))}\n"
    f"Buffer window memory conversation length: {len(tokenizer.encode(bufw_history))}"
)
# %%
# !pip install -qU networkx

conversation_kg = ConversationChain(llm=llm, memory=ConversationKGMemory(llm=llm))

count_tokens(
    conversation_kg, "I love programming in Python and exploring new technologies!"
)
# %%

conversation_kg.memory.kg.get_triples()
# %%
