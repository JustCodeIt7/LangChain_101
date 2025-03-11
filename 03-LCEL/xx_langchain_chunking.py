from langchain.document_loaders import ReadTheDocsLoader

loader = ReadTheDocsLoader("rtdocs")
docs = loader.load()
len(docs)


print(docs[0].page_content)

print(docs[5].page_content)

"""We can also find the source of each document:"""

docs[5].metadata["source"].replace("rtdocs/", "https://")

"""Looks good, we need to also consider the length of each page with respect to the number of tokens that will reasonably fit within the window of the latest LLMs. We will use `gpt-3.5-turbo` as an example.

To count the number of tokens that `gpt-3.5-turbo` will use for some text we need to initialize the `tiktoken` tokenizer.
"""

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


"""Note that for the tokenizer we defined the encoder as `"cl100k_base"`. This is a specific tiktoken encoder which is used by `gpt-3.5-turbo`, as well as `gpt-4`, and `text-embedding-ada-002` which are models supported by OpenAI at the time of this writing. Other encoders may be available, but are used with models that are now deprecated by OpenAI.

You can find more details in the [Tiktoken `model.py` script](https://github.com/openai/tiktoken/blob/main/tiktoken/model.py), or using `tiktoken.encoding_for_model`:
"""

tiktoken.encoding_for_model("gpt-3.5-turbo")

"""Using the `tiktoken_len` function, let's count and visualize the number of tokens across our webpages."""

token_counts = [tiktoken_len(doc.page_content) for doc in docs]

"""Let's see `min`, average, and `max` values:"""

print(f"""Min: {min(token_counts)}
Avg: {int(sum(token_counts) / len(token_counts))}
Max: {max(token_counts)}""")

"""Now visualize:"""

import matplotlib.pyplot as plt
import seaborn as sns

# set style and color palette for the plot
sns.set_style("whitegrid")
sns.set_palette("muted")

# create histogram
plt.figure(figsize=(12, 6))
sns.histplot(token_counts, kde=False, bins=50)

# customize the plot info
plt.title("Token Counts Histogram")
plt.xlabel("Token Count")
plt.ylabel("Frequency")

plt.show()

"""The vast majority of pages seem to contain a lower number of tokens. But our limits for the number of tokens to add to each chunk is actually smaller than some of the smaller pages. But, how do we decide what this number should be?

### Chunking the Text

At the time of writing, `gpt-3.5-turbo` supports a context window of 4096 tokens — that means that input tokens + generated ( / completion) output tokens, cannot total more than 4096 without hitting an error.

So we 100% need to keep below this. If we assume a very safe margin of ~2000 tokens for the input prompt into `gpt-3.5-turbo`, leaving ~2000 tokens for conversation history and completion.

With this ~2000 token limit we may want to include *five* snippets of relevant information, meaning each snippet can be no more than **400** token long.

To create these snippets we use the `RecursiveCharacterTextSplitter` from LangChain. To measure the length of snippets we also need a *length function*. This is a function that consumes text, counts the number of tokens within the text (after tokenization using the `gpt-3.5-turbo` tokenizer), and returns that number. We define it like so:

With the length function defined we can initialize our `RecursiveCharacterTextSplitter` object like so:
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,  # number of tokens overlap between chunks
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)

"""Then we split the text for a document like so:"""

chunks = text_splitter.split_text(docs[5].page_content)
len(chunks)

tiktoken_len(chunks[0]), tiktoken_len(chunks[1])

"""For `docs[5]` we created `2` chunks of token length `346` and `247`.

This is for a single document, we need to do this over all of our documents. While we iterate through the docs to create these chunks we will reformat them into a format that looks like:

```json
[
    {
        "id": "abc-0",
        "text": "some important document text",
        "source": "https://langchain.readthedocs.io/en/latest/glossary.html"
    },
    {
        "id": "abc-1",
        "text": "the next chunk of important document text",
        "source": "https://langchain.readthedocs.io/en/latest/glossary.html"
    }
    ...
]
```

The `"id"` will be created based on the URL of the text + it's chunk number.
"""

import hashlib

m = hashlib.md5()  # this will convert URL into unique ID

url = docs[5].metadata["source"].replace("rtdocs/", "https://")
print(url)

# convert URL to unique ID
m.update(url.encode("utf-8"))
uid = m.hexdigest()[:12]
print(uid)

"""Then use the `uid` alongside chunk number and actual `url` to create the format needed:"""

data = [
    {"id": f"{uid}-{i}", "text": chunk, "source": url} for i, chunk in enumerate(chunks)
]
data

"""Now we repeat the same logic across our full dataset:"""

from tqdm.auto import tqdm

documents = []

for doc in tqdm(docs):
    url = doc.metadata["source"].replace("rtdocs/", "https://")
    m.update(url.encode("utf-8"))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc.page_content)
    for i, chunk in enumerate(chunks):
        documents.append({"id": f"{uid}-{i}", "text": chunk, "source": url})

len(documents)

"""We're now left with `2201` documents. We can save them to a JSON lines (`.jsonl`) file like so:"""

import json

with open("train.jsonl", "w") as f:
    for doc in documents:
        f.write(json.dumps(doc) + "\n")

"""To load the data from file we'd write:"""

documents = []

with open("train.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

len(documents)

documents[0]
