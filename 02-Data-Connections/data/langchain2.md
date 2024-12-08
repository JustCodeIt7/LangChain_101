Title: Summarize Text | 🦜️🔗 LangChain

URL Source: https://python.langchain.com/docs/tutorials/summarization/

Markdown Content:
Suppose you have a set of documents (PDFs, Notion pages, customer questions, etc.) and you want to summarize the content.

LLMs are a great tool for this given their proficiency in understanding and synthesizing text.

In the context of [retrieval-augmented generation](https://python.langchain.com/docs/tutorials/rag/), summarizing text can help distill the information in a large number of retrieved documents to provide context for a LLM.

In this walkthrough we'll go over how to summarize content from multiple documents using LLMs.

![Image 11: Image description](https://python.langchain.com/assets/images/summarization_use_case_1-874f7b2c94f64216f1f967fb5aca7bc1.png)

Concepts[​](https://python.langchain.com/docs/tutorials/summarization/#concepts "Direct link to Concepts")
----------------------------------------------------------------------------------------------------------

Concepts we will cover are:

*   Using [language models](https://python.langchain.com/docs/concepts/chat_models/).
    
*   Using [document loaders](https://python.langchain.com/docs/concepts/document_loaders/), specifically the [WebBaseLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) to load content from an HTML webpage.
    
*   Two ways to summarize or otherwise combine documents.
    
    1.  [Stuff](https://python.langchain.com/docs/tutorials/summarization/#stuff), which simply concatenates documents into a prompt;
    2.  [Map-reduce](https://python.langchain.com/docs/tutorials/summarization/#map-reduce), for larger sets of documents. This splits documents into batches, summarizes those, and then summarizes the summaries.

Shorter, targeted guides on these strategies and others, including [iterative refinement](https://python.langchain.com/docs/how_to/summarize_refine/), can be found in the [how-to guides](https://python.langchain.com/docs/how_to/#summarization).

Setup[​](https://python.langchain.com/docs/tutorials/summarization/#setup "Direct link to Setup")
-------------------------------------------------------------------------------------------------

### Jupyter Notebook[​](https://python.langchain.com/docs/tutorials/summarization/#jupyter-notebook "Direct link to Jupyter Notebook")

This guide (and most of the other guides in the documentation) uses [Jupyter notebooks](https://jupyter.org/) and assumes the reader is as well. Jupyter notebooks are perfect for learning how to work with LLM systems because oftentimes things can go wrong (unexpected output, API down, etc) and going through guides in an interactive environment is a great way to better understand them.

This and other tutorials are perhaps most conveniently run in a Jupyter notebook. See [here](https://jupyter.org/install) for instructions on how to install.

### Installation[​](https://python.langchain.com/docs/tutorials/summarization/#installation "Direct link to Installation")

To install LangChain run:

*   Pip
*   Conda

For more details, see our [Installation guide](https://python.langchain.com/docs/how_to/installation/).

### LangSmith[​](https://python.langchain.com/docs/tutorials/summarization/#langsmith "Direct link to LangSmith")

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com/).

After you sign up at the link above, make sure to set your environment variables to start logging traces:

```
export LANGCHAIN_TRACING_V2="true"export LANGCHAIN_API_KEY="..."
```

Or, if in a notebook, you can set them with:

```
import getpassimport osos.environ["LANGCHAIN_TRACING_V2"] = "true"os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

Overview[​](https://python.langchain.com/docs/tutorials/summarization/#overview "Direct link to Overview")
----------------------------------------------------------------------------------------------------------

A central question for building a summarizer is how to pass your documents into the LLM's context window. Two common approaches for this are:

1.  `Stuff`: Simply "stuff" all your documents into a single prompt. This is the simplest approach (see [here](https://python.langchain.com/docs/how_to/summarize_stuff/) for more on the `create_stuff_documents_chain` constructor, which is used for this method).
    
2.  `Map-reduce`: Summarize each document on its own in a "map" step and then "reduce" the summaries into a final summary (see [here](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.map_reduce.MapReduceDocumentsChain.html) for more on the `MapReduceDocumentsChain`, which is used for this method).
    

Note that map-reduce is especially effective when understanding of a sub-document does not rely on preceding context. For example, when summarizing a corpus of many, shorter documents. In other cases, such as summarizing a novel or body of text with an inherent sequence, [iterative refinement](https://python.langchain.com/docs/how_to/summarize_refine/) may be more effective.

![Image 12: Image description](https://python.langchain.com/assets/images/summarization_use_case_2-f2a4d5d60980a79140085fb7f8043217.png)

Setup[​](https://python.langchain.com/docs/tutorials/summarization/#setup-1 "Direct link to Setup")
---------------------------------------------------------------------------------------------------

First set environment variables and install packages:

```
%pip install --upgrade --quiet tiktoken langchain langgraph beautifulsoup4 langchain-community# Set env var OPENAI_API_KEY or load from a .env file# import dotenv# dotenv.load_dotenv()
```

```
import osos.environ["LANGCHAIN_TRACING_V2"] = "true"
```

First we load in our documents. We will use [WebBaseLoader](https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) to load a blog post:

```
from langchain_community.document_loaders import WebBaseLoaderloader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")docs = loader.load()
```

Let's next select a LLM:

*   OpenAI
*   Anthropic
*   Azure
*   Google
*   AWS
*   Cohere
*   NVIDIA
*   FireworksAI
*   Groq
*   MistralAI
*   TogetherAI
*   Databricks

```
pip install -qU langchain-openai
```

```
import getpassimport osos.environ["OPENAI_API_KEY"] = getpass.getpass()from langchain_openai import ChatOpenAIllm = ChatOpenAI(model="gpt-4o-mini")
```

Stuff: summarize in a single LLM call[​](https://python.langchain.com/docs/tutorials/summarization/#stuff "Direct link to Stuff: summarize in a single LLM call")
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

We can use [create\_stuff\_documents\_chain](https://python.langchain.com/api_reference/langchain/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html), especially if using larger context window models such as:

*   128k token OpenAI `gpt-4o`
*   200k token Anthropic `claude-3-5-sonnet-20240620`

The chain will take a list of documents, insert them all into a prompt, and pass that prompt to an LLM:

```
from langchain.chains.combine_documents import create_stuff_documents_chainfrom langchain.chains.llm import LLMChainfrom langchain_core.prompts import ChatPromptTemplate# Define promptprompt = ChatPromptTemplate.from_messages(    [("system", "Write a concise summary of the following:\\n\\n{context}")])# Instantiate chainchain = create_stuff_documents_chain(llm, prompt)# Invoke chainresult = chain.invoke({"context": docs})print(result)
```
### Streaming[​](https://python.langchain.com/docs/tutorials/summarization/#streaming "Direct link to Streaming")

Note that we can also stream the result token-by-token:

```
for token in chain.stream({"context": docs}):    print(token, end="|")
```



### Go deeper[​](https://python.langchain.com/docs/tutorials/summarization/#go-deeper "Direct link to Go deeper")

*   You can easily customize the prompt.
*   You can easily try different LLMs, (e.g., [Claude](https://python.langchain.com/docs/integrations/chat/anthropic/)) via the `llm` parameter.

Map-Reduce: summarize long texts via parallelization[​](https://python.langchain.com/docs/tutorials/summarization/#map-reduce "Direct link to Map-Reduce: summarize long texts via parallelization")
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Let's unpack the map reduce approach. For this, we'll first map each document to an individual summary using an LLM. Then we'll reduce or consolidate those summaries into a single global summary.

Note that the map step is typically parallelized over the input documents.

[LangGraph](https://langchain-ai.github.io/langgraph/), built on top of `langchain-core`, supports [map-reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/) workflows and is well-suited to this problem:

*   LangGraph allows for individual steps (such as successive summarizations) to be streamed, allowing for greater control of execution;
*   LangGraph's [checkpointing](https://langchain-ai.github.io/langgraph/how-tos/persistence/) supports error recovery, extending with human-in-the-loop workflows, and easier incorporation into conversational applications.
*   The LangGraph implementation is straightforward to modify and extend, as we will see below.

### Map[​](https://python.langchain.com/docs/tutorials/summarization/#map "Direct link to Map")

Let's first define the prompt associated with the map step. We can use the same summarization prompt as in the `stuff` approach, above:

```
from langchain_core.prompts import ChatPromptTemplatemap_prompt = ChatPromptTemplate.from_messages(    [("system", "Write a concise summary of the following:\\n\\n{context}")])
```

We can also use the Prompt Hub to store and fetch prompts.

This will work with your [LangSmith API key](https://docs.smith.langchain.com/).

For example, see the map prompt [here](https://smith.langchain.com/hub/rlm/map-prompt).

```
from langchain import hubmap_prompt = hub.pull("rlm/map-prompt")
```

### Reduce[​](https://python.langchain.com/docs/tutorials/summarization/#reduce "Direct link to Reduce")

We also define a prompt that takes the document mapping results and reduces them into a single output.

```
# Also available via the hub: `hub.pull("rlm/reduce-prompt")`reduce_template = """The following is a set of summaries:{docs}Take these and distill it into a final, consolidated summaryof the main themes."""reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
```

### Orchestration via LangGraph[​](https://python.langchain.com/docs/tutorials/summarization/#orchestration-via-langgraph "Direct link to Orchestration via LangGraph")

Below we implement a simple application that maps the summarization step on a list of documents, then reduces them using the above prompts.

Map-reduce flows are particularly useful when texts are long compared to the context window of a LLM. For long texts, we need a mechanism that ensures that the context to be summarized in the reduce step does not exceed a model's context window size. Here we implement a recursive "collapsing" of the summaries: the inputs are partitioned based on a token limit, and summaries are generated of the partitions. This step is repeated until the total length of the summaries is within a desired limit, allowing for the summarization of arbitrary-length text.

First we chunk the blog post into smaller "sub documents" to be mapped:

```
from langchain_text_splitters import CharacterTextSplittertext_splitter = CharacterTextSplitter.from_tiktoken_encoder(    chunk_size=1000, chunk_overlap=0)split_docs = text_splitter.split_documents(docs)print(f"Generated {len(split_docs)} documents.")
```

```
Created a chunk of size 1003, which is longer than the specified 1000``````outputGenerated 14 documents.
```

Next, we define our graph. Note that we define an artificially low maximum token length of 1,000 tokens to illustrate the "collapsing" step.



LangGraph allows the graph structure to be plotted to help visualize its function:

```
from IPython.display import ImageImage(app.get_graph().draw_mermaid_png())
```

![Image 13](blob:https://python.langchain.com/e32009840dcd77fbf82e92b3fce23530)

When running the application, we can stream the graph to observe its sequence of steps. Below, we will simply print out the name of the step.

Note that because we have a loop in the graph, it can be helpful to specify a [recursion\_limit](https://langchain-ai.github.io/langgraph/reference/errors/#langgraph.errors.GraphRecursionError) on its execution. This will raise a specific error when the specified limit is exceeded.

```
async for step in app.astream(    {"contents": [doc.page_content for doc in split_docs]},    {"recursion_limit": 10},):    print(list(step.keys()))
```

```
['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['generate_summary']['collect_summaries']['collapse_summaries']['collapse_summaries']['generate_final_summary']
```



In the corresponding [LangSmith trace](https://smith.langchain.com/public/9d7b1d50-e1d6-44c9-9ab2-eabef621c883/r) we can see the individual LLM calls, grouped under their respective nodes.

### Go deeper[​](https://python.langchain.com/docs/tutorials/summarization/#go-deeper-1 "Direct link to Go deeper")

**Customization**

*   As shown above, you can customize the LLMs and prompts for map and reduce stages.

**Real-world use-case**

*   See [this blog post](https://blog.langchain.dev/llms-to-improve-documentation/) case-study on analyzing user interactions (questions about LangChain documentation)!
*   The blog post and associated [repo](https://github.com/mendableai/QA_clustering) also introduce clustering as a means of summarization.
*   This opens up another path beyond the `stuff` or `map-reduce` approaches that is worth considering.

![Image 14: Image description](https://python.langchain.com/assets/images/summarization_use_case_3-896f435bc48194ddaead73043027e16f.png)

Next steps[​](https://python.langchain.com/docs/tutorials/summarization/#next-steps "Direct link to Next steps")
----------------------------------------------------------------------------------------------------------------

We encourage you to check out the [how-to guides](https://python.langchain.com/docs/how_to/) for more detail on:

*   Other summarization strategies, such as [iterative refinement](https://python.langchain.com/docs/how_to/summarize_refine/)
*   Built-in [document loaders](https://python.langchain.com/docs/how_to/#document-loaders) and [text-splitters](https://python.langchain.com/docs/how_to/#text-splitters)
*   Integrating various combine-document chains into a [RAG application](https://python.langchain.com/docs/tutorials/rag/)
*   Incorporating retrieval into a [chatbot](https://python.langchain.com/docs/how_to/chatbots_retrieval/)

