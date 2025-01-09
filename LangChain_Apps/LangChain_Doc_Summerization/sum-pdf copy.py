from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.mapreduce import MapReduceChain, ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

# Replace 'your_document.pdf' with the actual path to your PDF file
pdf_path = "your_document.pdf"

# Load the PDF document
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Initialize the language model (replace with your preferred model and API key if needed)
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")

# --- Summarization Chain ---
# Define the prompt for summarizing each chunk
map_prompt_summary = """Write a concise summary of the following:
"{text}"
"""
map_prompt_summary_template = PromptTemplate(template=map_prompt_summary, input_variables=["text"])

# Define the prompt for combining the summaries
reduce_prompt_summary = """Write a detailed summary of the following text, which are summaries of different sections of a larger document:
"{text}"
"""
reduce_prompt_summary_template = PromptTemplate(
    template=reduce_prompt_summary, input_variables=["text"]
)

# Create the map chain
summary_map_chain = load_summarize_chain(
    llm, chain_type="stuff", prompt=map_prompt_summary_template
)

# Create the reduce chain
summary_reduce_chain = load_summarize_chain(
    llm, chain_type="stuff", prompt=reduce_prompt_summary_template
)

# Create the map-reduce chain for summarization
summary_chain = MapReduceDocumentsChain(
    llm_chain=summary_map_chain,
    reduce_documents_chain=summary_reduce_chain,
    collapse_documents_chain=summary_reduce_chain,  # Optional: for very large number of initial documents
    document_variable_name="text",
    return_intermediate_steps=False,
)

# --- Key Points Extraction Chain ---
# Define the prompt for extracting key points from each chunk
map_prompt_keypoints = """Identify the main key points and important information in the following text:
"{text}"
"""
map_prompt_keypoints_template = PromptTemplate(
    template=map_prompt_keypoints, input_variables=["text"]
)

# Define the prompt for combining the key points
reduce_prompt_keypoints = """Combine the following key points into a structured list of the most important takeaways from the document:
"{text}"
"""
reduce_prompt_keypoints_template = PromptTemplate(
    template=reduce_prompt_keypoints, input_variables=["text"]
)

# Create the map chain for key points
keypoints_map_chain = load_summarize_chain(
    llm, chain_type="stuff", prompt=map_prompt_keypoints_template
)

# Create the reduce chain for key points
keypoints_reduce_chain = load_summarize_chain(
    llm, chain_type="stuff", prompt=reduce_prompt_keypoints_template
)

# Create the map-reduce chain for key points
keypoints_chain = MapReduceDocumentsChain(
    llm_chain=keypoints_map_chain,
    reduce_documents_chain=keypoints_reduce_chain,
    collapse_documents_chain=keypoints_reduce_chain,  # Optional: for very large number of initial documents
    document_variable_name="text",
    return_intermediate_steps=False,
)

# --- Run the chains ---
summary_output = summary_chain.run(texts)
keypoints_output = keypoints_chain.run(texts)

# --- Structure the output for easy navigation ---
print("# Detailed Summary")
print(summary_output)
print("\n# Key Points")
print(keypoints_output)

print("\n# Notes")
print(
    "This summary and key points were generated using a language model. While it aims to be comprehensive, it's always recommended to refer back to the original document for complete information."
)
