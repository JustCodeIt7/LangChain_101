# key_points_extractor.py
from langchain.schema import HumanMessage


def extract_key_points(llm, summary: str) -> str:
    key_points_prompt = (
        "Given the following summary, extract the most important points as bullet points. "
        "Ensure each point is concise and captures essential information:\n\n"
        f"Summary:\n{summary}\n\n"
        "Now produce a clear, concise list of key points:"
    )

    key_points_message = [HumanMessage(content=key_points_prompt)]
    key_points_response = llm.invoke(key_points_message)
    return key_points_response.content.strip()
