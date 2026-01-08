
# 1. Logging Middleware Example

from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input, Output
from typing import Any

class LoggingMiddleware:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, *args, **kwargs):
        print("--- Request ---")
        print(f"Input: {args[0]}")
        
        result = self.runnable.invoke(*args, **kwargs)
        
        print("--- Response ---")
        print(f"Output: {result}")
        
        return result

# Example Usage
if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    # Simple chain to greet someone
    prompt = ChatPromptTemplate.from_template("Say hi to {name}")
    model = ChatOpenAI()
    parser = StrOutputParser()
    chain = prompt | model | parser

    # Wrap the chain with the logging middleware
    logging_chain = LoggingMiddleware(chain)

    # Invoke the chain
    logging_chain.invoke({"name": "Jim"})
