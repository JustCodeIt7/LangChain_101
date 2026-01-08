# 2. Validation Middleware Example

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ValidationError
from typing import Type

class ValidationMiddleware:
    def __init__(self, runnable: Runnable, input_model: Type[BaseModel], output_model: Type[BaseModel]):
        self.runnable = runnable
        self.input_model = input_model
        self.output_model = output_model

    def __call__(self, *args, **kwargs):
        # Validate input
        try:
            self.input_model.model_validate(args[0])
            print("Input validation successful.")
        except ValidationError as e:
            print(f"Input validation failed: {e}")
            return None # Or raise an error

        result = self.runnable.invoke(*args, **kwargs)

        # Validate output
        try:
            self.output_model.model_validate({"content": result})
            print("Output validation successful.")
        except ValidationError as e:
            print(f"Output validation failed: {e}")
            return None # Or raise an error
            
        return result

# Pydantic models for validation
class InputModel(BaseModel):
    name: str

class OutputModel(BaseModel):
    content: str

# Example Usage
if __name__ == "__main__":
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser

    # Simple chain
    prompt = ChatPromptTemplate.from_template("Say hi to {name}")
    model = ChatOpenAI()
    parser = StrOutputParser()
    chain = prompt | model | parser

    # Wrap the chain with the validation middleware
    validation_chain = ValidationMiddleware(chain, InputModel, OutputModel)

    # Invoke the chain with valid input
    print("--- Running with valid input ---")
    validation_chain.invoke({"name": "Jim"})

    # Invoke the chain with invalid input
    print("\n--- Running with invalid input ---")
    validation_chain.invoke({"name": 123})
