from typing import TypedDict

class State(TypedDict):
    query : str
    documents: str
    prompt: str 
    answer: str
