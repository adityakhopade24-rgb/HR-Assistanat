import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from state import State
load_dotenv()

# Load Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load ChromaDB

db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Load LLM

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5
)

# Node 1 : Retrieve Documents

def retrieve_node(state: State):
    print("User Query:", state["query"])
    docs = db.similarity_search(state["query"], k=3)
    print("Retrieved Docs:", len(docs))
    context = "\n\n".join(doc.page_content for doc in docs)
    print(context[:500])
    return {
        "documents": context
    }

# Node 2 : Create Prompt

def prompt_node(state: State):
    prompt = f"""
                You are an HR Assistant.
                Answer ONLY using the given context.
                Context:
                {state['documents']}
                Question:
                {state['query']}
                If the answer is not available in the context,
                simply reply:

                I don't know based on the provided document.

                Give the answer in bullet points.
               """
    return {"prompt": prompt}

# Node 3 : LLM

def llm_node(state: State):
    print(state["prompt"])
    response = llm.invoke(state["prompt"])
    print(response.content)
    return {"answer": response.content}

# Node 4 : Output

def output_node(state: State):
    return {"answer": state["answer"]}

# Build Graph

graph = StateGraph(State)

graph.add_node("retrieve", retrieve_node)
graph.add_node("prompt", prompt_node)
graph.add_node("llm", llm_node)
graph.add_node("output", output_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "prompt")
graph.add_edge("prompt", "llm")
graph.add_edge("llm", "output")
graph.add_edge("output", END)

app = graph.compile()