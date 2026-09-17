from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph import app as graph_app

# FastAPI App

app = FastAPI(
    title="HR Assistant using LangGraph",
    version="1.0"
)

# Request Model

class QueryRequest(BaseModel):
    query: str

# Response Model

class QueryResponse(BaseModel):
    answer: str

# Home Route

@app.get("/")
def home():
    return {"message": "HR Assistant is Running Successfully"}

# Ask Route

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    try:
        result = graph_app.invoke(
            {
                "query": request.query,
                "documents": "",
                "prompt": "",
                "answer": ""
            }
        )
        print(result)

        return QueryResponse(answer=result["answer"])

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )