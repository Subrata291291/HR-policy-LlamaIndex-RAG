from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag.pipeline import ask as run_rag
from app.rag.memory import clear_history


app = FastAPI(
    title="HR Policy RAG API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=1000,
    )

    session_id: str = Field(
        min_length=1,
        max_length=100,
    )


class ClearRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
    )


class AskResponse(BaseModel):
    answer: str
    source: str | None
    retrieval_score: float | None
    evidence: str | None


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "HR Policy RAG API",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    try:
        result = run_rag(
            request.query,
            session_id=request.session_id,
            debug=True,
        )

        return result

    except Exception as e:
        print(
            f"ASK ERROR: {type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        )


@app.post("/clear")
def clear_conversation(request: ClearRequest):
    try:
        clear_history(request.session_id)

        return {
            "status": "ok",
            "message": "Conversation memory cleared.",
        }

    except Exception as e:
        print(
            f"CLEAR ERROR: {type(e).__name__}: {e}",
            flush=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Could not clear conversation memory.",
        )