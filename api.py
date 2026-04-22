"""
api.py
──────
FastAPI REST API for GameGear AI.
Exposes the RAG pipeline as HTTP endpoints so any frontend,
mobile app, or external service can query it.

Endpoints:
  GET  /               → health check
  GET  /status         → index stats (chunk count, collection name)
  POST /ask            → main RAG query
  POST /index          → trigger re-indexing
  GET  /history        → retrieve session chat history
  DELETE /history      → clear session chat history

Run with:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi             import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic            import BaseModel, Field
from contextlib          import asynccontextmanager
import chromadb
from chromadb.config     import Settings
from datetime            import datetime

from src.pipeline.pipeline import ask
from src.retrieval.embed    import build_index
from src.config   import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
)


# ── In-memory chat history (per server session) ───────────────────────────────
# For production, swap this with Redis or a database.
_chat_history: list[dict] = []


# ── Lifespan: warm up the embedding model on startup ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-load the embedding model when the server starts.
    This avoids a slow first request — the model is ready immediately.
    """
    print("[api] Warming up embedding model...")
    from src.retrieval.retrieve import get_embedding_model
    get_embedding_model()
    print("[api] Ready.")
    yield


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "GameGear AI",
    description = "RAG-powered gaming components assistant API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# Allow requests from Streamlit (localhost:8501) and any other local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten this in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question : str  = Field(..., min_length=3, example="Is RTX 4070 good for 1440p?")
    top_k    : int  = Field(default=TOP_K, ge=1, le=20)

class ChunkResult(BaseModel):
    text       : str
    source     : str
    type       : str
    similarity : float

class AskResponse(BaseModel):
    question   : str
    answer     : str
    sources    : list[str]
    chunks     : list[ChunkResult]
    model_used : str
    timestamp  : str

class StatusResponse(BaseModel):
    status          : str
    chunk_count     : int
    collection_name : str
    embedding_model : str
    llm_model       : str

class IndexResponse(BaseModel):
    success : bool
    message : str

class HistoryItem(BaseModel):
    role      : str   # "user" | "assistant"
    content   : str
    timestamp : str


# ── Helper ────────────────────────────────────────────────────────────────────

def get_chunk_count() -> int:
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        return col.count()
    except Exception:
        return 0


def now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Basic health check — confirms the server is running."""
    return {"status": "ok", "service": "GameGear AI", "version": "1.0.0"}


@app.get("/status", response_model=StatusResponse, tags=["Health"])
def status():
    """
    Returns index stats and model configuration.
    Use this to verify the vector store is populated before querying.
    """
    return StatusResponse(
        status          = "ready" if get_chunk_count() > 0 else "empty",
        chunk_count     = get_chunk_count(),
        collection_name = CHROMA_COLLECTION,
        embedding_model = EMBEDDING_MODEL,
        llm_model       = LLM_MODEL,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(body: AskRequest):
    """
    Main RAG endpoint.

    Embeds the question, retrieves relevant chunks from ChromaDB,
    and generates a grounded answer via Groq LLM.

    - **question**: Natural language question about gaming components
    - **top_k**: Number of chunks to retrieve (1–20, default 5)
    """
    if get_chunk_count() == 0:
        raise HTTPException(
            status_code = 503,
            detail      = "Vector store is empty. Run POST /index first."
        )

    try:
        result = ask(body.question, top_k=body.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    timestamp = now()

    # Save to in-memory history
    _chat_history.append({"role": "user",      "content": body.question,    "timestamp": timestamp})
    _chat_history.append({"role": "assistant", "content": result["answer"], "timestamp": timestamp})

    return AskResponse(
        question   = result["question"],
        answer     = result["answer"],
        sources    = result["sources"],
        chunks     = [ChunkResult(**c) for c in result["chunks"]],
        model_used = LLM_MODEL,
        timestamp  = timestamp,
    )


@app.post("/index", response_model=IndexResponse, tags=["Indexing"])
def trigger_index():
    """
    Trigger a full re-index of all documents.
    Loads PDFs from data/pdfs/ and scrapes URLs from data/urls.txt.
    Safe to re-run — uses upsert so no duplicates are created.
    """
    try:
        build_index()
        return IndexResponse(
            success = True,
            message = f"Index built successfully. {get_chunk_count()} chunks stored."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.get("/history", response_model=list[HistoryItem], tags=["History"])
def get_history():
    """Returns the full in-memory chat history for this server session."""
    return [HistoryItem(**item) for item in _chat_history]


@app.delete("/history", tags=["History"])
def clear_history():
    """Clears the in-memory chat history."""
    _chat_history.clear()
    return {"message": "Chat history cleared."}