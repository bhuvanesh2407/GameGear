import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env file automatically

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR  = ROOT_DIR / "data"
PDF_DIR   = DATA_DIR / "pdfs"
URLS_FILE = DATA_DIR / "urls.txt"

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_DIR        = ROOT_DIR / "chroma_store"
CHROMA_COLLECTION = "gamegear_products"

# ── Embedding model (local, no API needed) ────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Chunking settings ─────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Groq LLM ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Model options (all free on Groq):
#   "llama-3.3-70b-versatile"  ← best quality, recommended
#   "llama-3.1-8b-instant"     ← fastest, lightweight
#   "mixtral-8x7b-32768"       ← largest context window (32k tokens)
LLM_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS  = 512
TEMPERATURE = 0.2   # low = factual, grounded answers

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5  # number of chunks to retrieve per query

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ── Ensure required directories exist on import ───────────────────────────────
for _dir in [PDF_DIR, CHROMA_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)