"""
retrieve.py
───────────
Phase 2 – Step 1: Given a user question,
  1. Embed the question using the SAME model used during indexing
  2. Query ChromaDB for the top-k most similar chunks
  3. Return those chunks for generate.py to use

Why same model?
  Embeddings only make sense when compared in the same vector space.
  If you index with all-MiniLM-L6-v2, you MUST query with it too.

Usage:
  from src.retrieve import retrieve_chunks
  chunks = retrieve_chunks("Is RTX 4070 compatible with B550 motherboard?")
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from src.config.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    TOP_K,
)


# ── Lazy singletons ───────────────────────────────────────────────────────────
# Load the model and DB client once — reused across calls in the same session.
# This avoids reloading the model on every query (slow).

_embedding_model: SentenceTransformer | None = None
_chroma_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[retrieve] Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[retrieve] Connected to ChromaDB collection "
              f"'{CHROMA_COLLECTION}' ({_chroma_collection.count()} chunks)")
    return _chroma_collection


# ── Core retrieval function ───────────────────────────────────────────────────

def retrieve_chunks(question: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the top-k most semantically similar chunks to the question.

    Args:
        question : User's natural language question
        top_k    : Number of chunks to return (default from config)

    Returns:
        List of chunk dicts, ranked by relevance (best first):
        [
          {
            "text":       <chunk text>,
            "source":     <PDF path or URL>,
            "type":       "pdf" | "web",
            "similarity": <cosine similarity score 0-1>
          },
          ...
        ]
    """
    if not question.strip():
        raise ValueError("[retrieve] Question cannot be empty.")

    # 1. Embed the question
    model          = get_embedding_model()
    question_vector = model.encode(question).tolist()

    # 2. Query ChromaDB
    collection = get_chroma_collection()

    if collection.count() == 0:
        print("[retrieve] WARNING: Vector store is empty. Run embed.py first.")
        return []

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=min(top_k, collection.count()),  # can't ask for more than exists
        include=["documents", "metadatas", "distances"],
    )

    # 3. Parse results into clean dicts
    # ChromaDB returns distances (lower = more similar for cosine space)
    # We convert distance → similarity score: similarity = 1 - distance
    chunks = []
    documents = results["documents"][0]   # [0] because we sent 1 query
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, distance in zip(documents, metadatas, distances):
        similarity = round(1 - distance, 4)
        chunks.append({
            "text":       text,
            "source":     meta.get("source", "unknown"),
            "type":       meta.get("type", "unknown"),
            "similarity": similarity,
        })

    return chunks


# ── Optional: filter by source type ──────────────────────────────────────────

def retrieve_chunks_filtered(
    question: str,
    source_type: str | None = None,   # "pdf" | "web" | None (all)
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Same as retrieve_chunks but optionally filter results by source type.
    Useful when you want answers only from PDFs (official specs)
    or only from web pages (pricing, availability).
    """
    chunks = retrieve_chunks(question, top_k=top_k * 2)  # fetch extra, then filter

    if source_type:
        chunks = [c for c in chunks if c["type"] == source_type]

    return chunks[:top_k]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "Is the RTX 4070 compatible with a B550 motherboard?",
        "What is the return policy for defective GPUs?",
        "Compare Corsair Vengeance vs G.Skill Trident DDR5 RAM",
    ]

    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question : {question}")
        print(f"{'='*60}")

        chunks = retrieve_chunks(question)

        if not chunks:
            print("No chunks retrieved. Make sure you ran embed.py first.")
            continue

        for i, chunk in enumerate(chunks, start=1):
            print(f"\n[Chunk {i}] Similarity: {chunk['similarity']} "
                  f"| Source: {chunk['source']}")
            print(f"Preview : {chunk['text'][:200].replace(chr(10), ' ')}...")