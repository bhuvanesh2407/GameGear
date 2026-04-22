"""
embed.py
────────
Phase 1 – Step 2: Take raw documents from ingest.py and:
  1. Split them into overlapping chunks (LangChain RecursiveCharacterTextSplitter)
  2. Generate vector embeddings  (sentence-transformers, runs on CPU)
  3. Store chunks + vectors in ChromaDB (persistent local store)

Usage:
  # Run the full indexing pipeline
  python -m src.embed

  # Or import and call programmatically
  from src.embed import build_index
  build_index()
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid

from src.config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.ingestion.ingest import load_all_documents


# ── Step 1: Chunking ──────────────────────────────────────────────────────────

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Split each document's text into overlapping chunks.

    Why RecursiveCharacterTextSplitter?
    It tries to split on paragraph breaks → sentences → words in order,
    so chunks stay semantically coherent rather than cutting mid-sentence.

    Returns list of chunk dicts:
      { "text": ..., "source": ..., "type": ..., "chunk_id": ... }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # priority order
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["text"])

        for i, text in enumerate(split_texts):
            chunks.append({
                "text":     text,
                "source":   doc["source"],
                "type":     doc["type"],
                "chunk_id": str(uuid.uuid4()),   # unique ID for ChromaDB
            })

    print(f"[embed] Total chunks created: {len(chunks)}")
    return chunks


# ── Step 2: Embedding ─────────────────────────────────────────────────────────

def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer model.
    Downloads once, cached locally by the library on subsequent runs.
    """
    print(f"[embed] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """
    Convert chunk texts into dense vectors.
    Returns a list of embedding vectors (one per chunk).
    """
    texts = [chunk["text"] for chunk in chunks]
    print(f"[embed] Embedding {len(texts)} chunks...")

    # batch_size=32 is efficient for CPU; increase if you have GPU
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings.tolist()


# ── Step 3: Store in ChromaDB ─────────────────────────────────────────────────

def get_chroma_collection():
    """
    Connect to (or create) a persistent ChromaDB collection.
    Data is saved to CHROMA_DIR on disk — survives restarts.
    """
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for retrieval
    )

    return collection


def store_in_chroma(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection,
) -> None:
    """
    Upsert chunks + embeddings into ChromaDB.
    Upsert = insert if new, update if chunk_id already exists.
    This makes re-indexing safe — no duplicates.
    """
    ids        = [c["chunk_id"] for c in chunks]
    documents  = [c["text"]     for c in chunks]
    metadatas  = [{"source": c["source"], "type": c["type"]} for c in chunks]

    # ChromaDB accepts batches of up to 5000; we batch at 500 to be safe
    batch_size = 500
    total      = len(ids)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        collection.upsert(
            ids        = ids[start:end],
            documents  = documents[start:end],
            embeddings = embeddings[start:end],
            metadatas  = metadatas[start:end],
        )
        print(f"[embed] Stored chunks {start+1}–{end} / {total}")

    print(f"[embed] ChromaDB collection '{CHROMA_COLLECTION}' now has "
          f"{collection.count()} total chunks.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_index() -> None:
    """
    Full Phase 1 pipeline:
      load → chunk → embed → store
    Run this once to build (or refresh) the vector index.
    """
    print("\n" + "="*60)
    print("  GameGear AI — Building Index (Phase 1)")
    print("="*60 + "\n")

    # 1. Load
    documents = load_all_documents()
    if not documents:
        print("[embed] No documents found. Add PDFs to data/pdfs/ "
              "or URLs to data/urls.txt and re-run.")
        return

    # 2. Chunk
    chunks = chunk_documents(documents)
    if not chunks:
        print("[embed] Chunking produced no output. Check document content.")
        return

    # 3. Embed
    model      = get_embedding_model()
    embeddings = embed_chunks(chunks, model)

    # 4. Store
    collection = get_chroma_collection()
    store_in_chroma(chunks, embeddings, collection)

    print("\n[embed] ✓ Index built successfully!")
    print(f"[embed] Vector store location: {CHROMA_DIR}")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_index()