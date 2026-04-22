"""
pipeline.py
───────────
Wires retrieve.py + generate.py into a single ask() function.
This is what both app.py (Streamlit) and api.py (FastAPI) will import.

Flow:
  question → embed question → similarity search → augment prompt → Groq LLM → answer

Usage:
  from src.pipeline import ask
  answer = ask("Does the RTX 4070 support DLSS 3?")
"""

from src.retrieval.retrieve import retrieve_chunks
from src.generation.generate import generate_answer
from src.config.config   import TOP_K


def ask(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG query pipeline.

    Args:
        question : User's natural language question
        top_k    : Number of chunks to retrieve (default from config)

    Returns:
        {
            "question": <original question>,
            "answer":   <LLM generated answer>,
            "sources":  [<source URLs / file paths used>],
            "chunks":   [<full retrieved chunk dicts>]
        }
    """
    # Step 1 — Retrieve relevant chunks from vector store
    chunks = retrieve_chunks(question, top_k=top_k)

    # Step 2 — Generate grounded answer via Groq
    answer = generate_answer(question=question, chunks=chunks)

    # Deduplicated list of sources used
    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "question": question,
        "answer":   answer,
        "sources":  sources,
        "chunks":   chunks,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    questions = [
        "Is the RTX 4070 good for 1440p gaming?",
        "What PSU wattage do I need for an RTX 4070 build?",
        "What is the return policy for defective products?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        result = ask(q)
        print(f"Q: {result['question']}")
        print(f"\nA: {result['answer']}")
        print(f"\nSources used:")
        for src in result['sources']:
            print(f"  • {src}")