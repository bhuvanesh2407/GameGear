"""
generate.py
───────────
Phase 2 – Step 2: Given a user question and retrieved chunks,
  1. Build an augmented prompt  (question + context chunks)
  2. Call Groq LLM              (free tier — Llama 3.3 70b by default)
  3. Return the grounded answer

Groq's Python client has the same interface as OpenAI's,
so this is easy to swap back if needed.

Usage:
  from src.generate import generate_answer
  answer = generate_answer(question="Is RTX 4070 good for 1440p?", chunks=[...])
"""

from groq import Groq
from src.config.config import GROQ_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are GameGear AI, a helpful and knowledgeable assistant \
for a gaming components store.

Your job is to answer customer questions accurately using ONLY the context \
provided below. The context comes from product manuals, spec sheets, and \
official product pages.

Rules:
- Answer based strictly on the provided context.
- If the context does not contain enough information, say:
  "I don't have enough information about that in my current knowledge base."
- Be concise, friendly, and technical when needed.
- If comparing products, use the specs from the context directly.
- Never make up specifications, prices, or compatibility claims.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Inject retrieved chunks into the prompt as numbered context blocks.
    Each block shows the source so the LLM can reference it.
    """
    if not chunks:
        context_text = "No relevant context found."
    else:
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "unknown")
            text   = chunk.get("text", "").strip()
            context_blocks.append(f"[Context {i} — Source: {source}]\n{text}")
        context_text = "\n\n".join(context_blocks)

    prompt = f"""Context:
{context_text}

---

Customer question: {question}

Answer:"""

    return prompt


# ── Groq LLM call ─────────────────────────────────────────────────────────────

def generate_answer(question: str, chunks: list[dict]) -> str:
    """
    Full generation step:
      1. Build the augmented prompt with retrieved chunks
      2. Send to Groq (free tier)
      3. Return the answer string

    Args:
        question : The user's natural language question
        chunks   : List of retrieved chunk dicts from retrieve.py
                   Each dict has keys: text, source, type

    Returns:
        answer (str) — grounded response from the LLM
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file: GROQ_API_KEY=your_key_here\n"
            "Get a free key at: https://console.groq.com"
        )

    client = Groq(api_key=GROQ_API_KEY)
    user_prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model       = LLM_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens  = MAX_TOKENS,
        temperature = TEMPERATURE,
    )

    answer = response.choices[0].message.content.strip()
    return answer


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test with dummy chunks — replace with real retrieved chunks
    test_chunks = [
        {
            "text": (
                "The NVIDIA GeForce RTX 4070 features 12GB GDDR6X memory, "
                "5888 CUDA cores, and a 192-bit memory bus. It delivers "
                "excellent 1440p gaming performance and supports DLSS 3."
            ),
            "source": "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070/",
            "type": "web",
        },
        {
            "text": (
                "Recommended PSU wattage for RTX 4070 builds is 650W. "
                "The card uses a single 16-pin power connector."
            ),
            "source": "rtx4070_spec_sheet.pdf",
            "type": "pdf",
        },
    ]

    question = "Is the RTX 4070 good for 1440p gaming and what PSU do I need?"

    print(f"Question: {question}\n")
    print("Generating answer via Groq...\n")

    answer = generate_answer(question=question, chunks=test_chunks)
    print(f"Answer:\n{answer}")