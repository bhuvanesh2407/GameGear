"""
app.py
──────
Streamlit chat interface for GameGear AI.
Provides a conversational UI where users can ask questions about
gaming components and get grounded answers with sources shown.

Run with:
  streamlit run app.py
"""

import streamlit as st
from src.pipeline.pipeline import ask
from src.retrieval.embed import build_index
from src.config.config   import CHROMA_DIR, CHROMA_COLLECTION

import chromadb
from chromadb.config import Settings


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "GameGear AI",
    page_icon  = "🎮",
    layout     = "centered",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_chunk_count() -> int:
    """Return how many chunks are currently indexed in ChromaDB."""
    try:
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection(CHROMA_COLLECTION)
        return col.count()
    except Exception:
        return 0


def render_sources(sources: list[str]) -> None:
    """Render a collapsible sources section below each answer."""
    if not sources:
        return
    with st.expander("📎 Sources used", expanded=False):
        for src in sources:
            # Show as a clickable link if it's a URL, plain text if it's a file
            if src.startswith("http"):
                st.markdown(f"- [{src}]({src})")
            else:
                st.markdown(f"- `{src}`")


def render_chunk_previews(chunks: list[dict]) -> None:
    """Render a collapsible section showing the raw retrieved chunks."""
    if not chunks:
        return
    with st.expander(f"🔍 Retrieved chunks ({len(chunks)})", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            st.markdown(f"**Chunk {i}** — similarity: `{chunk['similarity']}`  \n"
                        f"*Source: {chunk['source']}*")
            st.caption(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))
            if i < len(chunks):
                st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/controller.png", width=64)
    st.title("GameGear AI")
    st.caption("RAG-powered gaming components assistant")

    st.divider()

    # Index status
    chunk_count = get_chunk_count()
    if chunk_count > 0:
        st.success(f"✅ Index ready — {chunk_count} chunks loaded")
    else:
        st.warning("⚠️ Index is empty")
        st.caption("Add PDFs to `data/pdfs/` and URLs to `data/urls.txt`, "
                   "then click the button below.")

    # Re-index button
    if st.button("🔄 Build / Refresh Index", use_container_width=True):
        with st.spinner("Indexing documents... this may take a minute."):
            try:
                build_index()
                st.success("Index built successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Indexing failed: {e}")

    st.divider()

    # Settings
    st.subheader("⚙️ Settings")
    top_k = st.slider(
        label   = "Chunks to retrieve (top-k)",
        min_value = 1,
        max_value = 10,
        value   = 5,
        help    = "More chunks = more context for the LLM, but slower."
    )
    show_chunks = st.toggle("Show retrieved chunks", value=False)

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Groq · Streamlit")


# ── Main UI ───────────────────────────────────────────────────────────────────

st.title("🎮 GameGear AI")
st.caption("Ask me anything about gaming components — specs, compatibility, comparisons, policies.")

# Example questions
with st.expander("💡 Example questions", expanded=False):
    examples = [
        "Is the RTX 4070 good for 1440p gaming?",
        "What PSU wattage do I need for an RTX 4070 build?",
        "Compare Corsair Vengeance vs G.Skill Trident DDR5 RAM",
        "Is the RTX 4070 compatible with a B550 motherboard?",
        "What is the return policy for defective GPUs?",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True, key=f"ex_{i}"):
            st.session_state["prefill"] = example


# ── Chat history ──────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            render_sources(msg["sources"])
            if show_chunks and "chunks" in msg:
                render_chunk_previews(msg["chunks"])


# ── Chat input ────────────────────────────────────────────────────────────────

# Support pre-filled question from example buttons
prefill = st.session_state.pop("prefill", "")

user_input = st.chat_input(
    placeholder = "Ask about specs, compatibility, comparisons...",
)

# Use prefill if no direct input
question = user_input or prefill

if question:
    # Guard: don't answer if index is empty
    if get_chunk_count() == 0:
        st.warning("⚠️ The index is empty. Please build the index first using "
                   "the sidebar button.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Generate and show assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                result = ask(question, top_k=top_k)
                answer  = result["answer"]
                sources = result["sources"]
                chunks  = result["chunks"]

                st.markdown(answer)
                render_sources(sources)
                if show_chunks:
                    render_chunk_previews(chunks)

                # Save to history
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources,
                    "chunks":  chunks,
                })

            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": error_msg,
                })

# Clear chat button (bottom of sidebar)
with st.sidebar:
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()