# 🎮 GameGear AI — RAG-Powered Gaming Components Assistant

A production-style **Retrieval-Augmented Generation (RAG)** system built from scratch in Python.
Ask natural language questions about gaming components — specs, compatibility, comparisons, policies —
and get grounded, accurate answers backed by real product documents and web pages.

---

## What is RAG?

RAG (Retrieval-Augmented Generation) is a technique that stops LLMs from hallucinating by
grounding their answers in a trusted knowledge base. Instead of relying on the model's training
data alone, RAG retrieves the most relevant chunks from your own documents and injects them
into the prompt before generating an answer.

```
User question
     │
     ▼
Embed question ──► Similarity search (ChromaDB)
                         │
                    Top-k chunks
                         │
                         ▼
              Augmented prompt (question + context)
                         │
                         ▼
                  Groq LLM (Llama 3.3 70b)
                         │
                         ▼
              Grounded answer + sources
```

---

## Features

- 📄 **Multi-source ingestion** — PDFs (product manuals, spec sheets) + web scraping
- 🔍 **Semantic search** — local embeddings via `sentence-transformers`
- 🗄️ **Persistent vector store** — ChromaDB stored on disk, survives restarts
- ⚡ **Free LLM** — Groq API (Llama 3.3 70b), no OpenAI costs
- 💬 **Streamlit chat UI** — full conversation history, sources shown per answer
- 🚀 **FastAPI REST API** — `/ask`, `/index`, `/status`, `/history` endpoints
- 🖥️ **CPU-only** — no GPU required, runs on any laptop

---

## Project Structure

```
GameGear/
├── data/
│   ├── pdfs/           # Drop product PDFs here
│   └── urls.txt        # One URL per line to scrape
├── src/
│   ├── config
│       └── config.py   # All settings (paths, models, keys)
│   ├── generation
│   ├── ingestion
│       └── ingest.py
│   ├── pipeline        # PDF loader + web scraper
│   ├── retrieval
│       └── embed.py    # Chunking + embedding + ChromaDB storage
│       └── retrieve.py # Similarity search
│   ├── retrieve.py     # Similarity search
│   ├── generate.py     # Prompt builder + Groq LLM call
│   └── pipeline.py     # Wires retrieve + generate into ask()
├── app.py              # Streamlit chat UI
├── api.py              # FastAPI REST API
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/bhuvanesh2407/GameGear.git
cd GameGear
pip install -r requirements.txt
```

### 2. Set up your Groq API key

```bash
cp .env.example .env
# Edit .env and add your key — free at https://console.groq.com
```

### 3. Add your documents

```bash
# Drop PDF files into:
data/pdfs/

# Add product/FAQ page URLs to:
data/urls/urls.txt
```

### 4. Build the index

```bash
python -m src.retrieval.embed
```

### 5. Run the app

```bash
# Streamlit UI
streamlit run app.py

# OR FastAPI
uvicorn api:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | `/`        | Health check                       |
| GET    | `/status`  | Index stats and model info         |
| POST   | `/ask`     | Ask a question (main RAG endpoint) |
| POST   | `/index`   | Trigger re-indexing                |
| GET    | `/history` | Get chat history                   |
| DELETE | `/history` | Clear chat history                 |

**Example request:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Is the RTX 4070 compatible with a B550 motherboard?", "top_k": 5}'
```

---

## Tech Stack

| Layer         | Tool                          |
|---------------|-------------------------------|
| PDF parsing   | PyMuPDF                       |
| Web scraping  | requests + BeautifulSoup      |
| Chunking      | LangChain RecursiveTextSplitter |
| Embeddings    | sentence-transformers (CPU)   |
| Vector store  | ChromaDB (persistent, local)  |
| LLM           | Groq — Llama 3.3 70b (free)   |
| API           | FastAPI + Uvicorn             |
| UI            | Streamlit                     |

---

## License

MIT