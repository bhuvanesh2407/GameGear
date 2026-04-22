"""
ingest.py
─────────
Phase 1 – Step 1: Load raw text from two sources:
  1. PDF files  → parsed with PyMuPDF (fitz)
  2. Web pages  → scraped with requests + BeautifulSoup

Each document is returned as a dict:
  {
      "text":   <raw text string>,
      "source": <file path or URL>,
      "type":   "pdf" | "web"
  }

Usage:
  from src.ingest import load_all_documents
  docs = load_all_documents()
"""

import fitz                          # PyMuPDF
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from src.config.config import PDF_DIR, URLS_FILE


# ── PDF Loader ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> dict:
    """
    Extract all text from a PDF file page by page.
    Returns a single document dict with all pages joined.
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():              # skip blank pages
            full_text += f"\n[Page {page_num}]\n{page_text}"

    doc.close()

    return {
        "text":   full_text.strip(),
        "source": str(pdf_path),
        "type":   "pdf"
    }


def load_all_pdfs() -> list[dict]:
    """
    Load every .pdf found in the PDF_DIR folder.
    """
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[ingest] No PDFs found in {PDF_DIR}")
        return []

    documents = []
    for pdf_path in pdf_files:
        print(f"[ingest] Loading PDF: {pdf_path.name}")
        doc = load_pdf(pdf_path)

        if doc["text"]:                    # skip empty/corrupt files
            documents.append(doc)
        else:
            print(f"[ingest] WARNING: No text extracted from {pdf_path.name}")

    print(f"[ingest] Loaded {len(documents)} PDF(s)")
    return documents


# ── Web Scraper ───────────────────────────────────────────────────────────────

def scrape_url(url: str) -> dict | None:
    """
    Fetch a URL and extract clean visible text using BeautifulSoup.
    Strips nav/footer/script/style noise.
    Returns None if the request fails.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"[ingest] ERROR fetching {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noisy tags that don't carry useful content
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Get clean text — collapse excessive whitespace
    raw_text = soup.get_text(separator="\n")
    clean_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(clean_lines)

    return {
        "text":   clean_text,
        "source": url,
        "type":   "web"
    }


def load_all_urls() -> list[dict]:
    """
    Read URLs from urls.txt (one URL per line, # lines are comments).
    Scrape each and return list of document dicts.
    """
    if not URLS_FILE.exists():
        print(f"[ingest] No urls.txt found at {URLS_FILE}")
        return []

    urls = []
    for line in URLS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)

    if not urls:
        print("[ingest] urls.txt is empty or has only comments")
        return []

    documents = []
    for url in urls:
        print(f"[ingest] Scraping: {url}")
        doc = scrape_url(url)

        if doc and doc["text"]:
            documents.append(doc)
        else:
            print(f"[ingest] WARNING: No content scraped from {url}")

    print(f"[ingest] Scraped {len(documents)} web page(s)")
    return documents


# ── Main entry point ──────────────────────────────────────────────────────────

def load_all_documents() -> list[dict]:
    """
    Load ALL documents from both sources (PDFs + web).
    Returns a flat list of document dicts ready for chunking.
    """
    pdf_docs = load_all_pdfs()
    web_docs = load_all_urls()
    all_docs = pdf_docs + web_docs

    print(f"[ingest] Total documents loaded: {len(all_docs)}")
    return all_docs


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    docs = load_all_documents()
    for doc in docs:
        preview = doc["text"][:200].replace("\n", " ")
        print(f"\n{'─'*60}")
        print(f"Source : {doc['source']}")
        print(f"Type   : {doc['type']}")
        print(f"Preview: {preview}...")