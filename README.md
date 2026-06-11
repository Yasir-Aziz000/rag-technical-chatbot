# RAG Technical Documentation Chatbot

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink)
![Streamlit](https://img.shields.io/badge/Streamlit-1.5-red?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-purple)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203.1-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload technical documentation (PDF, DOCX, TXT) and ask questions about it — grounded answers with source citations, no hallucinations.

> Built as a portfolio project targeting AI Engineer / GenAI Engineer roles.

---

## Live Demo

🚀 **[Try it on HuggingFace Spaces](#)** ← *(link coming soon)*

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                     │
│                                                         │
│  PDF / DOCX / TXT                                       │
│       │                                                 │
│       ▼                                                 │
│  Document Loader  ──►  Text Chunker  ──►  Embeddings   │
│  (pdfplumber /         (500 tokens,       (all-MiniLM  │
│   python-docx /         50 overlap)        -L6-v2)     │
│   plain text)               │                          │
│                             ▼                          │
│                        ChromaDB                        │
│                      (Vector Store)                    │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│                    QUERY PIPELINE                        │
│                                                         │
│  User Question ──► Embed ──► Similarity Search          │
│                                    │                    │
│                              Top-K Chunks               │
│                                    │                    │
│                    Context + Question + History         │
│                                    │                    │
│                            Groq LLM API                 │
│                         (Llama 3.1 / Mixtral)           │
│                                    │                    │
│                    Answer + Source Citations            │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│                  STREAMLIT FRONTEND                      │
│  • Upload documents     • Chat interface                │
│  • Source citations     • Conversation history          │
│  • Model selector       • Per-doc management           │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | LangChain 0.3 (LCEL) | RAG pipeline orchestration |
| LLM | Groq API — Llama 3.1 8B | Answer generation (free tier) |
| Embeddings | `all-MiniLM-L6-v2` | Local, free, 384-dim vectors |
| Vector Store | ChromaDB | Persistent similarity search |
| PDF Parsing | pdfplumber | Page-level text extraction |
| Config | Pydantic-settings v2 | Type-safe environment config |
| Logging | loguru | Structured logging |
| Frontend | Streamlit | Chat UI with streaming |

---

## Features

- **Multi-format ingestion** — PDF (page-aware), DOCX, TXT
- **Content-hash deduplication** — re-uploading the same file is a no-op
- **Streaming answers** — tokens appear in real-time, not one big wait
- **Source citations** — every answer shows which document and page it came from
- **Conversation memory** — last 10 turns are passed as context
- **Per-document management** — delete individual documents from the vector store
- **No hallucinations** — prompt strictly instructs the model to answer only from context

---

## Project Structure

```
rag-technical-chatbot/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example                    # API key template
│
├── src/
│   ├── config.py                   # Pydantic settings (loads .env)
│   ├── models/schemas.py           # Document, Chunk, RAGResponse types
│   │
│   ├── ingestion/
│   │   ├── base.py                 # Abstract BaseDocumentLoader
│   │   ├── loaders/                # PDF, DOCX, TXT loaders
│   │   ├── chunker.py              # RecursiveCharacterTextSplitter
│   │   └── pipeline.py            # Orchestrates load → chunk → store
│   │
│   ├── embeddings/hf_embeddings.py # HuggingFace local embeddings
│   ├── vectorstore/chroma_store.py # ChromaDB wrapper
│   ├── retrieval/retriever.py      # Similarity search + context formatter
│   └── generation/
│       ├── prompts.py              # ChatPromptTemplate
│       └── chain.py               # LCEL RAG chain with streaming
│
└── tests/
    ├── unit/                       # Chunker, loader unit tests
    └── integration/                # Full pipeline integration tests
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Yasir-Aziz000/rag-technical-chatbot.git
cd rag-technical-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key

Sign up at **[console.groq.com](https://console.groq.com)** — no credit card required.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your key:
# GROQ_API_KEY=gsk_...
```

### 5. Run

```bash
streamlit run app.py
# Open http://localhost:8501
```

---

## Usage

1. **Upload documents** — use the sidebar to upload one or more PDF/DOCX/TXT files
2. **Process** — click "⚡ Process Documents" to ingest and embed
3. **Ask questions** — type in the chat box

### Sample queries

| Document type | Sample question |
|--------------|----------------|
| Python docs | *"How does the GIL affect multithreading?"* |
| API reference | *"What parameters does the authentication endpoint accept?"* |
| Research paper | *"What is the main contribution of this paper?"* |
| User manual | *"What are the system requirements for installation?"* |

---

## Design Decisions

**Why Groq over OpenAI?**
Free tier with fast inference (Llama 3.1). No credit card. Swap to any LangChain-compatible LLM in one line.

**Why `all-MiniLM-L6-v2` for embeddings?**
80MB model, runs on CPU, good quality for English technical text. Zero cost, no API calls.

**Why chunk at 500 tokens with 50 overlap?**
Balances retrieval precision (small chunks rank better) with enough context per chunk for the LLM to generate a coherent answer. Overlap prevents losing information at boundaries.

**Why content-hash deduplication?**
Prevents the vector store from growing with duplicate embeddings when users re-upload the same document.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## Roadmap

- [ ] Hybrid search (vector + BM25 keyword)
- [ ] Re-ranking with a cross-encoder
- [ ] RAGAS evaluation metrics
- [ ] Docker deployment
- [ ] Multi-language support

---

## License

MIT — free to use, modify, and distribute.

---

*Built by [Yasir Aziz](https://github.com/Yasir-Aziz000) — MSc Data Science*
