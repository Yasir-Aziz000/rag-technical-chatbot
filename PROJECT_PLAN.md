# End-to-End RAG Chatbot for Technical Documentation
### Portfolio Project Plan — 100% Free Stack

---

## Why This Project Is Strong for Job Applications

| Signal | Why It Matters |
|--------|---------------|
| Full RAG pipeline from scratch | Shows you understand the internals, not just wrappers |
| Multi-format document ingestion | Real-world use case, not toy data |
| Vector search + LLM fusion | Core skill in every GenAI/LLM job description |
| Deployed live demo | Recruiters click links, not README files |
| Clean GitHub repo | Used as a code interview proxy |

---

## Free Tech Stack Decision

| Component | Tool | Why Free / How Free |
|-----------|------|---------------------|
| Language | Python 3.10+ | Free |
| Framework | LangChain | Open source |
| Embeddings | `sentence-transformers` (HuggingFace) | Local, free forever |
| Embedding Model | `all-MiniLM-L6-v2` | Free HuggingFace model |
| Vector Database | ChromaDB | Local, open source |
| LLM | Groq API (Llama 3.1 / Mixtral) | Free tier, very fast |
| PDF Parsing | pdfplumber | Open source |
| DOCX Parsing | python-docx | Open source |
| Frontend | Streamlit | Open source |
| Deployment | HuggingFace Spaces | Free hosting |
| Version Control | GitHub | Free |

> **Why Groq?** Groq gives free API access to Llama 3.1 (Meta) and Mixtral — production-quality LLMs with no credit card required. Sign up at console.groq.com

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           INGESTION PIPELINE             │
                    │                                         │
  PDF / DOCX / TXT ─►  Document Loader  ►  Text Chunker      │
                    │                          │              │
                    │                    Embedding Model      │
                    │                  (all-MiniLM-L6-v2)    │
                    │                          │              │
                    │                    ChromaDB             │
                    │                  (Vector Store)         │
                    └──────────────────────────┬──────────────┘
                                               │
                    ┌──────────────────────────▼──────────────┐
                    │           QUERY PIPELINE                 │
                    │                                         │
  User Question ───►  Embed Question  ►  Similarity Search   │
                    │                          │              │
                    │                   Top-K Chunks          │
                    │                          │              │
                    │              Prompt = Context + Question │
                    │                          │              │
                    │                   Groq LLM API          │
                    │                  (Llama 3.1 / Mixtral)  │
                    │                          │              │
                    │                    Answer ◄─────────────┘
                    └─────────────────────────────────────────┘
                                               │
                    ┌──────────────────────────▼──────────────┐
                    │           STREAMLIT FRONTEND             │
                    │  - Upload Documents                      │
                    │  - Chat Interface                        │
                    │  - Source Citations                      │
                    │  - Conversation History                  │
                    └─────────────────────────────────────────┘
```

---

## Project Folder Structure

```
rag-chatbot/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .env.example              # API key template (no secrets in git)
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # Load PDF, DOCX, TXT
│   │   ├── chunker.py            # Split text into chunks
│   │   └── embedder.py           # Embed chunks, store in Chroma
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py          # Query vector store, get top-K
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm_chain.py          # Build prompt, call Groq LLM
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py             # Load env vars, constants
│
├── data/
│   └── sample_docs/              # Sample PDFs for demo
│
├── chroma_db/                    # Auto-created, local vector store
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
│
└── assets/
    └── architecture.png          # For README
```

---

## Phase 1 — Environment Setup (Day 1)

### Step 1.1 — Create GitHub Repo
```
1. Go to github.com → New Repository
2. Name: rag-technical-chatbot
3. Add README, .gitignore (Python), MIT License
4. Clone locally
```

### Step 1.2 — Python Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # Mac/Linux
```

### Step 1.3 — Install Dependencies
Create `requirements.txt`:
```txt
langchain==0.3.25
langchain-community==0.3.24
langchain-groq==0.3.2
chromadb==0.6.3
sentence-transformers==3.4.1
pdfplumber==0.11.4
python-docx==1.1.2
streamlit==1.45.1
python-dotenv==1.1.0
tiktoken==0.9.0
```
```bash
pip install -r requirements.txt
```

### Step 1.4 — Get Free Groq API Key
```
1. Go to: console.groq.com
2. Sign up (free, no credit card)
3. API Keys → Create Key → Copy it
4. Create .env file:
   GROQ_API_KEY=your_key_here
5. Add .env to .gitignore
```

---

## Phase 2 — Document Ingestion Pipeline (Day 2–3)

### Step 2.1 — Document Loader
**File:** `src/ingestion/document_loader.py`

Goal: Accept PDF, DOCX, or TXT file path and return raw text + metadata.

Key logic:
- PDF → use `pdfplumber`, extract text page by page, store page number as metadata
- DOCX → use `python-docx`, extract paragraphs
- TXT → read directly
- Return list of `{"text": ..., "source": filename, "page": N}`

### Step 2.2 — Text Chunker
**File:** `src/ingestion/chunker.py`

Goal: Split long documents into overlapping chunks for better retrieval.

Key decisions:
- Chunk size: **500 tokens** (good balance for technical docs)
- Overlap: **50 tokens** (prevents losing context at boundaries)
- Use `LangChain RecursiveCharacterTextSplitter`
- Preserve metadata (source file, page number) on each chunk

### Step 2.3 — Embedder + Vector Store
**File:** `src/ingestion/embedder.py`

Goal: Convert chunks to vectors and store in ChromaDB.

Key logic:
- Load model: `all-MiniLM-L6-v2` (384-dim, fast, free)
- For each chunk: generate embedding vector
- Store in ChromaDB with metadata
- Support adding new documents without re-ingesting old ones (check by source hash)

---

## Phase 3 — Retrieval Pipeline (Day 4)

### Step 3.1 — Retriever
**File:** `src/retrieval/retriever.py`

Goal: Given a user question, find the most relevant chunks.

Key logic:
- Embed the user question using the same model
- Query ChromaDB for top-K similar chunks (K=5 default)
- Return chunks with their metadata (source, page)
- Add a relevance score threshold to filter low-quality matches

---

## Phase 4 — Generation Pipeline (Day 5)

### Step 4.1 — LLM Chain
**File:** `src/generation/llm_chain.py`

Goal: Build a prompt from retrieved context and get LLM answer.

Prompt template:
```
You are a helpful assistant for technical documentation.
Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I don't have enough information."
Always cite the source document and page number.

Context:
{context}

Question: {question}

Answer:
```

Key logic:
- Use `langchain-groq` with model `llama-3.1-8b-instant` (free, fast)
- Pass retrieved chunks as context
- Return answer + list of source citations
- Keep conversation history using `ConversationBufferWindowMemory` (last 5 turns)

---

## Phase 5 — Streamlit Frontend (Day 6–7)

### Step 5.1 — Main App
**File:** `app.py`

Features to build in order:

**A. Sidebar — Document Upload**
- File uploader widget (PDF, DOCX, TXT, multi-file)
- "Process Documents" button
- Progress bar during ingestion
- List of already-loaded documents
- "Clear All Documents" button

**B. Main Area — Chat Interface**
- Chat message history display (user blue, assistant gray)
- Source citations expandable under each answer
- Text input at bottom
- "Clear Conversation" button

**C. Settings Panel (optional, adds polish)**
- Slider for number of retrieved chunks (K)
- Model selector (llama-3.1-8b-instant vs mixtral-8x7b)

### Step 5.2 — Session State Management
```python
# Streamlit reruns on every interaction — use st.session_state
st.session_state.messages        # conversation history
st.session_state.vectorstore     # loaded ChromaDB instance
st.session_state.loaded_docs     # list of ingested doc names
```

---

## Phase 6 — Testing (Day 8)

### Step 6.1 — Unit Tests

**test_ingestion.py**
- Load a sample PDF → assert text is non-empty
- Chunk a long text → assert chunks have correct overlap
- Embed a chunk → assert vector has correct dimension (384)

**test_retrieval.py**
- Ingest 3 sample docs → query → assert top result is relevant
- Query with no docs loaded → assert graceful error

**test_generation.py**
- Mock LLM response → assert citations are included
- Test "no context" case → assert correct fallback message

Run with:
```bash
pytest tests/ -v
```

---

## Phase 7 — Polish for Portfolio (Day 9)

### Step 7.1 — README.md
Must include:
- [ ] One-line project description
- [ ] Architecture diagram image
- [ ] Tech stack badges
- [ ] Setup instructions (clone → install → add API key → run)
- [ ] Screenshots of the UI
- [ ] 3 sample queries with answers
- [ ] Link to live demo
- [ ] What you learned / design decisions

### Step 7.2 — Sample Documents for Demo
Download free technical PDFs to use as demo content:
- Python official docs (export to PDF)
- LangChain docs
- Any open-source project documentation
- A company's public API documentation

### Step 7.3 — Code Quality
```bash
pip install black flake8
black src/ app.py          # auto-format code
flake8 src/ app.py         # check for issues
```

---

## Phase 8 — Deployment on HuggingFace Spaces (Day 10)

### Step 8.1 — Prepare for Deployment

1. Create `packages.txt` (system deps for HF Spaces):
```
build-essential
```

2. Update `requirements.txt` — remove any Windows-only packages

3. Add `README.md` header for HuggingFace:
```yaml
---
title: RAG Technical Documentation Chatbot
emoji: 📚
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
---
```

### Step 8.2 — Deploy to HuggingFace Spaces
```
1. Go to huggingface.co → New Space
2. Name: rag-technical-chatbot
3. SDK: Streamlit
4. Link your GitHub repo (auto-deploy on push)
5. Settings → Repository Secrets → Add GROQ_API_KEY
6. Space will build automatically (5–10 min)
```

### Step 8.3 — Add Demo Documents
Upload 2–3 sample PDFs directly in the Space so visitors can immediately test without uploading their own files.

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1. Setup | Day 1 | Repo, env, API key working |
| 2. Ingestion | Day 2–3 | Can load PDF/DOCX/TXT into ChromaDB |
| 3. Retrieval | Day 4 | Can query and get relevant chunks |
| 4. Generation | Day 5 | Can answer questions with citations |
| 5. Frontend | Day 6–7 | Full Streamlit UI working locally |
| 6. Testing | Day 8 | Tests passing |
| 7. Polish | Day 9 | Clean README, architecture diagram |
| 8. Deployment | Day 10 | Live demo link on HuggingFace Spaces |

**Total: 10 days of focused work**

---

## What Interviewers Will Ask — Be Ready

| Question | Your Answer (based on this project) |
|----------|-------------------------------------|
| What is RAG? | Retrieval-Augmented Generation — retrieve relevant context from a vector store, inject it into the LLM prompt to ground answers in real documents |
| Why ChromaDB over Pinecone? | ChromaDB is free and local; Pinecone is managed cloud. For a portfolio project and cost control, ChromaDB is correct. I know the tradeoffs. |
| Why chunk at 500 tokens with 50 overlap? | Balance between enough context per chunk and retrieval precision. Overlap prevents losing information at boundaries. |
| What embedding model did you use and why? | `all-MiniLM-L6-v2` — small (80MB), fast, good quality for English technical text. Compared alternatives (OpenAI ada, BGE) but this is free and runs locally. |
| How do you handle hallucinations? | Prompt instructs model to answer ONLY from provided context and explicitly say when it doesn't know. Evaluated on test queries. |
| How would you scale this? | Replace ChromaDB with Pinecone/Weaviate, use async ingestion queue, add Redis for conversation memory, containerize with Docker. |

---

## Stretch Goals (Add After Core is Done)

- [ ] **Hybrid search**: combine vector search + keyword search (BM25) for better retrieval
- [ ] **Re-ranking**: use a cross-encoder to re-rank top-K results before generating
- [ ] **Evaluation**: measure retrieval precision with RAGAS framework
- [ ] **Multi-language**: test with Italian technical documents
- [ ] **Document comparison**: ask questions across multiple documents
- [ ] **Export chat**: download conversation history as PDF

---

## Total Cost: €0

Everything in this stack is free. No credit card. No trial expiry.
