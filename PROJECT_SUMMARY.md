# MedSource RAG Chatbot — Technical Project Summary

## Overview

MedSource is a production-quality **medical question-answering system** built on a Retrieval-Augmented Generation (RAG) architecture. It retrieves relevant passages from a curated medical document corpus, re-ranks them for precision, and generates cited answers using Google Gemini — while being fully transparent about when the answer is grounded in verified sources versus general LLM knowledge.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Client Layer                           │
│   Streamlit UI (streamlit_app.py)  ·  FastAPI REST (/query)  │
└──────────────────────────┬───────────────────────────────────┘
                           │  HTTP  (JSON)
┌──────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend  (src/api.py)            │
│   POST /query   ·   GET /health   ·   GET /stats              │
│   X-API-Key auth  ·  CORS allowlist  ·  10 req/min rate limit │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    RAG Pipeline  (src/rag_pipeline.py)        │
│                                                               │
│  1. [Optional] Query Expansion                                │
│     LLM rewrites query into 3 variants → wider recall         │
│                                                               │
│  2. FAISS Vector Search                                       │
│     BGE asymmetric query prefix → cosine similarity search    │
│     Deduplication by doc_id across variants                   │
│                                                               │
│  3. [Optional] Cross-Encoder Re-rank                          │
│     ms-marco-MiniLM-L-6-v2  ·  raw CE scores preserved       │
│                                                               │
│  4. Confidence Scoring                                        │
│     BGE domain: linear normalisation (0.30 floor, 0.85 ceil) │
│     CE domain:  sigmoid on raw score (0 = boundary)          │
│                                                               │
│  5. Answer Generation                                         │
│     Above threshold → grounded prompt with [DOC_X] citations  │
│     Below threshold → labeled general-knowledge fallback      │
│                                                               │
│  6. Conversation History                                       │
│     Last N turns injected into prompt for follow-up questions │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  Vector Store  (src/vector_store.py)          │
│                                                               │
│  FAISS IndexFlatIP (exact inner-product = cosine on L2-norm)  │
│  Embedding model : BAAI/bge-base-en-v1.5  (768d)             │
│  Chunking        : 256 tokens / 40-token overlap              │
│  Token counter   : tiktoken cl100k_base                       │
│  Separators      : paragraph → sentence → clause → word       │
│  BGE prefixes    : passage prefix at index time               │
│                    query prefix at search time                 │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  Data Ingestion  (src/data_ingestion.py)      │
│                                                               │
│  DocumentLoader    — PDF, DOCX, TXT, JSON (local files)       │
│  PubMedLoader      — NCBI E-utilities API                     │
│  WHOGHOLoader      — WHO Global Health Observatory OData API  │
│  MedlinePlusLoader — NLM Connect API + ICD-10 lookup          │
│  OpenFDADrugLoader — openFDA Drug Label API                   │
│  ClinicalTrialsLoader — ClinicalTrials.gov REST API v2        │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Two-tier answer strategy
Most simple RAG systems either hallucinate when retrieval fails or refuse outright. MedSource uses a confidence gate:
- **Above threshold** (default 0.30): answer is generated strictly from retrieved context, with `[DOC_X]` inline citations.
- **Below threshold**: a general-knowledge answer is generated but clearly labeled as unverified, with no citations.

This gives users a useful response even for out-of-corpus questions, without silently mixing sourced and unsourced claims.

### 2. BGE asymmetric retrieval
The `BAAI/bge-base-en-v1.5` model uses separate instruction prefixes for passages and queries:
- Passages are embedded with `"Represent this passage: "` at ingestion time
- Queries are embedded with `"Represent this sentence for searching relevant passages: "` at search time

This asymmetric scheme is what makes BGE outperform symmetric models (e.g. MiniLM) on retrieval benchmarks.

### 3. Token-aware chunking
Chunk size is measured in **tokens** (via tiktoken `cl100k_base`), not characters. This prevents sentences from being cut mid-word when documents have variable sentence lengths, and ensures all chunks fit within BGE's 512-token context window with room for the passage prefix.

### 4. Optional cross-encoder re-ranking
When `ENABLE_RERANKER=true`, the initial FAISS pool (4× top_k candidates) is re-scored with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Raw CE scores are preserved (not min-max normalised), and the confidence calculation detects the CE score domain via `any(s > 1.0 or s < 0.0 for s in scores)` before applying a sigmoid instead of the BGE linear normalisation.

### 5. Optional query expansion
When `ENABLE_QUERY_EXPANSION=true`, each query is rewritten by the LLM into 3 variants (clinical, layman, symptom-focused vocabulary), all three are searched, and results are merged with deduplication by `doc_id`. This improves recall for short or ambiguous queries.

---

## Repository Map

| File | Purpose |
|---|---|
| `src/api.py` | FastAPI backend — `/query` (POST), `/health` (GET), `/stats` (GET) |
| `src/config.py` | Pydantic `Settings` class loaded from `.env` |
| `src/models.py` | `QueryRequest`, `QueryResponse`, `Source`, `HealthResponse` schemas |
| `src/data_ingestion.py` | 6 loader classes + `MedicalDocument` + `MedicalCorpusBuilder` |
| `src/vector_store.py` | FAISS index management, BGE embedding, token chunking |
| `src/rag_pipeline.py` | Full RAG pipeline — retrieval, re-ranking, confidence, generation |
| `streamlit_app.py` | Multi-tab UI: chat, 8 data source ingestors, insights dashboard |
| `scripts/run_ragas_eval.py` | RAGAS evaluation — faithfulness, answer relevance, precision, recall |
| `scripts/calibrate_confidence.py` | Measures real BGE score distribution on a live query set |
| `scripts/expand_corpus.py` | Targeted gap-fill: paediatrics, drug interactions, mental health |
| `scripts/bootstrap_data.py` | Full-corpus bootstrap from all 5 public API sources |
| `scripts/pubmed_append.py` | Resumable PubMed batch fetcher with checkpoint support |
| `_test_run.py` | Smoke test suite — 10 checks, no API key or vector store required |

---

## Technology Stack

### Core Python packages

| Package | Version | Role |
|---|---|---|
| `fastapi` | ≥ 0.109 | REST API framework |
| `uvicorn[standard]` | ≥ 0.27 | ASGI server |
| `pydantic` + `pydantic-settings` | ≥ 2.6 | Request validation + config |
| `streamlit` | ≥ 1.38 | Chat UI |
| `faiss-cpu` | ≥ 1.9 | Vector similarity search |
| `sentence-transformers` | ≥ 2.5 | BGE embedding + CrossEncoder |
| `google-genai` | ≥ 0.8 | Gemini 2.0 Flash SDK |
| `langchain-text-splitters` | ≥ 0.3 | Token-aware recursive splitting |
| `tiktoken` | ≥ 0.5 | Token counting for chunk sizing |
| `slowapi` | ≥ 0.1.9 | Per-IP rate limiting |

### Infrastructure

| Tool | Use |
|---|---|
| Docker (Python 3.11-slim) | Container image |
| Render.com free tier | Cloud deployment |
| `render.yaml` | Declarative Render config |
| `docker-compose.yml` | Local multi-container development |

---

## Current Vector Store State

| Metric | Value |
|---|---|
| Total chunks | ~3,100 |
| Unique documents | ~2,000 |
| Embedding model | `BAAI/bge-base-en-v1.5` (768d) |
| Chunk size | 256 tokens |
| Chunk overlap | 40 tokens |
| BGE prefix | enabled |

---

## API Reference

### `POST /query`

**Request:**
```json
{
  "query": "What are the symptoms of anaemia?",
  "top_k": 5,
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Response:**
```json
{
  "answer": "Anaemia is characterised by... [DOC_1]",
  "contexts": ["...retrieved passage..."],
  "sources": [{"doc_id": "DOC_1", "title": "...", "relevance_score": 0.74, "url": "..."}],
  "confidence": 0.74,
  "retrieval_time_ms": 18.4,
  "generation_time_ms": 3210.0,
  "total_time_ms": 3228.4,
  "warning": "Always consult with qualified healthcare professionals for medical decisions."
}
```

**Auth:** `X-API-Key` header. If `CHATBOT_API_KEY` is unset, auth is disabled (local dev only).

### `GET /health`

Returns service status and whether the vector store is loaded.

### `GET /stats`

Returns vector store dimensions, chunk count, and the active LLM configuration.

---

## Deployment (Render.com)

The `render.yaml` configures a free-tier Python web service:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: gunicorn src.api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT --timeout 120
```

Required environment secrets (set in Render dashboard, `sync: false`):
- `GEMINI_API_KEY`
- `CHATBOT_API_KEY`

All other variables have defaults defined in `render.yaml`.

---

## Running Smoke Tests

```bash
python _test_run.py
```

10 checks run without any API key or vector store on disk:
- Config defaults correct
- `QueryRequest` history field present
- BGE and CE confidence domain detection
- API routes respond correctly
- Dead imports removed
- Dockerfile Python version
- `.env.example` completeness
- `requirements.txt` SDK versions
- Config model name current
- `rag_pipeline.py` SDK migration

---

*Built as a student portfolio project demonstrating full-stack ML engineering: embedding models, vector databases, LLM integration, REST API design, evaluation pipelines, and cloud deployment.*
