# MedSource — Medical RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** system for medical question-answering. Every factual answer is grounded in a retrieved passage from a curated medical corpus and tagged with an inline `[DOC_X]` citation. When retrieval confidence falls below the configured threshold the response is explicitly labelled as general knowledge, not a sourced answer.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Medical Disclaimer:** This is an educational and research project. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## Problem Statement

General-purpose LLMs answer medical questions fluently but without verifiability. A model can generate a confident-sounding answer that is subtly incorrect, with no way for the reader to trace or challenge the claim.

MedSource addresses this by:

1. **Retrieving** semantically similar passages from a curated corpus before generating anything
2. **Citing** sources inline (`[DOC_1]`, `[DOC_2]`, …) so every factual claim is traceable
3. **Being transparent** about confidence — when retrieval scores fall below threshold, the answer is clearly labelled as unverified general knowledge
4. **Enforcing safety** — emergency patterns (chest pain, stroke, overdose, etc.) are detected and show an emergency notice before any AI response

---

## Key Features

**Implemented and verified:**

- **Semantic + Hybrid Retrieval** — BGE/FAISS dense retrieval fused with BM25 sparse retrieval via Reciprocal Rank Fusion (RRF). Enabled per-query by setting `ENABLE_HYBRID_RETRIEVAL=true`.
- **Semantic Domain Classifier** — Optional logistic-regression classifier (scikit-learn) on top of BGE embeddings that classifies queries as `medical`, `out_of_scope`, or `ambiguous`. Trained via `scripts/train_classifier.py`. Falls back to keyword gate when unavailable.
- **Evidence Sufficiency Assessment** — Optional pre-generation quality gate that assesses whether retrieved passages provide sufficient evidence to answer the query, beyond the confidence score threshold.
- **Live PubMed Retrieval** — Optional augmentation that queries NCBI PubMed at inference time when local retrieval confidence is low. Results are merged with local FAISS hits and marked `[LIVE]` in the UI. Enabled by `ENABLE_LIVE_PUBMED=true`.
- **Confidence Scoring** — Normalised [0–1] score derived from BGE cosine similarity. Displayed as a coloured badge (High / Medium / Low / General knowledge) per response.
- **Optional Cross-Encoder Re-ranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranks retrieved candidates before generation. Lazy-loaded (~23 MB), ~30 ms per query. Enabled by `ENABLE_RERANKER=true`.
- **Optional Query Expansion** — One Gemini call rewrites the query into 3 rephrased variants; results are merged by best score per document. Enabled by `ENABLE_QUERY_EXPANSION=true`.
- **Feedback Persistence** — Per-response thumbs-up/down stored in a local SQLite database (`data/feedback.db`) and persisted across Streamlit restarts.
- **Emergency Safety Layer** — 8 regex patterns (breathing difficulty, chest pain, stroke, unconsciousness, suicide, overdose, severe allergy, heavy bleeding) trigger an emergency notice with emergency numbers (112/108) before any AI response.
- **Medication Risk Detection** — Detects high-risk queries (overdose, pregnancy + drug, specific dangerous drug dosages) and shows a pharmacist referral notice.
- **Session-scoped Document Upload** — Users can upload PDF, DOCX, or TXT files for Q&A without modifying the shared index. Uploaded documents are never saved to disk.
- **Multi-turn Conversation** — Last 3 Q&A turns (token-budgeted at 400 tokens) are included in the prompt.
- **Symptom Checker Tab** — Structured symptom entry with severity classification.
- **Admin Panel** — Password-gated data ingestion, index reload, and corpus management via the Streamlit UI.
- **FastAPI REST Backend** — Async `/query`, `/health`, `/stats` endpoints with optional API-key auth and 10 req/min per-IP rate limiting.
- **Docker + Render.com Deployment** — `Dockerfile`, `docker-compose.yml`, and `render.yaml` included.

---

## Architecture

```
User question
      │
      ▼
┌─────────────────────────────────────────┐
│  1. Emergency / safety pattern check    │  8 regex patterns — runs before RAG
│     (emergency notice if matched)       │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  2. Domain gate                         │  Keyword gate (always)
│     (optional: semantic classifier)     │  + LogReg classifier on BGE embeddings
└─────────────────────────────────────────┘
      │ medical
      ▼
┌─────────────────────────────────────────┐
│  3. Retrieval                           │  Default: FAISS IndexFlatIP (exact cosine)
│     Dense: BGE/FAISS                    │  Optional: BM25 + RRF hybrid fusion
│     Sparse: BM25 (optional)             │  Optional: live PubMed augmentation
│     Live:   PubMed E-utilities (opt.)   │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  4. Optional: Cross-encoder re-rank     │  ms-marco-MiniLM-L-6-v2
│     (ENABLE_RERANKER=true)              │  lazy-loaded, ~30 ms CPU
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  5. Confidence gate                     │  BGE cosine → normalised 0–1
│     pass  → grounded generation         │  fail → labelled general-knowledge fallback
│     (optional: evidence assessment)     │  or hard refusal
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  6. Grounded generation                 │  Google Gemini 2.0 Flash
│     [DOC_X] citations injected          │  Free tier: no billing required
└─────────────────────────────────────────┘
      │
      ▼
  Answer + cited sources + confidence badge
```

---

## Technology Stack

| Layer | Technology | Notes |
|---|---|---|
| Frontend | **Streamlit** ≥ 1.38 | Multi-tab UI: Chat, Upload & Ask, Symptom Checker, Insights, Admin |
| Backend API | **FastAPI** ≥ 0.109 | Async; OpenAPI docs at `/docs` |
| LLM | **Google Gemini 2.0 Flash** via `google-genai` | Free tier — no billing needed |
| Embeddings | **BAAI/bge-base-en-v1.5** (768-dim) via sentence-transformers | Asymmetric query/passage prefixes |
| Vector store | **FAISS** `IndexFlatIP` | Exact cosine via L2-normalised vectors |
| Sparse retrieval | **BM25** via `rank-bm25` | Optional; fused with FAISS via RRF |
| Optional re-ranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Lazy-loaded 23 MB; ~30 ms/query |
| Semantic classifier | **scikit-learn** LogisticRegression on BGE | Optional; `scripts/train_classifier.py` |
| Text splitting | **LangChain** `RecursiveCharacterTextSplitter` | Token-aware, 256-tok chunks / 40-tok overlap |
| Config | **Pydantic Settings** (`pydantic-settings`) | `.env` file; `lru_cache` singleton |
| Rate limiting | **SlowAPI** | 10 req/min per IP |
| Containerisation | **Docker** python:3.11-slim + gunicorn | `Dockerfile` + `docker-compose.yml` |
| Deployment | **Render.com** free tier | `render.yaml` included |
| Python | **3.11+** | `runtime.txt` = 3.11.10 |

---

## RAG Workflow

### Retrieval

**Default (dense only):**
1. BGE encodes the query with a `"Represent this sentence: "` prefix.
2. FAISS `IndexFlatIP` returns the top `K×4` candidates by cosine similarity.
3. Candidates deduplicated by `doc_id` (best score retained per document).

**Hybrid retrieval** (`ENABLE_HYBRID_RETRIEVAL=true`):
1. Dense: FAISS as above.
2. Sparse: BM25 on tokenised chunk text.
3. RRF fusion: `score(d) = Σ_r weight_r / (k + rank_r(d))` where k=60 (Cormack 2009).
4. Results keyed on `chunk_id`; highest-scored chunk per document retained.

**Live PubMed augmentation** (`ENABLE_LIVE_PUBMED=true`):
- NCBI E-utilities (esearch + efetch) are queried at inference time.
- Results are deduplicated against indexed PMIDs and merged into the candidate pool.
- Live results are marked `provenance=live_pubmed` and shown with a `[LIVE]` badge.

### Confidence Gate

Each retrieval score (BGE cosine) is normalised to [0–1]:
- Scores at the threshold (`RETRIEVAL_CONFIDENCE_THRESHOLD`, default 0.30) map to 0.0.
- Scores at the practical maximum (0.85) map to 1.0.
- +10% bonus if the generated answer contains `[DOC_` citations.
- ×0.5 penalty if the model admits uncertainty in the response text.

When all scores fall below the threshold:
- `ENABLE_GENERAL_FALLBACK=true` (default): a general-knowledge answer is generated and clearly labelled as unverified.
- `ENABLE_GENERAL_FALLBACK=false`: a hard refusal is returned.

### Generation

The system prompt instructs Gemini to:
- Answer **only** from retrieved documents.
- Cite every factual claim as `[DOC_X]`.
- Use the structured format: `**Direct Answer:** / **Explanation:** / **Practical Guidance:**`.
- Refuse with a fixed phrase when documents do not answer the question.
- Never diagnose or prescribe.

Context window budget: 1,500 tokens for retrieved passages, 400 tokens for conversation history, ~300 tokens for the system prompt.

---

## Data Sources

MedSource can index from any combination of:

| Source | Method |
|---|---|
| **PubMed abstracts** | NCBI E-utilities (esearch + efetch XML); free, email address only |
| **WHO GHO statistics** | WHO Global Health Observatory OData API |
| **MedlinePlus (NIH)** | NLM Connect API + ICD-10 code lookup; HTML stripped |
| **FDA drug labels** | openFDA Drug Label API; 11 label sections per drug |
| **ClinicalTrials.gov** | Public REST API v2; completed studies, min_enrollment=50 |
| **Local files** | PDF (PyPDF2), DOCX (python-docx), TXT, JSON |
| **Manual paste** | Text or guideline excerpts via the Streamlit admin panel |

Ingestion chunks each document with `RecursiveCharacterTextSplitter` (256 token chunks, 40 token overlap, tiktoken `cl100k_base`), then embeds with BGE and adds to FAISS.

---

## Project Structure

```
medintel-rag-chatbot/
├── streamlit_app.py            # Streamlit UI — chat, upload, symptom checker, insights, admin
│
├── src/
│   ├── __init__.py             # Package init (version 1.0.0)
│   ├── api.py                  # FastAPI app — /query, /health, /stats
│   ├── config.py               # Pydantic settings (all env vars, lru_cache singleton)
│   ├── models.py               # QueryRequest, QueryResponse, Source schemas
│   ├── rag_pipeline.py         # Core RAG: retrieval → re-rank → confidence → generate
│   ├── vector_store.py         # FAISS + BGE embedding + token-aware chunking
│   ├── data_ingestion.py       # MedicalDocument + 6 loader classes + MedicalCorpusBuilder
│   ├── hybrid_retriever.py     # BM25 + FAISS dense retrieval fused via RRF
│   ├── bm25_retriever.py       # BM25 sparse retriever (rank-bm25)
│   ├── domain_gate.py          # Keyword-based domain classifier
│   ├── semantic_classifier.py  # LogReg classifier on BGE embeddings (optional)
│   ├── evidence_assessor.py    # Evidence sufficiency gate (optional)
│   ├── feedback_store.py       # SQLite feedback persistence
│   └── live_retrieval.py       # Live PubMed E-utilities retriever (optional)
│
├── scripts/
│   ├── ingest_data.py          # CLI ingestion runner (argparse, 7 source types)
│   ├── bootstrap_data.py       # Full-corpus bootstrap (all 5 public APIs)
│   ├── pubmed_append.py        # Resumable PubMed batch fetcher with checkpoint
│   ├── expand_corpus.py        # Gap-fill: paediatrics, drug interactions, mental health
│   ├── reembed_vector_store.py # Re-embed when switching embedding models
│   ├── build_versioned_index.py# Versioned index builder
│   ├── swap_index.py           # Hot-swap active index without restart
│   ├── validate_index.py       # Index integrity validation
│   ├── clean_corpus.py         # Corpus deduplication and cleaning
│   ├── train_classifier.py     # Train the semantic domain classifier
│   ├── calibrate_confidence.py # Measure BGE score distribution on live queries
│   ├── benchmark_chunk_sizes.py# Benchmark retrieval across chunk sizes
│   ├── audit_sources.py        # Verify source labels in the index
│   ├── eval_retrieval.py       # Retrieval quality evaluation
│   ├── run_eval.py             # General evaluation runner
│   ├── run_ragas_eval.py       # RAGAS evaluation (faithfulness, answer relevance)
│   ├── safe_ingest.sh          # macOS Apple Silicon threading workaround
│   ├── safe_start_api.sh       # macOS Apple Silicon threading workaround
│   ├── download_vector_store.sh# Fetch pre-built index on Render deploy
│   └── test_api.py             # API smoke test script
│
├── tests/                      # pytest test suite
│   ├── test_bm25_hybrid.py
│   ├── test_domain_gate.py
│   ├── test_evidence_assessor.py
│   ├── test_functional.py
│   ├── test_index_build.py
│   ├── test_medsource.py
│   ├── test_pipeline_reliability.py
│   ├── test_rag_eval_regression.py
│   ├── test_semantic_classifier.py
│   ├── test_semantic_hybrid_integration.py
│   └── test_ui_apptest.py
│
├── data/
│   ├── raw_documents/          # Source JSON saved after each ingest run
│   ├── vector_store/           # faiss.index + documents.pkl + config.pkl (git-ignored)
│   ├── vector_store_v1/        # Versioned index (git-ignored)
│   └── eval/                   # Evaluation datasets (domain classification, Q&A pairs)
│
├── examples/
│   ├── demo.py                 # HTTP demo — queries the running FastAPI server
│   └── direct_usage.py         # In-process demo — uses src/ directly, no server needed
│
├── reports/                    # Evaluation and audit reports (markdown)
│
├── .streamlit/config.toml      # Streamlit theme configuration
├── streamlit_app.py
├── .env.example                # Environment variable template (safe placeholders only)
├── requirements.txt            # Runtime dependencies
├── requirements-eval.txt       # Additional deps for RAGAS evaluation only
├── Dockerfile                  # Python 3.11-slim image
├── docker-compose.yml          # Local Docker setup
├── render.yaml                 # Render.com deployment config
├── runtime.txt                 # Python 3.11.10 (for Render)
├── setup.py                    # Package setup
├── _test_run.py                # Smoke test suite (no API key or vector store required)
├── _check_eval_prereqs.py      # Eval prerequisite checker
├── QUICKSTART.md               # 10-minute setup guide
├── DEPLOYMENT.md               # Render + Docker deployment guide
└── DATA_SOURCES_GUIDE.md       # How to add and manage data sources
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/medintel-rag-chatbot.git
cd medintel-rag-chatbot

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env    # then edit .env with your Gemini API key
```

---

## Configuration

The only required field is your Gemini API key. All other values have safe defaults.

```env
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a free key (no billing required) at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

See [`.env.example`](.env.example) for the full configuration reference. Key variables:

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Google Gemini API key |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | **Do not change after first ingest** |
| `TOP_K_DOCUMENTS` | `5` | Passages retrieved per query |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.30` | Below this → general-knowledge fallback |
| `ENABLE_GENERAL_FALLBACK` | `true` | Show labelled general answer vs. hard refusal |
| `ENABLE_HYBRID_RETRIEVAL` | `false` | BM25 + FAISS fusion via RRF |
| `ENABLE_RERANKER` | `false` | Cross-encoder re-ranking (+30 ms, better precision) |
| `ENABLE_QUERY_EXPANSION` | `false` | Rewrite query into 3 variants (+1 LLM call) |
| `ENABLE_SEMANTIC_CLASSIFIER` | `false` | ML domain gate (requires trained model) |
| `ENABLE_EVIDENCE_ASSESSMENT` | `false` | Evidence quality gate before generation |
| `ENABLE_LIVE_PUBMED` | `false` | Augment low-confidence queries from PubMed live |
| `CHATBOT_API_KEY` | _(blank)_ | `X-API-Key` header auth; blank = no auth |
| `ADMIN_PASSWORD` | _(blank)_ | Streamlit admin panel password; blank = hidden |

---

## Building the Knowledge Base

```bash
# PubMed abstracts for specific topics
python scripts/ingest_data.py --source pubmed \
  --pubmed-queries "diabetes management" "hypertension treatment" "asthma"

# Index local files from data/raw_documents/ (PDF, DOCX, TXT, JSON)
python scripts/ingest_data.py --source local

# Full bootstrap — PubMed + WHO GHO + MedlinePlus + FDA + ClinicalTrials
python scripts/bootstrap_data.py
```

Ingestion appends to the existing index by default. Use `--fresh` to wipe and rebuild.

Sources can also be added through the Streamlit UI → **Data Sources** tab (requires `ADMIN_PASSWORD`).

---

## Running

**Streamlit chat UI (recommended):**
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**FastAPI backend only:**
```bash
python -m src.api
# API docs at http://localhost:8000/docs
```

**Docker:**
```bash
# Copy and fill in your .env first
cp .env.example .env
docker-compose up
```

**macOS Apple Silicon — threading workaround:**
```bash
bash scripts/safe_ingest.sh --source pubmed
bash scripts/safe_start_api.sh
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-chatbot-api-key" \
  -d '{"query": "What are the symptoms of anaemia?", "top_k": 5}'
```

**Response structure:**
```json
{
  "answer": "Anaemia presents with fatigue, pallor, and shortness of breath [DOC_1]...",
  "sources": [
    {
      "doc_id": "DOC_1",
      "title": "Iron-Deficiency Anaemia",
      "year": 2023,
      "url": "https://pubmed.ncbi.nlm.nih.gov/...",
      "relevance_score": 0.74,
      "excerpt": "...",
      "provenance": "indexed"
    }
  ],
  "confidence": 0.74,
  "retrieval_time_ms": 18,
  "generation_time_ms": 520,
  "total_time_ms": 538,
  "warning": "Always consult with qualified healthcare professionals for medical decisions."
}
```

When no verified source matches: `sources` is `[]`, `confidence` is `0.0`, and the answer is prefixed with `"Note: This answer is based on general medical knowledge, not verified documents."`.

**Multi-turn conversation:**
```json
{
  "query": "What about treatment options?",
  "top_k": 5,
  "history": [
    {"role": "user", "content": "What are the symptoms of anaemia?"},
    {"role": "assistant", "content": "Anaemia presents with fatigue..."}
  ]
}
```

---

## Evaluation

```bash
# Install eval dependencies
pip install -r requirements-eval.txt

# Run RAGAS evaluation suite
python scripts/run_ragas_eval.py

# Run retrieval evaluation
python scripts/eval_retrieval.py

# Smoke test suite (no API key or vector store required)
python _test_run.py
```

Evaluation datasets are in `data/eval/`. The RAGAS evaluation reports metrics including **Grounded (cited) answers** rate, mean pipeline confidence, RAGAS faithfulness, and answer relevance. Results are written to `ragas_eval_results.csv`. See `reports/` for historical evaluation runs.

---

## Security

- API requests to `/query` and `/stats` require an `X-API-Key` header matched against `CHATBOT_API_KEY`. Leave blank to disable auth for local development.
- CORS is restricted to an explicit origin allowlist (`ALLOWED_ORIGINS`), not a wildcard.
- Rate-limited to 10 requests/minute per IP via SlowAPI.
- All server-side errors return a generic message to the client; full stack traces are logged server-side only.
- All API keys are read exclusively from `.env` — never from the UI or query parameters.
- `.env` is excluded from version control by `.gitignore`. **Never commit `.env` to any repository.**

---

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for full instructions. Summary for Render.com (free tier):

1. Push to GitHub (verify no secrets are staged first)
2. Connect the repo on [render.com](https://render.com) — `render.yaml` auto-configures the service
3. Archive `data/vector_store/` and host it publicly (Google Drive, Dropbox, or S3)
4. Set secrets in the Render dashboard: `GEMINI_API_KEY`, `CHATBOT_API_KEY`, `VECTOR_STORE_URL`
5. Click **Create Web Service** — first deploy takes ~5 minutes

---

## Current Limitations

- **FAISS is a local file** — not suitable for multi-instance horizontal scaling. A distributed store (Qdrant, Pinecone, Weaviate) would be needed for production multi-replica deployments.
- **Gemini free tier** — 15 RPM / 1 M TPM. Responses are synchronous; a slow Gemini call blocks the Streamlit thread.
- **Chat history is not persisted** across Streamlit server restarts (session state only).
- **RAGAS evaluation uses Gemini as both responder and judge** — results are partly self-referential.
- **Duplicate ingestion** — re-ingesting the same source will add duplicate chunks; no deduplication guard is in place. Use `--fresh` when re-running full ingestion.
- **Semantic classifier** requires training data and a separate `scripts/train_classifier.py` run before it can be enabled.
- **BGE embedding model change** requires full re-ingestion of the entire corpus (`scripts/reembed_vector_store.py`).

---

## Roadmap

- [ ] Response streaming (faster perceived latency)
- [ ] Deduplication guard in `VectorStore.add_documents()`
- [ ] Citation-to-source alignment verification post-generation
- [ ] In-app document management (rename, delete, filter by source)
- [ ] Migrate to a distributed vector store for multi-instance deployments
- [ ] Publish full RAGAS faithfulness and answer-relevance scores on a standard benchmark set

---

## License

MIT — see [LICENSE](LICENSE).

---

## Medical Disclaimer

MedSource is an **educational and research project**. It is not a substitute for professional medical advice, diagnosis, or treatment. The system does not accept patient data and does not store personal health information. Always consult a qualified healthcare provider about any medical condition or before making any healthcare decision.
