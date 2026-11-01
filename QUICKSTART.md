# Quick Start — MedSource RAG Chatbot

Get the chatbot running locally in under 10 minutes.

---

## Prerequisites

- Python 3.11+
- A free [Google AI Studio](https://aistudio.google.com/apikey) account (for the Gemini API key — no billing required)
- Git

---

## Step 1 — Clone and install

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/MedSource-rag-chatbot.git
cd MedSource-rag-chatbot

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

---

## Step 2 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set the minimum required fields:

```env
GEMINI_API_KEY=your-gemini-key-here
LLM_MODEL=gemini-2.0-flash
```

Get a free key (no billing) at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Everything else has a sensible default. See [`.env.example`](.env.example) for the full list.

---

## Step 3 — Build the knowledge base

The chatbot needs indexed medical documents before it can answer questions.

**Option A — Quick start with PubMed abstracts (recommended)**
```bash
python scripts/ingest_data.py --source pubmed \
  --pubmed-queries "diabetes management" "hypertension treatment" \
                   "asthma" "anemia" "depression"
```
This fetches ~100 real research abstracts per query from NCBI's free API.

**Option B — Add local files**

Place PDF, DOCX, TXT, or JSON files in `data/raw_documents/`, then:
```bash
python scripts/ingest_data.py --source local
```

**Option C — Full bootstrap (all public sources, takes ~10 min)**
```bash
python scripts/bootstrap_data.py
```
This pulls PubMed + WHO + MedlinePlus + FDA drug labels + ClinicalTrials.

Ingestion is safe to run in multiple batches — it always **appends** to the existing index.

---

## Step 4 — Start the server

```bash
# FastAPI backend only
python -m src.api
# API docs at http://localhost:8000/docs

# Or the full Streamlit chat UI (opens a browser tab)
streamlit run streamlit_app.py
```

---

## Step 5 — Test it

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of anaemia?", "top_k": 5}'
```

Expected response shape:

```json
{
  "answer": "Anaemia is characterised by fatigue, pallor, and shortness of breath [DOC_1]...",
  "contexts": ["...retrieved passage text..."],
  "sources": [
    {
      "doc_id": "DOC_1",
      "title": "Iron-Deficiency Anaemia — PubMed 38291234",
      "url": "https://pubmed.ncbi.nlm.nih.gov/38291234/",
      "relevance_score": 0.74
    }
  ],
  "confidence": 0.74,
  "retrieval_time_ms": 18.4,
  "generation_time_ms": 3210.0,
  "total_time_ms": 3228.4,
  "warning": "Always consult with qualified healthcare professionals for medical decisions."
}
```

If `CHATBOT_API_KEY` is set in `.env`, add the header:
```bash
-H "X-API-Key: your-chatbot-api-key"
```

---

## Run the smoke test suite

```bash
python _test_run.py
```

32 checks, no API key or vector store required. All should pass.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Vector store not found` | Run Step 3 first |
| `GEMINI_API_KEY not set` | Add key to `.env` |
| `422 Unprocessable Entity` | Check JSON field names — use `query`, not `question` |
| `401 Unauthorized` | Add `X-API-Key` header, or leave `CHATBOT_API_KEY` blank to disable auth |
| Port 8000 in use | Set `API_PORT=8001` in `.env` |
| OpenMP crash on macOS M1/M2/M3 | Use `bash scripts/safe_start_api.sh` instead |
| Slow first response (3–5 s) | Normal on Gemini free tier — model load + generation |

---

## macOS Apple Silicon

If you hit `OMP: Error #15` or threading crashes:

```bash
bash scripts/safe_ingest.sh --source pubmed   # instead of python scripts/ingest_data.py
bash scripts/safe_start_api.sh               # instead of python -m src.api
```

These set `OMP_NUM_THREADS=1` to avoid the known PyTorch/OpenBLAS conflict on M-series chips.

---

## Next steps

- **Streamlit UI** — `streamlit run streamlit_app.py` has a full data-ingestion panel with 8 source types
- **RAGAS evaluation** — `python scripts/run_ragas_eval.py` scores faithfulness + answer relevance
- **Deploy to Render** — see [`DEPLOYMENT.md`](DEPLOYMENT.md)
- **Tune retrieval** — adjust `RETRIEVAL_CONFIDENCE_THRESHOLD`, `TOP_K_DOCUMENTS`, and try enabling `ENABLE_RERANKER=true`
