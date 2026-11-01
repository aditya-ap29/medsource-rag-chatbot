# Deployment — MedSource RAG Chatbot

## Render.com (recommended — free tier)

MedSource includes a `render.yaml` that fully configures a free Render web service. The only manual steps are uploading secrets and the pre-built vector store.

### 1. Push code to GitHub

```bash
git add .
git commit -m "Deploy MedSource RAG chatbot"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/MedSource-rag-chatbot.git
git push -u origin main
```

> The `data/vector_store/` directory is in `.gitignore` — it must be uploaded separately (see step 3).

---

### 2. Create the Render service

1. Go to [render.com](https://render.com) → **New +** → **Web Service**
2. Connect your GitHub repository
3. Render auto-detects `render.yaml` — no manual configuration needed

---

### 3. Upload the vector store

The FAISS index is too large for git. Upload it to any public or signed-URL file host:

**Google Drive**
```
1. Upload data/vector_store/ as a .tar.gz archive
2. Share → Anyone with link → Get the file ID from the URL
3. Direct download URL: https://drive.google.com/uc?export=download&id=FILE_ID
```

**Dropbox**
```
1. Upload the archive → Share → Create link
2. Change ?dl=0 → ?dl=1 at the end of the URL
```

Then set the `VECTOR_STORE_URL` environment variable in the Render dashboard. The `render.yaml` build step runs `scripts/download_vector_store.sh` which fetches and extracts it automatically.

---

### 4. Set environment secrets

In the Render dashboard → your service → **Environment**, add:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | Your Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `CHATBOT_API_KEY` | A random string (e.g. `openssl rand -hex 32`) — required to call `/query` |
| `VECTOR_STORE_URL` | Download URL from step 3 |

All other variables are pre-set in `render.yaml` with correct defaults.

---

### 5. Deploy

Click **Create Web Service**. First deploy takes ~5 minutes (pip install + model download).

Your API will be live at:
```
https://MedSource-rag-api.onrender.com
```

Test it:
```bash
curl -X POST https://MedSource-rag-api.onrender.com/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-chatbot-api-key" \
  -d '{"query": "What are the symptoms of anaemia?", "top_k": 5}'
```

---

## Docker (local)

```bash
# Build
docker build -t MedSource-rag .

# Run (reads from .env automatically)
docker-compose up
```

The `docker-compose.yml` maps all required environment variables. Copy `.env.example` to `.env` and fill in your keys before running.

---

## Free-tier limitations (Render)

| Constraint | Details |
|---|---|
| Sleep | Service sleeps after 15 min of inactivity |
| Wake-up time | ~30 seconds on first request after sleep |
| Monthly hours | 750 free hours (enough for continuous uptime) |
| RAM | ~512 MB (the BGE embedding model uses ~300 MB) |
| CPU | Shared, no GPU |

**Keep-alive (optional):** Use [cron-job.org](https://cron-job.org) to ping `/health` every 10 minutes to prevent sleep.

---

## Configuration reference

All settings are environment variables. The most important ones:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `openai`, `mistral`, `qwen` |
| `GEMINI_API_KEY` | — | Required when `LLM_PROVIDER=gemini` |
| `LLM_MODEL` | `gemini-2.0-flash` | Model name passed to the provider |
| `CHATBOT_API_KEY` | _(blank = no auth)_ | Key clients must send in `X-API-Key` header |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Do not change unless re-embedding the index |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.30` | Below this score → general-knowledge fallback |
| `TOP_K_DOCUMENTS` | `5` | Passages retrieved per query |
| `ENABLE_RERANKER` | `false` | Cross-encoder re-ranking (+30 ms, better precision) |
| `ENABLE_QUERY_EXPANSION` | `false` | LLM rephrases query into 3 variants (+1 LLM call) |
| `OMP_NUM_THREADS` | `1` | Prevent OpenMP conflicts on CPU-only hosts |

Full list in [`.env.example`](.env.example).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails: `pip install` error | Check Python version is 3.11 in `render.yaml` |
| `Vector store not found` on startup | Check `VECTOR_STORE_URL` is set and accessible |
| `GEMINI_API_KEY` missing error | Add key in Render dashboard → Environment |
| 503 on `/query` | Service may still be initialising — check `/health` |
| Slow cold start | Normal — BGE model downloads ~440 MB on first boot |
| Out of memory crash | Reduce `EMBEDDING_BATCH_SIZE` to `16` in environment |
