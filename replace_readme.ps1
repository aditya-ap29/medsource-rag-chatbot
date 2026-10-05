$content = @'
# MedSource — AI-Powered Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that answers medical questions using real, citable research literature instead of relying purely on a language model's internal memory. Every answer is either grounded in a retrieved source with an inline citation, or clearly labeled as general knowledge when no verified source is available.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Why this exists

General-purpose LLMs can answer medical questions fluently, but that fluency is risky in healthcare: they can produce confident, plausible-sounding answers that are wrong, with no way for the reader to check the source. MedSource addresses this by retrieving from a real document corpus before generating an answer, citing sources inline, and being explicit about the difference between a verified, sourced answer and general knowledge.

**This is not a diagnostic tool.** It does not accept patient-specific data, does not prescribe treatment, and every response includes a disclaimer directing the user to a qualified healthcare professional.

## How it works

1. **Embed the question** using a sentence-transformer model.
2. **Vector search** with FAISS finds the most semantically similar chunks from the document corpus.
3. **Diversity + confidence filtering** removes duplicate chunks from the same source document and filters out low-similarity matches.
4. **Grounded generation**: if relevant context passed the threshold, the LLM is instructed to answer only from that context, with inline `[DOC_X]` citations.
5. **Fallback tier**: if nothing passes the threshold — or the model itself judges the retrieved context insufficient — the system answers from the LLM's general knowledge instead of a hard refusal, but clearly labels that answer as unverified, with no citations.

This two-tier design is the core idea: most simple RAG systems either hallucinate when retrieval fails, or refuse outright. MedSource instead gives a labeled middle ground.

## Tech stack

| Layer | Choice |
|---|---|
| Backend API | FastAPI, authenticated with an API key, rate-limited |
| Frontend | Streamlit |
| Vector search | FAISS |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), runs locally |
| LLM | Google Gemini (free tier) or OpenAI, configurable |
| Data | PubMed E-utilities API + curated patient-education summaries |
| Text splitting | LangChain (`langchain-text-splitters`) |

## Data sources

- **PubMed**: real research abstracts fetched from NCBI's free API, covering roughly 900 topics across infectious disease, cardiovascular, endocrine, mental health, musculoskeletal, dermatological, pediatric, and other categories. No API key required beyond a contact email.
- **Curated summaries**: a small set of hand-written, patient-friendly symptom overviews, added to cover common questions that research abstracts don't directly answer (research papers assume background knowledge rather than explaining it).

## Installation

```bash
git clone https://github.com/aditya-ap29/medsource-rag-chatbot.git
cd medsource-rag-chatbot

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
cp .env.example .env           # then fill in your API keys
```

## Configuration

Copy `.env.example` to `.env` and set at minimum:

```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
LLM_MODEL=gemini-2.0-flash
CHATBOT_API_KEY=generate-a-random-string-here
PUBMED_EMAIL=your-email@example.com
```

Get a free Gemini key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no billing required.

## Building the knowledge base

```bash
# Real PubMed research data
python scripts/ingest_data.py --source pubmed --pubmed-queries "diabetes management" "hypertension treatment"

# Your own local documents (PDF, DOCX, TXT, JSON) placed in data/raw_documents/
python scripts/ingest_data.py --source local
```

Ingestion appends to the existing vector store by default, so it can be run safely in multiple batches. Use `--fresh` to discard the existing store and start over.

## Running it

**Chat UI:**
```bash
streamlit run streamlit_app.py
```

**API server:**
```bash
python -m src.api
```
Interactive docs at `http://localhost:8000/docs`.

## API example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-chatbot-api-key" \
  -d '{"query": "What are the symptoms of anemia?", "top_k": 5}'
```

Response shape:
```json
{
  "answer": "...answer text with [DOC_1] style citations...",
  "contexts": ["...retrieved passage text..."],
  "sources": [
    {"doc_id": "DOC_1", "title": "...", "url": "...", "relevance_score": 0.77}
  ],
  "confidence": 0.77,
  "warning": "Always consult with qualified healthcare professionals for medical decisions."
}
```
When no verified source matches, `sources` and `contexts` are empty, `confidence` is `0.0`, and the `warning` field explains that the answer is general knowledge, not a cited answer.

## Security

- API requests require an `X-API-Key` header, checked against `CHATBOT_API_KEY` in `.env`.
- CORS is restricted to an explicit origin allowlist (`ALLOWED_ORIGINS` in `.env`), not a wildcard.
- The query endpoint is rate-limited (10 requests/minute per IP) to protect both the server and the LLM provider's quota.
- Server errors return a generic message to the client; details are logged server-side only, never exposed in the response.

## Known limitations

- Retrieval precision varies for short, conversational symptom questions matched against formal research abstracts.
- Free-tier Gemini is capped at 20 requests/day per model, and generation takes roughly 10–28 seconds per query.
- No conversation memory yet — each question is answered independently of prior turns.
- No automated evaluation wired in yet (a RAGAS-based evaluation script exists in `scripts/` but results haven't been published).
- General-knowledge fallback answers are, by definition, not verified against any source.

## Roadmap

- Run and publish RAGAS evaluation scores (faithfulness, answer relevancy)
- Conversation memory for follow-up questions
- User feedback logging (thumbs up/down)
- In-app document upload
- Cross-encoder re-ranking after initial retrieval

## Disclaimer

MedSource is an educational project and research prototype. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with questions about a medical condition.

## License

MIT License — see [LICENSE](LICENSE).

'@
Set-Content -Path 'README.md' -Value $content -Encoding utf8
Write-Host 'README.md has been replaced.'