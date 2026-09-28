# 📥 Feeding Data Sources into MedSource

MedSource only knows what's inside its **vector store**. Anything you add here becomes
retrievable evidence the chatbot can cite as `[DOC_X]`. There are two ways to add data:
the new **Streamlit "Data Sources" tab** (easiest, no code), or the **command-line
ingestion script** (best for bulk/automated loads).

---

## 1. Easiest way: the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the **📥 Data Sources** tab. It has four sub-tabs:

| Sub-tab | What it's for |
|---|---|
| 📄 Upload files | Medical **books/textbook chapters as PDF**, DOCX, TXT, or a JSON list of `{title, content, source, year, url}` |
| 🔬 PubMed API | Search + ingest abstracts straight from NCBI's public PubMed API |
| 🌐 WHO / CDC / NIH URL | Paste a fact-sheet URL and it scrapes + indexes the readable text |
| ✍️ Paste text | Manually paste a guideline excerpt or note |

Click the ingest button in a sub-tab → the text is chunked, embedded, and saved to
`./data/vector_store/` immediately. No server restart needed — the sidebar's chunk/document
counters update right away.

---

## 2. Data source options in detail

### A. PubMed API (already built in — `src/data_ingestion.py: PubMedLoader`)
Free, no API key required, but NCBI asks for a contact email with every request.

```bash
python scripts/ingest_data.py --source pubmed \
  --pubmed-queries "type 2 diabetes management" "vitamin D deficiency" \
  --max-pubmed-results 50
```
Set `PUBMED_EMAIL=you@example.com` in `.env` (or type it into the Streamlit PubMed tab).

**Docs:** https://www.ncbi.nlm.nih.gov/books/NBK25501/

### B. WHO — two different kinds of "official" data, two different ingestion paths

WHO doesn't expose one single API for everything. There are two distinct, both-authentic
sources, and the project now supports both from the Streamlit **Data Sources** tab:

**B1. WHO fact sheets (narrative text) — tab: "🌐 WHO / CDC / NIH URL"**
Best for Q&A prose. No key needed. The tab has a **quick-add list** of 10 official WHO
fact sheets (diabetes, hypertension, TB, malaria, HIV, etc.) — pick topics and click ingest.
You can also paste any other fact-sheet/CDC/MedlinePlus URL and it scrapes + indexes the text.

**B2. WHO Global Health Observatory (GHO) — structured statistics — tab: "🌍 WHO GHO Indicators"**
This is WHO's official, free, **no-API-key** statistics API:
```
GET https://ghoapi.azureedge.net/api/{IndicatorCode}
GET https://ghoapi.azureedge.net/api/{IndicatorCode}?$filter=SpatialDim eq 'IND'
```
It returns raw numeric rows (country, year, value) — not prose — so `WHOGHOLoader` in
`src/data_ingestion.py` (already added to your project) converts each country's rows into
short, citeable sentences like *"In 2021, the value was 71.4 (both sexes)."* before indexing.
The Streamlit tab has a dropdown of common indicators (life expectancy, obesity prevalence,
diabetes prevalence, TB/HIV/malaria incidence, immunization coverage, etc.) plus an optional
ISO3 country filter. Full indicator catalogue: https://www.who.int/data/gho/data/indicators

**B3. (Optional, advanced) WHO ICD-11 API — disease classification/coding**
If you later want authoritative disease definitions/codes (not built into this project yet),
WHO's ICD-11 API requires free registration:
1. Register at https://icd.who.int/icdapi → get a `client_id` / `client_secret`.
2. Exchange them for a bearer token (valid ~1 hour):
   ```python
   import requests
   token = requests.post(
       "https://icdaccessmanagement.who.int/connect/token",
       data={
           "client_id": "YOUR_CLIENT_ID",
           "client_secret": "YOUR_CLIENT_SECRET",
           "scope": "icdapi_access",
           "grant_type": "client_credentials",
       },
   ).json()["access_token"]
   ```
3. Call the API with `Authorization: Bearer <token>`:
   ```python
   resp = requests.get(
       "https://id.who.int/icd/release/11/2024-01/mms/search",
       params={"q": "diabetes"},
       headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "Accept-Language": "en"},
   )
   ```
   Wrap each result's definition text as a `MedicalDocument` the same way `WHOGHOLoader` does.

### C. Medical books / textbooks (PDF or DOCX)
Already supported end-to-end (`DocumentLoader.load_pdf` / `load_docx` in
`src/data_ingestion.py`). Two ways to load a book:

1. **Streamlit:** Data Sources → Upload files → drop the PDF/DOCX in.
2. **CLI (bulk):** drop files into `./data/raw_documents/` then run:
   ```bash
   python scripts/ingest_data.py --source local
   ```
   Each PDF page becomes its own citeable chunk (title = `<book name> — Page N`).

   ⚠️ Only use books you have the legal right to use (public-domain texts like NIH/WHO
   manuals, open-access textbooks, or your own licensed copies). Don't bulk-upload
   copyrighted commercial textbooks.

### D. Any other API (e.g. a hospital knowledge base, MedlinePlus Connect, DrugBank, etc.)
The pattern is always the same — turn each record into a `MedicalDocument` and hand it to
the vector store:

```python
from src.data_ingestion import MedicalDocument
from src.vector_store import VectorStore
from src.config import get_settings

settings = get_settings()
vs = VectorStore(settings.embedding_model, settings.chunk_size, settings.chunk_overlap)
vs.load(settings.vector_store_path)  # keep existing data

doc = MedicalDocument(
    content="<the article/abstract/section text>",
    title="<human-readable title>",
    source="<API or publisher name>",
    year="2024",
    url="<link back to original>",
)
vs.add_documents([doc])
vs.save(settings.vector_store_path)
```
Reuse this snippet for MedlinePlus Connect, DrugBank, ClinicalTrials.gov, an internal
hospital wiki export, etc. — anything that returns text.

---

## 3. Recommended `.env` setup

```bash
# Pick ONE generation provider
LLM_PROVIDER=gemini          # or: openai, mistral, qwen, huggingface
GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Optional, required only for PubMed ingestion
PUBMED_EMAIL=you@example.com

# Retrieval tuning
TOP_K_DOCUMENTS=5
RETRIEVAL_CONFIDENCE_THRESHOLD=0.75
```

## 4. Sanity checklist after adding new sources
1. Sidebar **Chunks / Documents** counters go up.
2. **📊 Insights** tab table lists the new title(s).
3. Ask a question in **💬 Ask MedSource** related to the new content and confirm the
   answer cites the new `[DOC_X]` with the right title in the sources panel.
