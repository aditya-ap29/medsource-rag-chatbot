# Data Sources Guide — MedSource

MedSource only knows what's inside its **vector store**. Anything you add here becomes
retrievable evidence the chatbot can cite as `[DOC_X]`. There are two ways to add data:
the **Streamlit "📥 Data Sources" tab** (easiest, no code), or the **command-line
ingestion script** (best for bulk/automated loads).

---

## 1. Easiest way: the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open the **📥 Data Sources** tab. It has eight sub-tabs:

| Sub-tab | What it's for | API key? |
|---|---|---|
| 📄 Upload files | Medical books/textbook chapters as PDF, DOCX, TXT, or a JSON list of `{title, content, source, year, url}` | No |
| 🔬 PubMed API | Peer-reviewed research abstracts from NCBI's E-utilities | Email only |
| 🌍 WHO GHO Indicators | Official WHO Global Health Observatory statistics | No |
| 🌐 WHO / CDC / NIH URL | Scrape any public fact-sheet URL | No |
| 📖 MedlinePlus (NIH) | NLM patient-education articles, free-text or ICD-10 code lookup | No |
| 💊 FDA Drug Labels | FDA-approved drug label sections (openFDA API) | No (optional) |
| 🧪 Clinical Trials | Completed study summaries from ClinicalTrials.gov | No |
| ✍️ Paste text | Manually paste a guideline excerpt or note | No |

Click the ingest button in a sub-tab → text is chunked, embedded, and saved to
`./data/vector_store/` immediately. No server restart needed — the sidebar's
Chunks / Documents counters update right away.

---

## 2. Data source options in detail

### A. PubMed (NCBI E-utilities)

Free, no API key required; NCBI asks for a contact email with every request.

```bash
# Custom queries
python scripts/ingest_data.py --source pubmed \
  --pubmed-queries "type 2 diabetes management" "vitamin D deficiency" \
  --max-pubmed-results 50

# Broad preset — 55 curated queries covering the most common conditions
python scripts/ingest_data.py --source pubmed-broad
```

Set `PUBMED_EMAIL=you@example.com` in `.env` (or enter it in the Streamlit PubMed tab).

**Docs:** https://www.ncbi.nlm.nih.gov/books/NBK25501/

---

### B. MedlinePlus (NLM / NIH) — patient-education articles

MedlinePlus is the U.S. National Library of Medicine's free patient-education database.
Two ingestion paths, both with no API key needed:

**B1. Free-text search** (`MedlinePlusLoader.search(query)`)
```bash
python scripts/ingest_data.py --source medlineplus \
  --medlineplus-queries "diabetes" "hypertension" "anemia" "stroke"
```

**B2. ICD-10 code lookup** (`MedlinePlusLoader.fetch_by_icd10(code)`) — available only in
the Streamlit tab ("📖 MedlinePlus → Or look up by ICD-10 code").

In Python:
```python
from src.data_ingestion import MedlinePlusLoader
loader = MedlinePlusLoader()
docs = loader.search("anemia", max_results=10)
docs += loader.fetch_by_icd10("D50")  # Iron-deficiency anaemia
```

**Docs:** https://medlineplus.gov/connect/service.html

---

### C. FDA Drug Labels (openFDA)

Official FDA-approved prescribing information. The `indications_and_usage`, `warnings`,
`adverse_reactions`, `drug_interactions`, `dosage_and_administration`, and other sections
are each indexed as a separate searchable document.

No API key needed (240 req/min); optional free key from https://open.fda.gov/apis/authentication/
raises the limit to 1 000 req/min.

```bash
python scripts/ingest_data.py --source openfda \
  --drugs "metformin" "lisinopril" "atorvastatin" "amoxicillin" "sertraline"
```

In Python:
```python
from src.data_ingestion import OpenFDADrugLoader
loader = OpenFDADrugLoader()          # or OpenFDADrugLoader(api_key="your-key")
docs = loader.search_drug("metformin", max_results=3)
```

**Docs:** https://open.fda.gov/apis/drug/label/

---

### D. ClinicalTrials.gov

Completed clinical trial summaries — study title, brief summary, eligibility criteria,
primary outcomes, and lead sponsor. Useful for evidence-based treatment questions.

No API key required.

```bash
python scripts/ingest_data.py --source clinical-trials \
  --clinical-trial-queries "type 2 diabetes" "hypertension" "asthma"
```

In Python:
```python
from src.data_ingestion import ClinicalTrialsLoader
loader = ClinicalTrialsLoader()
docs = loader.search(
    "type 2 diabetes GLP-1",
    max_results=5,
    status="COMPLETED",
    min_enrollment=100,   # skip very small studies
)
```

**Docs:** https://clinicaltrials.gov/data-api/api

---

### E. WHO — two different kinds of official data

WHO doesn't expose one single API for everything.

**E1. WHO fact sheets (narrative text) — Streamlit tab: "🌐 WHO / CDC / NIH URL"**

Best for Q&A prose. The tab has a quick-add list of 10 official WHO fact sheets (diabetes,
hypertension, TB, malaria, HIV, etc.) — pick topics and click ingest. You can also paste
any other fact-sheet/CDC/MedlinePlus URL and it scrapes + indexes the text.

**E2. WHO Global Health Observatory (GHO) — structured statistics — Streamlit tab: "🌍 WHO GHO Indicators"**

Official, free, no-API-key statistics API:
```
GET https://ghoapi.azureedge.net/api/{IndicatorCode}
GET https://ghoapi.azureedge.net/api/{IndicatorCode}?$filter=SpatialDim eq 'IND'
```
`WHOGHOLoader` (already in `src/data_ingestion.py`) converts each country's rows into
short, citeable prose sentences before indexing. The Streamlit tab has a dropdown of
common indicators (life expectancy, obesity prevalence, diabetes prevalence, TB/HIV/malaria
incidence, immunization coverage, etc.) plus an optional ISO3 country filter.

Full indicator catalogue: https://www.who.int/data/gho/data/indicators

**E3. (Optional, advanced) WHO ICD-11 API — disease classification**
Requires free registration at https://icd.who.int/icdapi. Exchange `client_id` /
`client_secret` for a bearer token, then call:
```python
resp = requests.get(
    "https://id.who.int/icd/release/11/2024-01/mms/search",
    params={"q": "diabetes"},
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Accept-Language": "en",
    },
)
```
Wrap each result's definition text as a `MedicalDocument` the same way `WHOGHOLoader` does.

---

### F. Medical books / textbooks (PDF or DOCX)

Already supported end-to-end. Two ways:

1. **Streamlit:** Data Sources → Upload files → drop the PDF/DOCX in.
2. **CLI (bulk):** drop files into `./data/raw_documents/` then run:
   ```bash
   python scripts/ingest_data.py --source local
   ```

⚠️ Only use books you have the legal right to use (public-domain texts, NIH/WHO manuals,
open-access textbooks, or your own licensed copies).

---

### G. Any other API or custom source

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

Reuse this pattern for MedlinePlus Connect, DrugBank, ClinicalTrials.gov, an internal
hospital wiki export, etc.

---

## 3. CLI quick-reference

| Goal | Command |
|---|---|
| Bootstrap with curated summaries + MedlinePlus | `python scripts/ingest_data.py --source all` |
| Add broad PubMed coverage (~1 100 abstracts) | `python scripts/ingest_data.py --source pubmed-broad` |
| Add FDA drug labels for common drugs | `python scripts/ingest_data.py --source openfda` |
| Add clinical trial evidence | `python scripts/ingest_data.py --source clinical-trials` |
| Add your own local files | `python scripts/ingest_data.py --source local` |
| Full rebuild from scratch | `python scripts/ingest_data.py --source all --fresh` |
| Audit + remove unverified sources | `python scripts/audit_sources.py --apply` |

---

## 4. Recommended `.env` setup

```bash
# Pick ONE generation provider
LLM_PROVIDER=gemini          # or: openai, mistral, qwen, huggingface
GEMINI_API_KEY=your_key_here
# OPENAI_API_KEY=your_key_here

# Required for PubMed ingestion
PUBMED_EMAIL=you@example.com

# Retrieval tuning
TOP_K_DOCUMENTS=5
RETRIEVAL_CONFIDENCE_THRESHOLD=0.3
```

---

## 5. Sanity checklist after adding new sources

1. Sidebar **Chunks / Documents** counters go up.
2. **📊 Insights** tab table lists the new title(s).
3. Ask a question in **💬 Chat** related to the new content and confirm the
   answer cites the new `[DOC_X]` with the right title in the sources panel.
4. For FDA drug labels, try "What are the side effects of metformin?" and confirm a
   `[DOC_X]` with title containing "FDA Label" appears.
5. For clinical trials, try "What treatments have been studied for type 2 diabetes?" and
   confirm a ClinicalTrials.gov citation appears.
