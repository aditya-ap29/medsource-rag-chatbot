# MedSource RAG Pipeline — Audit Report

> **Scope:** Read-only inspection of source files, persisted vector store, and test suite.
> No code was modified. No live API calls were made. No embeddings were rebuilt.
> All statistics were measured directly from artifacts on disk.
>
> **Verification key:**  
> `[verified]` = confirmed in source code or measured data  
> `[inferred]` = strongly implied by surrounding code but not directly measured  
> `[unknown]` = insufficient information available without running the app

---

## 1. Executive Summary

MedSource is a medical Retrieval-Augmented Generation (RAG) chatbot backed by Google Gemini 2.0 Flash, a FAISS `IndexFlatIP` vector store, and BGE-base-en-v1.5 (768-dim) sentence embeddings. The Streamlit UI provides five tabs (Chat, Upload & Ask, Symptom Checker, Insights, Admin). A FastAPI server exposes the same pipeline as a `/query` REST endpoint.

**The codebase is in a notably healthy state relative to an earlier version** (documented in `MEDSOURCE_PROJECT_UNDERSTANDING.md`). All four P0 bugs identified in the prior analysis have been fixed:

| Prior bug | Status in current code |
|---|---|
| `_generate()` returned raw error string | **Fixed** — now raises `RuntimeError` |
| `TypeError: unhashable type: 'list'` on second question | **Fixed** — cache key uses only `(role, content)` |
| `add_documents()` lacked dedup guard | **Fixed** — `existing_doc_ids` check added |
| `_build_doc_pipeline()` threshold was `0.0` | **Fixed** — now `0.20` |

**Remaining confirmed issues are lower severity.** The most actionable are:

1. **`.env` sets `LLM_MODEL=gemini-3.8-flash`** — a non-existent model name. Every query call will fail at the Gemini API layer. `[verified]`
2. **Chunk-size mismatch**: the persisted index was built with `chunk_size=500` tokens; `config.py` defaults to `256`. A fresh ingest or re-embed would produce a differently-chunked index without warning. `[verified]`
3. **917 MedlinePlus chunks contain raw HTML** (`<span class="qt0">…</span>`) — these degrade semantic retrieval and pollute excerpts shown to users. `[verified]`
4. **70% of chunks have no year metadata** — most MedlinePlus, patient-education, and FDA chunks carry `year=None`. `[verified]`
5. **`LLM_MODEL` override in `.env` is a single point of breakage** — there is no validation that the configured model name is a valid Gemini model. `[verified]`

All 73 unit and regression tests (across `test_medsource.py` and `test_pipeline_reliability.py`) pass with zero failures.

---

## 2. Architecture and Execution-Flow Diagram

```
Browser / HTTP client
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Streamlit UI  (streamlit_app.py)                               │
│  • Session state: messages, vector_store, user_doc_vs           │
│  • Safety: _detect_emergency(), _detect_medication_risk()       │
│  • Routes: Chat / Upload&Ask / Symptom / Insights / Admin       │
│  • _execute_query()  ──────────────────────────────────────┐   │
└───────────────────────────────────────────────────────────────┘ │
                                                                   │
  FastAPI (src/api.py)  POST /query  ──────── also calls ─────────┘
                                                                   │
                                                          ▼
                                             ┌────────────────────┐
                                             │  RAGPipeline       │
                                             │  (rag_pipeline.py) │
                                             │                    │
                                             │  1. Cache check    │
                                             │     OrderedDict    │
                                             │     (128-entry LRU)│
                                             │                    │
                                             │  2. Retrieval      │
                                             │    ┌────────────┐  │
                                             │    │ VectorStore│  │
                                             │    │ search()   │  │
                                             │    │  BGE query │  │
                                             │    │  prefix +  │  │
                                             │    │  encode    │  │
                                             │    │  FAISS     │  │
                                             │    │  IndexFlatIP│  │
                                             │    └────────────┘  │
                                             │                    │
                                             │  [optional]        │
                                             │  Live PubMed       │
                                             │  (disabled by def.)│
                                             │                    │
                                             │  [optional]        │
                                             │  CrossEncoder      │
                                             │  reranker          │
                                             │  (disabled by def.)│
                                             │                    │
                                             │  3. Confidence gate│
                                             │     score≥threshold│
                                             │     → grounded     │
                                             │     else → fallback│
                                             │                    │
                                             │  4. _generate()    │
                                             │     Gemini API     │
                                             │     (3 retries,    │
                                             │      exp backoff)  │
                                             │                    │
                                             │  5. Build sources  │
                                             │     (cited only)   │
                                             │                    │
                                             │  6. Return         │
                                             │     QueryResponse  │
                                             └────────────────────┘
```

---

## 3. Component-by-Component Implementation Map

### 3.1 Entry Points

| Component | File | Key Lines | Notes |
|---|---|---|---|
| Streamlit page setup | `streamlit_app.py:46-51` | `st.set_page_config()`, `load_dotenv()` | Must be first Streamlit call |
| Session state init | `streamlit_app.py:860-883` | All 9 session keys initialised before widgets | `[verified]` |
| Vector store load | `streamlit_app.py:400-413` | `@st.cache_resource`, calls `vs.load()` if path exists | `[verified]` |
| Pipeline build | `streamlit_app.py:416-437` | `@st.cache_resource`, keyed by `vs_chunk_count` | Cache invalidates when docs added `[verified]` |
| Chat query handler | `streamlit_app.py:459-537` | `_execute_query()` — single authoritative path | `[verified]` |
| FastAPI startup | `src/api.py:53-100` | `lifespan()` asynccontextmanager | Separate object instances from Streamlit `[verified]` |
| FastAPI query route | `src/api.py:~167` | `POST /query` → `rag_pipeline.query()` | `[verified]` |

### 3.2 Query Execution Flow (`streamlit_app.py:_execute_query`)

1. **Active VS selection** (line 470): prefers `user_doc_vs` (uploaded doc) over global `vector_store`.
2. **Empty KB guard** (line 471): raises `RuntimeError` immediately if `active_vs.documents` is empty.
3. **Pipeline selection** (lines 478-482): `_build_doc_pipeline()` for user docs, `build_pipeline()` for global.
4. **Bypass-cache for regeneration** (lines 486-496): pops the cache key so a fresh call is made.
5. **`pipe.query()` call** (line 498): passes `question`, `top_k`, and `st.session_state.messages` as history.
6. **Post-processing** (lines 509-537): safety detection, `_parse_answer()`, `_get_suggestions()`, assemble dict.

### 3.3 RAGPipeline.query (`src/rag_pipeline.py:543-750`)

| Step | Code location | Detail |
|---|---|---|
| Cache check | lines 565-574 | Key = `(q.lower(), k, history_key)`. `history_key` uses only `(role, content)` per turn. |
| Retrieval (no expansion) | lines 581-592 | `VectorStore.search(q, top_k=k*4, score_threshold=threshold)` → dedup by `doc_id` (best score per doc) |
| Retrieval (with expansion) | lines 578-580 | `_retrieve_with_expansion()` — extra Gemini call for 3 query variants |
| Live PubMed merge | lines 597-605 | Only when `enable_live_pubmed=True` (default: False) |
| Reranker | lines 607-610 | Only when `enable_reranker=True` (default: False) |
| Confidence gate | lines 615-634 | Empty or all scores < threshold → fallback or hard refusal |
| Grounded generation | lines 662-675 | `SYSTEM_PROMPT.format(history, context, question)` → `_generate()` |
| Uncertainty re-check | lines 677-700 | If model says "don't have enough" etc. despite having docs → demoted to fallback |
| Confidence scoring | lines 702-537 | Normalised avg score ± citation/uncertainty adjustors |
| Source building | lines 709-732 | Only docs cited as `[DOC_X]` in the answer; fallback to all docs if none cited |
| Cache write | lines 745-748 | Evicts oldest if >128 entries |

### 3.4 VectorStore (`src/vector_store.py`)

| Method | Lines | Behaviour |
|---|---|---|
| `__init__` | 87-120 | Loads `SentenceTransformer`, builds `RecursiveCharacterTextSplitter` (token-aware, tiktoken cl100k_base) |
| `_chunk_documents` | 123-146 | Splits by tokens, separators `\n\n → character`, `keep_separator=False` |
| `_encode_passages` | 148-158 | BGE prefix `"Represent this passage: "` + L2-normalised batch encode |
| `_encode_query` | 160-168 | BGE prefix `"Represent this sentence for searching relevant passages: "` |
| `add_documents` | 170-207 | **Deduplication by `doc_id`** (lines 186-193); skip existing `doc_id`s |
| `search` | 209-233 | Requests `k*3` (min `k+10`) vectors from FAISS; filters to `top_k` above threshold |
| `save` / `load` | 235-295 | Pickle for docs + config; binary faiss format; load restores `use_bge_prefix` from saved config |

### 3.5 Data Ingestion (`src/data_ingestion.py`)

| Loader | Source | Key behaviour |
|---|---|---|
| `PubMedLoader` | NCBI E-utilities | esearch → efetch XML; 0.4 s rate-limit sleep; `doc_id=PMID_{pmid}` |
| `WHOGHOLoader` | WHO GHO OData API | Converts numeric rows to prose sentences per country |
| `MedlinePlusLoader` | NLM HealthTopics XML | HTML is supposedly stripped but `<span class="qt…">` tags survive |
| `OpenFDADrugLoader` | openFDA Drug Label API | 11 sections per drug label; sections >3000 chars truncated |
| `ClinicalTrialsLoader` | ClinicalTrials.gov REST v2 | COMPLETED status only; min enrollment 50 |
| `DocumentLoader` | Local files | PDF (PyPDF2), DOCX (python-docx), TXT, JSON |

### 3.6 Safety Layer (`streamlit_app.py`)

| Function | Lines | Detail |
|---|---|---|
| `_detect_emergency` | 574-580 | 8 compiled regex patterns; runs before RAG |
| `_detect_medication_risk` | 581-582 | 5 compiled regex patterns; runs before RAG |
| `_classify_response` | 585-598 | `grounded / general / uncertain / conflicting` based on sources + confidence + answer text |
| Emergency notice | 683-699 | Shows 112/108 numbers + "call for help" before answer |
| Medication caution | 730-741 | "consult your doctor" caution before answer |
| Disclaimer | 743-750 | Appended to every response via `_clean_answer()` + rendered in UI footer |

---

## 4. Dataset and Source-Quality Findings

> All statistics measured from `data/vector_store/documents.pkl` and `data/vector_store/config.pkl`. `[verified]`

### 4.1 Corpus Statistics

| Metric | Value |
|---|---|
| Total chunks in FAISS index | **3,123** |
| Unique `doc_id`s | **2,018** |
| FAISS index type | `IndexFlatIP` (exact cosine) |
| Embedding dimension | **768** (BAAI/bge-base-en-v1.5) |
| Stored chunk_size (config.pkl) | **500 tokens** |
| Current code default chunk_size | **256 tokens** ← MISMATCH |
| Stored chunk_overlap | 50 tokens |
| Average chunk length | ~174 words |
| Min chunk words | 3 |
| Max chunk words | 406 |

### 4.2 Source Distribution

| Source label | Chunks | % |
|---|---|---|
| FDA Drug Label (openFDA) | 1,198 | 38.4% |
| MedlinePlus (NLM/NIH) | 922 | 29.5% |
| PubMed | 824 | 26.4% |
| ClinicalTrials.gov | 118 | 3.8% |
| Patient Education Summary | 61 | 2.0% |

### 4.3 Metadata Quality

| Field | Coverage | Notes |
|---|---|---|
| `url` | 3,062 / 3,123 (98%) | 61 Patient Education chunks have no URL |
| `year` | 940 / 3,123 (30%) | 2,183 chunks have `year=None` |
| `metadata.provenance` | Sparse | Only PubMed and FDA chunks have metadata beyond category/type |

#### Year distribution anomalies
- **793 chunks dated 2026** — these are FDA drug-label records; FDA labels use the approval/revision date, which can be current or future-facing. Not necessarily wrong, but worth noting.
- **1 chunk dated 2027** (PubMed: insulin delivery cannula study). Possible pre-print date.
- **2,183 chunks have `year=None`** — MedlinePlus, Patient Education, and some PubMed records. Users see "N/A" in source citations.

### 4.4 Content Quality Issues

**HTML contamination in MedlinePlus chunks** `[verified — measured]`

917 chunks (29.4% of the corpus) contain raw HTML markup, primarily from MedlinePlus. Examples:

```
What is <span class="qt0">diabetes</span>?<p><span class="qt0">Diabetes</span>, 
also known as <span class="qt0">diabetes mellitus</span>...
```

The HTML is embedded in the `content` field and is passed verbatim to the BGE encoder and shown to users as source excerpts. This contaminates the embedding space because the model encodes `<span class="qt0">diabetes</span>` differently from `diabetes`, reducing retrieval precision for MedlinePlus documents.

**Very short chunks** `[verified]`

28 chunks have fewer than 20 words. Three examples:
- `"It is rare"` (MedlinePlus)
- `"of Health and Human Services Office on Women's Health</p>"` (HTML fragment)
- `"Then throw away the tissue and wash your hands. Staying home when sick. Centers "` (partial sentence)

These pass the FAISS index with no quality filter and could be returned as top-k results.

**Duplicate content (approximate)** `[verified]`

129 chunks share the same first 200 characters with at least one other chunk. This occurs because `doc_id` deduplication is based on the whole-document `doc_id`, not chunk content. If two different source documents include the same boilerplate paragraph (e.g., FDA label standard disclaimers), both are indexed.

**Processed corpora not in the FAISS index**

The `data/processed/` directory holds 3,498 items across 8 JSON files (including `pubmed_corpus.json` with 1,429 items and `medlineplus_corpus.json` with 335 items). The FAISS index contains only 3,123 chunks (2,018 unique doc_ids). **It is unknown whether these processed files are subsets of what is indexed, supersets, or partially overlapping.** `[unknown]`

---

## 5. Retrieval and Generation Findings

### 5.1 Top-k Selection and Thresholds

| Parameter | Value | Source |
|---|---|---|
| `top_k` (documents to LLM) | 5 | `config.py:37` |
| FAISS candidates requested | `max(k*4, 20) = 20` minimum | `rag_pipeline.py:582-583` |
| `confidence_threshold` | **0.30** (BGE cosine) | `config.py:38` |
| User-upload pipeline threshold | **0.20** | `streamlit_app.py:449` |
| `BGE_PRACTICAL_MAX` (normalisation ceiling) | 0.85 | `rag_pipeline.py:506` |

The FAISS `search()` method applies `score_threshold` as a pre-filter: any chunk scoring below 0.30 is dropped before `top_k` trimming. The `VectorStore.search()` method requests `min(k*3, k+10)` — actually `k_request = min(max(k*3, k+10), len(documents))` — candidates from FAISS (line 220), then filters by threshold and limits to `top_k`. In `RAGPipeline.query()`, it requests `top_k=k*4` from the store, then deduplicates by `doc_id`, keeping the best-scored chunk per document.

**Implication**: a document with 14 chunks (the busiest in the index) will only contribute one representative chunk to the final context, regardless of how many chunks score above threshold. This is deliberate (dedup by `doc_id`) but means that for a long document, only one passage is shown to the LLM even if multiple sections are relevant.

### 5.2 Confidence Scoring Mechanics

```
raw_scores = [score_1, score_2, ..., score_k]    (BGE cosine, [0.30, 0.85])
avg_raw = mean(raw_scores)
normalised = (avg_raw - 0.30) / (0.85 - 0.30)   # → [0.0, 1.0]
if has_citations: normalised *= 1.1 (capped at 1.0)
if admits_uncertainty: normalised *= 0.5
confidence = round(normalised, 3)
```

**Limitations of this approach:**
- A query scoring exactly at threshold (0.30) normalises to 0.0 — even though a result was retrieved and cited.
- The +10% citation bonus is awarded for any `[DOC_` occurrence in the answer, including citations copied verbatim from the context block format. The LLM would have to completely omit all citation syntax to not receive the bonus.
- The ×0.5 uncertainty penalty applies to phrases in the answer text (e.g., "don't have enough"). These phrases are also present in the refusal response the LLM is instructed to use when evidence is insufficient, so the penalty may double-penalise cases already routed to the general fallback.
- Cross-encoder (reranker) scores are in an unbounded domain (e.g., −6 to +6). When enabled, `is_ce_domain` is detected heuristically by checking if any score is `>1.0` or `<0.0`. This detection can fail if all CE scores happen to fall in `[0.0, 1.0]`, causing the sigmoid normalisation to be skipped.

### 5.3 Empty Retrieval and Fallback Handling

Two confidence gates exist:

**Gate 1** (line 615): triggered when `retrieved_docs` is empty or all scores < threshold:
- `enable_general_fallback=False` → hard refusal (`QueryResponse` with fixed text, no generation).
- `enable_general_fallback=True` → `GENERAL_KNOWLEDGE_PROMPT` → Gemini generation with `sources=[], confidence=0.0`.

**Gate 2** (line 677): triggered after grounded generation if the answer contains uncertainty phrases:
- Always demotes to general fallback regardless of `enable_general_fallback`.
- The uncertainty phrases checked: `"don't have enough"`, `"cannot answer"`, `"insufficient information"`, `"not enough verified"`.
- These are the exact same phrases the system prompt instructs the LLM to use as refusal. In practice, if the LLM uses the prescribed refusal phrasing, Gate 2 catches it correctly.

**Risk**: the generic "not enough verified information" warning is triggered whenever retrieval fails (no documents above threshold) and the fallback answer itself doesn't match the uncertainty phrases. In that case, the response has a `warning` field set but `confidence=0.0` and `sources=[]`. The UI `_classify_response()` function detects `not response.sources` → `"general"`, which triggers the `_render_uncertainty_notice()` block. This path is correct.

### 5.4 Conversation History Construction

- Last `max_history_turns=3` assistant+user pairs are included.
- Token budget: 400 tokens for history (line 29).
- Format: `"Prior conversation (most recent last):\nUser: …\nAssistant: …\n"`.
- History passed to `pipe.query()` is `st.session_state.messages` — **the full list including all assistant message dicts** (which contain `sources`, `suggestions`, `badge_html`, etc.). The pipeline extracts only `role` and `content` when building the cache key (lines 565-568) and when calling `_format_history()` (line 662). This is correct and tested.

**Risk**: the history passed to `_format_history()` is `st.session_state.messages`, which can contain user messages beginning with `[Doc]` or `[Symptom Checker]` prefixes. These prefixes are stripped for sidebar display (lines 953-954) but are passed verbatim to the LLM in history. This may cause the LLM to produce slightly odd history-aware responses.

### 5.5 Cache Key and Isolation

- Cache key: `(question.strip().lower(), k, history_key)` where `history_key = tuple((role, content) for each turn)`.
- **Correctly isolated**: different history → different key. Same question asked by different users with different histories → different cache entries.
- **Risk**: the pipeline instance is shared across all Streamlit sessions via `@st.cache_resource`. The `_cache_lock` (line 138) guards concurrent reads/writes. However, `_query_cache` is cleared when the admin clicks "Reload index" (`st.cache_resource.clear()`, line 982), which destroys the pipeline object and creates a new one — the old cache is garbage-collected. This is correct behaviour.
- **Regeneration cache bypass** (lines 486-496): pops the key from `pipe._query_cache` before calling `query()`. Uses an identical key construction to the pipeline's own key, so the bypass is reliable.
- **Edge case**: if `st.session_state.messages` grows very long (>128 message pairs), the `history_key` tuple becomes large and hashable but may dominate cache memory. There is no per-key size limit.

---

## 6. Gemini and Answer Generation Audit

### 6.1 Model Configuration

| Parameter | Value | Source |
|---|---|---|
| Model name | **`gemini-3.8-flash`** (from `.env`) | `settings.llm_model` at runtime |
| Code default (if unset) | `gemini-2.0-flash` | `rag_pipeline.py:111` |
| Temperature | 0.1 | `config.py:17` |
| Max output tokens | 1,000 | `config.py:18` |
| API client | `google.genai.Client` | `rag_pipeline.py:140` |

**CONFIRMED ISSUE**: The active `.env` sets `LLM_MODEL=gemini-3.8-flash`. As of the time of this audit, `gemini-3.8-flash` is not a recognised Google Gemini model name. Every live query will fail at `_client.models.generate_content()`. The failure mode is a `RuntimeError` with the message `"The AI service returned an error. Please try again."` — which is surfaced correctly to the user, but the root cause is invisible without reading the `.env`. `[verified — observed via settings at runtime]`

### 6.2 Prompt Construction

**Grounded SYSTEM_PROMPT** (`rag_pipeline.py:41-67`):
```
You are MedSource ... Answer ONLY from the retrieved documents below.
RULES: 1. Cite as [DOC_X]. 2. No Sources list. 3. No diagnose/prescribe. 4. Refusal phrase if insufficient.
OUTPUT FORMAT: **Direct Answer:** / **Explanation:** / **Practical Guidance:**
{history}Retrieved Context:
{context}
User Query: {question}
```

**Context format** (lines 322-333):
```
[DOC_1] Title (Source, Year)
Content: <chunk text>
Relevance Score: 0.XXX
```
Context is token-capped at 1,500 tokens (line 25). Low-scoring chunks are dropped when this limit is reached (not just the last one — the loop breaks as soon as `used + block_tokens > max_tokens`).

**General knowledge fallback prompt** (`rag_pipeline.py:70-85`): starts with "Note: This answer is based on general medical knowledge…". No citations are expected or possible.

### 6.3 Retry and Error Handling

| Signal | Retried? | Max attempts | Notes |
|---|---|---|---|
| HTTP 429 / quota / resource_exhausted | Yes | 4 (1 + 3 retries) | Exponential backoff ×2, ±25% jitter, cap 16 s |
| HTTP 503 / internal / unavailable | Yes | 4 | Same backoff |
| timeout / deadline | Yes | 4 | Same backoff |
| Invalid API key | No | 1 | Treated as non-transient |
| Other exceptions | No | 1 | Not in `_TRANSIENT_SIGNALS` |

After exhausting retries, `_generate()` raises `RuntimeError` with a **human-readable** message (never the raw SDK error text). The three message categories are: rate-limited, timeout, temporarily unavailable. A fourth catch-all produces "The AI service returned an error. Please try again."

**Gap**: The `_TRANSIENT_SIGNALS` tuple matches on substrings of `str(exc).lower()`. A custom exception whose `.args[0]` does not contain those strings is treated as non-transient even if it is (e.g., connection reset by peer). No logging of the exception class type — only the string.

### 6.4 Citation Generation and Source-to-Claim Mapping

The LLM is instructed to write `[DOC_X]` inline. After generation, the pipeline:

1. Scans `answer` for `[DOC_(\d+)]` occurrences (line 710).
2. Builds `cited_indices: Set[int]`.
3. Builds `Source` objects only for retrieved docs whose 1-based index is in `cited_indices`.
4. **If `cited_indices` is empty (model cited nothing)**: falls back to including ALL retrieved docs as sources (lines 716-720). This prevents a source-less response but can produce a misleading source list if the answer is entirely general.

**Out-of-range citation guard**: line 712 checks `1 <= idx <= len(retrieved_docs)`. A `[DOC_6]` citation when only 5 docs were retrieved is silently ignored. No warning is logged.

**Source `doc_id` remapping**: sources are labelled `"DOC_{i}"` based on their 1-based position in `retrieved_docs` (line 723) — not their original `doc_id` from the vector store. This means the displayed `DOC_1` in the source panel refers to position 1 in the result set, not a stable document identifier. If the same question is asked twice and retrieval order changes, `DOC_1` refers to a different document.

### 6.5 Conflict and Insufficiency Handling

- **Conflicting sources**: detected in `_classify_response()` by a regex on the answer text (line 591). The classification does not inspect whether two source documents actually contradict each other — it relies entirely on the LLM having used conflict-indicating language.
- **Insufficient evidence**: Gate 1 (no retrieval) → general fallback or hard refusal. Gate 2 (model admits uncertainty) → demoted to fallback with `sources=[]`.
- **Hallucination prevention**: the system prompt rule 1 ("Answer ONLY from retrieved documents") and rule 4 (exact refusal phrase) are enforced through prompting only. There is no post-generation fact-check against retrieved chunks.

---

## 7. Confirmed Bugs and Supporting Evidence

### Bug 1 — `LLM_MODEL=gemini-3.8-flash` in `.env` (CRITICAL — confirmed runtime)

**Evidence**: `python -c "from src.config import get_settings; print(get_settings().llm_model)"` outputs `gemini-3.8-flash`. This model name does not correspond to any announced Google Gemini model. Every query will fail at the generation step.

**Impact**: All users get `"The AI service returned an error. Please try again."` on every query. The application is non-functional in the current deployment environment.

**File**: `.env` (runtime configuration, not in source control by default)  
**Fix**: Change `LLM_MODEL=gemini-3.8-flash` to `LLM_MODEL=gemini-2.0-flash` (or another valid model name).

---

### Bug 2 — Chunk-size mismatch between stored index and config default (MEDIUM)

**Evidence**: `data/vector_store/config.pkl` records `chunk_size=500`. `src/config.py` defaults to `chunk_size=256`. These differ.

**Impact**: If anyone runs a fresh ingest (e.g., `scripts/ingest_data.py`) or calls `VectorStore.add_documents()` via the admin panel without explicitly setting `CHUNK_SIZE=500` in `.env`, new chunks will be 256 tokens while existing chunks are 500 tokens. The FAISS index will contain mixed-granularity chunks, making retrieval scores less comparable across sources. The `load()` method logs a warning only for model name mismatches, not chunk-size mismatches.

**Files**: `src/config.py:70` (default), `data/vector_store/config.pkl` (stored value)  
**Fix**: Either set `CHUNK_SIZE=500` in `.env` to match the stored index, or document that the index must be rebuilt if `CHUNK_SIZE` changes.

---

### Bug 3 — HTML tags in 917 MedlinePlus chunks contaminate embeddings (MEDIUM)

**Evidence**: Measured directly from `documents.pkl`. 917 of 922 MedlinePlus chunks (99.5%) contain `<span class="qt…">` tags and `<p>` elements. Sample: `What is <span class="qt0">diabetes</span>?<p>...`

**Impact**: BGE encodes these with the HTML tags present. The span tags reduce semantic similarity between related queries and these chunks, degrading recall for MedlinePlus content specifically. The chunks also display raw HTML to users in the source excerpt panel.

**Root cause**: `MedlinePlusLoader._parse_nlm_xml()` uses BeautifulSoup to parse the XML response but the extracted `content` field is the raw result of `.get_text()` on an HTML-structured XML node, which retains inner span tags.

**Files**: `src/data_ingestion.py:464-491`

---

### Bug 4 — `pydantic` `class Config` deprecation warning (LOW — non-breaking)

**Evidence**: Test output shows `PydanticDeprecatedSince20: Support for class-based 'config' is deprecated, use ConfigDict instead.`

**File**: `src/config.py:100-103`  
**Fix**: Replace inner `class Config` with `model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")`.

---

### Bug 5 — `PyPDF2` deprecation warning (LOW — non-breaking)

**Evidence**: Test output shows `DeprecationWarning: PyPDF2 is deprecated. Please move to the pypdf library instead.`

**Files**: `src/data_ingestion.py:11`, `requirements.txt:21`

---

## 8. Risks, Limitations, and Missing Observability

### Risks

| Risk | Severity | Notes |
|---|---|---|
| `gemini-3.8-flash` model name in `.env` | Critical | App is non-functional right now |
| Single FAISS flat index — no ANN approximation | Low | Acceptable for 3,123 chunks on CPU; becomes a bottleneck above ~100K chunks |
| `@st.cache_resource` pipeline shared across Streamlit sessions | Medium | Thread-safe via `_cache_lock`, but `genai.Client` thread-safety is unverified |
| Chat history not persisted across server restarts | Medium | Session state only; restart = lost conversation |
| Gemini free tier rate limit (15 RPM) | Medium | Under concurrent use, retries will stack up, delaying responses |
| No streaming responses | Low | Long answers block the Streamlit thread for the full generation time |
| RAGAS eval uses Gemini as both judge and responder | Low | Self-referential; inflates faithfulness and relevance metrics |
| `documents.pkl` is a Python pickle | Low | Version incompatibilities between Python versions would corrupt the store silently |

### Missing Observability

- **No application logs** — `logs/` directory exists but is empty. Logging is configured at module level (`logging.basicConfig(level=logging.INFO)` in `src/api.py`) but Streamlit does not surface these logs to a file by default. All query-level errors, retrieval timings, and model errors go to stderr only.
- **No metrics collection** — retrieval time, generation time, and confidence are returned in `QueryResponse` but not aggregated or stored anywhere.
- **No alert on generation failure** — if Gemini returns errors repeatedly (e.g., exhausted quota), there is no alerting mechanism.
- **No feedback analysis pipeline** — `FeedbackStore` persists thumbs up/down votes to `data/feedback.db` but there is no analysis script or dashboard for reviewing them.
- **`data/feedback.db-wal` and `data/feedback.db-shm` present** — WAL mode is active and the database has been used.

---

## 9. Test Coverage and Unverified Behaviour

### Test Results (all measured)

| Suite | Tests | Passed | Failed |
|---|---|---|---|
| `tests/test_medsource.py` | 22 | **22** | 0 |
| `tests/test_pipeline_reliability.py` | 51 | **51** | 0 |
| `tests/test_functional.py` | Not run (requires embedding model cold start) | — | — |
| `tests/test_ui_apptest.py` | Not run | — | — |

### What Is Well-Tested

- `RAGPipeline._generate()` raises `RuntimeError` on API failure (not returns error string) `[test_medsource.py:TestGenerationErrorPropagation]`
- Citation alignment: only cited `[DOC_X]` indices appear in `sources` `[test_medsource.py:TestCitationAlignment]`
- `VectorStore.add_documents()` deduplication `[test_medsource.py:TestDuplicateIngestionGuard]`
- Confidence score boundary (at-threshold passes gate) `[test_medsource.py:TestConfidenceBoundary]`
- `_build_doc_pipeline` threshold > 0 `[test_medsource.py:TestDocPipelineThreshold]`
- `TypeError: unhashable type: 'list'` regression (history with list-valued fields) `[test_pipeline_reliability.py:TestUnhashableHistoryRegression]`
- Cache bypass for regeneration `[test_pipeline_reliability.py:TestRegenerationCacheBypass]`
- Retry logic and exponential backoff `[test_pipeline_reliability.py:TestAPIResilience]`
- Feedback store SQLite persistence `[test_pipeline_reliability.py:TestFeedbackStore]`
- Duplicate submission guard `[test_pipeline_reliability.py:TestDuplicateSubmissionPrevention]`
- HTML injection prevention in error messages `[test_pipeline_reliability.py:TestHTMLInjectionPrevention]`
- Regeneration failure restores original assistant message `[test_pipeline_reliability.py:TestRegenFailureRestore]`
- Concurrent cache writes (thread safety) `[test_pipeline_reliability.py:TestCacheIsolation]`

### What Is Not Tested

| Gap | Risk |
|---|---|
| Live FAISS search with real embeddings (all tests use mock VectorStore) | End-to-end retrieval quality is never verified in CI |
| HTML-stripped MedlinePlus content quality | Silent degradation — tests don't detect it |
| Behaviour when `LLM_MODEL` is an invalid model name | The app fails silently at runtime; not caught in tests |
| Chunk-size mismatch between stored index and new ingests | Could produce inconsistent retrieval results undetected |
| `QueryResponse.contexts` field | Always `[]` in the current pipeline (not populated); tests don't catch this |
| Source URL correctness after retrieval | Not validated |
| `_format_history()` with `[Doc]`/`[Symptom Checker]` prefixed user messages | Untested |
| FastAPI `/query` endpoint integration | No integration test; only smoke tests in `_test_run.py` |
| Behaviour under Gemini rate limiting (429 responses) | Mocked in one test but real back-off timing not tested |
| `WHOGHOLoader` and `ClinicalTrialsLoader` ingestion | No unit tests |

### Unverified Claims

- `QueryResponse.contexts` field is defined in `models.py` as `List[str]` with `default_factory=list`. In `RAGPipeline.query()`, this field is never populated — it is always an empty list. The API consumers expecting contexts in the response will get `[]`. `[verified — by inspection, not tested]`
- The cross-encoder reranker's `is_ce_domain` detection (any score > 1.0 or < 0.0) could misclassify if all CE scores fall in [0.0, 1.0]. `[verified by code inspection; not tested with real CE scores]`

---

## 10. Prioritised Recommendations

> Ordered by severity. Estimated complexity: S = hours, M = 1–2 days, L = 3–5 days.

### P0 — Fix Now (blocking)

#### R1 — Fix `LLM_MODEL` in `.env` (S)
Change `LLM_MODEL=gemini-3.8-flash` to `LLM_MODEL=gemini-2.0-flash` in `.env`. Add a startup validation in `RAGPipeline.__init__()` (or `src/config.py`) that logs a warning if the model name doesn't match known Gemini model name patterns.

#### R2 — Document and align `CHUNK_SIZE` in `.env` (S)
Add `CHUNK_SIZE=500` to `.env` to match the stored index. Add a startup check in `VectorStore.load()` that compares `config.pkl.chunk_size` against the current `self.chunk_size` and logs a `WARNING` when they differ (not just for model name).

---

### P1 — High priority (quality and correctness)

#### R3 — Strip HTML from MedlinePlus chunks (M)
Add a `_strip_html(text: str) -> str` helper using `re.sub(r'<[^>]+>', '', text)` and call it on all extracted content in `MedlinePlusLoader._parse_nlm_xml()`. **The existing index must be rebuilt** after fixing the loader because stored chunks already contain HTML. Add a test asserting that MedlinePlus content is HTML-free after parsing.

#### R4 — Filter out very short chunks (S)
Add a minimum content length filter in `VectorStore._chunk_documents()`: skip any chunk with fewer than 50 characters (or 10 tokens). Add the threshold as a configurable parameter.

#### R5 — Validate `contexts` field population (S)
Populate `QueryResponse.contexts` from the retrieved chunk content strings in `RAGPipeline.query()`. This field is part of the public API contract (`models.py`) but is always `[]`. Any API consumer relying on it will get empty responses.

---

### P2 — Medium priority (observability and robustness)

#### R6 — Structured logging to file (M)
Configure a `RotatingFileHandler` in `streamlit_app.py` and `src/api.py` to write structured logs to `logs/medsource.log`. Include: query text (hashed), retrieval time, generation time, confidence score, model used, and any errors. This enables post-hoc debugging and quality monitoring.

#### R7 — Add model name validation in config (S)
Add a `@validator` or `model_validator` in `Settings` that warns if `llm_model` is not one of the known valid Gemini model names. This prevents silent runtime failures from `.env` typos.

#### R8 — Add a startup self-check (M)
Add a test query (`"What is diabetes?"`) at startup (in `lifespan()` for FastAPI; in `load_vector_store()` for Streamlit) that verifies retrieval works and logs the number of results found. This catches index/model mismatches before the first user query.

#### R9 — Replace `class Config` with `model_config = ConfigDict(…)` (S)
Fix the Pydantic v2 deprecation warning in `src/config.py:100-103`.

#### R10 — Replace `PyPDF2` with `pypdf` (S)
Replace `PyPDF2` with `pypdf` in `requirements.txt` and `src/data_ingestion.py`. API is compatible.

---

### P3 — Enhancements (after P0-P2 are done)

#### R11 — Re-ingest MedlinePlus with HTML-stripped content (M–L)
After R3 is implemented, re-run the MedlinePlus ingestion script to refresh the 922 MedlinePlus chunks in the index. Requires `scripts/reembed_vector_store.py` or a targeted re-ingest.

#### R12 — Add feedback analytics endpoint/script (M)
Write a script that reads `data/feedback.db` and produces a summary (total votes, positive rate, message IDs with negative votes). These votes are currently collected but never reviewed.

#### R13 — Add end-to-end smoke test for retrieval (M)
Add a pytest test that loads the real FAISS index and runs a known query (e.g., `"what is diabetes?"`) with a mock Gemini call, asserting that at least one source is returned and the confidence score is > 0. This would have caught the `LLM_MODEL` bug earlier.

---

## 11. Questions That Must Be Answered Before Implementation

1. **What Gemini model name should be used?** `gemini-2.0-flash` is the code default; confirm this is the intended production model. (R1 depends on this.)

2. **Should the existing index be rebuilt with `chunk_size=256` or left at `chunk_size=500`?** Rebuilding requires re-running all ingestion scripts and costs embedding compute time. Keeping at 500 requires updating `config.py` default and `.env`. (R2 depends on this.)

3. **Is the `data/processed/` corpus meant to supplement the FAISS index or is it already fully indexed?** If there are documents in `processed/` that are not in the FAISS index, retrieval is missing those sources. (Cannot determine without running a cross-reference query.)

4. **What is the expected Gemini API quota and rate limit for the target deployment?** The free tier is 15 RPM; if concurrent use is expected, rate-limit errors will be frequent and the 3-retry policy may need tuning.

5. **Is live PubMed retrieval intended to be enabled in production?** It is disabled (`ENABLE_LIVE_PUBMED=false`) and fully implemented. Enabling it adds latency (5 s timeout) and requires a valid `PUBMED_EMAIL` in `.env`.

6. **Are the `data/feedback.db` votes from real users or synthetic testing?** If real, they should be analysed before further changes to understand which response types users find unhelpful.

7. **Is the FastAPI server (`src/api.py`) deployed alongside the Streamlit app, or only the Streamlit app?** The `render.yaml` deploys only the Streamlit app. The FastAPI server starts separately. Confirm which is the intended primary interface.

---

## Appendix A — File Reference

| File | Purpose | Key lines |
|---|---|---|
| `streamlit_app.py` | UI, safety, routing | ~2,033 lines |
| `src/rag_pipeline.py` | Core RAG logic | 750 lines |
| `src/vector_store.py` | FAISS + BGE | 314 lines |
| `src/data_ingestion.py` | 6 loader classes | ~789 lines |
| `src/config.py` | Pydantic Settings | 109 lines |
| `src/models.py` | Pydantic models | 62 lines |
| `src/api.py` | FastAPI backend | ~200 lines |
| `src/feedback_store.py` | SQLite feedback | 116 lines |
| `src/live_retrieval.py` | Live PubMed | 229 lines |
| `tests/test_medsource.py` | Unit tests (22) | 477 lines |
| `tests/test_pipeline_reliability.py` | Regression tests (51) | ~1,165 lines |
| `tests/test_functional.py` | AppTest functional | ~300 lines |
| `tests/test_ui_apptest.py` | AppTest UI | ~200 lines |

---

*Generated by read-only source inspection. No files were modified. No API calls were made.*
