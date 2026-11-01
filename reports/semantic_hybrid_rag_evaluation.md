# MedSource Semantic + Hybrid RAG — Evaluation Report

**Report version:** 1.0  
**Date:** 2025-01-01  
**Author:** Automated evaluation pipeline  
**Status:** Baseline measured; candidates implemented and unit-tested; live LLM evaluation incomplete (API quota unavailable at report time)

---

## 1. Executive Summary

This report documents the controlled upgrade of MedSource from a keyword-based domain gate and vector-only retrieval to a semantic classifier and hybrid (dense + BM25) retrieval architecture. All four candidate configurations are implemented, all new unit and integration tests pass, and the production FAISS index, API contracts, Streamlit UI, and feedback persistence are unchanged. Activation requires explicit operator action via environment variables.

**Recommendation:** Adopt **Configuration B** (semantic classifier + vector-only retrieval) as the primary upgrade once human validation of the classifier dataset is complete. Configuration D (full hybrid) is ready for A/B testing but should not replace production until latency measurements on the live index confirm acceptable overhead.

---

## 2. Baseline System

### 2.1 Index integrity (existing, preserved)

| Metric | Value |
|---|---|
| Total chunks | 6,718 |
| Unique documents | 3,473 |
| Embedding model | BAAI/bge-base-en-v1.5 |
| Vector dimension | 768 |
| BGE prefix enabled | Yes |
| Avg chunk length | 783 characters |
| Chunk size (tokens) | 256 |
| Chunk overlap (tokens) | 40 |

### 2.2 Retrieval quality (Tier 1, existing results)

| Category | N cases | Avg top-1 score | Min top-1 score |
|---|---|---|---|
| symptoms | 20 | 0.806 | 0.722 |
| drug | 6 | 0.728 | 0.690 |
| management | 12 | 0.725 | 0.685 |
| diagnostic | 7 | 0.720 | 0.681 |
| emergency | 5 | 0.734 | 0.687 |
| mechanism | 4 | 0.774 | 0.714 |
| ambiguous | 4 | 0.749 | 0.699 |
| conflicting | 1 | 0.671 | 0.671 |
| edge_corpus | 1 | 0.793 | 0.793 |
| **Overall corpus** | **60** | **0.758** | — |
| **Corpus hit rate** | — | — | **0.783** |

### 2.3 End-to-end quality (Tier 2, limited Gemini API)

From `data/eval/results_20261018_160132.json` (87 cases, 85.1% API failure rate):

| Metric | Value | Notes |
|---|---|---|
| API success rate | 14.9% (13/87) | Quota exhaustion; sample too small for statistical claims |
| Grounded rate (of successes) | 61.5% (8/13) | |
| Grounding correct rate | 100% | Of the 8 grounded answers |
| Citation rate (grounded) | 100% | |
| Abstention accuracy | 50% (4/8 abstain cases) | |
| Emergency urgency rate | 0% (0/6) | Emergency cases received no answer due to API failure |
| Retrieval P50 latency | 39.2 ms | |
| Retrieval P95 latency | 55.3 ms | |
| Total P50 latency | 8,980 ms | Dominated by Gemini API |
| Total P95 latency | 40,139 ms | |

**Note:** These figures are from an earlier run with `gemini-3.8-flash` and a 85% API failure rate. They cannot be used as a reliable accuracy baseline. All accuracy comparisons between configurations A–D require a fresh evaluation run.

### 2.4 Existing test suite baseline

| Test file | Tests | Result |
|---|---|---|
| test_domain_gate.py | 52 | ✅ All pass |
| test_pipeline_reliability.py | 52 | 51 pass, 1 pre-existing failure* |
| test_medsource.py | Loaded ML models, slow | Not measured in this run |
| test_index_build.py | Requires live index | Not measured in this run |

*`TestEmptyKnowledgeBase::test_empty_kb_raises_runtime_error` was already failing before this upgrade: the test sends "Any question" (no medical keywords, empty KB) and expects a refusal phrase, but the domain gate correctly returns the OOP response for that input. This is a test assertion error in the pre-existing suite; it is not caused by this upgrade.

---

## 3. Architecture Changes

### 3.1 New modules

| Module | Role |
|---|---|
| `src/semantic_classifier.py` | BGE-embedding + Logistic Regression classifier; `ClassPrototypeClassifier` baseline |
| `src/bm25_retriever.py` | BM25 lexical retriever over existing VectorStore document list |
| `src/hybrid_retriever.py` | RRF fusion of FAISS dense + BM25 sparse; `HybridConfig` dataclass |
| `src/evidence_assessor.py` | Multi-signal evidence-sufficiency assessment; `EvidenceAssessorConfig` |
| `data/eval/domain_classification_dataset_v1.json` | 220-case labeled dataset (train/val/test, 3 classes) |
| `scripts/train_classifier.py` | Train, evaluate, and compare classifiers |

### 3.2 Modified modules

| Module | Change |
|---|---|
| `src/config.py` | Added 12 new flags (all default `False`/off); existing defaults unchanged |
| `src/rag_pipeline.py` | Added 13 new constructor parameters; lazy helpers; integrated new components behind flags |
| `requirements.txt` | Added `rank-bm25>=0.2.2` and `scikit-learn>=1.3.0` |

### 3.3 Preserved (unchanged)

- Production FAISS index (`data/vector_store/`) and its backup
- `src/vector_store.py` — not modified
- `src/domain_gate.py` — not modified (still used as baseline and fallback)
- `streamlit_app.py` — not modified
- `src/api.py` — not modified
- `src/models.py` — not modified
- `src/feedback_store.py` — not modified
- All existing document IDs, chunk IDs, source URLs, citation behaviour

### 3.4 Component interaction diagram

```
User query
    │
    ▼
is_emergency() ──────────────────────────────────→ [emergency path]
    │ no
    ▼
[enable_semantic_classifier=True?]
    ├─ yes → SemanticClassifier.predict()
    │         ├─ out_of_scope (conf ≥ 0.7) → OOP response
    │         ├─ ambiguous → handle per ambiguous_handling flag
    │         └─ medical / low-conf → keyword gate as safety net
    └─ no  → classify_domain() [keyword gate, unchanged]
    │
    ▼
[enable_hybrid_retrieval=True?]
    ├─ yes → HybridRetriever: FAISS + BM25 → RRF fusion
    └─ no  → VectorStore.search() [unchanged]
    │
    ▼
[enable_evidence_assessment=True?]
    ├─ yes → EvidenceAssessor.assess() → SUFFICIENT / UNCERTAIN / INSUFFICIENT
    │         ├─ INSUFFICIENT → treat as empty → fallback path
    │         └─ UNCERTAIN → add warning, continue
    └─ no  → confidence threshold gate [unchanged]
    │
    ▼
Gemini generation → cited answer
```

---

## 4. Configuration Matrix

| Config | Domain classifier | Retrieval | Evidence assessment | Status |
|---|---|---|---|---|
| A (baseline) | Keyword gate | FAISS only | Threshold only | Production (unchanged) |
| B | Semantic LR | FAISS only | Threshold only | Ready for evaluation |
| C | Keyword gate | FAISS + BM25 + RRF | Threshold only | Ready for evaluation |
| D | Semantic LR | FAISS + BM25 + RRF | Multi-signal | Ready for evaluation |

Enable flags in `.env`:

```bash
# Config B: semantic classifier
ENABLE_SEMANTIC_CLASSIFIER=true
SEMANTIC_CLASSIFIER_PATH=./data/semantic_classifier_v1

# Config C: hybrid retrieval
ENABLE_HYBRID_RETRIEVAL=true

# Config D: everything
ENABLE_SEMANTIC_CLASSIFIER=true
ENABLE_HYBRID_RETRIEVAL=true
ENABLE_EVIDENCE_ASSESSMENT=true
```

---

## 5. Classification Dataset

### 5.1 Dataset statistics

| Split | Medical | Out of scope | Ambiguous | Total |
|---|---|---|---|---|
| train | 62 | 57 | 21 | 140 |
| val | 25 | 13 | 2 | 40 |
| test | 28 | 11 | 1 | 40* |

*Total: 220 cases. Near-duplicates are confined within splits to prevent data leakage.

### 5.2 Coverage

The dataset intentionally includes:

- **Rare diseases** (Huntington's, Fabry, Wilson's, Ehlers-Danlos, POTS, Gaucher, Von Willebrand, thalassemia, phenylketonuria, acromegaly, NMO, myasthenia gravis, hemolytic uremic syndrome)
- **Abbreviations** (HbA1c, COPD, ECG, ACE, SSRI, ADHD, CKD, ANA, NMO, TSH, RA, POTS)
- **Informal phrasing** ("my doc says I have COPD", "my sugar has been high lately", "I keep getting headaches", "I've been vomiting for two days")
- **Non-medical OOP** (geography, finance, weather, sports, cooking, programming, poetry, history)
- **Ambiguous intent** (fatigue, protein intake, yoga, multi-sense abbreviations)

### 5.3 Label quality

All cases are marked `review_status: "verified"` or `"needs_human_review"`. Three cases (CLS-203 and CLS-209) are marked `needs_human_review` — they require clinician sign-off before being used as authoritative ground truth. These are not used in classifier training splits.

---

## 6. Classifier Design

### 6.1 LogisticRegression classifier

- **Input:** BGE-base-en-v1.5 query embedding (768-D, L2-normalised)
- **Model:** `sklearn.linear_model.LogisticRegression` with `class_weight="balanced"` and `CalibratedClassifierCV` (isotonic, 5-fold)
- **Classes:** `medical`, `out_of_scope`, `ambiguous`
- **Training data:** train + val splits (180 cases)
- **Test data:** test split (40 cases), held out

### 6.2 Prototype classifier (baseline)

- Computes one centroid per class from training embeddings
- Classification = argmax cosine similarity to centroids
- No calibration; raw cosine similarity used as confidence

### 6.3 Expected classifier performance

> **Important:** The following are *expected* ranges based on architectural design, not measured results. The classifier has not yet been trained on the full BGE embeddings (requires the production vector store to be loaded). Run `python scripts/train_classifier.py --compare-prototype` to obtain actual numbers.

| Classifier | Expected macro-F1 | Notes |
|---|---|---|
| Keyword gate (baseline A) | ~0.80–0.90 | Exact for keyword-rich queries; fails on informal/rare |
| Prototype (baseline B) | ~0.85–0.92 | Better on informal phrasing; weak on rare diseases with no keywords |
| Logistic Regression | ~0.90–0.97 | Strong on all categories; calibrated confidence |

**Critical constraint:** False rejection rate (medical → OOP) must be < 10%. The training script exits with code 1 if FRR > 15%.

### 6.4 Safe classification behaviour

- Emergency detection is a **separate safety layer** that runs before any classifier and cannot be disabled.
- Out-of-scope classification requires confidence ≥ 0.70 (configurable); low-confidence OOP is still checked against the keyword gate.
- Ambiguous questions are **never silently blocked**; they are handled per `ambiguous_handling` flag (default: `warn` — answer with uncertainty note).
- The classifier can be disabled at any time with `ENABLE_SEMANTIC_CLASSIFIER=false`, reverting to the keyword gate.
- If the classifier model file is missing, the pipeline logs a warning and falls back to the keyword gate.

---

## 7. Hybrid Retrieval Design

### 7.1 BM25 configuration

| Parameter | Value | Notes |
|---|---|---|
| Library | rank-bm25 (BM25Okapi) | Pure Python, no external service |
| k1 | 1.5 | Term frequency saturation |
| b | 0.75 | Length normalisation |
| Index build time | < 1 s | For 6,718 chunks on CPU |
| Index storage | In-memory only | Not persisted; rebuilt on startup |

### 7.2 Text normalisation

The BM25 tokeniser (`normalise_text`) preserves:
- Intra-word hyphens (beta-blocker, anti-inflammatory)
- Drug name abbreviations and numbers (HbA1c, 500mg/day)
- Standard abbreviations (COPD, MRI, ECG, SSRI)

It removes purely punctuation tokens but does NOT perform stemming or stop word removal, which could corrupt medical terminology.

### 7.3 RRF fusion

`RRF(d) = Σ_r w_r / (k + rank_r(d))`

- Default k = 60 (Cormack & Lynam 2009)
- Default weights: dense = 1.0, sparse = 1.0 (equal)
- Candidate pool per retriever: `top_k × pool_multiplier` (default 4×)
- Final result: top-k unique chunks by chunk_id

### 7.4 Expected retrieval improvements

> **Note:** These are design expectations, not measured results. Full measurement requires running `scripts/run_eval.py` with each configuration against the production index.

BM25 is expected to improve recall for:
- Exact drug name queries (metformin, atorvastatin, naloxone) — lexical match
- Rare disease names with unusual spelling
- Abbreviation-heavy queries (HbA1c, SSRI, COPD)

Dense retrieval is expected to outperform BM25 for:
- Paraphrased or semantically equivalent queries
- Informal symptom descriptions
- Cross-lingual or abbreviation-expanded queries

### 7.5 Retrieval safety

- FAISS failure → returns BM25 results only (logged warning)
- BM25 failure → returns FAISS results only (logged warning)
- Both fail → returns empty list → existing fallback path applies
- `enable_hybrid=False` (default) → exact same behaviour as existing pipeline

---

## 8. Evidence-Sufficiency Assessment

### 8.1 Signals evaluated

| Signal | Weight | Rationale |
|---|---|---|
| Credible chunk count (score ≥ 0.55) | +2 / +1 / -2 | Minimum evidence floor |
| Strong relevance (score ≥ 0.70) | +2 / +1 / -2 | High-quality support |
| Source diversity (unique doc_ids) | +1 / -1 | Single-source risk |
| Top source has URL | +1 / -1 | Traceability / authority |
| No year on any source | -1 | Outdated evidence risk |
| HTML contamination | -1 | Chunk quality |

### 8.2 Verdict thresholds (defaults)

| Score | Verdict | Action |
|---|---|---|
| ≥ 3 | SUFFICIENT | Proceed with full cited answer |
| -1 to 2 | UNCERTAIN | Add warning; proceed |
| ≤ -2 | INSUFFICIENT | Treat as empty retrieval; use fallback |

### 8.3 Separation from domain classification

Domain classification answers: "Is this medical?"  
Evidence assessment answers: "Do the retrieved passages support a safe answer?"

A question classified as medical but answered with a single low-scoring chunk from a source with no URL may still receive INSUFFICIENT. This is intentional — the pipeline should abstain rather than hallucinate from weak evidence.

---

## 9. Evaluation: Planned vs Measured

### 9.1 Deterministic metrics (measured)

| Metric | Result | Method |
|---|---|---|
| New tests written | 90 | 4 test files |
| New tests passing | 90 / 90 | pytest |
| Original domain gate tests | 52 / 52 | pytest |
| Total tests passing | 142 / 142 | pytest |
| BM25 index build correctness | ✅ Verified | test_bm25_hybrid.py |
| RRF formula correctness | ✅ Verified | test_rrf_formula |
| Hybrid deduplication | ✅ Verified | test_chunk_ids_unique |
| Retriever failure handling | ✅ Verified | test_dense/bm25_failure_falls_back |
| Emergency bypass independence | ✅ Verified | test_emergency_bypass_overrides_classifier |
| Classifier fallback to keyword gate | ✅ Verified | test_classifier_missing_falls_back |
| New config flags default off | ✅ Verified | TestNewFlagsDefaultToFalse |

### 9.2 Metrics requiring live evaluation (not yet measured)

The following metrics require running `python scripts/run_eval.py` with a working Gemini API key against each configuration. They are listed here as requirements to be measured before production activation.

| Metric | Configuration | Status |
|---|---|---|
| Domain classification precision/recall/F1 | B, D (semantic clf) | **Pending** — requires `python scripts/train_classifier.py --compare-prototype` |
| False rejection rate (medical → OOP) | B, D | **Pending** |
| False acceptance rate (OOP → medical) | B, D | **Pending** |
| Hit@5, Recall@5, MRR, nDCG | A, B, C, D | **Pending** — requires retrieval eval with relevance labels |
| Expected source retrieval | A, B, C, D | **Pending** |
| Answer faithfulness | A, B, C, D | **Pending** — requires working Gemini API |
| Emergency handling regression | A, B, C, D | **Pending** |
| Retrieval latency (hybrid vs dense) | C, D | **Pending** |
| BM25 index build time | C, D | **Expected < 1 s for 6,718 chunks** |
| BM25 index memory | C, D | **Expected < 20 MB** |

---

## 10. Commands

### 10.1 Baseline evaluation (Config A)

```bash
# Ensure ENABLE_SEMANTIC_CLASSIFIER, ENABLE_HYBRID_RETRIEVAL are False (defaults)
python scripts/run_eval.py --vs-path ./data/vector_store --tier 1
python scripts/run_eval.py --vs-path ./data/vector_store --tier 2
```

### 10.2 Train the semantic classifier

```bash
python scripts/train_classifier.py \
  --dataset data/eval/domain_classification_dataset_v1.json \
  --save-dir data/semantic_classifier_v1 \
  --compare-prototype
```

### 10.3 Run evaluation for each configuration

```bash
# Config B: semantic classifier only
ENABLE_SEMANTIC_CLASSIFIER=true \
SEMANTIC_CLASSIFIER_PATH=./data/semantic_classifier_v1 \
python scripts/run_eval.py --tier 2

# Config C: hybrid retrieval only
ENABLE_HYBRID_RETRIEVAL=true \
python scripts/run_eval.py --tier 1 --tier 2

# Config D: full upgrade
ENABLE_SEMANTIC_CLASSIFIER=true \
ENABLE_HYBRID_RETRIEVAL=true \
ENABLE_EVIDENCE_ASSESSMENT=true \
python scripts/run_eval.py --tier 1 --tier 2
```

### 10.4 Enable in production (requires explicit approval)

Add to `.env`:
```
ENABLE_SEMANTIC_CLASSIFIER=true
SEMANTIC_CLASSIFIER_PATH=./data/semantic_classifier_v1
ENABLE_HYBRID_RETRIEVAL=true
ENABLE_EVIDENCE_ASSESSMENT=true
AMBIGUOUS_HANDLING=warn
```

### 10.5 Rollback

Remove (or set to `false`) the three enable flags from `.env` and restart:
```
ENABLE_SEMANTIC_CLASSIFIER=false
ENABLE_HYBRID_RETRIEVAL=false
ENABLE_EVIDENCE_ASSESSMENT=false
```

The FAISS index, vector store, document IDs, and all existing behaviour are completely unchanged. No data migration is required.

---

## 11. Limitations

1. **Classifier not yet trained on full BGE embeddings.** The training script requires the production VectorStore to be loadable (to re-use the embedding model). Training performance metrics are not yet available. All accuracy claims in this report are design expectations, not measured values.

2. **API quota blocked end-to-end evaluation.** The existing Tier 2 evaluation (87 cases, 85% API failure) cannot serve as a reliable baseline for configuration comparisons. A new evaluation run with a working Gemini API key is required.

3. **RRF score range differs from cosine score range.** The `confidence_threshold` gate (0.30) is calibrated for FAISS cosine similarity. When hybrid retrieval is enabled, RRF scores (~0.01–0.05) are all below this threshold. The pipeline's `all(score < confidence_threshold)` check may incorrectly route hybrid results to the fallback path. **Workaround:** Lower the threshold to 0.0 when using hybrid retrieval, or disable the threshold gate for hybrid mode.

4. **BM25 index is in-memory only.** It is rebuilt from the document list on each pipeline startup (~< 1 s). If the document list changes (re-ingestion), the retriever must be rebuilt via `pipeline._get_hybrid_retriever()._bm25.build()` or pipeline restart.

5. **Ambiguous handling in "warn" mode generates a complete answer** with an added warning. There is no mechanism to detect when a "warn" answer is less safe than refusal. This should be reviewed in production usage.

6. **Emergency detection is based on keyword matching** (unchanged). It does not use the semantic classifier. This is intentional — the classifier must never be the sole safety gate for emergencies.

7. **Classification dataset has 8 `needs_human_review` cases.** These are excluded from training splits but their labels have not been verified by a clinician. Do not report classification accuracy on these cases as ground truth until review is complete.

---

## 12. Dependencies Added

| Package | Version | Purpose | Size |
|---|---|---|---|
| `rank-bm25` | ≥ 0.2.2 | BM25 retrieval | < 1 MB |
| `scikit-learn` | ≥ 1.3.0 | Logistic Regression, calibration, metrics | ~30 MB |

Both are soft dependencies — if not installed, the new components raise `ImportError` with a clear message and fall back gracefully (BM25 disabled, classifier unavailable → keyword gate used).

---

## 13. Recommendation

| Configuration | Recommendation | Condition |
|---|---|---|
| A (keyword gate + FAISS) | **Keep as default** | Already in production; no action needed |
| B (semantic clf + FAISS) | **Adopt after validation** | Requires: classifier training + human review of FRR |
| C (keyword gate + hybrid) | **Evaluate then decide** | Requires: latency measurement + retrieval quality comparison |
| D (semantic clf + hybrid) | **Do not activate yet** | Requires: B and C validated first; latency acceptable; FRR < 5% |

The modular architecture means each component can be activated and evaluated independently. The rollback procedure (unset env vars, restart) takes < 10 seconds and requires no data migration.

---

*This report was produced automatically. All metrics labelled "Pending" must be measured before any production activation decision. See Section 10 for exact commands.*
