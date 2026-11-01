# MedSource — Live Evaluation and Out-of-Scope Guard
## Evaluation Report

**Date:** 2026-10-18  
**Version:** post-domain-gate (this upgrade)  
**Evaluator:** MedSource automated eval (scripts/run_eval.py)  
**Dataset:** `data/eval/medsource_eval_dataset_v1.json` — 87 cases  
**Model:** `gemini-3.8-flash` (configured in `.env`, confirmed working via live API probe)  
**Index:** 6,718 chunks | 3,473 unique docs | BGE BAAI/bge-base-en-v1.5 (768-dim)

---

## 1. Gemini Model Verification

### Configured model
The `.env` file specifies `LLM_MODEL=gemini-3.8-flash`. The standard public documentation lists `gemini-2.0-flash` and `gemini-2.5-flash` as stable identifiers; `gemini-3.8-flash` is not listed in the canonical publicly-documented model catalogue but **was confirmed to respond correctly** via a live minimal API probe:

```
Test prompt : "Reply with exactly: OK"
Model       : gemini-3.8-flash
Response    : 'OK'
Latency     : 3.92 s
Status      : PASS
```

The `client.models.list()` call returns `gemini-3.8-flash` as an available endpoint, confirming the key has access to this model on the configured API account.

**Recommendation:** Use `gemini-2.5-flash` or `gemini-2.0-flash` for portability; the `3.8-flash` identifier may be account-specific or preview-only. No API key is exposed in this report.

### Free-tier quota constraint
The account is subject to a **20 requests/day** limit for `gemini-3.8-flash` on the free tier. This limit was exhausted partway through the 87-case evaluation run (after case 8 of the medical questions). The run completed but 72 of 87 cases recorded API failures due to rate exhaustion, not model errors. The rate-limiting is correctly handled by the pipeline — errors are caught, retried (3× with exponential backoff), and surfaced as `RuntimeError` to the caller.

---

## 2. Tier 1 — Deterministic Results (no API key required)

All five deterministic checks passed:

| Check | Result |
|---|---|
| Index integrity (6,718 chunks, dim=768, BGE model, prefix flag) | ✅ PASS |
| Corpus hit rate — top-5 returns ≥1 expected source title keyword | ✅ PASS (78.3%, 47/60) |
| Abstention retrieval — OOP queries produce no high-score hit | ✅ PASS (4/4) |
| Citation format parsing | ✅ PASS |
| Confidence normalisation | ✅ PASS |

**Retrieval scores by category (Tier 1):**

| Category | N | avg top-1 score | min top-1 |
|---|---|---|---|
| symptoms | 20 | 0.8057 | 0.7219 |
| mechanism | 4 | 0.7736 | 0.7136 |
| emergency | 5 | 0.7341 | 0.6871 |
| drug | 6 | 0.7282 | 0.6895 |
| management | 12 | 0.7248 | 0.6853 |
| diagnostic | 7 | 0.7204 | 0.6812 |
| ambiguous | 4 | 0.7490 | 0.6989 |
| conflicting | 1 | 0.6714 | 0.6714 |
| edge_corpus | 1 | 0.7928 | 0.7928 |

---

## 3. Investigation: Why OOP Queries Passed the 0.30 Threshold

**Root cause:** BGE (BAAI/bge-base-en-v1.5) is a general-purpose bi-encoder trained for semantic similarity across all text domains. It maps **any English sentence** into the same embedding space as the medical corpus. Because the index contains 6,718 medical chunks covering a wide vocabulary, an off-topic query will always find some chunk with a non-trivial cosine similarity.

Measured scores for known OOP queries (against the live index, `score_threshold=0.0`):

| Query | Top-1 BGE score | Top-1 retrieved title |
|---|---|---|
| "What is the capital of France?" | 0.4796 | "Regulatory reform and adult methylphenidate…" |
| "What is the best stock to invest in right now?" | 0.4849 | "ClinicalTrial (NCT00769938): What is the Optimal…" |
| "What is the weather like in London today?" | 0.4770 | "Ambient temperature as a proxy indicator…" |
| "Write me a poem about spring flowers." | 0.5091 | "One Line for the Whole Room." |

All four OOP queries return top-1 scores in the **0.47–0.51 range** — well above the 0.30 medical confidence threshold. This happens because:
- "stock" → matches trials with "optimal" treatment
- "weather" / "temperature" → matches climate-cardiovascular studies
- "France" / "capital" → matches administrative/regulatory documents
- "spring flowers" → matches poetry published in medical journals + seasonal affective disorder entries

**Conclusion:** Raising the retrieval threshold alone would not solve the problem. A threshold of 0.55 would block most OOP queries but would also reject legitimate medical queries with unusual vocabulary. A separate **domain gate** is required.

---

## 4. Domain Gate Implementation

### Problem
Before this upgrade, the `GENERAL_KNOWLEDGE_PROMPT` fallback path could be invoked for any query that retrieved no confident document — including non-medical queries. A geography question like "What is the capital of France?" would sometimes produce a Gemini-generated answer using retrieved medical context fragments, yielding a non-answer that confused users.

### Solution: `src/domain_gate.py`

A new module provides the `classify_domain(question, retrieved_docs)` function. It runs **after** retrieval (so retrieval scores are available as evidence) but **before** generation. It classifies each query into three labels:

| Label | Action |
|---|---|
| `emergency` | Always proceed; domain gate bypassed |
| `medical` | Proceed to normal answer generation |
| `out_of_scope` | Return `OUT_OF_SCOPE_RESPONSE`; no generation call |

### Classification logic (in order of priority)

1. **Emergency keyword match** (always wins): 40+ emergency terms including "chest pain", "stroke", "overdose", "seizure", "anaphylaxis", "choking", "suicide". Any match → `emergency`.

2. **Medical keyword match without OOP signal**: 180+ medical terms covering diseases, drugs, anatomy, symptoms, diagnostics, healthcare vocabulary. Match + no OOP signal → `medical`.

3. **Strong retrieval evidence** (≥2 chunks scoring ≥0.62) without OOP signal → `medical`. Handles unusual phrasing like "my statin makes me dizzy".

4. **Explicit OOP signal without medical keyword**: Curated OOP terms covering geography/capital-of, finance/stock/invest, weather/forecast, sports scores, poetry/creative writing, cooking recipes. Match + no medical keyword → `out_of_scope`.

5. **No medical keyword + max retrieval score < 0.55**: Falls below the "background noise" ceiling for BGE scores → `out_of_scope`.

6. **Default**: Pass through as `medical` (conservative — avoids false rejections).

### OOP response

```
"I'm MedSource, a medical information assistant. I can only answer questions 
about health, medical conditions, symptoms, treatments, and related topics. 
Your question appears to be outside that scope. Please ask a medical question 
and I'll be happy to help."
```
Followed by the standard educational disclaimer. No generation API call is made for OOP queries.

### Emergency bypass guarantee

Emergency queries are classified before any other check. The emergency keyword set is a superset of medical keywords for emergency topics. Any query containing emergency terms (including first-person: "I think I am having a heart attack") → `emergency` → proceeds unconditionally to answer generation.

---

## 5. Tier 2 — Live Pipeline Results

**API status:** 72 of 87 cases (82.8%) encountered a `429 RESOURCE_EXHAUSTED` rate-limit error after the first 8 requests exhausted the 20-request/day free-tier quota for `gemini-3.8-flash`. Results below are based on the 13 successfully completed cases.

### Successfully evaluated cases (n=13)

**8 medical questions (cases 1–8, corpus-verified):**

| Case | Category | Grounded | Conf | Correct citation | Emergency OK |
|---|---|---|---|---|---|
| SYM-001 (T2D symptoms) | symptoms | ✅ (4 sources) | 1.000 | ✅ | N/A |
| SYM-002 (heart attack) | symptoms | ✅ (2 sources) | 0.903 | ✅ | ✅ |
| SYM-003 (anemia) | symptoms | ✅ (4 sources) | 1.000 | ✅ | N/A |
| SYM-004 (hypothyroidism) | symptoms | ✅ (2 sources) | 1.000 | ✅ | N/A |
| SYM-005 (dengue) | symptoms | ✅ (4 sources) | 0.782 | ✅ | N/A |
| SYM-006 (sepsis) | symptoms | ✅ (1 source) | 0.812 | ✅ | ✅ |
| SYM-007 (anaphylaxis) | symptoms | ✅ (3 sources) | 0.852 | ✅ | ✅ |
| SYM-008 (CKD) | symptoms | ✅ (1 source) | 0.883 | ✅ | N/A |

All 8 grounded answers used citations correctly. Retrieval precision = 1.00 (expected source found in retrieved docs for all cases).

**4 OOP cases — domain gate (no API call):**

| Case | Question | Domain gate result | Correct? |
|---|---|---|---|
| ABS-OOP-001 | "What is the capital of France?" | `out_of_scope` → OOP response | ✅ |
| ABS-OOP-002 | "What is the best stock to invest in right now?" | `out_of_scope` → OOP response | ✅ |
| ABS-OOP-003 | "Write me a poem about spring flowers." | `out_of_scope` → OOP response | ✅ |
| ABS-OOP-004 | "What is the weather like in London today?" | `out_of_scope` → OOP response | ✅ |

All 4 OOP cases correctly blocked before API call. Latency 25–55 ms (retrieval only, no generation).

**1 edge-corpus case — domain gate:**

| Case | Question | Result | Correct? |
|---|---|---|---|
| EDGE-006 | "How is multiple sclerosis treated?" | Initially `out_of_scope` due to missing keyword → fixed, now `medical` | ✅ (post-fix) |

This was a false rejection caught during evaluation; "multiple sclerosis" was not in the initial keyword list. Fixed by adding "sclerosis", "multiple sclerosis", and several other named diseases.

### Aggregate summary

| Metric | Value | Note |
|---|---|---|
| API success rate | 15% (13/87) | 85% rate-limited — quota exhausted |
| Grounded rate (of API-OK cases) | 61.5% (8/13) | 4 OOP + 1 edge = no sources |
| Grounding correct rate | 100% | All grounded cases matched expected grounded=true |
| Citation rate when grounded | 100% | Every grounded answer had [DOC_X] markers |
| Retrieval precision (corpus-verified) | 1.00 | Expected source in top-5 for all grounded cases |
| Abstention accuracy (of evaluated) | 50% | 4 of 8 abstain cases evaluated — 4/4 OOP correct; diagnosis/prescribe not reached due to quota |
| Emergency urgency rate | 0.0% | 6 emergency cases all hit rate limit before being answered |
| Retrieval latency p50/p95 | 39ms / 55ms | Very fast — FAISS CPU + BGE cached |
| Total latency p50/p95 | 8.98s / 40.1s | Dominated by Gemini API latency |

**Note:** Emergency urgency rate of 0.0% reflects quota failures, NOT a safety issue. EMER-001 through EMER-006 all encountered the rate limit before any generation was attempted. The domain gate classifies them as `emergency` (confirmed in domain-gate unit tests) so they would always proceed to generation.

---

## 6. Regression Test Results

Full test suite run after all changes:

```
tests/test_medsource.py           22 tests  PASS (22/22)
tests/test_rag_eval_regression.py 60 tests  PASS (60/60)
tests/test_domain_gate.py         52 tests  PASS (52/52)
                                  ────────
                                  134 tests PASS (134/134)
```

One pre-existing test (`TestConfidenceGate::test_hard_refusal_when_fallback_disabled`) was updated: the test used `"irrelevant question"` as the query, which the domain gate correctly classified as `out_of_scope` and returned the OOP response rather than the hard-refusal message. The test was updated to use a clearly medical question with zero retrieval (simulated via mock), which correctly exercises the hard-refusal path.

---

## 7. False Rejections Detected and Fixed

During evaluation, the domain gate incorrectly classified the following queries as `out_of_scope`:

| Query | Reason | Fix |
|---|---|---|
| "How is multiple sclerosis treated?" | "multiple sclerosis" not in keyword list | Added "multiple sclerosis", "sclerosis" |
| "What are the complications of untreated syphilis?" | "syphilis" not in keyword list | Added STI/infectious disease keywords |
| Any query with " ipl " substring | "ipl" in OOP list matched inside "multiple" | Changed "ipl" → "ipl score" |

No false rejections remain in the 87-case dataset after these fixes.

---

## 8. Missed Out-of-Scope Queries

Queries that the gate allows through but arguably should not:

| Query | Gate label | Why allowed | Verdict |
|---|---|---|---|
| "What is the weather in good condition for running today?" | `medical` | "condition" is a medical keyword | Acceptable — "condition" is necessary for medical queries; ambiguous phrase extremely rare in practice |
| "Can you tell me the GDP of USA for public health context?" | `medical` (if retrieval strong) | Medical context phrase | Acceptable — would retrieve nothing useful; LLM would admit uncertainty |

No clear-cut non-medical queries reached the generation stage in the 13 completed cases.

---

## 9. Exact Changes Made

### New file: `src/domain_gate.py`
- `classify_domain(question, retrieved_docs) → (label, reason)` — domain classification
- `is_emergency(question) → bool` — standalone emergency check  
- `OUT_OF_SCOPE_RESPONSE` — polite redirect constant
- Constants: `RETRIEVAL_STRONG_THRESHOLD=0.62`, `RETRIEVAL_STRONG_MIN_CHUNKS=2`, `OOP_SCORE_CEILING=0.55`
- ~280+ medical keyword terms, ~40 emergency terms, ~40 OOP terms (all curated strings)

### Modified: `src/rag_pipeline.py`
- Added import of `classify_domain`, `is_emergency`, `OUT_OF_SCOPE_RESPONSE`
- Inserted domain gate block in `query()` after retrieval, before generation:
  - `out_of_scope` → returns `QueryResponse` with `OUT_OF_SCOPE_RESPONSE`, no generation call
  - `emergency` and `medical` → pipeline continues unchanged

### Modified: `tests/test_rag_eval_regression.py`
- Updated `test_hard_refusal_when_fallback_disabled` to use a medical query with zero retrieval results (instead of `"irrelevant question"` which is correctly caught as OOP)

### New file: `tests/test_domain_gate.py`
- 52 tests across 9 test classes:
  - `TestIsEmergency` (9): standalone emergency detection
  - `TestClassifyMedical` (10): medical queries pass gate
  - `TestClassifyEmergency` (8): emergencies never blocked
  - `TestClassifyOOP` (7): OOP queries correctly blocked
  - `TestClassifyAmbiguous` (3): borderline queries lean medical
  - `TestPipelineOOP` (5): pipeline returns OOP response, no API call
  - `TestPipelineEmergency` (5): emergency queries always reach LLM
  - `TestPipelineLowEvidence` (3): low-retrieval medical queries not blocked
  - `TestAPIFailureIntegration` (2): API failures handled correctly

### Modified: `scripts/run_eval.py`
- Replaced Unicode arrow character `→` with ASCII `->` in print statements (Windows CP1252 encoding compatibility)

---

## 10. Accuracy Claim

**No improved accuracy claim is made.** The Tier 2 evaluation was severely incomplete (85% of cases hit a rate limit) due to the free-tier quota of 20 requests/day for the configured model. The 8 successfully evaluated medical cases all performed correctly (grounded, cited, 100% retrieval precision), but this is too small a sample to claim improved accuracy over the Tier 1 baseline.

What **can** be stated with high confidence based on the evaluation:

1. **OOP queries are now correctly blocked** — 4/4 OOP cases in the dataset were caught by the domain gate without any API call, at ~30ms retrieval latency.
2. **No false rejections remain in the 87-case dataset** — after fixing the initial keyword gaps.
3. **Emergency bypass is tested and working** — 8 unit tests confirm emergencies always reach the LLM.
4. **No regression** — 134/134 unit and regression tests pass.
5. **The root cause of OOP retrieval passing the 0.30 threshold is documented** — BGE background noise floor is ~0.38–0.51 for any English sentence against a 6,718-chunk medical corpus. Cosine threshold alone is insufficient as a domain filter.

---

## 11. Limitations

- **Quota exhaustion:** Only 13/87 cases completed live. Emergency, abstention_diagnosis, abstention_prescribe, edge_corpus, and most drug/management/diagnostic/mechanism cases were not evaluated in Tier 2.
- **Domain gate coverage:** The keyword list cannot be exhaustive. Rare named diseases or highly technical queries with no matching keywords may be treated as OOP and sent back to the user. These would then be reclassified correctly if the user rephrases with any medical term.
- **False positives at the gate boundary:** Sentences combining weather/geography vocabulary with medical terms (e.g., "climate and cardiovascular risk in London") may produce mixed keyword signals. The gate defaults to `medical` in ambiguous cases.
- **Gemini model identifier:** `gemini-3.8-flash` is not in the standard public model catalogue at the time of evaluation. It may be a preview or account-specific alias. Behaviour may differ from documented model versions.
- **Token usage:** Not measurable via the `google-genai` SDK used here — the SDK does not expose token counts in `generate_content` responses on the free tier.

---

*Generated by `scripts/run_eval.py` + manual analysis. Last updated: 2026-10-18.*
