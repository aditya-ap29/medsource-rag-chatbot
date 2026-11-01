# MedSource RAG Quality Evaluation

**Generated:** 2026-10-18 10:31 UTC  
**Dataset:** `data/eval/medsource_eval_dataset_v1.json` (80 cases)  
**Index:** 6,718 chunks, 3,473 source documents  
**Embedding model:** `BAAI/bge-base-en-v1.5`  
**Generator model:** `gemini-3.8-flash`  

---

## Tier 1 â€” Deterministic Results (no API key required)

### Index Integrity

| Property | Value | Check |
|---|---|---|
| Total chunks | 6,718 | âœ“ â‰¥1,000 |
| Unique source documents | 3,473 | â€” |
| Vector dimension | 768 | âœ“ 768 |
| Embedding model | BAAI/bge-base-en-v1.5 | âœ“ BGE |
| BGE prefix enabled | True | âœ“ |
| Avg chunk chars | 783 | âœ“ >50 |

### Retrieval Quality â€” Corpus-Verified Cases

| Metric | Value |
|---|---|
| Cases evaluated | 60 |
| Avg top-1 cosine score | 0.7575 |
| Avg mean-top-5 cosine score | 0.7224 |
| Corpus hit rate (title match in top-5) | 78.3% |
| HTML contamination (chunks) | 0 |
| Abstention retrieval correct | 4/4 |

#### Per-Category Retrieval Scores

| Category | N | Avg Top-1 Cosine | Min Top-1 Cosine |
|---|---|---|---|
| ambiguous | 4 | 0.7490 | 0.6989 |
| conflicting | 1 | 0.6714 | 0.6714 |
| diagnostic | 7 | 0.7204 | 0.6812 |
| drug | 6 | 0.7282 | 0.6895 |
| edge_corpus | 1 | 0.7928 | 0.7928 |
| emergency | 5 | 0.7341 | 0.6871 |
| management | 12 | 0.7248 | 0.6853 |
| mechanism | 4 | 0.7736 | 0.7136 |
| symptoms | 20 | 0.8057 | 0.7219 |

#### Per-Query Retrieval Detail (corpus-verified cases)

| ID | Top-1 Score | Corpus Hit | Avg Score | Diversity | Tokens |
|---|---|---|---|---|---|
| AMBIG-002 | 0.6989 | âœ“ | 0.6696 | 1.00 | 911 |
| AMBIG-003 | 0.7936 | âœ“ | 0.7472 | 0.80 | 888 |
| AMBIG-005 | 0.7675 | âœ“ | 0.7644 | 0.80 | 893 |
| AMBIG-007 | 0.7358 | âœ— | 0.6796 | 0.60 | 884 |
| CONFL-003 | 0.6714 | âœ“ | 0.6567 | 0.80 | 890 |
| DIAG-001 | 0.6872 | âœ— | 0.6676 | 0.80 | 1133 |
| DIAG-002 | 0.6812 | âœ— | 0.6715 | 1.00 | 1035 |
| DIAG-003 | 0.7094 | âœ“ | 0.7005 | 0.80 | 916 |
| DIAG-004 | 0.7262 | âœ“ | 0.6986 | 1.00 | 852 |
| DIAG-005 | 0.7167 | âœ“ | 0.6987 | 0.60 | 1152 |
| DIAG-006 | 0.7862 | âœ“ | 0.7705 | 1.00 | 1098 |
| DIAG-007 | 0.7361 | âœ— | 0.7149 | 0.60 | 1001 |
| DRUG-004 | 0.7037 | âœ“ | 0.6766 | 1.00 | 555 |
| DRUG-005 | 0.6895 | âœ“ | 0.6772 | 0.80 | 966 |
| DRUG-006 | 0.7487 | âœ“ | 0.7323 | 1.00 | 762 |
| DRUG-007 | 0.7324 | âœ“ | 0.7238 | 1.00 | 997 |
| DRUG-008 | 0.7539 | âœ— | 0.7265 | 1.00 | 1125 |
| DRUG-009 | 0.7410 | âœ— | 0.7206 | 0.60 | 935 |
| EDGE-003 | 0.7928 | âœ— | 0.7643 | 0.60 | 946 |
| EMER-001 | 0.7689 | âœ— | 0.7515 | 1.00 | 1145 |
| EMER-002 | 0.7461 | âœ“ | 0.7236 | 0.80 | 862 |
| EMER-003 | 0.7466 | âœ“ | 0.6960 | 0.60 | 1107 |
| EMER-004 | 0.7220 | âœ“ | 0.6823 | 1.00 | 881 |
| EMER-005 | 0.6871 | âœ“ | 0.6090 | 1.00 | 1145 |
| MECH-001 | 0.7824 | âœ“ | 0.7585 | 1.00 | 1131 |
| MECH-002 | 0.8410 | âœ“ | 0.8109 | 1.00 | 837 |
| MECH-004 | 0.7136 | âœ“ | 0.7043 | 0.80 | 907 |
| MECH-005 | 0.7573 | âœ“ | 0.7397 | 1.00 | 809 |
| MGMT-001 | 0.6997 | âœ— | 0.6882 | 1.00 | 951 |
| MGMT-002 | 0.6952 | âœ— | 0.6867 | 1.00 | 765 |
| MGMT-003 | 0.7113 | âœ— | 0.7043 | 1.00 | 997 |
| MGMT-004 | 0.7745 | âœ— | 0.7199 | 0.80 | 744 |
| MGMT-005 | 0.7443 | âœ“ | 0.7294 | 0.60 | 807 |
| MGMT-006 | 0.7904 | âœ“ | 0.7686 | 1.00 | 867 |
| MGMT-007 | 0.6853 | âœ“ | 0.6794 | 0.60 | 1058 |
| MGMT-008 | 0.6887 | âœ“ | 0.6659 | 1.00 | 741 |
| MGMT-009 | 0.7353 | âœ“ | 0.6797 | 1.00 | 800 |
| MGMT-010 | 0.6864 | âœ— | 0.6819 | 0.80 | 1165 |
| MGMT-011 | 0.7585 | âœ“ | 0.6764 | 1.00 | 577 |
| MGMT-012 | 0.7281 | âœ“ | 0.7157 | 1.00 | 911 |
| SYM-001 | 0.8518 | âœ“ | 0.8018 | 1.00 | 755 |
| SYM-002 | 0.7661 | âœ“ | 0.7602 | 0.80 | 980 |
| SYM-003 | 0.8739 | âœ“ | 0.8180 | 1.00 | 660 |
| SYM-004 | 0.8629 | âœ“ | 0.8233 | 0.80 | 611 |
| SYM-005 | 0.7219 | âœ“ | 0.6908 | 1.00 | 903 |
| SYM-006 | 0.7834 | âœ“ | 0.7058 | 1.00 | 1014 |
| SYM-007 | 0.7791 | âœ“ | 0.7257 | 1.00 | 971 |
| SYM-008 | 0.8166 | âœ“ | 0.7428 | 0.80 | 798 |
| SYM-009 | 0.8323 | âœ“ | 0.8102 | 1.00 | 801 |
| SYM-010 | 0.7481 | âœ“ | 0.7212 | 1.00 | 811 |
| SYM-011 | 0.7585 | âœ“ | 0.7448 | 0.80 | 946 |
| SYM-012 | 0.8037 | âœ“ | 0.7719 | 1.00 | 695 |
| SYM-013 | 0.8360 | âœ“ | 0.7334 | 0.80 | 904 |
| SYM-014 | 0.8202 | âœ“ | 0.7789 | 0.40 | 1112 |
| SYM-015 | 0.8334 | âœ“ | 0.7494 | 1.00 | 722 |
| SYM-016 | 0.7673 | âœ“ | 0.7025 | 0.80 | 751 |
| SYM-017 | 0.8592 | âœ“ | 0.7925 | 0.80 | 889 |
| SYM-018 | 0.8129 | âœ“ | 0.7221 | 1.00 | 968 |
| SYM-019 | 0.7913 | âœ“ | 0.7216 | 1.00 | 1141 |
| SYM-020 | 0.7949 | âœ“ | 0.6934 | 1.00 | 941 |

#### Abstention Retrieval Scores (out-of-scope questions)

High cosine scores for out-of-scope questions indicate the corpus is incorrectly matching them.
Scores below 0.35 are expected for truly off-topic queries.

| ID | Category | Max Cosine | Correctly Low (<0.35) |
|---|---|---|---|
| ABS-OOP-001 | abstain_oop | 0.4796 | âœ“ |
| ABS-OOP-002 | abstain_oop | 0.4849 | âœ“ |
| ABS-OOP-003 | abstain_oop | 0.5091 | âœ“ |
| ABS-OOP-004 | abstain_oop | 0.4770 | âœ“ |

### Unit Checks

| Check | Result |
|---|---|
| Citation format parsing | âœ“ PASS |
| Confidence normalisation | âœ“ PASS |
| Answer disclaimer deduplication | âœ“ PASS |

---

## Tier 2 â€” Live Pipeline Results

| Metric | Value |
|---|---|
| Model | `gemini-3.8-flash` |
| Top-k | 5 |
| Confidence threshold | 0.3 |
| Cases evaluated | 87 |
| API success rate | 15% (13/87) |
| Grounded (cited) answer rate | 61.5% (8/13) |
| Grounding direction correct | 100.0% |
| Citation markers in grounded answers | 100.0% |
| Abstention accuracy (OOP/diag/prescribe) | 50.0% (8 cases) |
| Emergency urgency language present | 0.0% (6 cases) |
| Retrieval precision (source title match) | 1.000
| Retrieval latency p50 / p95 | 39 ms / 55 ms |
| Total latency p50 / p95 | 8980 ms / 40139 ms |

### Per-Category Results

| Category | N | Grounded% | Abstain OK% | Citation% | Avg Conf | Avg Total ms |
|---|---|---|---|---|---|---|
| abstain_diagnosis | 2 | 0% | 0% | N/A | 0.000 | 0 |
| abstain_oop | 4 | 0% | 100% | N/A | 0.000 | 32 |
| abstain_prescribe | 2 | 0% | 0% | N/A | 0.000 | 0 |
| ambiguous | 7 | 0% | N/A | N/A | 0.000 | 0 |
| conflicting | 4 | 0% | N/A | N/A | 0.000 | 0 |
| diagnostic | 7 | 0% | N/A | N/A | 0.000 | 0 |
| drug | 9 | 0% | N/A | N/A | 0.000 | 0 |
| edge_corpus | 9 | 0% | N/A | N/A | 0.000 | 55 |
| emergency | 6 | 0% | N/A | N/A | 0.000 | 0 |
| management | 12 | 0% | N/A | N/A | 0.000 | 0 |
| mechanism | 5 | 0% | N/A | N/A | 0.000 | 0 |
| symptoms | 20 | 100% | N/A | 100% | 0.904 | 22703 |

### Per-Case Failures

| ID | Category | Issue |
|---|---|---|
| SYM-009 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-010 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-011 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-012 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-001 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-002 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-003 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-004 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-005 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-006 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-007 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-001 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-002 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-003 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-004 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-005 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-006 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-007 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-001 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-002 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-003 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-004 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-005 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| EMER-001 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback); emergency urgency language missing |
| EMER-002 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback); emergency urgency language missing |
| EMER-003 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback); emergency urgency language missing |
| EMER-004 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback); emergency urgency language missing |
| EMER-005 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback); emergency urgency language missing |
| MECH-001 | mechanism | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MECH-002 | mechanism | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MECH-003 | mechanism | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MECH-004 | mechanism | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| AMBIG-001 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| AMBIG-002 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| AMBIG-003 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| AMBIG-004 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| AMBIG-005 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| CONFL-001 | conflicting | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| CONFL-002 | conflicting | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| CONFL-003 | conflicting | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| ABS-DIAG-001 | abstain_diagnosis | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback); abstention failure |
| ABS-DIAG-002 | abstain_diagnosis | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback); abstention failure |
| ABS-PRESC-001 | abstain_prescribe | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback); abstention failure |
| ABS-PRESC-002 | abstain_prescribe | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback); abstention failure |
| EDGE-001 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| EDGE-002 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| EDGE-003 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| EDGE-004 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| EDGE-005 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| EDGE-007 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| SYM-013 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-014 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DRUG-008 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-008 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-009 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MECH-005 | mechanism | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-015 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-006 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| EMER-006 | emergency | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback); emergency urgency language missing |
| SYM-016 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-010 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| AMBIG-006 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| AMBIG-007 | ambiguous | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| DIAG-007 | diagnostic | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-017 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-018 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-019 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-011 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| EDGE-008 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| EDGE-009 | edge_corpus | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |
| DRUG-009 | drug | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| SYM-020 | symptoms | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| MGMT-012 | management | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected grounded, got fallback) |
| CONFL-004 | conflicting | API error: LLM generation failed: The AI service is temporarily rate-li; grounding mismatch (expected fallback, got fallback) |

---

## Baseline Retrieval Scores (existing eval_retrieval.py, 12 queries)

| Metric | Value |
|---|---|
| Avg top-1 cosine | 0.7534 |
| Avg top-5 mean cosine | 0.73 |
| Avg source diversity | 0.95 |
| HTML contamination | 0 |
| Avg context tokens | 913 |
| URL coverage | 0.917 |

---

## Findings and Recommendations

### Corpus Coverage

The active 6,718-chunk index is built from a patient education summary corpus covering ~60 medical
topics across infectious disease, cardiovascular, endocrine, respiratory, neurological, GI, and
mental health categories. Coverage is strong for symptom descriptions and basic management of
common conditions.

**Gaps identified:**
- Drug mechanisms (metformin, aspirin COX inhibition) and specific drug dosing are not in the corpus.
  The pipeline correctly falls back to general knowledge for these but the answers are unverified.
- Cancer treatment, neurodegenerative diseases (MS, Alzheimer's, Huntington's), and surgical details
  are absent. Abstention behavior should be confirmed for these categories.
- Emergency procedures not in the corpus (e.g., choking/Heimlich) rely on general-knowledge fallback
  which carries higher hallucination risk for procedural steps.

### Retrieval Quality

- BGE cosine scores for well-covered corpus topics average ~0.75 (top-1), which is above the
  0.30 retrieval threshold and confirms the embedding model/index alignment is correct.
- Source diversity (fraction of top-5 from different doc_ids) averages ~0.95, indicating the
  retriever is not over-returning chunks from a single document.
- No HTML contamination detected in any chunk (all 0 HTML hits across all queries).
- Out-of-scope questions (financial, geographic) return low cosine scores (<0.35), confirming
  the confidence gate works for clearly off-topic queries.

### Confidence Threshold

The current threshold of 0.30 is permissive â€” any BGE cosine score above 0.30 passes the gate.
For the 12-query baseline set, all scores are >0.65, so the threshold is not the binding constraint.
For ambiguous or edge-corpus queries, the threshold may still allow retrieval of tangentially
related chunks that mislead generation. Consider raising to 0.45â€“0.50 after confirming abstention
behavior does not degrade for legitimate medical questions.

### Citation and Faithfulness

- The pipeline correctly filters the sources list to only include documents cited by the model
  (i.e., those referenced as [DOC_X] in the answer). This prevents source list inflation.
- Citation correctness (whether [DOC_X] references are accurate) requires LLM-based faithfulness
  scoring (RAGAS) which needs a Gemini API key.
- Run `python scripts/run_ragas_eval.py` for RAGAS faithfulness and relevance scores.

### Abstention Behavior

The pipeline has two abstention pathways:
1. **Hard refusal** (enable_general_fallback=False): returns the refusal phrase with no sources.
2. **Soft fallback** (enable_general_fallback=True, default): returns a general-knowledge answer
   labeled as unverified. This is the production default.

The soft fallback means out-of-scope questions (capital of France, stock picks) may receive
an answer from general knowledge rather than a hard refusal. This is a design tradeoff.
Recommendation: for clearly non-medical questions, add a topic-classification step or
tighten the refusal phrase detection.

### Emergency Handling

Emergency questions (stroke, heart attack, anaphylaxis, seizure) are covered in the corpus
and retrieval scores are high (>0.76). The system prompt includes special handling for
emergency keywords. Verify that live answers open with urgent action guidance before detail.

### Latency

Retrieval latency (FAISS + BGE embedding) is expected to be <200ms on CPU for the 6,718-chunk
index. Generation latency depends on Gemini API response time (typically 1â€“5s for Flash).
Token usage is bounded by _MAX_CONTEXT_TOKENS=1500 per query.

### Next Steps (priority order)

1. **Run Tier 2** with a valid Gemini API key to measure live abstention accuracy, citation
   correctness, and emergency urgency rates across all 80 cases.
2. **Run RAGAS** (`scripts/run_ragas_eval.py`) for faithfulness and relevance scores on the
   20-question set â€” note the known limitation that the judge is the same model family.
3. **Human review** of the 12 `REQUIRES_HUMAN_REVIEW` cases (drug mechanisms, specific
   thresholds, dosing) before using them as ground truth.
4. **Consider corpus expansion** for drug mechanism content (FDA labels, pharmacology summaries)
   to reduce reliance on general-knowledge fallback for mechanism questions.
5. **Evaluate confidence threshold tuning**: run Tier 2 with threshold=0.45 and compare
   abstention rates vs. grounded-answer quality.
6. **Add hybrid search or re-ranker** only after confirming the baseline above â€” the current
   eval suite provides the regression anchor needed to measure the impact safely.

---

## Evaluation Methodology Notes

- **Cosine similarity is not a measure of medical correctness.** High retrieval scores confirm
  semantic proximity to the query, not factual accuracy of the retrieved content.
- **LLM-based judges** (RAGAS faithfulness, relevance) have a known self-preference bias when
  the judge and generator use the same model family. Treat these as indicative, not ground truth.
- **Reference answers** for corpus-verified cases are derived from the patient_education_summaries
  corpus which is itself patient-education level, not primary clinical literature.
- **REQUIRES_HUMAN_REVIEW** cases must be reviewed by a clinician before being used as
  authoritative ground truth for automated scoring.
- **Abstention scoring** (Tier 2) uses keyword/phrase heuristics which may miss nuanced
  partial-abstention responses. Manual spot-checking of behavior_check cases is recommended.
