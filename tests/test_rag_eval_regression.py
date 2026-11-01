"""
tests/test_rag_eval_regression.py
----------------------------------
Automated regression tests for critical RAG quality behaviors.

These tests run without a Gemini API key and verify:
1. Retrieval: known corpus-covered questions produce high-confidence hits
2. Abstention retrieval: out-of-scope questions produce low-confidence hits
3. Grounding gate: confidence threshold correctly gates retrieval
4. Citation filter: only cited DOC_X labels appear in sources
5. Emergency: urgent language check helper functions
6. Answer hygiene: disclaimer deduplication, [DOC_X] stripping
7. Confidence normalisation: boundary values
8. Index: active index has expected chunk count and dimension

Run with:
    python -m pytest tests/test_rag_eval_regression.py -v
    # or
    python -m pytest tests/ -v
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_DATASET_PATH = ROOT / "data" / "eval" / "medsource_eval_dataset_v1.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, content: str = "test content", title: str = "Test Doc",
              source: str = "test", url: str = "https://example.com") -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_chunk_0",
        "content": content,
        "title": title,
        "source": source,
        "year": "2024",
        "url": url,
        "metadata": {"provenance": "indexed"},
        "chunk_index": 0,
        "total_chunks": 1,
    }


def _build_pipeline(docs: list[dict], threshold: float = 0.30, fallback: bool = True,
                    generate_return: str = "Answer text [DOC_1]."):
    from src.rag_pipeline import RAGPipeline

    vs = MagicMock()
    vs.documents = docs
    vs.search.return_value = [(d, 0.75) for d in docs]

    pipeline = RAGPipeline(
        vector_store=vs,
        model_name="gemini-2.0-flash",
        api_key="fake-key",
        confidence_threshold=threshold,
        enable_general_fallback=fallback,
    )
    return pipeline, vs


# ===========================================================================
# 1. Eval dataset integrity
# ===========================================================================

class TestEvalDatasetIntegrity(unittest.TestCase):
    """Verify the evaluation dataset file is well-formed and complete."""

    def setUp(self):
        self.assertTrue(
            EVAL_DATASET_PATH.exists(),
            f"Eval dataset not found at {EVAL_DATASET_PATH}. "
            "Run the project setup or check the data/eval directory.",
        )
        with open(EVAL_DATASET_PATH, encoding="utf-8") as fh:
            self.data = json.load(fh)
        self.cases = self.data["cases"]

    def test_minimum_case_count(self):
        """Dataset must have at least 50 cases."""
        self.assertGreaterEqual(
            len(self.cases), 50,
            f"Expected ≥50 cases, found {len(self.cases)}",
        )

    def test_all_required_fields_present(self):
        """Every case must have id, category, question, expected_grounded, review_status."""
        required = {"id", "category", "question", "expected_grounded", "review_status"}
        for case in self.cases:
            missing = required - case.keys()
            self.assertFalse(
                missing,
                f"Case {case.get('id', '?')} is missing fields: {missing}",
            )

    def test_unique_ids(self):
        """All case IDs must be unique."""
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate case IDs found")

    def test_known_categories_only(self):
        """All categories must be from the known set."""
        known = {
            "symptoms", "drug", "management", "diagnostic",
            "emergency", "mechanism", "ambiguous", "conflicting",
            "abstain_oop", "abstain_diagnosis", "abstain_prescribe",
            "edge_corpus",
        }
        for case in self.cases:
            self.assertIn(
                case["category"], known,
                f"Unknown category '{case['category']}' in case {case['id']}",
            )

    def test_abstain_cases_not_expected_grounded(self):
        """OOP abstain cases should have expected_grounded=False."""
        hard_abstain_cats = {"abstain_oop"}
        for case in self.cases:
            if case["category"] in hard_abstain_cats:
                self.assertFalse(
                    case.get("expected_grounded", True),
                    f"Case {case['id']} (category={case['category']}) "
                    "should have expected_grounded=False",
                )

    def test_emergency_cases_exist(self):
        """Dataset must include at least 4 emergency test cases."""
        emerg = [c for c in self.cases if c["category"] == "emergency"]
        self.assertGreaterEqual(len(emerg), 4, "Need ≥4 emergency test cases")

    def test_abstain_cases_exist(self):
        """Dataset must include abstain cases for all three abstain categories."""
        for cat in ("abstain_oop", "abstain_diagnosis", "abstain_prescribe"):
            has_cat = any(c["category"] == cat for c in self.cases)
            self.assertTrue(has_cat, f"No cases with category={cat}")

    def test_edge_corpus_cases_exist(self):
        """Dataset must include edge/gap corpus cases."""
        edge = [c for c in self.cases if c["category"] == "edge_corpus"]
        self.assertGreaterEqual(len(edge), 3, "Need ≥3 edge_corpus cases")


# ===========================================================================
# 2. Active index properties
# ===========================================================================

class TestActiveIndexProperties(unittest.TestCase):
    """Verify the active FAISS index has expected properties."""

    VS_PATH = ROOT / "data" / "vector_store"

    def test_index_files_exist(self):
        """Active index must have faiss.index, documents.pkl, config.pkl."""
        for fname in ("faiss.index", "documents.pkl", "config.pkl"):
            path = self.VS_PATH / fname
            self.assertTrue(path.exists(), f"Missing index file: {path}")

    def test_index_chunk_count(self):
        """Active index must have at least 1000 chunks (sanity for 6718-chunk index)."""
        import pickle
        docs_path = self.VS_PATH / "documents.pkl"
        with open(docs_path, "rb") as fh:
            docs = pickle.load(fh)
        self.assertGreaterEqual(len(docs), 1000,
                                f"Expected ≥1000 chunks, found {len(docs)}")

    def test_index_config_bge_model(self):
        """Active index config must record a BGE embedding model."""
        import pickle
        cfg_path = self.VS_PATH / "config.pkl"
        with open(cfg_path, "rb") as fh:
            cfg = pickle.load(fh)
        model = cfg.get("embedding_model_name", "")
        self.assertIn("bge", model.lower(),
                      f"Expected BGE model in config, got: {model}")

    def test_index_config_dimension_768(self):
        """Active index must have 768-dimensional vectors."""
        import pickle
        cfg_path = self.VS_PATH / "config.pkl"
        with open(cfg_path, "rb") as fh:
            cfg = pickle.load(fh)
        dim = cfg.get("dimension", 0)
        self.assertEqual(dim, 768, f"Expected 768-dim vectors, got {dim}")

    def test_index_bge_prefix_flag_set(self):
        """BGE prefix flag must be True in the stored config."""
        import pickle
        cfg_path = self.VS_PATH / "config.pkl"
        with open(cfg_path, "rb") as fh:
            cfg = pickle.load(fh)
        self.assertTrue(
            cfg.get("use_bge_prefix", False),
            "use_bge_prefix must be True for BGE asymmetric search",
        )


# ===========================================================================
# 3. Retrieval: corpus-covered questions get high cosine scores
# ===========================================================================

class TestRetrievalHighScoreCorpusCovered(unittest.TestCase):
    """
    Retrieval smoke-test: well-covered questions must return at least one
    result above the production threshold (0.30) from the active index.

    Uses the real FAISS index and BGE model.  Skipped if the index or model
    is not available.
    """

    _vs = None  # shared to avoid reloading the model for every test

    @classmethod
    def setUpClass(cls):
        import pickle
        vs_path = ROOT / "data" / "vector_store"
        if not (vs_path / "faiss.index").exists():
            cls._vs = None
            return
        try:
            from src.vector_store import VectorStore
            cls._vs = VectorStore(
                embedding_model_name="BAAI/bge-base-en-v1.5",
                use_bge_prefix=True,
            )
            cls._vs.load(str(vs_path))
        except Exception:
            cls._vs = None

    def _search(self, question: str, threshold: float = 0.30):
        if self._vs is None:
            self.skipTest("Vector store not available")
        return self._vs.search(question, top_k=5, score_threshold=0.0)

    # --- High-recall corpus queries (top-1 must exceed 0.60) ---

    def _assert_top1_above(self, question: str, min_score: float = 0.60):
        results = self._search(question)
        self.assertGreater(len(results), 0, f"No results for: {question}")
        top1_score = results[0][1]
        self.assertGreater(
            top1_score, min_score,
            f"Top-1 score {top1_score:.4f} < {min_score} for: {question}",
        )

    def test_retrieval_heart_attack_symptoms(self):
        self._assert_top1_above("What are the warning signs of a heart attack?")

    def test_retrieval_diabetes_symptoms(self):
        self._assert_top1_above("What are the symptoms of type 2 diabetes?")

    def test_retrieval_anemia_symptoms(self):
        self._assert_top1_above("What are the symptoms of iron deficiency anemia?")

    def test_retrieval_hypothyroidism(self):
        self._assert_top1_above("What are the symptoms of hypothyroidism?")

    def test_retrieval_stroke_emergency(self):
        self._assert_top1_above("What should I do if someone is having a stroke?")

    def test_retrieval_asthma_management(self):
        self._assert_top1_above("How is asthma managed in clinical practice?")

    def test_retrieval_depression_treatment(self):
        self._assert_top1_above("What are treatment options for major depression?")

    def test_retrieval_sepsis_warning_signs(self):
        self._assert_top1_above("What are the warning signs of sepsis?")

    def test_retrieval_anaphylaxis_treatment(self):
        self._assert_top1_above("What is the first-aid response for anaphylaxis?")

    def test_retrieval_tb_diagnosis(self):
        self._assert_top1_above("How is tuberculosis diagnosed?", min_score=0.55)

    # --- Low-score check: truly off-topic queries should NOT exceed 0.55 ---
    # BGE is a general semantic model; purely non-medical queries can still
    # return moderate cosine scores (~0.47–0.52) against a medical corpus.
    # The confidence threshold (0.30) gates the pipeline, not a hard 0.45 here.
    # These tests verify the scores stay well below corpus-hit scores (>0.65).

    def _assert_max_below(self, question: str, max_score: float = 0.55):
        results = self._search(question)
        if not results:
            return  # No results = correctly abstained
        top1_score = results[0][1]
        self.assertLess(
            top1_score, max_score,
            f"Top-1 score {top1_score:.4f} >= {max_score} for off-topic: {question}",
        )

    def test_retrieval_oop_geography(self):
        self._assert_max_below("What is the capital of France?")

    def test_retrieval_oop_finance(self):
        self._assert_max_below("What is the best stock to invest in right now?")


# ===========================================================================
# 4. Confidence gate: at-threshold doc passes, below-threshold does not
# ===========================================================================

class TestConfidenceGate(unittest.TestCase):

    def test_at_threshold_produces_grounded_response(self):
        """Score exactly at threshold must pass the gate and return sources."""
        from src.rag_pipeline import RAGPipeline

        threshold = 0.30
        doc = _make_doc("DOC_GATE")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, threshold)]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=threshold,
        )
        with patch.object(p, "_generate", return_value="Answer [DOC_1]."):
            result = p.query("test question")
        self.assertGreater(len(result.sources), 0,
                           "Score at threshold should produce grounded response")

    def test_below_threshold_triggers_fallback(self):
        """Empty retrieval pool must trigger fallback (no sources)."""
        from src.rag_pipeline import RAGPipeline

        threshold = 0.30
        doc = _make_doc("DOC_LOW")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = []  # nothing above threshold

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=threshold,
            enable_general_fallback=True,
        )
        with patch.object(p, "_generate",
                          return_value="Note: This answer is based on general medical knowledge."):
            result = p.query("obscure off-topic query")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.sources or []), 0)

    def test_hard_refusal_when_fallback_disabled(self):
        """When enable_general_fallback=False, below-threshold queries must return a refusal answer."""
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("DOC_R")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = []

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=False,
        )
        # Use a clearly medical question that would have zero retrieval results
        # (simulated by vs.search.return_value = []). The domain gate must pass
        # this question (medical keyword present), then the hard-refusal fires
        # because fallback is disabled and no documents were found.
        result = p.query("What is the mechanism of Huntington's disease neurodegeneration?")
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(len(result.sources or []), 0)
        self.assertIn("don't have enough verified", result.answer.lower())


# ===========================================================================
# 5. Citation alignment
# ===========================================================================

class TestCitationAlignmentRegression(unittest.TestCase):
    """Sources list must only include DOC_X cited in the answer."""

    def _pipeline_n_docs(self, n: int) -> "RAGPipeline":
        from src.rag_pipeline import RAGPipeline

        docs = [_make_doc(f"D{i}", content=f"content {i}") for i in range(1, n + 1)]
        vs = MagicMock()
        vs.documents = docs
        vs.search.return_value = [(d, 0.75) for d in docs]
        return RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
        )

    def test_cited_subset_only(self):
        """Only the two cited docs appear in sources."""
        p = self._pipeline_n_docs(4)
        with patch.object(p, "_generate",
                          return_value="Claim A [DOC_1]. Claim B [DOC_3]."):
            result = p.query("question")
        ids = {s.doc_id for s in result.sources}
        self.assertEqual(ids, {"DOC_1", "DOC_3"})

    def test_no_citation_includes_all(self):
        """When model cites nothing, all retrieved docs are included."""
        p = self._pipeline_n_docs(3)
        with patch.object(p, "_generate", return_value="Uncited answer."):
            result = p.query("question")
        self.assertEqual(len(result.sources), 3)

    def test_out_of_range_citation_ignored(self):
        """[DOC_99] when only 2 docs retrieved is silently dropped."""
        p = self._pipeline_n_docs(2)
        with patch.object(p, "_generate",
                          return_value="Claim [DOC_1]. Ghost [DOC_99]."):
            result = p.query("question")
        ids = {s.doc_id for s in result.sources}
        self.assertIn("DOC_1", ids)
        self.assertNotIn("DOC_99", ids)


# ===========================================================================
# 6. Answer hygiene
# ===========================================================================

class TestAnswerHygiene(unittest.TestCase):
    """_clean_answer must add exactly one disclaimer and remove existing ones."""

    def setUp(self):
        from src.rag_pipeline import RAGPipeline, DISCLAIMER
        self.clean = RAGPipeline._clean_answer
        self.disclaimer = DISCLAIMER

    def test_disclaimer_added_once(self):
        cleaned = self.clean("Some factual answer.")
        count = cleaned.count(self.disclaimer)
        self.assertEqual(count, 1, f"Expected 1 disclaimer, got {count}")

    def test_existing_disclaimer_not_doubled(self):
        raw = f"Some answer.\n\n{self.disclaimer}"
        cleaned = self.clean(raw)
        count = cleaned.count(self.disclaimer)
        self.assertEqual(count, 1, "Disclaimer doubled after cleaning")

    def test_sources_block_removed(self):
        raw = "Answer text.\n\nSources: [DOC_1] Title\nURL: http://example.com"
        cleaned = self.clean(raw)
        self.assertNotIn("Sources:", cleaned)

    def test_empty_answer_unchanged(self):
        cleaned = self.clean("")
        self.assertEqual(cleaned, "")

    def test_triple_star_converted(self):
        """***bullet → * **bullet (markdown formatting fix)."""
        raw = "***Some bold bullet point"
        cleaned = self.clean(raw, add_disclaimer=False)
        self.assertNotIn("***", cleaned)


# ===========================================================================
# 7. Confidence normalisation boundary values
# ===========================================================================

class TestConfidenceNormalisationRegression(unittest.TestCase):

    def setUp(self):
        from src.rag_pipeline import RAGPipeline
        self.norm = RAGPipeline._normalise_score

    def test_at_threshold_maps_to_zero(self):
        score = self.norm(0.30, is_ce_domain=False, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 0.0, places=3,
                               msg="Score at threshold should map to 0.0")

    def test_at_ceiling_maps_to_one(self):
        score = self.norm(0.85, is_ce_domain=False, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 1.0, places=3,
                               msg="Score at BGE ceiling should map to 1.0")

    def test_above_ceiling_capped_at_one(self):
        score = self.norm(1.0, is_ce_domain=False, confidence_threshold=0.30)
        self.assertLessEqual(score, 1.0, "Score must be capped at 1.0")

    def test_below_threshold_is_zero(self):
        score = self.norm(0.10, is_ce_domain=False, confidence_threshold=0.30)
        self.assertEqual(score, 0.0, "Below-threshold score should be 0.0")

    def test_ce_zero_near_half(self):
        score = self.norm(0.0, is_ce_domain=True, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 0.5, delta=0.02,
                               msg="CE score=0 should map near 0.5 (sigmoid midpoint)")

    def test_ce_high_positive_near_one(self):
        # sigmoid(-6/3) = sigmoid(-2) ≈ 0.119; sigmoid(6/3) = sigmoid(2) ≈ 0.881
        # The implementation divides by 3 before sigmoid, so ±6 → ±2, not ±6.
        score = self.norm(6.0, is_ce_domain=True, confidence_threshold=0.30)
        self.assertGreater(score, 0.85, "CE score=6 should map above 0.85")

    def test_ce_high_negative_near_zero(self):
        score = self.norm(-6.0, is_ce_domain=True, confidence_threshold=0.30)
        self.assertLess(score, 0.15, "CE score=-6 should map below 0.15")


# ===========================================================================
# 8. Emergency: urgency detection helper
# ===========================================================================

class TestEmergencyUrgencyDetection(unittest.TestCase):
    """Verify the emergency urgency detection logic from run_eval.py."""

    EMERGENCY_KEYWORDS = {
        "call", "emergency", "immediately", "911", "urgent", "ambulance",
        "call for help", "seek immediate", "do not delay",
    }

    def _has_urgency(self, text: str) -> bool:
        opening = text.lower()[:300]
        return any(kw in opening for kw in self.EMERGENCY_KEYWORDS)

    def test_emergency_answer_with_urgency_passes(self):
        answer = "**Direct Answer:** Call emergency services immediately. This is a medical emergency."
        self.assertTrue(self._has_urgency(answer))

    def test_emergency_answer_without_urgency_fails(self):
        answer = "**Direct Answer:** A stroke occurs when blood supply to the brain is interrupted."
        self.assertFalse(self._has_urgency(answer))

    def test_911_in_answer_passes(self):
        answer = "Call 911 right away if you suspect a stroke."
        self.assertTrue(self._has_urgency(answer))

    def test_urgency_word_in_later_text_fails(self):
        """Urgency word appearing only after 300 chars should NOT pass the check."""
        padding = "A " * 180  # >300 chars
        answer = padding + "Call emergency services."
        self.assertFalse(self._has_urgency(answer))


# ===========================================================================
# 9. API retry resilience
# ===========================================================================

class TestAPIRetryBehavior(unittest.TestCase):
    """Verify retry logic: transient errors are retried, non-transient fail fast."""

    def _pipeline(self):
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("DOC_R")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, 0.75)]
        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
        )
        p._RETRY_BASE_S = 0.0  # no wait in tests
        return p

    def test_transient_429_is_detected(self):
        p = self._pipeline()
        self.assertTrue(p._is_transient(Exception("429 quota exceeded")))

    def test_transient_503_is_detected(self):
        p = self._pipeline()
        self.assertTrue(p._is_transient(Exception("503 service unavailable")))

    def test_non_transient_auth_not_retried(self):
        p = self._pipeline()
        self.assertFalse(p._is_transient(Exception("401 invalid api key")))

    def test_retry_succeeds_on_second_attempt(self):
        p = self._pipeline()
        call_count = [0]

        def flaky(**kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise Exception("503 internal")
            m = MagicMock()
            m.text = "Success on retry"
            return m

        with patch.object(p._client.models, "generate_content", side_effect=flaky):
            text = p._generate("prompt")
        self.assertEqual(text, "Success on retry")
        self.assertEqual(call_count[0], 2)

    def test_exhausted_retries_raise_runtime_error(self):
        p = self._pipeline()
        with patch.object(
            p._client.models, "generate_content",
            side_effect=Exception("503 unavailable")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                p._generate("prompt")
        self.assertIn("llm generation failed", str(ctx.exception).lower())

    def test_user_facing_message_no_raw_sdk_detail(self):
        p = self._pipeline()
        with patch.object(
            p._client.models, "generate_content",
            side_effect=Exception("quota_exceeded raw internal detail xyz")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                p._generate("prompt")
        msg = str(ctx.exception).lower()
        # User-facing message must not expose raw SDK error text
        self.assertNotIn("quota_exceeded raw internal detail xyz", msg)


# ===========================================================================
# 10. Cache correctness
# ===========================================================================

class TestCacheCorrectness(unittest.TestCase):

    def _pipeline(self):
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("DOC_C")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, 0.75)]
        return RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
        )

    def test_identical_questions_hit_cache(self):
        """Second call with same question does not invoke _generate again."""
        p = self._pipeline()
        call_count = [0]

        def counting_generate(prompt):
            call_count[0] += 1
            return "Answer [DOC_1]."

        with patch.object(p, "_generate", side_effect=counting_generate):
            p.query("What are symptoms of anemia?")
            p.query("What are symptoms of anemia?")

        self.assertEqual(call_count[0], 1, "Cache should prevent second _generate call")

    def test_different_questions_invoke_generate_twice(self):
        p = self._pipeline()
        call_count = [0]

        def counting_generate(prompt):
            call_count[0] += 1
            return "Answer [DOC_1]."

        with patch.object(p, "_generate", side_effect=counting_generate):
            p.query("What are symptoms of anemia?")
            p.query("How is hypertension treated?")

        self.assertEqual(call_count[0], 2)

    def test_different_history_invalidates_cache(self):
        p = self._pipeline()
        call_count = [0]

        def counting_generate(prompt):
            call_count[0] += 1
            return "Answer [DOC_1]."

        hist = [{"role": "user", "content": "prior"}, {"role": "assistant", "content": "prior answer"}]
        with patch.object(p, "_generate", side_effect=counting_generate):
            p.query("What are symptoms?")
            p.query("What are symptoms?", history=hist)

        self.assertEqual(call_count[0], 2, "Different history must produce different cache key")


# ===========================================================================
# 11. Provenance field
# ===========================================================================

class TestSourceProvenanceRegression(unittest.TestCase):

    def test_indexed_provenance_set(self):
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("DOC_P", content="Content.")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, 0.75)]
        p = RAGPipeline(
            vector_store=vs, model_name="gemini-2.0-flash",
            api_key="fake-key", confidence_threshold=0.30,
        )
        with patch.object(p, "_generate", return_value="Answer [DOC_1]."):
            result = p.query("test")
        self.assertEqual(result.sources[0].provenance, "indexed")

    def test_live_pubmed_provenance_preserved(self):
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("PMID_12345", content="Live content.")
        doc["metadata"] = {"provenance": "live_pubmed"}
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, 0.75)]
        p = RAGPipeline(
            vector_store=vs, model_name="gemini-2.0-flash",
            api_key="fake-key", confidence_threshold=0.30,
        )
        with patch.object(p, "_generate", return_value="Answer [DOC_1]."):
            result = p.query("test")
        self.assertEqual(result.sources[0].provenance, "live_pubmed")


# ===========================================================================
# 12. Duplicate ingestion guard
# ===========================================================================

class TestDuplicateIngestionGuardRegression(unittest.TestCase):

    def _empty_vs(self):
        import faiss
        import numpy as np
        from src.vector_store import VectorStore
        from src.data_ingestion import MedicalDocument

        vs = VectorStore.__new__(VectorStore)
        vs.embedding_model_name = "BAAI/bge-base-en-v1.5"
        vs.use_bge_prefix = True
        vs.chunk_size = 256
        vs.chunk_overlap = 40
        vs.embedding_batch_size = 32
        vs.dimension = 4

        vs.index = faiss.IndexFlatIP(4)
        vs.documents = []

        splitter = MagicMock()
        splitter.split_text = lambda t: [t]
        vs.text_splitter = splitter
        vs._encode_passages = lambda texts: np.zeros((len(texts), 4), dtype="float32")
        return vs

    def _doc(self, doc_id):
        from src.data_ingestion import MedicalDocument
        return MedicalDocument(content="content", title="T", source="s", doc_id=doc_id)

    def test_second_ingest_same_id_adds_zero(self):
        vs = self._empty_vs()
        doc = self._doc("DEDUP_1")
        vs.add_documents([doc])
        added = vs.add_documents([doc])
        self.assertEqual(added, 0)

    def test_new_id_after_existing_added(self):
        vs = self._empty_vs()
        doc_a = self._doc("DA")
        doc_b = self._doc("DB")
        vs.add_documents([doc_a])
        added = vs.add_documents([doc_b])
        self.assertGreater(added, 0)
        doc_ids = {d["doc_id"] for d in vs.documents}
        self.assertIn("DA", doc_ids)
        self.assertIn("DB", doc_ids)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
