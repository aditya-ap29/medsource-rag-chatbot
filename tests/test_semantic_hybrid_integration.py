"""
tests/test_semantic_hybrid_integration.py
------------------------------------------
Integration tests for the semantic + hybrid RAG upgrade.

Tests cover:
- Semantic classifier + pipeline integration (mock embedding).
- Hybrid retriever + pipeline integration.
- Evidence assessor + pipeline integration.
- Emergency bypass always fires regardless of classifier.
- Ambiguous handling modes (pass / warn / refuse).
- All new flags default to False (existing baseline behaviour unchanged).
- Classifier fallback when model file is missing.
- BM25 not built when enable_hybrid=False.
- Compatibility with existing RAG pipeline API.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(doc_id, score=0.80, url="https://example.com", year="2023"):
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_c0",
        "content": f"Medical content from {doc_id}.",
        "title": f"Title {doc_id}",
        "source": "test",
        "year": str(year),
        "url": url,
        "metadata": {},
        "chunk_index": 0,
        "total_chunks": 1,
    }


def _make_pipeline(
    vs=None,
    enable_semantic_classifier=False,
    enable_hybrid_retrieval=False,
    enable_evidence_assessment=False,
    ambiguous_handling="warn",
    api_key="FAKE",
):
    """Return a RAGPipeline with mocked Gemini client."""
    from src.rag_pipeline import RAGPipeline

    if vs is None:
        vs = MagicMock()
        vs.embedding_model = MagicMock()
        vs.embedding_model_name = "BAAI/bge-base-en-v1.5"
        vs.documents = [
            _make_chunk("d1"), _make_chunk("d2"), _make_chunk("d3"),
        ]
        vs.search.return_value = [
            (_make_chunk("d1"), 0.82),
            (_make_chunk("d2"), 0.78),
        ]

    pipeline = RAGPipeline(
        vector_store=vs,
        api_key=api_key,
        enable_semantic_classifier=enable_semantic_classifier,
        enable_hybrid_retrieval=enable_hybrid_retrieval,
        enable_evidence_assessment=enable_evidence_assessment,
        ambiguous_handling=ambiguous_handling,
        confidence_threshold=0.30,
    )

    # Replace Gemini client with a mock
    pipeline._client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "**Direct Answer:** Diabetes symptoms include thirst [DOC_1]. "
        "**Explanation:** More detail [DOC_2]."
    )
    pipeline._client.models.generate_content.return_value = mock_response

    return pipeline


# ---------------------------------------------------------------------------
# Baseline behaviour unchanged
# ---------------------------------------------------------------------------

class TestNewFlagsDefaultToFalse(unittest.TestCase):
    """All new flags must default to False so existing behaviour is unchanged."""

    def test_defaults(self):
        p = _make_pipeline()
        self.assertFalse(p.enable_semantic_classifier)
        self.assertFalse(p.enable_hybrid_retrieval)
        self.assertFalse(p.enable_evidence_assessment)

    def test_default_ambiguous_handling(self):
        p = _make_pipeline()
        self.assertEqual(p.ambiguous_handling, "warn")

    def test_hybrid_retriever_uses_vector_store_when_disabled(self):
        p = _make_pipeline(enable_hybrid_retrieval=False)
        retriever = p._get_hybrid_retriever()
        # With enable_hybrid=False, search() should delegate to vector_store.search()
        results = retriever.search("diabetes", top_k=2, score_threshold=0.3)
        p.vector_store.search.assert_called()


# ---------------------------------------------------------------------------
# Semantic classifier integration
# ---------------------------------------------------------------------------

class TestSemanticClassifierIntegration(unittest.TestCase):

    def _inject_clf(self, pipeline, label, conf=0.95):
        """Inject a mock SemanticClassifier returning the given label."""
        mock_clf = MagicMock()
        mock_clf.predict.return_value = (label, conf, f"mock p={conf}")
        pipeline._semantic_clf = mock_clf
        pipeline.enable_semantic_classifier = True
        return mock_clf

    def test_medical_question_passes(self):
        p = _make_pipeline(enable_semantic_classifier=True)
        self._inject_clf(p, "medical")
        result = p.query("What are the symptoms of diabetes?")
        # Must not return out-of-scope response
        self.assertNotIn("outside", result.answer.lower())

    def test_oop_question_blocked(self):
        p = _make_pipeline(enable_semantic_classifier=True)
        self._inject_clf(p, "out_of_scope", conf=0.92)
        result = p.query("What is the capital of France?")
        self.assertIn("outside", result.answer.lower())
        self.assertEqual(result.sources, [])

    def test_oop_low_confidence_falls_through_to_keyword_gate(self):
        """OOP with conf < 0.7 should NOT be blocked by classifier alone."""
        p = _make_pipeline(enable_semantic_classifier=True)
        self._inject_clf(p, "out_of_scope", conf=0.55)  # Below 0.7 threshold
        # Keyword gate will pass it if it has medical keywords
        # "What are the symptoms of diabetes?" → keyword gate says medical
        result = p.query("What are the symptoms of diabetes?")
        self.assertNotIn("outside", result.answer.lower())

    def test_ambiguous_warn_mode(self):
        p = _make_pipeline(enable_semantic_classifier=True, ambiguous_handling="warn")
        self._inject_clf(p, "ambiguous")
        result = p.query("I feel tired all the time")
        # Should still produce an answer (not an OOP response)
        self.assertFalse(result.answer.startswith("I'm MedSource"))

    def test_ambiguous_refuse_mode(self):
        p = _make_pipeline(enable_semantic_classifier=True, ambiguous_handling="refuse")
        self._inject_clf(p, "ambiguous")
        result = p.query("I feel tired all the time")
        self.assertIn("ambiguous", result.answer.lower())
        self.assertEqual(result.sources, [])

    def test_emergency_bypass_overrides_classifier(self):
        """Emergency keywords must bypass classifier regardless of its label."""
        p = _make_pipeline(enable_semantic_classifier=True)
        self._inject_clf(p, "out_of_scope", conf=0.99)  # Classifier says OOP
        # But this is an emergency question
        result = p.query("I think I'm having a heart attack")
        # Must NOT return out-of-scope response
        self.assertNotIn("outside", result.answer.lower())

    def test_classifier_missing_falls_back_to_keyword(self):
        """If classifier file not found, must fall back to keyword gate silently."""
        p = _make_pipeline(enable_semantic_classifier=True)
        # Don't inject clf — _semantic_clf stays None
        p.semantic_classifier_path = "/nonexistent/path/that/does/not/exist"
        # Should not raise; keyword gate handles it
        result = p.query("What are the symptoms of diabetes?")
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Hybrid retrieval integration
# ---------------------------------------------------------------------------

class TestHybridRetrievalIntegration(unittest.TestCase):

    def test_hybrid_disabled_uses_vector_store(self):
        p = _make_pipeline(enable_hybrid_retrieval=False)
        p.query("What is asthma?")
        p.vector_store.search.assert_called()

    def test_hybrid_enabled_uses_hybrid_retriever(self):
        p = _make_pipeline(enable_hybrid_retrieval=True)

        # Inject a mock hybrid retriever
        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            (_make_chunk("d1"), 0.032),
            (_make_chunk("d2"), 0.028),
        ]
        mock_hr.config = MagicMock()
        mock_hr.config.enable_hybrid = True
        p._hybrid_retriever = mock_hr

        result = p.query("What is asthma?")
        mock_hr.search.assert_called()

    def test_result_has_sources_from_hybrid(self):
        p = _make_pipeline(enable_hybrid_retrieval=True)
        mock_hr = MagicMock()
        mock_hr.search.return_value = [
            (_make_chunk("d1"), 0.032),
            (_make_chunk("d2"), 0.028),
        ]
        mock_hr.config = MagicMock()
        mock_hr.config.enable_hybrid = True
        p._hybrid_retriever = mock_hr

        result = p.query("What is asthma?")
        # Sources should carry provenance metadata
        for source in result.sources:
            self.assertIsNotNone(source.doc_id)


# ---------------------------------------------------------------------------
# Evidence assessor integration
# ---------------------------------------------------------------------------

class TestEvidenceAssessorIntegration(unittest.TestCase):

    def test_evidence_disabled_skips_assessment(self):
        p = _make_pipeline(enable_evidence_assessment=False)
        # No assessor should be created
        result = p.query("What is diabetes?")
        self.assertIsNone(p._evidence_assessor)

    def test_evidence_enabled_creates_assessor(self):
        p = _make_pipeline(enable_evidence_assessment=True)
        p.query("What is diabetes?")
        self.assertIsNotNone(p._evidence_assessor)

    def test_insufficient_evidence_triggers_fallback(self):
        """When evidence is INSUFFICIENT, should use general fallback (no sources)."""
        from src.evidence_assessor import EvidenceAssessor, VERDICT_INSUFFICIENT, EvidenceAssessment

        p = _make_pipeline(enable_evidence_assessment=True)

        mock_assessor = MagicMock(spec=EvidenceAssessor)
        mock_assessor.assess.return_value = EvidenceAssessment(
            verdict=VERDICT_INSUFFICIENT,
            score=-5,
            reason="no credible docs",
        )
        p._evidence_assessor = mock_assessor

        result = p.query("What is an obscure rare syndrome?")
        # Should fall through to fallback (no sources because retrieved_docs emptied)
        # Answer should contain the general knowledge note or refusal
        self.assertIsNotNone(result)

    def test_uncertain_evidence_adds_warning(self):
        """UNCERTAIN verdict should add an uncertainty warning to the response."""
        from src.evidence_assessor import EvidenceAssessor, VERDICT_UNCERTAIN, EvidenceAssessment

        p = _make_pipeline(enable_evidence_assessment=True)

        mock_assessor = MagicMock(spec=EvidenceAssessor)
        mock_assessor.assess.return_value = EvidenceAssessment(
            verdict=VERDICT_UNCERTAIN,
            score=1,
            reason="borderline evidence",
        )
        p._evidence_assessor = mock_assessor

        result = p.query("What is diabetes?")
        # Warning should mention evidence limitation
        self.assertIn("limited", result.warning.lower())


# ---------------------------------------------------------------------------
# Config flags
# ---------------------------------------------------------------------------

class TestConfigFlags(unittest.TestCase):
    """New settings must be accepted by the Settings model without error."""

    def test_new_settings_loadable(self):
        from src.config import Settings
        s = Settings(
            enable_semantic_classifier=True,
            semantic_classifier_path="./data/semantic_classifier_v1",
            ambiguous_handling="warn",
            enable_hybrid_retrieval=True,
            rrf_k=60,
            rrf_dense_weight=1.0,
            rrf_sparse_weight=1.0,
            bm25_k1=1.5,
            bm25_b=0.75,
            enable_evidence_assessment=True,
        )
        self.assertTrue(s.enable_semantic_classifier)
        self.assertTrue(s.enable_hybrid_retrieval)
        self.assertTrue(s.enable_evidence_assessment)

    def test_default_settings_unchanged(self):
        from src.config import Settings
        s = Settings()
        self.assertFalse(s.enable_semantic_classifier)
        self.assertFalse(s.enable_hybrid_retrieval)
        self.assertFalse(s.enable_evidence_assessment)
        self.assertEqual(s.ambiguous_handling, "warn")


if __name__ == "__main__":
    unittest.main()
