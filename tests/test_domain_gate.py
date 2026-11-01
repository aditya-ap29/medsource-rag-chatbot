"""
tests/test_domain_gate.py
--------------------------
Tests for the medical-domain relevance gate (src/domain_gate.py) and
its integration with RAGPipeline.query().

Categories tested:
  1. Clear medical questions         → 'medical' label
  2. Emergency medical questions     → 'emergency' label (never blocked)
  3. Explicit out-of-scope questions → 'out_of_scope' label
  4. Ambiguous / borderline queries  → correct fallback behaviour
  5. Low-evidence medical queries    → pass (keyword match wins)
  6. API failure mid-query           → RuntimeError propagated correctly
  7. OOP query via pipeline          → OUT_OF_SCOPE_RESPONSE returned
  8. Emergency query via pipeline    → urgent answer returned (not blocked)

Run with:
    python -m pytest tests/test_domain_gate.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_gate import (
    classify_domain,
    is_emergency,
    OUT_OF_SCOPE_RESPONSE,
    RETRIEVAL_STRONG_THRESHOLD,
    OOP_SCORE_CEILING,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(doc_id: str, score: float, title: str = "Test Doc"):
    """Return a (doc_dict, score) pair mimicking VectorStore output."""
    return (
        {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_0",
            "content": "Medical content.",
            "title": title,
            "source": "test",
            "year": "2024",
            "url": None,
            "metadata": {"provenance": "indexed"},
            "chunk_index": 0,
            "total_chunks": 1,
        },
        score,
    )


def _pipeline(retrieved_score: float = 0.75, answer: str = "Answer [DOC_1]."):
    """Return a RAGPipeline with a mock vector store."""
    from src.rag_pipeline import RAGPipeline

    doc = _doc("DOC_TEST", retrieved_score)[0]
    vs = MagicMock()
    vs.documents = [doc]
    vs.search.return_value = [_doc("DOC_TEST", retrieved_score)]

    p = RAGPipeline(
        vector_store=vs,
        model_name="gemini-2.0-flash",
        api_key="fake-key",
        confidence_threshold=0.30,
        enable_general_fallback=True,
    )
    return p, answer


# ===========================================================================
# 1. is_emergency() — standalone helper
# ===========================================================================

class TestIsEmergency(unittest.TestCase):
    """is_emergency() should detect emergency keywords reliably."""

    def test_chest_pain(self):
        self.assertTrue(is_emergency("I have chest pain"))

    def test_stroke(self):
        self.assertTrue(is_emergency("What should I do if someone is having a stroke?"))

    def test_overdose(self):
        self.assertTrue(is_emergency("opioid overdose treatment"))

    def test_anaphylaxis(self):
        self.assertTrue(is_emergency("severe anaphylaxis, give epinephrine"))

    def test_choking(self):
        self.assertTrue(is_emergency("child near me is choking"))

    def test_seizure(self):
        self.assertTrue(is_emergency("person is having a seizure"))

    def test_non_emergency_medical(self):
        self.assertFalse(is_emergency("What are the symptoms of type 2 diabetes?"))

    def test_geography_not_emergency(self):
        self.assertFalse(is_emergency("What is the capital of France?"))

    def test_finance_not_emergency(self):
        self.assertFalse(is_emergency("What is the best stock to invest in?"))


# ===========================================================================
# 2. classify_domain() — medical questions
# ===========================================================================

class TestClassifyMedical(unittest.TestCase):
    """Medical questions should be labelled 'medical' or 'emergency'."""

    def _strong_docs(self, n: int = 3):
        return [_doc(f"D{i}", RETRIEVAL_STRONG_THRESHOLD + 0.05) for i in range(n)]

    def _weak_docs(self, n: int = 3):
        return [_doc(f"D{i}", 0.40) for i in range(n)]

    def test_diabetes_symptoms(self):
        label, _ = classify_domain("What are the symptoms of type 2 diabetes?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_hypertension_treatment(self):
        label, _ = classify_domain("How is hypertension treated?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_metformin_side_effects(self):
        label, _ = classify_domain("What are the side effects of metformin?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_tb_diagnosis(self):
        label, _ = classify_domain("How is tuberculosis diagnosed?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_unusual_phrasing_statin_dizzy(self):
        """Unusual phrasing should not be blocked if medical keywords present."""
        label, _ = classify_domain("I feel lightheaded after my statin", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_copd_management(self):
        label, _ = classify_domain("How is COPD managed?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_anemia_low_score(self):
        """Low retrieval scores should not block a keyword-matched medical query."""
        label, _ = classify_domain("What are the symptoms of iron deficiency anemia?", self._weak_docs())
        self.assertEqual(label, "medical")

    def test_medical_keyword_overrides_low_retrieval(self):
        """Explicit medical keyword wins even with zero retrieval docs."""
        label, _ = classify_domain("I have chest tightness and fatigue", [])
        self.assertIn(label, ("medical", "emergency"))

    def test_pregnancy_fatigue(self):
        label, _ = classify_domain("Is it normal to feel tired during pregnancy?", self._strong_docs())
        self.assertEqual(label, "medical")

    def test_depression_antidepressants(self):
        label, _ = classify_domain("Are antidepressants effective for depression?", self._strong_docs())
        self.assertEqual(label, "medical")


# ===========================================================================
# 3. classify_domain() — emergency questions (always pass)
# ===========================================================================

class TestClassifyEmergency(unittest.TestCase):
    """Emergency questions must never be blocked."""

    def _no_docs(self):
        return []

    def test_heart_attack_first_person(self):
        """First-person emergency must be classified as emergency."""
        label, _ = classify_domain("I think I am having a heart attack", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_stroke_recognition(self):
        label, _ = classify_domain("Someone has sudden face drooping and arm weakness", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_anaphylaxis_epipen(self):
        label, _ = classify_domain("Use EpiPen for anaphylaxis", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_overdose(self):
        label, _ = classify_domain("opioid overdose how to treat", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_seizure_duration(self):
        label, _ = classify_domain("seizure lasting more than 5 minutes", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_choking_child(self):
        label, _ = classify_domain("A child near me is choking what should I do", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_emergency_overrides_oop_signal(self):
        """Emergency keyword beats any OOP signal."""
        label, _ = classify_domain("capital of france heart attack now", self._no_docs())
        self.assertEqual(label, "emergency")

    def test_suicide_mention(self):
        label, _ = classify_domain("I am having thoughts of suicide", self._no_docs())
        self.assertEqual(label, "emergency")


# ===========================================================================
# 4. classify_domain() — out-of-scope questions
# ===========================================================================

class TestClassifyOOP(unittest.TestCase):
    """Non-medical questions must be labelled 'out_of_scope'."""

    def _low_docs(self):
        """Simulate BGE background noise: scores between 0.35–0.50."""
        return [_doc(f"D{i}", 0.38 + i * 0.03) for i in range(3)]

    def test_geography_capital(self):
        label, _ = classify_domain("What is the capital of France?", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_stock_investment(self):
        label, _ = classify_domain("What is the best stock to invest in right now?", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_weather_london(self):
        label, _ = classify_domain("What is the weather like in London today?", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_poem_spring(self):
        label, _ = classify_domain("Write me a poem about spring flowers.", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_recipe(self):
        label, _ = classify_domain("Give me a recipe for chocolate cake", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_sports_score(self):
        label, _ = classify_domain("What was the cricket score last night?", self._low_docs())
        self.assertEqual(label, "out_of_scope")

    def test_no_docs_no_keywords(self):
        """Unrecognised topic with no retrieval → out of scope."""
        label, _ = classify_domain("Tell me the exchange rate for USD to EUR", [])
        self.assertEqual(label, "out_of_scope")


# ===========================================================================
# 5. classify_domain() — ambiguous / borderline
# ===========================================================================

class TestClassifyAmbiguous(unittest.TestCase):
    """Borderline queries should lean towards 'medical' rather than blocking."""

    def _medium_docs(self, n: int = 2):
        return [_doc(f"D{i}", 0.55) for i in range(n)]

    def test_tired_all_the_time(self):
        """'Tired all the time' is ambiguous but contains enough medical signal."""
        label, _ = classify_domain("I am tired all the time", self._medium_docs())
        # 'fatigue' is in the medical keyword set
        self.assertIn(label, ("medical",))

    def test_chest_pain_ambiguous(self):
        """Chest pain is both a medical term and an emergency signal."""
        label, _ = classify_domain("Is my chest pain serious?", self._medium_docs())
        self.assertIn(label, ("medical", "emergency"))

    def test_aspirin_daily(self):
        """Aspirin question with conflicting evidence."""
        label, _ = classify_domain("Should I take aspirin every day?", self._medium_docs())
        self.assertEqual(label, "medical")


# ===========================================================================
# 6. Pipeline integration — OOP queries return OOP response
# ===========================================================================

class TestPipelineOOP(unittest.TestCase):
    """RAGPipeline.query() must return OUT_OF_SCOPE_RESPONSE for OOP queries."""

    def _pipeline_oop(self, q: str):
        """Build a pipeline that returns a low BGE score (OOP noise range)."""
        from src.rag_pipeline import RAGPipeline

        doc = _doc("OOP_DOC", 0.40)[0]
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [_doc("OOP_DOC", 0.40)]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=True,
        )
        return p

    def test_geography_returns_oop_response(self):
        p = self._pipeline_oop("What is the capital of France?")
        result = p.query("What is the capital of France?")
        self.assertIn("MedSource", result.answer)
        self.assertIn("medical", result.answer.lower())
        self.assertEqual(result.sources, [])
        self.assertEqual(result.confidence, 0.0)

    def test_finance_returns_oop_response(self):
        p = self._pipeline_oop("What is the best stock to buy?")
        result = p.query("What is the best stock to buy?")
        self.assertIn("MedSource", result.answer)
        self.assertEqual(result.sources, [])

    def test_weather_returns_oop_response(self):
        p = self._pipeline_oop("What is the weather like in London today?")
        result = p.query("What is the weather like in London today?")
        # Domain gate should block this; OOP response must mention MedSource
        self.assertIn("MedSource", result.answer)
        self.assertEqual(result.sources, [])

    def test_oop_answer_contains_disclaimer(self):
        """OOP response must include the standard educational disclaimer."""
        p = self._pipeline_oop("What is the capital of France?")
        result = p.query("What is the capital of France?")
        self.assertIn("educational purposes only", result.answer)

    def test_oop_does_not_call_generate(self):
        """_generate() must never be called for an OOP query."""
        p = self._pipeline_oop("Write me a poem about flowers.")
        with patch.object(p, "_generate") as mock_gen:
            p.query("Write me a poem about flowers.")
        mock_gen.assert_not_called()


# ===========================================================================
# 7. Pipeline integration — emergency questions are NEVER blocked
# ===========================================================================

class TestPipelineEmergency(unittest.TestCase):
    """Emergency queries must always reach the LLM and return urgency language."""

    def _pipeline_emergency(self):
        """Pipeline with high-scoring docs (like a real emergency retrieval)."""
        from src.rag_pipeline import RAGPipeline

        doc = _doc("EMER_DOC", 0.78, "Heart Attack Emergency Response")[0]
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [_doc("EMER_DOC", 0.78, "Heart Attack Emergency Response")]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=True,
        )
        return p

    def test_heart_attack_not_blocked(self):
        """Heart attack query must reach _generate() and not be OOP-blocked."""
        p = self._pipeline_emergency()
        with patch.object(p, "_generate", return_value="Call 911 immediately [DOC_1].") as mock_gen:
            result = p.query("I think I am having a heart attack")
        mock_gen.assert_called_once()
        self.assertIn("Call 911", result.answer)

    def test_stroke_not_blocked(self):
        p = self._pipeline_emergency()
        with patch.object(p, "_generate", return_value="Call emergency services immediately [DOC_1].") as mock_gen:
            result = p.query("Someone is having a stroke. What should I do?")
        mock_gen.assert_called_once()

    def test_overdose_not_blocked(self):
        p = self._pipeline_emergency()
        with patch.object(p, "_generate", return_value="Administer naloxone immediately [DOC_1].") as mock_gen:
            result = p.query("Someone has an opioid overdose")
        mock_gen.assert_called_once()

    def test_seizure_not_blocked(self):
        p = self._pipeline_emergency()
        with patch.object(p, "_generate", return_value="Do not restrain the person [DOC_1].") as mock_gen:
            result = p.query("What should I do if someone has a seizure?")
        mock_gen.assert_called_once()

    def test_anaphylaxis_not_blocked(self):
        p = self._pipeline_emergency()
        with patch.object(p, "_generate", return_value="Epinephrine first [DOC_1].") as mock_gen:
            result = p.query("anaphylaxis first aid epinephrine")
        mock_gen.assert_called_once()


# ===========================================================================
# 8. Pipeline integration — low-evidence medical queries still pass
# ===========================================================================

class TestPipelineLowEvidence(unittest.TestCase):
    """Medical questions with weak retrieval must not be incorrectly blocked."""

    def _pipeline_low_evidence(self):
        from src.rag_pipeline import RAGPipeline

        doc = _doc("LOW_DOC", 0.35)[0]
        vs = MagicMock()
        vs.documents = [doc]
        # Return a doc that passes the confidence gate
        vs.search.return_value = [_doc("LOW_DOC", 0.35)]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=True,
        )
        return p

    def test_rare_disease_not_blocked(self):
        """A medical question about a disease in corpus should not be blocked."""
        p = self._pipeline_low_evidence()
        with patch.object(p, "_generate", return_value="Scurvy is caused by vitamin C deficiency."):
            result = p.query("What are the symptoms of scurvy?")
        # Should NOT return OOP response
        self.assertNotIn("outside", result.answer.lower())

    def test_unusual_phrasing_medical(self):
        """'My statin makes me dizzy' should not be blocked."""
        p = self._pipeline_low_evidence()
        with patch.object(p, "_generate", return_value="Statins can cause dizziness."):
            result = p.query("My statin makes me feel lightheaded")
        self.assertNotIn("outside", result.answer.lower())

    def test_vaccination_question(self):
        p = self._pipeline_low_evidence()
        with patch.object(p, "_generate", return_value="Vaccination prevents infections."):
            result = p.query("Should I get a flu vaccine this year?")
        self.assertNotIn("outside", result.answer.lower())


# ===========================================================================
# 9. API failure integration
# ===========================================================================

class TestAPIFailureIntegration(unittest.TestCase):
    """API failures should propagate as RuntimeError, not silently swallowed."""

    def _pipeline(self):
        from src.rag_pipeline import RAGPipeline

        doc = _doc("DOC_API", 0.75)[0]
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [_doc("DOC_API", 0.75)]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=True,
        )
        return p

    def test_api_failure_raises_runtime_error(self):
        p = self._pipeline()
        with patch.object(p, "_generate", side_effect=RuntimeError("LLM generation failed")):
            with self.assertRaises(RuntimeError):
                p.query("What are the symptoms of diabetes?")

    def test_oop_query_does_not_require_api(self):
        """OOP queries must not make any API call even if the key is invalid."""
        from src.rag_pipeline import RAGPipeline

        doc = _doc("OOP_API", 0.40)[0]
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [_doc("OOP_API", 0.40)]

        p = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="INVALID_KEY",  # would fail if called
            confidence_threshold=0.30,
        )
        # Should return normally (no API call → no failure)
        result = p.query("What is the capital of France?")
        self.assertIn("MedSource", result.answer)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
