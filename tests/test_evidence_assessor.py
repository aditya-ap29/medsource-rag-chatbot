"""
tests/test_evidence_assessor.py
---------------------------------
Unit tests for the EvidenceAssessor.

Tests cover:
- Empty retrieval → INSUFFICIENT.
- High-quality evidence → SUFFICIENT.
- Single-source diversity penalty.
- Low-score penalty.
- URL presence bonus / penalty.
- HTML contamination penalty.
- RRF score handling.
- Config overrides.
- Verdict and score contract.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _doc(doc_id: str, score: float, url: str = "https://example.com", year: str = "2023", content: str = "Medical content about the condition.") -> tuple:
    return (
        {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_c0",
            "content": content,
            "title": "Test",
            "source": "test",
            "year": str(year),
            "url": url,
            "metadata": {},
        },
        score,
    )


class TestEvidenceAssessorEmpty(unittest.TestCase):

    def setUp(self):
        from src.evidence_assessor import EvidenceAssessor
        self.assessor = EvidenceAssessor()

    def test_empty_docs_is_insufficient(self):
        from src.evidence_assessor import VERDICT_INSUFFICIENT
        result = self.assessor.assess("What is diabetes?", [])
        self.assertEqual(result.verdict, VERDICT_INSUFFICIENT)

    def test_empty_docs_negative_score(self):
        result = self.assessor.assess("What is diabetes?", [])
        self.assertLess(result.score, 0)

    def test_empty_docs_reason_present(self):
        result = self.assessor.assess("What is diabetes?", [])
        self.assertIn("no documents", result.reason)


class TestEvidenceAssessorSufficient(unittest.TestCase):

    def setUp(self):
        from src.evidence_assessor import EvidenceAssessor
        self.assessor = EvidenceAssessor()

    def test_multiple_high_score_docs_sufficient(self):
        from src.evidence_assessor import VERDICT_SUFFICIENT
        docs = [
            _doc("d1", 0.82),
            _doc("d2", 0.78),
            _doc("d3", 0.75),
        ]
        result = self.assessor.assess("What is diabetes?", docs)
        self.assertEqual(result.verdict, VERDICT_SUFFICIENT)

    def test_diverse_sources_sufficient(self):
        from src.evidence_assessor import VERDICT_SUFFICIENT
        docs = [
            _doc("d1", 0.80),
            _doc("d2", 0.78),
            _doc("d3", 0.76),
        ]
        result = self.assessor.assess("What is diabetes?", docs)
        # Three distinct doc_ids → high diversity
        self.assertGreater(result.signals.get("source_diversity", 0), 0.5)

    def test_score_is_integer(self):
        docs = [_doc("d1", 0.80), _doc("d2", 0.78)]
        result = self.assessor.assess("test", docs)
        self.assertIsInstance(result.score, int)

    def test_signals_populated(self):
        docs = [_doc("d1", 0.80)]
        result = self.assessor.assess("test", docs)
        self.assertIn("credible_chunks", result.signals)
        self.assertIn("n_with_url", result.signals)


class TestEvidenceAssessorInsufficient(unittest.TestCase):

    def setUp(self):
        from src.evidence_assessor import EvidenceAssessor, EvidenceAssessorConfig
        cfg = EvidenceAssessorConfig(
            credibility_floor=0.65,
            min_credible_chunks=2,
            strong_relevance_threshold=0.72,
            min_strong_chunks=1,
            min_diversity=0.5,
            sufficient_threshold=3,
            insufficient_threshold=-1,
        )
        self.assessor = EvidenceAssessor(config=cfg)

    def test_low_score_single_doc(self):
        from src.evidence_assessor import VERDICT_INSUFFICIENT
        docs = [_doc("d1", 0.30)]
        result = self.assessor.assess("test", docs)
        self.assertIn(result.verdict, ("insufficient", "uncertain"))

    def test_no_url_penalty_applied(self):
        docs = [_doc("d1", 0.80, url="")]
        result = self.assessor.assess("test", docs)
        self.assertIn("n_with_url", result.signals)
        self.assertEqual(result.signals["n_with_url"], 0)

    def test_single_source_penalised(self):
        docs = [
            _doc("d1", 0.80),
            _doc("d1", 0.78),  # Same doc_id — two chunks, one source
            _doc("d1", 0.75),
        ]
        result = self.assessor.assess("test", docs)
        self.assertEqual(result.signals.get("unique_doc_ids"), 1)

    def test_html_contamination_penalised(self):
        html_content = "<p>Some text about <b>hypertension</b> management.</p>"
        docs = [_doc("d1", 0.80, content=html_content)]
        result = self.assessor.assess("test", docs)
        self.assertGreater(result.signals.get("html_chunks", 0), 0)


class TestEvidenceAssessorConfig(unittest.TestCase):
    """Ensure EvidenceAssessorConfig overrides are respected."""

    def test_custom_thresholds(self):
        from src.evidence_assessor import EvidenceAssessor, EvidenceAssessorConfig, VERDICT_SUFFICIENT
        # Very lax config — single doc with any score = sufficient
        cfg = EvidenceAssessorConfig(
            credibility_floor=0.0,
            min_credible_chunks=1,
            strong_relevance_threshold=0.0,
            min_strong_chunks=1,
            min_diversity=0.0,
            sufficient_threshold=0,   # Even score=0 is sufficient
            insufficient_threshold=-99,
        )
        assessor = EvidenceAssessor(config=cfg)
        docs = [_doc("d1", 0.10)]  # Very low score
        result = assessor.assess("test", docs)
        self.assertNotEqual(result.verdict, "insufficient")


class TestEvidenceAssessorRRF(unittest.TestCase):
    """Assessor handles RRF scores (small float values) gracefully."""

    def setUp(self):
        from src.evidence_assessor import EvidenceAssessor
        self.assessor = EvidenceAssessor()

    def test_rrf_scores_handled(self):
        # RRF scores are typically 0.01–0.05 (1/61 ≈ 0.016)
        docs = [
            _doc("d1", 0.032),
            _doc("d2", 0.028),
            _doc("d3", 0.024),
        ]
        # Must not raise; verdict may be anything
        result = self.assessor.assess("What is diabetes?", docs, confidence_threshold=0.30)
        self.assertIn(result.verdict, ("sufficient", "insufficient", "uncertain"))

    def test_mixed_rrf_and_cosine(self):
        # Should handle mixed pool gracefully
        docs = [_doc("d1", 0.80), _doc("d2", 0.03)]
        result = self.assessor.assess("test", docs)
        self.assertIsNotNone(result.verdict)


class TestEvidenceAssessorVerdictContract(unittest.TestCase):
    """All verdicts must be one of the defined constants."""

    def setUp(self):
        from src.evidence_assessor import EvidenceAssessor
        self.assessor = EvidenceAssessor()

    def test_verdict_is_valid(self):
        from src.evidence_assessor import VERDICT_SUFFICIENT, VERDICT_INSUFFICIENT, VERDICT_UNCERTAIN
        valid = {VERDICT_SUFFICIENT, VERDICT_INSUFFICIENT, VERDICT_UNCERTAIN}
        test_cases = [
            [],
            [_doc("d1", 0.90)],
            [_doc("d1", 0.90), _doc("d2", 0.85), _doc("d3", 0.80)],
            [_doc("d1", 0.10)],
        ]
        for docs in test_cases:
            result = self.assessor.assess("q", docs)
            self.assertIn(result.verdict, valid, f"Unexpected verdict '{result.verdict}'")


if __name__ == "__main__":
    unittest.main()
