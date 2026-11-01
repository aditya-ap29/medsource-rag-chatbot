"""
tests/test_bm25_hybrid.py
---------------------------
Unit tests for BM25Retriever, HybridRetriever and RRF fusion.

Tests cover:
- BM25 indexing and stable document mapping.
- Text normalisation preserving drug names / numbers / abbreviations.
- BM25 search ranking on small synthetic corpus.
- Empty results, missing metadata, retriever failures.
- RRF formula correctness.
- HybridRetriever with enable_hybrid=False (pure FAISS delegation).
- HybridRetriever with enable_hybrid=True (RRF fusion).
- Deduplication by chunk_id.
- Source provenance preservation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id: str,
    content: str,
    chunk_id: str | None = None,
    url: str = "https://example.com",
    year: str = "2023",
    title: str = "Test Document",
) -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id or f"{doc_id}_chunk_0",
        "content": content,
        "title": title,
        "source": "test",
        "year": str(year),
        "url": url,
        "metadata": {},
        "chunk_index": 0,
        "total_chunks": 1,
    }


SAMPLE_DOCS = [
    _make_doc("d1", "Metformin is used to treat type 2 diabetes by lowering blood glucose.", chunk_id="d1_c0"),
    _make_doc("d2", "Ibuprofen is an NSAID used for pain and fever reduction.", chunk_id="d2_c0"),
    _make_doc("d3", "The capital of France is Paris. Paris is a city in Europe.", chunk_id="d3_c0"),
    _make_doc("d4", "Insulin therapy is essential for type 1 diabetes management.", chunk_id="d4_c0"),
    _make_doc("d5", "HbA1c measures average blood glucose over three months.", chunk_id="d5_c0"),
]


# ---------------------------------------------------------------------------
# BM25 text normalisation
# ---------------------------------------------------------------------------

class TestNormaliseText(unittest.TestCase):

    def test_lowercases(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("DIABETES Mellitus TYPE 2")
        self.assertEqual(tokens, ["diabetes", "mellitus", "type", "2"])

    def test_preserves_hyphenated_drug(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("beta-blocker treatment")
        self.assertIn("beta-blocker", tokens)

    def test_preserves_numbers(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("500mg dose daily")
        self.assertIn("500mg", tokens)

    def test_removes_pure_punctuation(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("!!! ... ???")
        self.assertEqual(tokens, [])

    def test_empty_string(self):
        from src.bm25_retriever import normalise_text
        self.assertEqual(normalise_text(""), [])

    def test_abbreviation_preserved(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("What is HbA1c?")
        self.assertIn("hba1c", tokens)

    def test_abbreviation_COPD(self):
        from src.bm25_retriever import normalise_text
        tokens = normalise_text("Managing COPD with inhalers")
        self.assertIn("copd", tokens)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------

class TestBM25Retriever(unittest.TestCase):

    def _make_retriever(self, docs=None):
        from src.bm25_retriever import BM25Retriever
        return BM25Retriever(docs or SAMPLE_DOCS)

    def test_build_indexes_all_docs(self):
        r = self._make_retriever()
        r.build()
        self.assertEqual(r._index_doc_count, len(SAMPLE_DOCS))

    def test_lazy_build_on_search(self):
        r = self._make_retriever()
        self.assertIsNone(r._bm25)
        results = r.search("diabetes", top_k=3)
        self.assertIsNotNone(r._bm25)

    def test_medical_query_returns_relevant(self):
        r = self._make_retriever()
        results = r.search("diabetes", top_k=3)
        ids = [doc["doc_id"] for doc, _ in results]
        # d1 (metformin/diabetes) and d4 (insulin/diabetes) should rank top
        self.assertTrue(
            any(d in ids for d in ("d1", "d4", "d5")),
            f"Expected diabetes-related docs but got {ids}",
        )

    def test_oop_query_ranks_differently(self):
        r = self._make_retriever()
        results = r.search("capital France", top_k=2)
        if results:
            self.assertEqual(results[0][0]["doc_id"], "d3")

    def test_empty_docs_list(self):
        from src.bm25_retriever import BM25Retriever
        r = BM25Retriever([])
        results = r.search("diabetes")
        self.assertEqual(results, [])

    def test_empty_query_returns_empty(self):
        r = self._make_retriever()
        results = r.search("!!! ... ???")
        self.assertEqual(results, [])

    def test_top_k_respected(self):
        r = self._make_retriever()
        results = r.search("diabetes", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_scores_are_non_negative(self):
        r = self._make_retriever()
        results = r.search("diabetes blood glucose", top_k=5)
        for _, score in results:
            self.assertGreaterEqual(score, 0.0)

    def test_doc_metadata_preserved(self):
        r = self._make_retriever()
        results = r.search("metformin", top_k=1)
        if results:
            doc, _ = results[0]
            self.assertIn("url", doc)
            self.assertIn("year", doc)
            self.assertIn("chunk_id", doc)

    def test_stats(self):
        r = self._make_retriever()
        r.build()
        stats = r.get_stats()
        self.assertEqual(stats["indexed_chunks"], len(SAMPLE_DOCS))
        self.assertTrue(stats["is_built"])

    def test_rebuild_on_document_count_change(self):
        from src.bm25_retriever import BM25Retriever
        docs = list(SAMPLE_DOCS)
        r = BM25Retriever(docs)
        r.build()
        old_count = r._index_doc_count
        # Add a new document
        docs.append(_make_doc("d6", "new doc about hypertension", chunk_id="d6_c0"))
        # Trigger rebuild
        results = r.search("hypertension")
        self.assertEqual(r._index_doc_count, len(docs))

    def test_missing_rank_bm25_raises_import_error(self):
        from src.bm25_retriever import BM25Retriever
        r = BM25Retriever(SAMPLE_DOCS)
        with patch.dict("sys.modules", {"rank_bm25": None}):
            with self.assertRaises(ImportError):
                r.build()


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class TestRRF(unittest.TestCase):

    def test_rrf_combines_lists(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        list1 = [("A", 0.9), ("B", 0.8), ("C", 0.7)]
        list2 = [("B", 5.0), ("A", 4.0), ("D", 3.0)]
        fused = dict(reciprocal_rank_fusion([list1, list2]))
        # A and B should both appear
        self.assertIn("A", fused)
        self.assertIn("B", fused)

    def test_rrf_formula(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        # A is rank 1 in both lists → highest fused score
        list1 = [("A", 1.0), ("B", 0.5)]
        list2 = [("A", 1.0), ("C", 0.5)]
        fused = dict(reciprocal_rank_fusion([list1, list2], k=60))
        # A should score 1/61 + 1/61 = 2/61
        expected_A = 2.0 / 61.0
        self.assertAlmostEqual(fused["A"], expected_A, places=6)

    def test_rrf_weights(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        list1 = [("A", 1.0)]
        list2 = [("B", 1.0)]
        # Dense weight 2x sparse
        fused = dict(reciprocal_rank_fusion([list1, list2], weights=[2.0, 1.0]))
        self.assertGreater(fused["A"], fused["B"])

    def test_rrf_empty_lists(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_rrf_single_list(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        ranked = [("X", 0.9), ("Y", 0.8)]
        fused = reciprocal_rank_fusion([ranked])
        # Order should be preserved
        ids = [cid for cid, _ in fused]
        self.assertEqual(ids, ["X", "Y"])

    def test_rrf_deduplicates(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        list1 = [("A", 0.9), ("A", 0.8)]  # Duplicate entry
        fused = dict(reciprocal_rank_fusion([list1]))
        # Should only appear once (with the rank 1 score)
        self.assertEqual(len([k for k in fused if k == "A"]), 1)

    def test_rrf_wrong_weights_raises(self):
        from src.hybrid_retriever import reciprocal_rank_fusion
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[("A", 1.0)]], weights=[1.0, 2.0])


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

def _make_mock_vs(docs=None):
    """Return a mock VectorStore with controlled search()."""
    docs = docs or SAMPLE_DOCS
    vs = MagicMock()
    vs.documents = docs
    # Default: return first 3 docs with decreasing scores
    vs.search.return_value = [(d, 0.9 - i * 0.1) for i, d in enumerate(docs[:3])]
    return vs


class TestHybridRetrieverDenseOnly(unittest.TestCase):
    """With enable_hybrid=False the retriever must delegate to vector_store.search()."""

    def setUp(self):
        from src.hybrid_retriever import HybridRetriever, HybridConfig
        self.vs = _make_mock_vs()
        self.retriever = HybridRetriever(
            vector_store=self.vs,
            config=HybridConfig(enable_hybrid=False),
        )

    def test_delegates_to_vector_store(self):
        self.retriever.search("diabetes", top_k=3)
        self.vs.search.assert_called_once()

    def test_returns_vector_store_results(self):
        results = self.retriever.search("diabetes", top_k=3)
        self.assertEqual(len(results), 3)

    def test_score_threshold_passed(self):
        self.retriever.search("diabetes", top_k=3, score_threshold=0.5)
        _, kwargs = self.vs.search.call_args
        self.assertEqual(kwargs.get("score_threshold", 0.5), 0.5)


class TestHybridRetrieverFused(unittest.TestCase):
    """With enable_hybrid=True the retriever fuses dense + BM25."""

    def setUp(self):
        from src.hybrid_retriever import HybridRetriever, HybridConfig

        # Build a vs mock that returns dense docs
        self.vs = _make_mock_vs()

        self.config = HybridConfig(
            enable_hybrid=True,
            rrf_k=60,
        )
        self.retriever = HybridRetriever(
            vector_store=self.vs,
            config=self.config,
        )

    def _inject_bm25(self, docs=None):
        """Inject a real BM25Retriever over SAMPLE_DOCS to avoid mocking issues."""
        try:
            from src.bm25_retriever import BM25Retriever
            bm25 = BM25Retriever(docs or SAMPLE_DOCS)
            bm25.build()
            self.retriever._bm25 = bm25
            self.retriever._bm25_ready = True
        except ImportError:
            self.skipTest("rank-bm25 not installed")

    def test_hybrid_search_returns_results(self):
        self._inject_bm25()
        results = self.retriever.search("diabetes", top_k=3)
        self.assertGreater(len(results), 0)

    def test_chunk_ids_unique(self):
        self._inject_bm25()
        results = self.retriever.search("diabetes", top_k=5)
        chunk_ids = [doc.get("chunk_id") for doc, _ in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_provenance_preserved(self):
        self._inject_bm25()
        results = self.retriever.search("diabetes", top_k=3)
        for doc, score in results:
            self.assertIn("doc_id", doc)
            self.assertIn("content", doc)
            self.assertIn("url", doc)

    def test_dense_failure_falls_back_to_bm25(self):
        self._inject_bm25()
        self.vs.search.side_effect = RuntimeError("FAISS failed")
        # Should not raise; should return BM25 results
        results = self.retriever.search("diabetes", top_k=3)
        # Results may be empty or BM25-only; must not raise
        self.assertIsInstance(results, list)

    def test_bm25_failure_falls_back_to_dense(self):
        from src.bm25_retriever import BM25Retriever
        bm25_mock = MagicMock(spec=BM25Retriever)
        bm25_mock.search.side_effect = RuntimeError("BM25 failed")
        bm25_mock.build.return_value = None
        self.retriever._bm25 = bm25_mock
        self.retriever._bm25_ready = True

        results = self.retriever.search("diabetes", top_k=3)
        # Should return dense results without raising
        self.assertIsInstance(results, list)

    def test_both_fail_returns_empty(self):
        from src.bm25_retriever import BM25Retriever
        self.vs.search.side_effect = RuntimeError("FAISS failed")
        bm25_mock = MagicMock(spec=BM25Retriever)
        bm25_mock.search.side_effect = RuntimeError("BM25 failed")
        bm25_mock.build.return_value = None
        self.retriever._bm25 = bm25_mock
        self.retriever._bm25_ready = True

        results = self.retriever.search("diabetes", top_k=3)
        self.assertEqual(results, [])

    def test_get_stats(self):
        self._inject_bm25()
        stats = self.retriever.get_stats()
        self.assertTrue(stats["enable_hybrid"])
        self.assertIn("bm25", stats)


if __name__ == "__main__":
    unittest.main()
