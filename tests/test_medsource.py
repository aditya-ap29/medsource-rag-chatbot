"""
tests/test_medsource.py — Focused unit and integration tests for MedSource.

Covers:
  - Generation error propagation (no longer leaks error string into answer)
  - Citation-to-source alignment (only cited DOC_X present in sources)
  - Duplicate ingestion guard (doc_id deduplication in VectorStore.add_documents)
  - Confidence score boundary (at-threshold doc passes gate)
  - Session-scoped document pipeline uses confidence_threshold > 0
  - Live PubMed retriever deduplication helpers
  - Source provenance field is set correctly

Run with:
    python -m pytest tests/ -v
    # or without pytest installed:
    python tests/test_medsource.py
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers for constructing lightweight fakes
# ---------------------------------------------------------------------------

def _make_doc(doc_id: str, content: str = "test content", title: str = "Test Doc") -> dict:
    """Return a minimal doc dict compatible with VectorStore.documents format."""
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_chunk_0",
        "content": content,
        "title": title,
        "source": "test",
        "year": "2024",
        "url": f"https://example.com/{doc_id}",
        "metadata": {"provenance": "indexed"},
        "chunk_index": 0,
        "total_chunks": 1,
    }


def _make_medical_doc(doc_id: str, content: str = "test content", title: str = "Test") -> "MedicalDocument":
    from src.data_ingestion import MedicalDocument
    doc = MedicalDocument(content=content, title=title, source="test", doc_id=doc_id)
    return doc


# ===========================================================================
# 1. Generation error propagation
# ===========================================================================

class TestGenerationErrorPropagation(unittest.TestCase):
    """_generate() should raise RuntimeError; query() should not return an error string as the answer."""

    def _build_pipeline_with_mock_vs(self):
        """Build a RAGPipeline with a minimal mock VectorStore."""
        from src.rag_pipeline import RAGPipeline

        vs = MagicMock()
        vs.documents = [_make_doc("DOC_1")]
        vs.search.return_value = [(_make_doc("DOC_1"), 0.75)]

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=True,
        )
        return pipeline

    def test_generate_raises_runtime_error_on_api_failure(self):
        """_generate() must raise RuntimeError with a safe user-facing message.

        Since _generate() now translates SDK error text (e.g. '429 quota exceeded')
        into a user-friendly string, the raw SDK text must NOT appear in the message,
        but a human-readable description must.
        """
        pipeline = self._build_pipeline_with_mock_vs()
        # Zero the retry delay so the test runs fast.
        pipeline._RETRY_BASE_S = 0.0

        with patch.object(pipeline._client.models, "generate_content", side_effect=Exception("quota exceeded")):
            with self.assertRaises(RuntimeError) as ctx:
                pipeline._generate("test prompt")
            msg = str(ctx.exception).lower()
            # Must raise RuntimeError with a user-friendly message.
            self.assertIn("llm generation failed", msg)
            # The message must describe the problem in plain English.
            self.assertTrue(
                any(word in msg for word in ("rate-limited", "unavailable", "error", "time")),
                f"Expected user-friendly error message, got: {msg}",
            )
            # The raw SDK detail must not be exposed verbatim.
            self.assertNotIn("quota exceeded", msg)

    def test_query_raises_on_generation_failure(self):
        """query() should propagate the RuntimeError from _generate(), not swallow it."""
        pipeline = self._build_pipeline_with_mock_vs()

        with patch.object(pipeline, "_generate", side_effect=RuntimeError("LLM generation failed: quota exceeded")):
            with self.assertRaises(RuntimeError):
                pipeline.query("What are the symptoms of anemia?")

    def test_query_answer_is_not_error_string(self):
        """query() with a working _generate() should return a real answer, not an error string."""
        pipeline = self._build_pipeline_with_mock_vs()

        with patch.object(pipeline, "_generate", return_value="Anemia causes fatigue [DOC_1]."):
            result = pipeline.query("What are the symptoms of anemia?")
        self.assertNotIn("Error generating", result.answer)
        self.assertIn("Anemia", result.answer)


# ===========================================================================
# 2. Citation-to-source alignment
# ===========================================================================

class TestCitationAlignment(unittest.TestCase):
    """Sources list must only include docs whose [DOC_X] index appears in the answer."""

    def _pipeline_with_n_docs(self, n: int):
        from src.rag_pipeline import RAGPipeline

        docs = [_make_doc(f"PX_{i}", content=f"content {i}", title=f"Doc {i}") for i in range(1, n + 1)]
        vs = MagicMock()
        vs.documents = docs
        vs.search.return_value = [(d, 0.75) for d in docs]

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
            enable_general_fallback=False,
        )
        return pipeline

    def test_only_cited_sources_returned(self):
        """When answer cites only [DOC_1] and [DOC_3], sources list has exactly 2 entries."""
        pipeline = self._pipeline_with_n_docs(3)
        answer_text = "Fatigue is common [DOC_1]. Pallor may occur [DOC_3]."

        with patch.object(pipeline, "_generate", return_value=answer_text):
            result = pipeline.query("What are symptoms of anemia?")

        cited_ids = {s.doc_id for s in result.sources}
        self.assertEqual(cited_ids, {"DOC_1", "DOC_3"},
                         f"Expected {{DOC_1, DOC_3}}, got {cited_ids}")
        self.assertEqual(len(result.sources), 2)

    def test_no_citation_returns_all_sources(self):
        """When the model cites nothing, all retrieved docs are included (safe fallback)."""
        pipeline = self._pipeline_with_n_docs(3)
        answer_text = "Anemia causes fatigue."  # no [DOC_X] citations

        with patch.object(pipeline, "_generate", return_value=answer_text):
            result = pipeline.query("What are symptoms of anemia?")

        self.assertEqual(len(result.sources), 3,
                         "All 3 docs should be included when model cites none")

    def test_out_of_range_citation_ignored(self):
        """[DOC_6] when only 3 docs were retrieved is silently ignored."""
        pipeline = self._pipeline_with_n_docs(3)
        answer_text = "Anemia [DOC_1] is described [DOC_6]."

        with patch.object(pipeline, "_generate", return_value=answer_text):
            result = pipeline.query("What are symptoms of anemia?")

        cited_ids = {s.doc_id for s in result.sources}
        self.assertIn("DOC_1", cited_ids)
        self.assertNotIn("DOC_6", cited_ids)


# ===========================================================================
# 3. Duplicate ingestion guard
# ===========================================================================

class TestDuplicateIngestionGuard(unittest.TestCase):
    """add_documents() must skip doc_ids already present in the store."""

    def _empty_vs(self):
        """Return a VectorStore with a real FAISS index but no model (uses mock encode)."""
        import numpy as np

        from src.vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        # Minimal attribute setup without downloading the model
        vs.embedding_model_name = "BAAI/bge-base-en-v1.5"
        vs.use_bge_prefix = True
        vs.chunk_size = 256
        vs.chunk_overlap = 40
        vs.embedding_batch_size = 32
        vs.dimension = 4  # tiny for speed

        import faiss
        vs.index = faiss.IndexFlatIP(vs.dimension)
        vs.documents = []

        # Patch text splitter to return content as a single chunk
        splitter_mock = MagicMock()
        splitter_mock.split_text = lambda text: [text]
        vs.text_splitter = splitter_mock

        # Patch _encode_passages to return zero vectors
        vs._encode_passages = lambda texts: np.zeros((len(texts), vs.dimension), dtype="float32")

        return vs

    def test_first_ingest_adds_chunks(self):
        vs = self._empty_vs()
        doc = _make_medical_doc("DOC_FIRST", content="First document content.")
        added = vs.add_documents([doc])
        self.assertGreater(added, 0)
        self.assertEqual(len(vs.documents), added)

    def test_duplicate_ingest_skipped(self):
        vs = self._empty_vs()
        doc = _make_medical_doc("DOC_DUP", content="Duplicate document.")
        vs.add_documents([doc])
        initial_count = len(vs.documents)

        # Ingest the same document again — should be a no-op
        added = vs.add_documents([doc])
        self.assertEqual(added, 0, "Re-ingesting the same doc_id should add 0 chunks")
        self.assertEqual(len(vs.documents), initial_count,
                         "Document list should not grow on duplicate ingest")

    def test_partial_dedup(self):
        """Mix of new and duplicate docs: only new ones are added."""
        vs = self._empty_vs()
        doc_a = _make_medical_doc("DOC_A", content="Content A.")
        doc_b = _make_medical_doc("DOC_B", content="Content B.")
        vs.add_documents([doc_a])
        initial_count = len(vs.documents)

        added = vs.add_documents([doc_a, doc_b])  # DOC_A is a duplicate
        self.assertGreater(added, 0)
        new_doc_ids = {d["doc_id"] for d in vs.documents}
        self.assertIn("DOC_A", new_doc_ids)
        self.assertIn("DOC_B", new_doc_ids)
        # Total count = initial (DOC_A chunks) + added (DOC_B chunks)
        self.assertEqual(len(vs.documents), initial_count + added)


# ===========================================================================
# 4. Confidence score boundary
# ===========================================================================

class TestConfidenceBoundary(unittest.TestCase):
    """A doc with score exactly at threshold should pass the confidence gate."""

    def test_at_threshold_is_grounded(self):
        from src.rag_pipeline import RAGPipeline

        threshold = 0.30
        doc = _make_doc("DOC_THR")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, threshold)]  # score == threshold exactly

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=threshold,
            enable_general_fallback=True,
        )

        with patch.object(pipeline, "_generate", return_value="Answer [DOC_1]."):
            result = pipeline.query("test question")

        # Should be grounded (sources not empty) since score == threshold passes
        self.assertTrue(len(result.sources) > 0,
                        "Score at threshold should produce a grounded (sourced) response")

    def test_below_threshold_triggers_fallback(self):
        from src.rag_pipeline import RAGPipeline

        threshold = 0.30
        doc = _make_doc("DOC_LOW")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = []  # nothing above threshold

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=threshold,
            enable_general_fallback=True,
        )

        with patch.object(pipeline, "_generate", return_value="Note: This answer is based on general medical knowledge."):
            result = pipeline.query("obscure query with no matches")

        self.assertEqual(len(result.sources or []), 0,
                         "Below-threshold query with fallback should return no sources")
        self.assertEqual(result.confidence, 0.0)


# ===========================================================================
# 5. Session-scoped doc pipeline threshold
# ===========================================================================

class TestDocPipelineThreshold(unittest.TestCase):
    """_build_doc_pipeline must use confidence_threshold > 0 (was 0.0 — bug fixed)."""

    def test_doc_pipeline_threshold_not_zero(self):
        """Verify _build_doc_pipeline confidence threshold is > 0."""
        import ast

        src = Path("streamlit_app.py").read_text(encoding="utf-8")
        tree = ast.parse(src)

        # Walk AST looking for _build_doc_pipeline's return statement
        threshold_value = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_doc_pipeline":
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.keyword) and subnode.arg == "confidence_threshold":
                        if isinstance(subnode.value, ast.Constant):
                            threshold_value = subnode.value.value
                        break

        self.assertIsNotNone(threshold_value,
                             "confidence_threshold kwarg not found in _build_doc_pipeline")
        self.assertGreater(threshold_value, 0.0,
                           f"_build_doc_pipeline confidence_threshold should be > 0, got {threshold_value}")


# ===========================================================================
# 6. Live PubMed retriever deduplication helpers
# ===========================================================================

class TestLivePubMedDeduplication(unittest.TestCase):
    """PubMedLiveRetriever.get_indexed_pmids extracts PMIDs from doc lists."""

    def test_pmid_extraction_from_doc_id(self):
        from src.live_retrieval import PubMedLiveRetriever

        retriever = PubMedLiveRetriever()
        docs = [
            {"doc_id": "PMID_12345678", "metadata": {}},
            {"doc_id": "PMID_99999999", "metadata": {}},
            {"doc_id": "LOCAL_ABC", "metadata": {}},
        ]
        pmids = retriever.get_indexed_pmids(docs)
        self.assertIn("12345678", pmids)
        self.assertIn("99999999", pmids)
        self.assertNotIn("LOCAL_ABC", pmids)

    def test_pmid_extraction_from_metadata(self):
        from src.live_retrieval import PubMedLiveRetriever

        retriever = PubMedLiveRetriever()
        docs = [
            {"doc_id": "custom_id", "metadata": {"pmid": "55555555"}},
        ]
        pmids = retriever.get_indexed_pmids(docs)
        self.assertIn("55555555", pmids)

    def test_empty_docs(self):
        from src.live_retrieval import PubMedLiveRetriever

        retriever = PubMedLiveRetriever()
        self.assertEqual(retriever.get_indexed_pmids([]), set())


# ===========================================================================
# 7. Source provenance field
# ===========================================================================

class TestSourceProvenance(unittest.TestCase):
    """Source objects should include the provenance field."""

    def test_indexed_source_has_provenance(self):
        from src.rag_pipeline import RAGPipeline

        doc = _make_doc("DOC_PROV", content="Indexed content.")
        vs = MagicMock()
        vs.documents = [doc]
        vs.search.return_value = [(doc, 0.75)]

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
        )

        with patch.object(pipeline, "_generate", return_value="Content [DOC_1]."):
            result = pipeline.query("test")

        self.assertTrue(len(result.sources) > 0)
        src = result.sources[0]
        self.assertIsNotNone(src.provenance)
        self.assertEqual(src.provenance, "indexed")

    def test_live_pubmed_source_provenance(self):
        """A doc with provenance='live_pubmed' in metadata gets that provenance on the Source."""
        from src.rag_pipeline import RAGPipeline

        live_doc = _make_doc("PMID_99999", content="Live abstract content.")
        live_doc["metadata"] = {"provenance": "live_pubmed"}
        live_doc["source"] = "PubMed (live)"

        vs = MagicMock()
        vs.documents = [live_doc]
        vs.search.return_value = [(live_doc, 0.75)]

        pipeline = RAGPipeline(
            vector_store=vs,
            model_name="gemini-2.0-flash",
            api_key="fake-key",
            confidence_threshold=0.30,
        )

        with patch.object(pipeline, "_generate", return_value="Live result [DOC_1]."):
            result = pipeline.query("test")

        self.assertTrue(len(result.sources) > 0)
        src = result.sources[0]
        self.assertEqual(src.provenance, "live_pubmed",
                         f"Expected 'live_pubmed', got '{src.provenance}'")


# ===========================================================================
# 8. Normalise score correctness
# ===========================================================================

class TestNormaliseScore(unittest.TestCase):
    """_normalise_score maps BGE scores to [0,1] correctly."""

    def setUp(self):
        from src.rag_pipeline import RAGPipeline
        self.norm = RAGPipeline._normalise_score

    def test_at_threshold_maps_to_zero(self):
        score = self.norm(0.30, is_ce_domain=False, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 0.0, places=3)

    def test_at_ceiling_maps_to_one(self):
        score = self.norm(0.85, is_ce_domain=False, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 1.0, places=3)

    def test_above_ceiling_capped_at_one(self):
        score = self.norm(1.0, is_ce_domain=False, confidence_threshold=0.30)
        self.assertLessEqual(score, 1.0)

    def test_ce_zero_maps_to_half(self):
        score = self.norm(0.0, is_ce_domain=True, confidence_threshold=0.30)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_ce_positive_above_half(self):
        score = self.norm(3.0, is_ce_domain=True, confidence_threshold=0.30)
        self.assertGreater(score, 0.5)


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    # Coloured summary when run directly
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
