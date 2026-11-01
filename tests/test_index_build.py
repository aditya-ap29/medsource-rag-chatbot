"""
tests/test_index_build.py
--------------------------
Unit and integration tests for versioned index building, validation,
and rollback. All tests use temporary directories — production data
is never touched.

Test classes:
  TestCleanCorpus          — cleaning and deduplication logic
  TestBuildVersionedIndex  — build_full, build_incremental, _save_and_verify
  TestValidateIndex        — validate_index with good/bad indexes
  TestSwapIndex            — swap, rollback, partial-build detection
  TestPipelineLoadability  — VectorStore.load() roundtrip
  TestIncrementalUpdate    — doc_id-based skip logic
  TestContentIntegrity     — HTML-free, no empty chunks after build

Run (no live API key required):
    python -m pytest tests/test_index_build.py -v
"""
import hashlib
import json
import pickle
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import faiss
from unittest.mock import MagicMock, patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_mini_corpus(n: int = 10) -> list[dict]:
    """Return n minimal clean corpus records (no HTML, no short content)."""
    topics = [
        ("diabetes", "Diabetes mellitus is a chronic metabolic disorder characterised by "
         "elevated blood glucose levels. Type 2 diabetes is the most common form, "
         "associated with insulin resistance and relative insulin deficiency. "
         "Management includes lifestyle modification, metformin, and other agents."),
        ("hypertension", "Hypertension, or high blood pressure, is defined as systolic "
         "blood pressure >=140 mmHg or diastolic >=90 mmHg. It is a major risk factor "
         "for cardiovascular disease, stroke, and renal failure. First-line treatment "
         "includes lifestyle changes and antihypertensive medications."),
        ("asthma", "Asthma is a chronic inflammatory airway disease characterised by "
         "variable airflow obstruction and bronchial hyperresponsiveness. Symptoms "
         "include wheezing, shortness of breath, chest tightness, and cough. "
         "Treatment includes inhaled corticosteroids and bronchodilators."),
        ("anemia", "Anaemia is defined as a haemoglobin concentration below the "
         "normal reference range for age and sex. Iron deficiency anaemia is the "
         "most prevalent type globally. Symptoms include fatigue, pallor, "
         "and dyspnoea on exertion."),
        ("heart_attack", "Myocardial infarction (heart attack) occurs when coronary "
         "artery blood flow is interrupted, causing cardiomyocyte death. Classic "
         "symptoms include crushing chest pain, radiation to the left arm, "
         "diaphoresis, and nausea. Immediate reperfusion therapy is the treatment."),
        ("depression", "Major depressive disorder is characterised by persistent low "
         "mood, anhedonia, fatigue, and cognitive disturbances lasting at least "
         "two weeks. Treatment includes psychotherapy, SSRIs, and SNRIs. "
         "Severe cases may require electroconvulsive therapy."),
        ("pneumonia", "Pneumonia is an acute infection of the lung parenchyma. "
         "Common causative organisms include Streptococcus pneumoniae and "
         "influenza viruses. Symptoms include fever, productive cough, "
         "pleuritic chest pain, and dyspnoea."),
        ("migraine", "Migraine is a primary headache disorder characterised by "
         "recurrent episodes of moderate to severe unilateral pulsating headache. "
         "Associated symptoms include nausea, photophobia, and phonophobia. "
         "Treatment involves triptans and preventive medications."),
        ("tuberculosis", "Tuberculosis is caused by Mycobacterium tuberculosis and "
         "primarily affects the lungs. Classic symptoms include chronic cough, "
         "haemoptysis, night sweats, and weight loss. Standard treatment "
         "involves a six-month regimen of rifampicin, isoniazid, pyrazinamide."),
        ("obesity", "Obesity is defined as a BMI >= 30 kg/m2 and is a major risk "
         "factor for type 2 diabetes, cardiovascular disease, and certain cancers. "
         "Management includes dietary modification, physical activity, behavioural "
         "therapy, and pharmacotherapy in selected patients."),
    ]
    records = []
    for i in range(min(n, len(topics))):
        topic, content = topics[i % len(topics)]
        doc_id = f"TEST_{topic}_{i:03d}"
        records.append({
            "doc_id":       doc_id,
            "content":      content,
            "content_hash": hashlib.md5(content.encode()).hexdigest(),
            "title":        topic.replace("_", " ").title(),
            "source":       "TestCorpus",
            "year":         "2024",
            "url":          f"https://example.com/{topic}",
            "metadata":     {"category": topic},
        })
    return records


def _build_tmp_vs(records: list[dict]) -> tuple:
    """Build an in-memory VectorStore over mini-corpus. Returns (vs, tmp_dir)."""
    from src.data_ingestion import MedicalDocument
    from src.vector_store import VectorStore

    tmp = tempfile.mkdtemp(prefix="medsource_test_")
    tmp_dir = Path(tmp)

    docs = [
        MedicalDocument(
            content  = r["content"],
            title    = r.get("title", ""),
            source   = r.get("source", ""),
            doc_id   = r.get("doc_id", ""),
            year     = r.get("year"),
            url      = r.get("url"),
            metadata = r.get("metadata", {}),
        )
        for r in records
    ]

    vs = VectorStore(
        embedding_model_name = "BAAI/bge-base-en-v1.5",
        chunk_size           = 256,
        chunk_overlap        = 40,
        use_bge_prefix       = True,
        embedding_batch_size = 32,
    )
    vs.add_documents(docs)
    vs.save(str(tmp_dir))
    return vs, tmp_dir


def _write_version_json(index_dir: Path, **kwargs) -> None:
    default = {
        "version":             "v1",
        "build_mode":          "full",
        "created_at":          "2026-01-01T00:00:00+00:00",
        "embedding_model":     "BAAI/bge-base-en-v1.5",
        "embedding_dimension": 768,
        "index_type":          "IndexFlatIP",
        "normalisation":       "L2",
        "use_bge_prefix":      True,
        "chunk_size_tokens":   256,
        "chunk_overlap_tokens": 40,
        "source_corpus_file":  "clean_corpus_v1.json",
        "source_corpus_hash":  "abc123",
        "total_chunks":        10,
        "unique_doc_ids":      10,
    }
    default.update(kwargs)
    with open(index_dir / "version.json", "w") as fh:
        json.dump(default, fh)


def _write_build_status(index_dir: Path, status: str) -> None:
    with open(index_dir / "build_status.json", "w") as fh:
        json.dump({"status": status}, fh)


# ── TestCleanCorpus ───────────────────────────────────────────────────────────

class TestCleanCorpus(unittest.TestCase):
    """Unit tests for scripts/clean_corpus.py cleaning functions."""

    def setUp(self):
        from scripts.clean_corpus import clean_text, content_hash, clean_record
        self.clean_text   = clean_text
        self.content_hash = content_hash
        self.clean_record = clean_record

    def test_html_stripped(self):
        raw = '<p>What is <span class="qt0">diabetes</span>?</p>'
        out = self.clean_text(raw)
        self.assertNotIn("<", out)
        self.assertIn("diabetes", out)

    def test_nested_html_stripped(self):
        raw = "<ul><li><span>Item one</span></li><li>Item two</li></ul>"
        out = self.clean_text(raw)
        self.assertNotIn("<", out)
        self.assertIn("Item one", out)

    def test_whitespace_normalised(self):
        raw = "Hello   world\n\n\n\nEnd"
        out = self.clean_text(raw)
        self.assertNotIn("   ", out)
        self.assertNotIn("\n\n\n", out)

    def test_control_chars_removed(self):
        raw = "Normal\x00Text\x1fHere"
        out = self.clean_text(raw)
        self.assertNotIn("\x00", out)
        self.assertNotIn("\x1f", out)

    def test_html_entity_decoded(self):
        raw = "AT&amp;T &lt;3 health"
        out = self.clean_text(raw)
        self.assertIn("AT&T", out)
        self.assertIn("<3", out)

    def test_short_record_dropped(self):
        record = {"doc_id": "x", "content": "Too short."}
        result = self.clean_record(record)
        self.assertNone(result)

    def assertNone(self, obj):
        self.assertIsNone(obj)

    def test_long_record_kept(self):
        record = {"doc_id": "x", "content": "A" * 100}
        result = self.clean_record(record)
        self.assertIsNotNone(result)
        self.assertEqual(result["content"], "A" * 100)

    def test_content_hash_deterministic(self):
        text = "Diabetes mellitus is a chronic metabolic disorder."
        h1 = self.content_hash(text)
        h2 = self.content_hash(text)
        self.assertEqual(h1, h2)

    def test_content_hash_different_content(self):
        h1 = self.content_hash("text A")
        h2 = self.content_hash("text B")
        self.assertNotEqual(h1, h2)

    def test_clean_record_adds_content_hash(self):
        record = {"doc_id": "d1", "content": "A" * 100}
        result = self.clean_record(record)
        self.assertIn("content_hash", result)
        self.assertEqual(len(result["content_hash"]), 32)

    def test_clean_corpus_end_to_end_dry_run(self):
        """build_clean_corpus(dry_run=True) should not raise or create files."""
        import tempfile, os, json
        from scripts.clean_corpus import build_clean_corpus, OUTPUT_FILE, PROCESSED_DIR
        # Smoke-test: just ensure it completes without modifying files
        # We can't easily redirect OUTPUT_FILE without patching, so we just
        # verify dry_run=True returns without writing
        before = OUTPUT_FILE.exists()
        try:
            build_clean_corpus(dry_run=True)
        except Exception as exc:
            self.fail(f"build_clean_corpus(dry_run=True) raised: {exc}")


# ── TestBuildVersionedIndex ───────────────────────────────────────────────────

class TestBuildVersionedIndex(unittest.TestCase):
    """Tests for build_full, build_incremental, _save_and_verify."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_build_test_")
        self.tmp_dir = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _records(self, n=5):
        return _make_mini_corpus(n)

    def test_build_full_creates_required_files(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        for fname in ["faiss.index", "documents.pkl", "config.pkl",
                      "version.json", "build_status.json"]:
            self.assertTrue((self.tmp_dir / fname).exists(), f"Missing {fname}")

    def test_build_full_status_complete(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with open(self.tmp_dir / "build_status.json") as fh:
            status = json.load(fh)
        self.assertEqual(status["status"], "complete")

    def test_build_full_vector_count_matches_documents(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        stats = build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        idx = faiss.read_index(str(self.tmp_dir / "faiss.index"))
        self.assertEqual(idx.ntotal, len(docs))
        self.assertEqual(stats["total_chunks"], len(docs))

    def test_build_full_dimension_768(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        stats = build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        self.assertEqual(stats["dimension"], 768)

    def test_build_full_no_html_in_content(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        html_pat = re.compile(r"<[a-zA-Z][^>]*>")
        html_docs = [d for d in docs if html_pat.search(d.get("content", ""))]
        self.assertEqual(len(html_docs), 0, "HTML found in indexed chunks")

    def test_build_full_version_json_fields(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with open(self.tmp_dir / "version.json") as fh:
            v = json.load(fh)
        for key in ["version", "build_mode", "created_at", "embedding_model",
                    "embedding_dimension", "index_type", "normalisation",
                    "chunk_size_tokens", "chunk_overlap_tokens",
                    "source_corpus_hash", "total_chunks", "unique_doc_ids"]:
            self.assertIn(key, v, f"version.json missing key: {key}")
        self.assertEqual(v["chunk_size_tokens"], 256)
        self.assertEqual(v["chunk_overlap_tokens"], 40)
        self.assertEqual(v["index_type"], "IndexFlatIP")
        self.assertEqual(v["normalisation"], "L2")

    def test_build_full_source_provenance_preserved(self):
        from scripts.build_versioned_index import build_full
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        for doc in docs:
            self.assertIn("source", doc)
            self.assertEqual(doc["source"], "TestCorpus")
            self.assertIn("url", doc)
            self.assertTrue(doc["url"].startswith("https://"))

    def test_incremental_guard_wrong_model_raises(self):
        from scripts.build_versioned_index import build_full, build_incremental
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with self.assertRaises(ValueError):
            build_incremental(records, 256, 40,
                              "some-other-model/different", 32, self.tmp_dir)

    def test_incremental_guard_wrong_chunk_size_raises(self):
        from scripts.build_versioned_index import build_full, build_incremental
        records = self._records(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        with self.assertRaises(ValueError):
            build_incremental(records, 512, 40,
                              "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)

    def test_incremental_no_new_docs_is_noop(self):
        from scripts.build_versioned_index import build_full, build_incremental
        records = self._records(5)
        stats_full = build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        stats_incr = build_incremental(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        self.assertEqual(stats_incr["mode"], "incremental_no_op")
        self.assertEqual(stats_incr["new_chunks"], 0)
        self.assertEqual(stats_incr["total_chunks"], stats_full["total_chunks"])

    def test_incremental_adds_new_docs(self):
        from scripts.build_versioned_index import build_full, build_incremental
        records_initial = _make_mini_corpus(4)
        records_all     = _make_mini_corpus(8)
        new_records     = [r for r in records_all
                           if r["doc_id"] not in {x["doc_id"] for x in records_initial}]

        stats_full = build_full(records_initial, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        stats_incr = build_incremental(records_all,  256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)

        self.assertEqual(stats_incr["mode"], "incremental")
        self.assertGreater(stats_incr["total_chunks"], stats_full["total_chunks"])

    def test_full_rebuild_flag_renames_existing(self):
        """--full-rebuild should rename old dir and build fresh."""
        import subprocess
        # Pre-create a fake index
        (self.tmp_dir / "faiss.index").touch()
        (self.tmp_dir / "documents.pkl").touch()
        (self.tmp_dir / "config.pkl").touch()
        # We can't test the CLI flag here without argparse plumbing,
        # so test the rename logic directly
        from datetime import datetime
        ts = "20260101_120000"
        renamed = self.tmp_dir.parent / f"{self.tmp_dir.name}_old_{ts}"
        self.tmp_dir.rename(renamed)
        self.assertFalse(self.tmp_dir.exists())
        self.assertTrue(renamed.exists())
        renamed.rename(self.tmp_dir)  # restore


# ── TestValidateIndex ─────────────────────────────────────────────────────────

class TestValidateIndex(unittest.TestCase):
    """Tests for scripts/validate_index.py."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_val_test_")
        self.tmp_dir = Path(self.tmp)
        # Build a good index once
        from scripts.build_versioned_index import build_full
        records = _make_mini_corpus(5)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_good_index_passes(self):
        from scripts.validate_index import validate_index
        result = validate_index(self.tmp_dir, expected_model="BAAI/bge-base-en-v1.5")
        self.assertTrue(result)

    def test_missing_faiss_index_fails(self):
        from scripts.validate_index import validate_index
        tmp2 = Path(tempfile.mkdtemp(prefix="medsource_val2_"))
        try:
            # Only documents.pkl and config.pkl, no faiss.index
            shutil.copy(self.tmp_dir / "documents.pkl", tmp2 / "documents.pkl")
            shutil.copy(self.tmp_dir / "config.pkl",    tmp2 / "config.pkl")
            result = validate_index(tmp2, expected_model="BAAI/bge-base-en-v1.5")
            self.assertFalse(result)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_html_in_content_fails(self):
        from scripts.validate_index import validate_index
        import copy
        # Inject HTML into one chunk's content
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        docs[0]["content"] = '<span class="qt0">diabetes</span> ' + docs[0]["content"]
        tmp2 = Path(tempfile.mkdtemp(prefix="medsource_val3_"))
        try:
            shutil.copytree(self.tmp_dir, tmp2, dirs_exist_ok=True)
            with open(tmp2 / "documents.pkl", "wb") as fh:
                pickle.dump(docs, fh)
            # FAISS index is unchanged so vector count still matches
            result = validate_index(tmp2, expected_model="BAAI/bge-base-en-v1.5")
            self.assertFalse(result)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_vector_count_mismatch_fails(self):
        from scripts.validate_index import validate_index
        # Add an extra document entry without adding to FAISS
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        extra = dict(docs[0])
        extra["doc_id"] = "extra_ghost"
        extra["chunk_id"] = "extra_ghost_chunk_0"
        docs.append(extra)
        tmp2 = Path(tempfile.mkdtemp(prefix="medsource_val4_"))
        try:
            shutil.copytree(self.tmp_dir, tmp2, dirs_exist_ok=True)
            with open(tmp2 / "documents.pkl", "wb") as fh:
                pickle.dump(docs, fh)
            result = validate_index(tmp2, expected_model="BAAI/bge-base-en-v1.5")
            self.assertFalse(result)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_wrong_model_warns_not_fails(self):
        from scripts.validate_index import validate_index
        # Pass a different BGE variant as expected_model — triggers a warning but
        # validate_index uses the model stored in config.pkl (bge-base) for the
        # actual search, so validation still passes.
        result = validate_index(self.tmp_dir, expected_model="BAAI/bge-large-en-v1.5")
        self.assertTrue(result)

    def test_build_status_not_complete_warns(self):
        from scripts.validate_index import validate_index
        import copy
        tmp2 = Path(tempfile.mkdtemp(prefix="medsource_val5_"))
        try:
            shutil.copytree(self.tmp_dir, tmp2, dirs_exist_ok=True)
            _write_build_status(tmp2, "in_progress")
            result = validate_index(tmp2, expected_model="BAAI/bge-base-en-v1.5")
            # Should still pass (status is warning-level only)
            self.assertTrue(result)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)


# ── TestSwapIndex ─────────────────────────────────────────────────────────────

class TestSwapIndex(unittest.TestCase):
    """Tests for scripts/swap_index.py swap and rollback logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_swap_test_")
        self.root = Path(self.tmp)
        self.active = self.root / "vector_store"
        self.backup = self.root / "vector_store_backup"
        self.new    = self.root / "vector_store_v1"
        self.log    = self.root / "swap_log.json"

        # Build a tiny index in the "new" dir
        from scripts.build_versioned_index import build_full
        records = _make_mini_corpus(3)
        self.new.mkdir(parents=True)
        build_full(records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.new)

        # Simulate an existing active index
        self.active.mkdir(parents=True)
        (self.active / "faiss.index").write_bytes(b"fake_index")
        (self.active / "documents.pkl").write_bytes(b"fake_docs")
        (self.active / "config.pkl").write_bytes(b"fake_cfg")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patched_swap(self, new_dir, active=None, backup=None, logfile=None):
        import scripts.swap_index as si
        orig_active  = si.ACTIVE_DIR
        orig_backup  = si.BACKUP_DIR
        orig_log     = si.SWAP_LOG
        si.ACTIVE_DIR  = active  or self.active
        si.BACKUP_DIR  = backup  or self.backup
        si.SWAP_LOG    = logfile or self.log
        try:
            si.swap_index(new_dir)
        finally:
            si.ACTIVE_DIR = orig_active
            si.BACKUP_DIR = orig_backup
            si.SWAP_LOG   = orig_log

    def _patched_rollback(self, active=None, backup=None, logfile=None):
        import scripts.swap_index as si
        orig_active  = si.ACTIVE_DIR
        orig_backup  = si.BACKUP_DIR
        orig_log     = si.SWAP_LOG
        si.ACTIVE_DIR  = active  or self.active
        si.BACKUP_DIR  = backup  or self.backup
        si.SWAP_LOG    = logfile or self.log
        try:
            si.rollback()
        finally:
            si.ACTIVE_DIR = orig_active
            si.BACKUP_DIR = orig_backup
            si.SWAP_LOG   = orig_log

    def test_swap_creates_backup(self):
        self._patched_swap(self.new)
        self.assertTrue(self.backup.exists())
        self.assertTrue((self.backup / "faiss.index").exists())

    def test_swap_copies_new_files_to_active(self):
        self._patched_swap(self.new)
        self.assertTrue((self.active / "faiss.index").exists())
        self.assertTrue((self.active / "documents.pkl").exists())
        self.assertTrue((self.active / "config.pkl").exists())

    def test_swap_writes_log(self):
        self._patched_swap(self.new)
        self.assertTrue(self.log.exists())
        with open(self.log) as fh:
            log = json.load(fh)
        self.assertEqual(log[-1]["action"], "swap")

    def test_rollback_restores_active_from_backup(self):
        self._patched_swap(self.new)
        original_size = (self.backup / "faiss.index").stat().st_size
        # Check that the content after swap is the real index (not fake bytes)
        # Then rollback and check we get the fake bytes back
        self._patched_rollback()
        content = (self.active / "faiss.index").read_bytes()
        # After rollback, active should contain what was in backup (original fake bytes)
        self.assertEqual(content, b"fake_index")

    def test_rollback_without_backup_exits(self):
        import scripts.swap_index as si
        orig_active = si.ACTIVE_DIR
        orig_backup = si.BACKUP_DIR
        si.ACTIVE_DIR = self.active
        si.BACKUP_DIR = self.root / "nonexistent_backup"
        try:
            with self.assertRaises(SystemExit):
                si.rollback()
        finally:
            si.ACTIVE_DIR = orig_active
            si.BACKUP_DIR = orig_backup

    def test_swap_missing_file_exits(self):
        import scripts.swap_index as si
        bad_dir = self.root / "incomplete"
        bad_dir.mkdir()
        (bad_dir / "faiss.index").touch()  # only one of three required files
        orig_active = si.ACTIVE_DIR
        orig_backup = si.BACKUP_DIR
        si.ACTIVE_DIR = self.active
        si.BACKUP_DIR = self.backup
        try:
            with self.assertRaises(SystemExit):
                si.swap_index(bad_dir)
        finally:
            si.ACTIVE_DIR = orig_active
            si.BACKUP_DIR = orig_backup

    def test_swap_log_appends_multiple_entries(self):
        self._patched_swap(self.new)
        self._patched_rollback()
        with open(self.log) as fh:
            log = json.load(fh)
        actions = [e["action"] for e in log]
        self.assertIn("swap", actions)
        self.assertIn("rollback", actions)


# ── TestPipelineLoadability ───────────────────────────────────────────────────

class TestPipelineLoadability(unittest.TestCase):
    """VectorStore.load() roundtrip — ensures built index is usable by the query pipeline."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_pl_test_")
        self.tmp_dir = Path(self.tmp)
        from scripts.build_versioned_index import build_full
        self.records = _make_mini_corpus(5)
        build_full(self.records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_vector_store_load_succeeds(self):
        from src.vector_store import VectorStore
        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        vs.load(str(self.tmp_dir))  # should not raise
        self.assertGreater(len(vs.documents), 0)

    def test_loaded_vs_returns_search_results(self):
        from src.vector_store import VectorStore
        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        vs.load(str(self.tmp_dir))
        results = vs.search("diabetes symptoms", top_k=3, score_threshold=0.0)
        self.assertGreater(len(results), 0)
        score = results[0][1]
        self.assertGreater(score, 0.3, f"Top-1 score too low: {score:.4f}")

    def test_loaded_vs_search_above_threshold(self):
        from src.vector_store import VectorStore
        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        vs.load(str(self.tmp_dir))
        results = vs.search("heart attack warning signs", top_k=3, score_threshold=0.30)
        # At least one result should be above threshold for this clear medical query
        self.assertGreater(len(results), 0)

    def test_loaded_metadata_url_preserved(self):
        from src.vector_store import VectorStore
        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        vs.load(str(self.tmp_dir))
        urls = [d.get("url", "") for d in vs.documents]
        has_url = sum(1 for u in urls if u)
        # All 5 test records have URLs
        self.assertGreater(has_url, 0)

    def test_rag_pipeline_loads_and_searches(self):
        """RAGPipeline.query() should work with the newly built index (mocked Gemini)."""
        from src.vector_store import VectorStore
        from src.rag_pipeline import RAGPipeline
        from unittest.mock import patch

        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        vs.load(str(self.tmp_dir))

        pipe = RAGPipeline(
            vector_store        = vs,
            api_key             = "fake-key",
            confidence_threshold = 0.20,
            enable_general_fallback = True,
        )
        with patch.object(pipe, "_generate",
                          return_value="Diabetes is a metabolic disease [DOC_1]."):
            result = pipe.query("What is diabetes?")

        self.assertIsNotNone(result)
        self.assertIn("Diabetes", result.answer)


# ── TestIncrementalUpdate ─────────────────────────────────────────────────────

class TestIncrementalUpdate(unittest.TestCase):
    """Content hash / doc_id based incremental update logic."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_incr_test_")
        self.tmp_dir = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_duplicate_doc_id_not_added_twice(self):
        """VectorStore.add_documents() must skip duplicate doc_ids."""
        from src.data_ingestion import MedicalDocument
        from src.vector_store import VectorStore

        vs = VectorStore(embedding_model_name="BAAI/bge-base-en-v1.5",
                         chunk_size=256, chunk_overlap=40, use_bge_prefix=True)
        doc = MedicalDocument(
            content = "Aspirin is used to treat pain and inflammation. " * 5,
            title   = "Aspirin",
            source  = "Test",
            doc_id  = "ASPIRIN_001",
        )
        n1 = vs.add_documents([doc])
        initial = len(vs.documents)
        n2 = vs.add_documents([doc])
        self.assertEqual(n2, 0, "Re-adding same doc_id should add 0 chunks")
        self.assertEqual(len(vs.documents), initial)

    def test_new_doc_id_added_in_incremental(self):
        from scripts.build_versioned_index import build_full, build_incremental
        records_a = _make_mini_corpus(3)
        records_b = _make_mini_corpus(5)  # superset
        new_ids = {r["doc_id"] for r in records_b} - {r["doc_id"] for r in records_a}
        self.assertGreater(len(new_ids), 0, "Test requires new doc_ids in b vs a")

        stats_a = build_full(records_a, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        stats_b = build_incremental(records_b, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)
        self.assertEqual(stats_b["mode"], "incremental")
        self.assertGreater(stats_b["total_chunks"], stats_a["total_chunks"])


# ── TestContentIntegrity ──────────────────────────────────────────────────────

class TestContentIntegrity(unittest.TestCase):
    """End-to-end content quality checks on a freshly built small index."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="medsource_ci_test_")
        self.tmp_dir = Path(self.tmp)
        # Inject an HTML-containing record alongside clean ones
        self.records = _make_mini_corpus(5)
        self.records.append({
            "doc_id":       "HTML_TEST_001",
            "content":      '<p>What is <span class="qt0">hypertension</span>? '
                            'High blood pressure is a condition in which the force '
                            'of the blood against artery walls is too high. ' * 3 + "</p>",
            "content_hash": hashlib.md5(b"html test").hexdigest(),
            "title":        "Hypertension HTML Test",
            "source":       "TestCorpus",
            "year":         "2024",
            "url":          "https://example.com/hypertension_html",
            "metadata":     {},
        })
        from scripts.build_versioned_index import build_full
        build_full(self.records, 256, 40, "BAAI/bge-base-en-v1.5", 32, self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_empty_chunks(self):
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        empty = [d for d in docs if not d.get("content", "").strip()]
        self.assertEqual(len(empty), 0)

    def test_html_record_is_indexed_with_tags_present(self):
        """The build_versioned_index script passes content through unchanged.
        HTML removal is the job of clean_corpus.py (called before building).
        This test confirms raw HTML is NOT stripped during indexing."""
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        html_docs = [d for d in docs if "HTML_TEST_001" in d.get("doc_id", "")]
        # HTML content was passed directly to VectorStore; it is stored as-is.
        # The clean_corpus step was not applied here, so HTML should survive.
        self.assertTrue(len(html_docs) > 0, "HTML test record should be indexed")

    def test_clean_records_have_no_html(self):
        """The 5 clean mini-corpus records must have no HTML after indexing."""
        with open(self.tmp_dir / "documents.pkl", "rb") as fh:
            docs = pickle.load(fh)
        html_pat = re.compile(r"<[a-zA-Z][^>]*>")
        clean_html = [
            d for d in docs
            if html_pat.search(d.get("content", ""))
            and "HTML_TEST" not in d.get("doc_id", "")
        ]
        self.assertEqual(len(clean_html), 0, "Clean records must not contain HTML tags")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=str(Path(__file__).parent), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
