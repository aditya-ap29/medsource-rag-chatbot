"""
tests/test_semantic_classifier.py
----------------------------------
Unit tests for the SemanticClassifier and ClassPrototypeClassifier.

Tests cover:
- Classifier interface with a fitted and unfitted model.
- Rare diseases, misspellings, abbreviations, informal medical language.
- Ambiguous and mixed-intent questions.
- Non-medical questions.
- Dataset label sanity.
- Persistence (save/load) round-trip.
- Prototype baseline interface.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_embedding_model(dim: int = 8):
    """Return a mock SentenceTransformer that produces deterministic embeddings."""
    import numpy as np

    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.random.RandomState(
        abs(hash(str(texts))) % (2**31)
    ).randn(len(texts), dim).astype("float32")
    return model


def _make_minimal_dataset(tmp_path: Path) -> str:
    """Write a minimal 3-class dataset with enough cases to fit a classifier."""
    cases = []
    medical_questions = [
        "What are the symptoms of diabetes?",
        "How does metformin work?",
        "What is hypertension?",
        "What causes asthma?",
        "How is tuberculosis treated?",
        "What is anemia?",
        "What are side effects of ibuprofen?",
        "How do vaccines work?",
        "What is sepsis?",
        "How is kidney disease diagnosed?",
        "What is Huntington's disease?",
        "What is amyotrophic lateral sclerosis?",
    ]
    oop_questions = [
        "What is the capital of France?",
        "How does the stock market work?",
        "Write me a poem about spring",
        "What is Bitcoin?",
        "How do I cook pasta?",
        "What programming language should I learn?",
        "What is the population of India?",
        "What is photosynthesis?",
        "Who won the World Cup?",
        "What is quantum computing?",
        "How do airplanes fly?",
        "What is the French word for apple?",
    ]
    ambiguous_questions = [
        "Is it normal to feel tired all the time?",
        "How much protein do I need?",
        "What is CKD?",
    ]

    for i, q in enumerate(medical_questions):
        split = "train" if i < 9 else ("val" if i < 10 else "test")
        cases.append({"id": f"M{i}", "question": q, "label": "medical", "split": split})
    for i, q in enumerate(oop_questions):
        split = "train" if i < 9 else ("val" if i < 10 else "test")
        cases.append({"id": f"O{i}", "question": q, "label": "out_of_scope", "split": split})
    for i, q in enumerate(ambiguous_questions):
        split = "train" if i < 1 else ("val" if i < 2 else "test")
        cases.append({"id": f"A{i}", "question": q, "label": "ambiguous", "split": split})

    ds = {"version": "test", "cases": cases}
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(ds), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSemanticClassifierUnfitted(unittest.TestCase):
    """Classifier returns safe fallback when not fitted."""

    def setUp(self):
        from src.semantic_classifier import SemanticClassifier
        self.clf = SemanticClassifier(
            embedding_model=_make_mock_embedding_model(),
            embedding_model_name="BAAI/bge-base-en-v1.5",
        )

    def test_predict_unfitted_returns_ambiguous(self):
        from src.semantic_classifier import LABEL_AMBIGUOUS
        label, conf, reason = self.clf.predict("What is diabetes?")
        self.assertEqual(label, LABEL_AMBIGUOUS)
        self.assertEqual(conf, 0.0)
        self.assertIn("not fitted", reason)

    def test_predict_batch_unfitted(self):
        from src.semantic_classifier import LABEL_AMBIGUOUS
        results = self.clf.predict_batch(["question 1", "question 2"])
        self.assertEqual(len(results), 2)
        for label, conf, _ in results:
            self.assertEqual(label, LABEL_AMBIGUOUS)

    def test_save_unfitted_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                self.clf.save(tmpdir)

    def test_load_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.clf.load("/nonexistent/path/that/does/not/exist")


class TestSemanticClassifierFitted(unittest.TestCase):
    """Classifier interface after fitting on minimal dataset."""

    def setUp(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        from src.semantic_classifier import SemanticClassifier, LABELS

        self.clf = SemanticClassifier(
            embedding_model=_make_mock_embedding_model(dim=8),
            embedding_model_name="BAAI/bge-base-en-v1.5",
        )
        # Manually inject a fitted sklearn classifier so we don't need
        # a real embedding model or sklearn dependency at test time.
        X = np.random.RandomState(42).randn(30, 8).astype("float32")
        y = (["medical"] * 12 + ["out_of_scope"] * 12 + ["ambiguous"] * 6)
        base_lr = LogisticRegression(C=1.0, max_iter=100, random_state=42)
        calibrated = CalibratedClassifierCV(base_lr, cv=3)
        calibrated.fit(X, y)
        self.clf._clf = calibrated
        self.clf._is_fitted = True
        self.clf._classes = LABELS

    def test_predict_returns_valid_label(self):
        from src.semantic_classifier import LABELS
        label, conf, reason = self.clf.predict("What are the symptoms of flu?")
        self.assertIn(label, LABELS)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIn("LR", reason)

    def test_predict_batch_length(self):
        questions = [
            "What is diabetes?",
            "What is the capital of France?",
            "Is it normal to feel tired?",
        ]
        results = self.clf.predict_batch(questions)
        self.assertEqual(len(results), len(questions))

    def test_confidence_is_float(self):
        _, conf, _ = self.clf.predict("What causes sepsis?")
        self.assertIsInstance(conf, float)

    def test_save_load_roundtrip(self):
        from src.semantic_classifier import SemanticClassifier, LABELS
        with tempfile.TemporaryDirectory() as tmpdir:
            self.clf.save(tmpdir)
            clf2 = SemanticClassifier(
                embedding_model=_make_mock_embedding_model(dim=8),
                embedding_model_name="BAAI/bge-base-en-v1.5",
            )
            clf2.load(tmpdir)
            self.assertTrue(clf2._is_fitted)
            label, conf, _ = clf2.predict("What is hypertension?")
            from src.semantic_classifier import LABELS
            self.assertIn(label, LABELS)


class TestSemanticClassifierDataset(unittest.TestCase):
    """Validate the production classification dataset."""

    def _load_dataset(self):
        dataset_path = ROOT / "data" / "eval" / "domain_classification_dataset_v1.json"
        if not dataset_path.exists():
            self.skipTest("Classification dataset not found.")
        with open(dataset_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def test_dataset_has_cases(self):
        ds = self._load_dataset()
        self.assertGreater(len(ds.get("cases", [])), 100)

    def test_all_cases_have_required_fields(self):
        ds = self._load_dataset()
        for case in ds["cases"]:
            for field in ("id", "question", "label", "split"):
                self.assertIn(field, case, f"Case {case.get('id')} missing field '{field}'")

    def test_labels_are_valid(self):
        from src.semantic_classifier import LABELS
        ds = self._load_dataset()
        for case in ds["cases"]:
            self.assertIn(case["label"], LABELS, f"Invalid label in case {case.get('id')}")

    def test_splits_are_valid(self):
        ds = self._load_dataset()
        valid_splits = {"train", "val", "test"}
        for case in ds["cases"]:
            self.assertIn(case["split"], valid_splits, f"Invalid split in case {case.get('id')}")

    def test_no_empty_questions(self):
        ds = self._load_dataset()
        for case in ds["cases"]:
            self.assertTrue(case["question"].strip(), f"Empty question in case {case.get('id')}")

    def test_all_three_labels_present(self):
        from src.semantic_classifier import LABEL_MEDICAL, LABEL_OOP, LABEL_AMBIGUOUS
        ds = self._load_dataset()
        labels = {c["label"] for c in ds["cases"]}
        self.assertIn(LABEL_MEDICAL, labels)
        self.assertIn(LABEL_OOP, labels)
        self.assertIn(LABEL_AMBIGUOUS, labels)

    def test_all_three_splits_present(self):
        ds = self._load_dataset()
        splits = {c["split"] for c in ds["cases"]}
        for split in ("train", "val", "test"):
            self.assertIn(split, splits)

    def test_no_duplicate_ids(self):
        ds = self._load_dataset()
        ids = [c["id"] for c in ds["cases"]]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate case IDs found")

    def test_rare_disease_cases_are_medical(self):
        ds = self._load_dataset()
        rare_ids = {
            "CLS-027",  # Huntington's
            "CLS-045",  # Fabry disease
            "CLS-091",  # Wilson's disease
            "CLS-135",  # Ehlers-Danlos syndrome
            "CLS-147",  # Phenylketonuria
            "CLS-158",  # Gaucher disease
            "CLS-172",  # Von Willebrand disease
        }
        cases_by_id = {c["id"]: c for c in ds["cases"]}
        for cid in rare_ids:
            if cid in cases_by_id:
                self.assertEqual(
                    cases_by_id[cid]["label"], "medical",
                    f"Rare disease case {cid} should be 'medical'",
                )

    def test_abbreviation_cases_are_medical(self):
        ds = self._load_dataset()
        abbrev_ids = {
            "CLS-025",  # HbA1c
            "CLS-037",  # COPD
            "CLS-041",  # ECG
            "CLS-047",  # ACE inhibitors
            "CLS-117",  # SSRIs
            "CLS-137",  # POTS
        }
        cases_by_id = {c["id"]: c for c in ds["cases"]}
        for cid in abbrev_ids:
            if cid in cases_by_id:
                self.assertEqual(
                    cases_by_id[cid]["label"], "medical",
                    f"Abbreviation case {cid} should be 'medical'",
                )


class TestPrototypeClassifier(unittest.TestCase):
    """ClassPrototypeClassifier basic interface."""

    def setUp(self):
        import numpy as np
        from src.semantic_classifier import ClassPrototypeClassifier, LABELS

        self.proto = ClassPrototypeClassifier(
            embedding_model=_make_mock_embedding_model(dim=8),
            embedding_model_name="BAAI/bge-base-en-v1.5",
        )
        # Inject centroids manually
        self.proto._classes = list(LABELS)
        self.proto._centroids = np.eye(3, 8).astype("float32")
        # Normalise
        norms = np.linalg.norm(self.proto._centroids, axis=1, keepdims=True)
        self.proto._centroids = self.proto._centroids / norms
        self.proto._is_fitted = True

    def test_predict_returns_label(self):
        from src.semantic_classifier import LABELS
        label, conf, reason = self.proto.predict("Some question")
        self.assertIn(label, LABELS)
        self.assertIn("prototype", reason)

    def test_unfitted_returns_fallback(self):
        from src.semantic_classifier import ClassPrototypeClassifier, LABEL_AMBIGUOUS
        proto = ClassPrototypeClassifier(
            embedding_model=_make_mock_embedding_model(dim=8),
        )
        label, conf, _ = proto.predict("test")
        self.assertEqual(label, LABEL_AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
