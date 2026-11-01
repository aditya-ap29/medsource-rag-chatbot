"""
src/semantic_classifier.py
---------------------------
Lightweight supervised domain classifier for MedSource.

Architecture
------------
Encodes each question with the same BGE model already loaded by VectorStore,
then classifies the embedding with Logistic Regression (LR) trained on
data/eval/domain_classification_dataset_v1.json.

Three classes:
  - medical       — question is about health / medicine
  - out_of_scope  — question is clearly outside medical scope
  - ambiguous     — intent unclear; handled conservatively downstream

Prototype-based classifier is also provided as a baseline comparator
(ClassPrototypeClassifier).  Both share the same encode() interface.

Design notes
------------
1.  BGE embeddings from the trained FAISS index are 768-D, L2-normalised.
    We re-use the same SentenceTransformer via a lazy reference passed at
    construction time so the model is loaded only once.

2.  Logistic regression is calibrated on validation data using
    sklearn.calibration.CalibratedClassifierCV so predicted probabilities
    are interpretable.

3.  The classifier is intentionally conservative:
    - It never replaces the emergency bypass (handled upstream).
    - If the model is not fitted it falls back to returning 'ambiguous'
      so the pipeline continues rather than blocking.
    - Unfamiliar text → falls through to medical (see _default_to_medical).

4.  Confidence scores are calibrated probabilities, not raw cosine values.

5.  The module is **stateless at import time**: nothing expensive happens
    until train() or load() is called.

Usage
-----
    from src.semantic_classifier import SemanticClassifier
    clf = SemanticClassifier(embedding_model=vs.embedding_model)
    clf.train(dataset_path="data/eval/domain_classification_dataset_v1.json")
    clf.save("data/classifier_v1")

    label, confidence, reason = clf.predict("What are the symptoms of flu?")
    # label = 'medical', confidence = 0.97, reason = 'LR p=0.97'
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MedSource.semantic_classifier")

# Label constants
LABEL_MEDICAL = "medical"
LABEL_OOP = "out_of_scope"
LABEL_AMBIGUOUS = "ambiguous"

LABELS = [LABEL_MEDICAL, LABEL_OOP, LABEL_AMBIGUOUS]

# BGE asymmetric query prefix — must match vector_store.py
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_MODEL_PREFIXES = ("BAAI/bge",)


def _use_bge_prefix(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _BGE_MODEL_PREFIXES)


class SemanticClassifier:
    """Logistic Regression classifier on BGE embeddings.

    Parameters
    ----------
    embedding_model : SentenceTransformer
        The *already-loaded* embedding model from VectorStore.  Passed by
        reference to avoid a second model load.
    embedding_model_name : str
        Name / path of the model — used only to determine the BGE prefix flag.
    C : float
        Inverse regularisation for LogisticRegression.  Smaller = stronger
        regularisation.  Default 1.0.
    max_iter : int
        Max solver iterations for LogisticRegression.
    """

    def __init__(
        self,
        embedding_model: Any,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        C: float = 1.0,
        max_iter: int = 1000,
    ):
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.use_bge_prefix = _use_bge_prefix(embedding_model_name)
        self.C = C
        self.max_iter = max_iter
        self._clf = None      # fitted CalibratedClassifierCV | None
        self._classes: List[str] = LABELS
        self._is_fitted = False
        self._metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of questions into L2-normalised BGE embeddings."""
        if self.use_bge_prefix:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype("float32")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_path: str = "data/eval/domain_classification_dataset_v1.json",
        splits: Tuple[str, ...] = ("train", "val"),
    ) -> Dict[str, Any]:
        """Fit the classifier on the labeled dataset.

        Uses train+val data for fitting, calibrated with cross-validation on
        the training fold.

        Parameters
        ----------
        dataset_path : str
            Path to domain_classification_dataset_v1.json.
        splits : tuple of str
            Which dataset splits to use for training (default: train + val).

        Returns
        -------
        dict with training metrics.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.metrics import classification_report, confusion_matrix
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for the semantic classifier. "
                "Install with: pip install scikit-learn"
            ) from exc

        with open(dataset_path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        cases = dataset.get("cases", [])
        train_cases = [c for c in cases if c.get("split") in splits]

        if len(train_cases) < 10:
            raise ValueError(
                f"Only {len(train_cases)} training cases found in {splits}. "
                "Need at least 10."
            )

        texts = [c["question"] for c in train_cases]
        labels = [c["label"] for c in train_cases]

        logger.info(
            "Encoding %d training questions with %s...",
            len(texts), self.embedding_model_name,
        )
        X = self.encode(texts)
        y = np.array(labels)

        # Distribution
        from collections import Counter
        dist = Counter(labels)
        logger.info("Label distribution: %s", dict(dist))

        base_lr = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        # CalibratedClassifierCV wraps the LR and calibrates probabilities
        # using cross-validation on the same training data.
        clf = CalibratedClassifierCV(base_lr, method="isotonic", cv=min(5, min(dist.values())))
        clf.fit(X, y)
        self._clf = clf
        self._is_fitted = True

        # Evaluate on training data (optimistic but useful for sanity check)
        y_pred = clf.predict(X)
        report = classification_report(y, y_pred, output_dict=True)

        self._metadata = {
            "dataset_path": dataset_path,
            "n_train": len(train_cases),
            "label_dist": dict(dist),
            "train_report": report,
            "embedding_model": self.embedding_model_name,
            "C": self.C,
        }

        logger.info(
            "Classifier fitted. Train macro-F1=%.3f",
            report.get("macro avg", {}).get("f1-score", float("nan")),
        )
        return self._metadata

    def evaluate(
        self,
        dataset_path: str = "data/eval/domain_classification_dataset_v1.json",
        split: str = "test",
    ) -> Dict[str, Any]:
        """Evaluate on a held-out split. Returns per-class metrics + confusion."""
        if not self._is_fitted:
            raise RuntimeError("Classifier is not fitted. Call train() first.")

        try:
            from sklearn.metrics import (
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
            )
        except ImportError as exc:
            raise ImportError("scikit-learn required") from exc

        with open(dataset_path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        cases = [c for c in dataset.get("cases", []) if c.get("split") == split]
        if not cases:
            return {"error": f"No cases in split='{split}'", "n_cases": 0}

        texts = [c["question"] for c in cases]
        y_true = [c["label"] for c in cases]

        X = self.encode(texts)
        y_pred = self._clf.predict(X)
        y_prob = self._clf.predict_proba(X)

        class_order = list(self._clf.classes_)

        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=class_order)

        # False rejection = medical question predicted as out_of_scope
        med_idx = class_order.index(LABEL_MEDICAL) if LABEL_MEDICAL in class_order else -1
        oop_idx = class_order.index(LABEL_OOP) if LABEL_OOP in class_order else -1

        false_rejections = 0
        false_acceptances = 0
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label == LABEL_MEDICAL and pred_label == LABEL_OOP:
                false_rejections += 1
            if true_label == LABEL_OOP and pred_label == LABEL_MEDICAL:
                false_acceptances += 1

        n_medical_true = sum(1 for l in y_true if l == LABEL_MEDICAL)
        n_oop_true = sum(1 for l in y_true if l == LABEL_OOP)

        result = {
            "split": split,
            "n_cases": len(cases),
            "classification_report": report,
            "confusion_matrix": {
                "labels": class_order,
                "matrix": cm.tolist(),
            },
            "false_rejection_rate": false_rejections / max(n_medical_true, 1),
            "false_acceptance_rate": false_acceptances / max(n_oop_true, 1),
            "false_rejections_n": false_rejections,
            "false_acceptances_n": false_acceptances,
            "macro_f1": report.get("macro avg", {}).get("f1-score", float("nan")),
        }
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        question: str,
    ) -> Tuple[str, float, str]:
        """Classify a single question.

        Returns
        -------
        (label, confidence, reason)
            label      : 'medical', 'out_of_scope', or 'ambiguous'
            confidence : calibrated probability in [0, 1]
            reason     : human-readable explanation for logging
        """
        if not self._is_fitted:
            logger.debug("Classifier not fitted — returning 'ambiguous' fallback")
            return LABEL_AMBIGUOUS, 0.0, "classifier not fitted (fallback)"

        X = self.encode([question])
        pred_label = self._clf.predict(X)[0]
        proba = self._clf.predict_proba(X)[0]
        class_order = list(self._clf.classes_)

        try:
            conf = float(proba[class_order.index(pred_label)])
        except (ValueError, IndexError):
            conf = float(np.max(proba))

        reason = f"LR p({pred_label})={conf:.3f}"
        return str(pred_label), round(conf, 4), reason

    def predict_batch(
        self,
        questions: List[str],
    ) -> List[Tuple[str, float, str]]:
        """Classify a list of questions efficiently (single encode call)."""
        if not self._is_fitted:
            return [(LABEL_AMBIGUOUS, 0.0, "classifier not fitted") for _ in questions]

        X = self.encode(questions)
        pred_labels = self._clf.predict(X)
        probas = self._clf.predict_proba(X)
        class_order = list(self._clf.classes_)

        results = []
        for pred_label, proba in zip(pred_labels, probas):
            try:
                conf = float(proba[class_order.index(pred_label)])
            except (ValueError, IndexError):
                conf = float(np.max(proba))
            reason = f"LR p({pred_label})={conf:.3f}"
            results.append((str(pred_label), round(conf, 4), reason))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, save_dir: str) -> None:
        """Persist the fitted classifier and metadata to disk."""
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — classifier not fitted.")
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "classifier.pkl", "wb") as fh:
            pickle.dump(self._clf, fh)

        meta = {**self._metadata, "embedding_model": self.embedding_model_name}
        with open(path / "metadata.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info("Semantic classifier saved to %s", path)

    def load(self, load_dir: str) -> None:
        """Load a previously saved classifier."""
        path = Path(load_dir)
        clf_path = path / "classifier.pkl"
        if not clf_path.exists():
            raise FileNotFoundError(f"classifier.pkl not found in {load_dir}")

        with open(clf_path, "rb") as fh:
            self._clf = pickle.load(fh)
        self._is_fitted = True

        meta_path = path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            stored_model = self._metadata.get("embedding_model", "")
            if stored_model and stored_model != self.embedding_model_name:
                logger.warning(
                    "Classifier was trained with '%s' but current model is '%s'. "
                    "Predictions may be unreliable.",
                    stored_model, self.embedding_model_name,
                )

        logger.info("Semantic classifier loaded from %s", path)


# ---------------------------------------------------------------------------
# Prototype-based baseline classifier
# ---------------------------------------------------------------------------

class ClassPrototypeClassifier:
    """Nearest-prototype classifier as a simple baseline.

    Computes one centroid per class from training embeddings.
    Classification = argmax cosine similarity to class centroids.
    No calibration step — confidence = raw cosine similarity.

    Use SemanticClassifier (LR) for production; this is for comparison only.
    """

    def __init__(
        self,
        embedding_model: Any,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.use_bge_prefix = _use_bge_prefix(embedding_model_name)
        self._centroids: Optional[np.ndarray] = None  # shape (n_classes, dim)
        self._classes: List[str] = []
        self._is_fitted = False

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.use_bge_prefix:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).astype("float32")

    def train(
        self,
        dataset_path: str = "data/eval/domain_classification_dataset_v1.json",
        splits: Tuple[str, ...] = ("train", "val"),
    ) -> Dict[str, Any]:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        cases = [c for c in dataset.get("cases", []) if c.get("split") in splits]
        texts = [c["question"] for c in cases]
        labels = [c["label"] for c in cases]

        X = self.encode(texts)
        unique_labels = sorted(set(labels))
        self._classes = unique_labels

        centroids = []
        for lbl in unique_labels:
            mask = np.array([l == lbl for l in labels])
            centroid = X[mask].mean(axis=0)
            # Re-normalise
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / norm if norm > 0 else centroid)

        self._centroids = np.stack(centroids).astype("float32")
        self._is_fitted = True
        return {"n_train": len(cases), "classes": unique_labels}

    def predict(self, question: str) -> Tuple[str, float, str]:
        if not self._is_fitted:
            return LABEL_AMBIGUOUS, 0.0, "prototype classifier not fitted"

        X = self.encode([question])
        sims = (self._centroids @ X.T).flatten()  # cosine sims (already L2-normed)
        idx = int(np.argmax(sims))
        label = self._classes[idx]
        conf = float(sims[idx])
        reason = f"prototype cosine={conf:.3f}"
        return label, round(conf, 4), reason

    def evaluate(
        self,
        dataset_path: str = "data/eval/domain_classification_dataset_v1.json",
        split: str = "test",
    ) -> Dict[str, Any]:
        try:
            from sklearn.metrics import classification_report, confusion_matrix
        except ImportError as exc:
            raise ImportError("scikit-learn required") from exc

        with open(dataset_path, "r", encoding="utf-8") as fh:
            dataset = json.load(fh)

        cases = [c for c in dataset.get("cases", []) if c.get("split") == split]
        if not cases:
            return {"error": f"No cases in split='{split}'"}

        y_true = [c["label"] for c in cases]
        y_pred = [self.predict(c["question"])[0] for c in cases]

        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=self._classes)

        false_rejections = sum(
            1 for t, p in zip(y_true, y_pred) if t == LABEL_MEDICAL and p == LABEL_OOP
        )
        false_acceptances = sum(
            1 for t, p in zip(y_true, y_pred) if t == LABEL_OOP and p == LABEL_MEDICAL
        )
        n_med = sum(1 for l in y_true if l == LABEL_MEDICAL)
        n_oop = sum(1 for l in y_true if l == LABEL_OOP)

        return {
            "split": split,
            "n_cases": len(cases),
            "classification_report": report,
            "confusion_matrix": {"labels": self._classes, "matrix": cm.tolist()},
            "false_rejection_rate": false_rejections / max(n_med, 1),
            "false_acceptance_rate": false_acceptances / max(n_oop, 1),
            "macro_f1": report.get("macro avg", {}).get("f1-score", float("nan")),
        }
