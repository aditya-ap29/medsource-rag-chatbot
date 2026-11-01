#!/usr/bin/env python3
"""
scripts/train_classifier.py
-----------------------------
Train and evaluate the MedSource semantic domain classifier.

Usage
-----
# Train on train+val splits, evaluate on test split, save model
python scripts/train_classifier.py

# Evaluate only (model must already be trained)
python scripts/train_classifier.py --eval-only

# Compare LR classifier against prototype-based baseline
python scripts/train_classifier.py --compare-prototype

# Custom paths
python scripts/train_classifier.py \\
    --dataset data/eval/domain_classification_dataset_v1.json \\
    --save-dir data/semantic_classifier_v1

Output
------
- Trained model saved to data/semantic_classifier_v1/
- Evaluation results saved to data/eval/classifier_eval_results.json
- Summary printed to stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_classifier")


def _load_vector_store(vs_path: str | None = None):
    """Load the production VectorStore (needed for the embedding model)."""
    from src.vector_store import VectorStore
    from src.config import get_settings

    settings = get_settings()
    vs = VectorStore(embedding_model_name=settings.embedding_model)
    path = vs_path or settings.vector_store_path
    try:
        vs.load(path)
        logger.info("VectorStore loaded from %s (%d chunks)", path, len(vs.documents))
    except Exception as exc:
        logger.warning(
            "Could not load VectorStore from %s (%s). "
            "Will create a standalone embedding model for classification only.",
            path, exc,
        )
    return vs


def train(
    dataset_path: str,
    save_dir: str,
    vs_path: str | None = None,
    train_splits: tuple[str, ...] = ("train", "val"),
) -> dict:
    """Train the semantic classifier and save it.

    Returns a dict of training metrics.
    """
    from src.semantic_classifier import SemanticClassifier

    vs = _load_vector_store(vs_path)

    clf = SemanticClassifier(
        embedding_model=vs.embedding_model,
        embedding_model_name=vs.embedding_model_name,
    )

    logger.info(
        "Training classifier on splits=%s from %s ...", train_splits, dataset_path
    )
    metrics = clf.train(dataset_path=dataset_path, splits=train_splits)

    clf.save(save_dir)
    logger.info("Classifier saved to %s", save_dir)
    return metrics


def evaluate(
    dataset_path: str,
    save_dir: str,
    vs_path: str | None = None,
    test_split: str = "test",
) -> dict:
    """Load the saved classifier and evaluate it on the test split."""
    from src.semantic_classifier import SemanticClassifier

    vs = _load_vector_store(vs_path)

    clf = SemanticClassifier(
        embedding_model=vs.embedding_model,
        embedding_model_name=vs.embedding_model_name,
    )
    clf.load(save_dir)

    results = clf.evaluate(dataset_path=dataset_path, split=test_split)
    return results


def compare_prototype(
    dataset_path: str,
    save_dir: str,
    vs_path: str | None = None,
) -> dict:
    """Train prototype classifier and compare against LR on test split."""
    from src.semantic_classifier import ClassPrototypeClassifier, SemanticClassifier

    vs = _load_vector_store(vs_path)

    # Prototype
    proto = ClassPrototypeClassifier(
        embedding_model=vs.embedding_model,
        embedding_model_name=vs.embedding_model_name,
    )
    proto.train(dataset_path=dataset_path, splits=("train", "val"))
    proto_eval = proto.evaluate(dataset_path=dataset_path, split="test")

    # LR
    clf = SemanticClassifier(
        embedding_model=vs.embedding_model,
        embedding_model_name=vs.embedding_model_name,
    )
    clf.load(save_dir)
    lr_eval = clf.evaluate(dataset_path=dataset_path, split="test")

    return {
        "logistic_regression": lr_eval,
        "prototype": proto_eval,
        "lr_vs_prototype_macro_f1": {
            "logistic_regression": lr_eval.get("macro_f1"),
            "prototype": proto_eval.get("macro_f1"),
        },
    }


def print_summary(eval_results: dict, title: str = "Classifier Evaluation") -> None:
    """Print a human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Split         : {eval_results.get('split', 'N/A')}")
    print(f"  Cases         : {eval_results.get('n_cases', 'N/A')}")
    print(f"  Macro F1      : {eval_results.get('macro_f1', 0):.3f}")
    print(f"  False rejection rate  : {eval_results.get('false_rejection_rate', 0):.3f}  "
          f"(medical → out_of_scope, n={eval_results.get('false_rejections_n', 0)})")
    print(f"  False acceptance rate : {eval_results.get('false_acceptance_rate', 0):.3f}  "
          f"(out_of_scope → medical, n={eval_results.get('false_acceptances_n', 0)})")

    report = eval_results.get("classification_report", {})
    for label in ("medical", "out_of_scope", "ambiguous"):
        if label in report:
            r = report[label]
            print(
                f"  {label:>15s} — P={r.get('precision', 0):.3f}  "
                f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}  "
                f"support={r.get('support', 0)}"
            )

    cm = eval_results.get("confusion_matrix", {})
    if cm:
        labels = cm.get("labels", [])
        matrix = cm.get("matrix", [])
        print(f"\n  Confusion matrix (rows=true, cols=predicted):")
        print(f"  {'':>15s}  " + "  ".join(f"{l:>13s}" for l in labels))
        for i, row in enumerate(matrix):
            true_label = labels[i] if i < len(labels) else str(i)
            print(f"  {true_label:>15s}  " + "  ".join(f"{v:>13d}" for v in row))
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MedSource semantic domain classifier."
    )
    parser.add_argument(
        "--dataset",
        default="data/eval/domain_classification_dataset_v1.json",
        help="Path to the classification dataset JSON.",
    )
    parser.add_argument(
        "--save-dir",
        default="data/semantic_classifier_v1",
        help="Directory to save/load the trained classifier.",
    )
    parser.add_argument(
        "--vs-path",
        default=None,
        help="Override vector store path (default: from config).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; only evaluate the saved model.",
    )
    parser.add_argument(
        "--compare-prototype",
        action="store_true",
        help="Also compare against the prototype-based baseline.",
    )
    parser.add_argument(
        "--results-out",
        default="data/eval/classifier_eval_results.json",
        help="Path to write machine-readable evaluation results.",
    )
    args = parser.parse_args(argv)

    dataset_path = args.dataset
    save_dir = args.save_dir
    results_out = Path(args.results_out)

    if not args.eval_only:
        train_metrics = train(dataset_path, save_dir, vs_path=args.vs_path)
        logger.info(
            "Training complete. Train macro-F1=%.3f",
            train_metrics.get("train_report", {})
            .get("macro avg", {})
            .get("f1-score", 0),
        )

    logger.info("Evaluating on test split...")
    eval_results = evaluate(
        dataset_path, save_dir, vs_path=args.vs_path, test_split="test"
    )
    print_summary(eval_results)

    all_results: dict = {"lr_test": eval_results}

    if args.compare_prototype:
        logger.info("Comparing against prototype classifier...")
        comparison = compare_prototype(dataset_path, save_dir, vs_path=args.vs_path)
        all_results["comparison"] = comparison

        print_summary(
            comparison.get("logistic_regression", {}),
            title="LR Classifier (test split)",
        )
        print_summary(
            comparison.get("prototype", {}),
            title="Prototype Classifier (test split)",
        )
        delta = (
            (comparison.get("logistic_regression", {}).get("macro_f1") or 0)
            - (comparison.get("prototype", {}).get("macro_f1") or 0)
        )
        print(f"  LR vs Prototype macro-F1 delta: {delta:+.3f}")

    results_out.parent.mkdir(parents=True, exist_ok=True)
    with open(results_out, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("Evaluation results saved to %s", results_out)

    # Return non-zero exit code if false rejection rate is dangerously high
    frr = eval_results.get("false_rejection_rate", 0.0)
    if frr > 0.15:
        logger.error(
            "False rejection rate %.3f exceeds 0.15 safety threshold! "
            "Review classifier before enabling in production.",
            frr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
