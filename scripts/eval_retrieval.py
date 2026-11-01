"""
scripts/eval_retrieval.py
--------------------------
Compare retrieval quality between two FAISS indexes using a fixed set of
representative medical queries.

Metrics per query:
  - top-1 cosine similarity score
  - avg top-5 cosine similarity
  - source diversity (fraction of top-5 from different doc_ids)
  - HTML in any top-5 excerpt (0 = clean, >0 = contaminated)
  - context token cost (sum of top-5 chunk tokens)
  - has URL for each top-5 result

Outputs:
  - Console table comparing old vs new
  - data/processed/eval_retrieval_results.json

Usage (from project root):
    python scripts/eval_retrieval.py \\
        --old-index  data/vector_store \\
        --new-index  data/vector_store_v1
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "data" / "processed" / "eval_retrieval_results.json"

# Evaluation dataset — 12 representative medical queries spanning different
# question types (symptom, drug, mechanism, emergency, chronic condition)
EVAL_QUERIES = [
    # Symptom-focused
    ("symptom", "What are the symptoms of type 2 diabetes?"),
    ("symptom", "What are warning signs of a heart attack?"),
    ("symptom", "What are symptoms of iron deficiency anemia?"),
    # Drug/mechanism
    ("drug",    "How does metformin work to lower blood sugar?"),
    ("drug",    "What are common side effects of lisinopril?"),
    ("drug",    "What is the mechanism of action of aspirin?"),
    # Condition management
    ("management", "How is hypertension treated and managed?"),
    ("management", "How is asthma managed in clinical practice?"),
    ("management", "What are treatment options for major depression?"),
    # Diagnostics / lab
    ("diagnostic", "What does a high HbA1c level indicate?"),
    ("diagnostic", "What lab tests are used to diagnose anemia?"),
    # Emergency
    ("emergency", "What should I do if someone has a stroke?"),
]


def _token_count(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return len(text.split())


_HTML_RE = re.compile(r"<[a-zA-Z][^>]*>")


def _evaluate_one_query(vs, query: str, top_k: int = 5) -> dict:
    results = vs.search(query, top_k=top_k, score_threshold=0.0)
    if not results:
        return {
            "n_results":     0,
            "top1_score":    0.0,
            "avg_score":     0.0,
            "diversity":     0.0,
            "html_count":    0,
            "ctx_tokens":    0,
            "url_count":     0,
        }

    scores    = [s for _, s in results]
    doc_ids   = [d["doc_id"] for d, _ in results]
    diversity = len(set(doc_ids)) / len(doc_ids)
    html_c    = sum(1 for d, _ in results if _HTML_RE.search(d.get("content", "")))
    ctx_tok   = sum(_token_count(d["content"]) for d, _ in results)
    url_c     = sum(1 for d, _ in results if d.get("url"))

    return {
        "n_results":  len(results),
        "top1_score": round(scores[0], 4),
        "avg_score":  round(sum(scores) / len(scores), 4),
        "diversity":  round(diversity, 3),
        "html_count": html_c,
        "ctx_tokens": ctx_tok,
        "url_count":  url_c,
        "top1_source": results[0][0].get("source", "?"),
        "top1_title":  results[0][0].get("title", "?")[:60],
    }


def load_vs(index_dir: Path, chunk_size: int = 256, chunk_overlap: int = 40):
    """Load a VectorStore from disk, respecting the stored config."""
    import pickle
    cfg_path = index_dir / "config.pkl"
    if cfg_path.exists():
        with open(cfg_path, "rb") as fh:
            cfg = pickle.load(fh)
        chunk_size    = cfg.get("chunk_size",    chunk_size)
        chunk_overlap = cfg.get("chunk_overlap", chunk_overlap)
        use_prefix    = cfg.get("use_bge_prefix", True)
    else:
        use_prefix = True

    from src.vector_store import VectorStore
    vs = VectorStore(
        embedding_model_name = "BAAI/bge-base-en-v1.5",
        chunk_size           = chunk_size,
        chunk_overlap        = chunk_overlap,
        use_bge_prefix       = use_prefix,
    )
    vs.load(str(index_dir))
    return vs


def evaluate_index(vs, label: str, queries: list[tuple[str, str]], top_k: int = 5) -> dict:
    print(f"\n  Evaluating '{label}' ({len(vs.documents):,} chunks) ...")
    per_query = {}
    for q_type, q_text in queries:
        per_query[q_text] = _evaluate_one_query(vs, q_text, top_k=top_k)

    scores    = [v["top1_score"] for v in per_query.values()]
    avg_scores= [v["avg_score"]  for v in per_query.values()]
    diversities=[v["diversity"]  for v in per_query.values()]
    html_total= sum(v["html_count"] for v in per_query.values())
    ctx_total = [v["ctx_tokens"] for v in per_query.values()]
    url_total = sum(v["url_count"] for v in per_query.values())
    total_results = sum(v["n_results"] for v in per_query.values())

    return {
        "label":           label,
        "total_chunks":    len(vs.documents),
        "n_queries":       len(queries),
        "avg_top1_score":  round(sum(scores) / len(scores), 4),
        "avg_avg_score":   round(sum(avg_scores) / len(avg_scores), 4),
        "avg_diversity":   round(sum(diversities) / len(diversities), 3),
        "total_html_hits": html_total,
        "avg_ctx_tokens":  int(sum(ctx_total) / len(ctx_total)),
        "url_coverage":    round(url_total / max(total_results, 1), 3),
        "per_query":       per_query,
    }


def print_comparison(old: dict, new: dict) -> None:
    print("\n" + "=" * 76)
    print(f"  RETRIEVAL QUALITY COMPARISON")
    print(f"  Old: {old['label']} ({old['total_chunks']:,} chunks)")
    print(f"  New: {new['label']} ({new['total_chunks']:,} chunks)")
    print("-" * 76)
    print(f"  {'Metric':<35} {'Old':>10} {'New':>10} {'Delta':>10}")
    print("-" * 76)

    rows = [
        ("Avg top-1 cosine",   "avg_top1_score", ".4f"),
        ("Avg top-5 mean cos", "avg_avg_score",  ".4f"),
        ("Avg source diversity","avg_diversity",  ".3f"),
        ("HTML contamination (total)", "total_html_hits", "d"),
        ("Avg context tokens (top-5)", "avg_ctx_tokens", "d"),
        ("URL coverage (fraction)",    "url_coverage",   ".3f"),
    ]
    for label, key, fmt in rows:
        ov = old.get(key, 0)
        nv = new.get(key, 0)
        delta = nv - ov if isinstance(nv, (int, float)) else "n/a"
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else (f"{delta:+d}" if isinstance(delta, int) else str(delta))
        ov_str = format(ov, fmt)
        nv_str = format(nv, fmt)
        print(f"  {label:<35} {ov_str:>10} {nv_str:>10} {delta_str:>10}")

    print("-" * 76)
    print(f"  {'Per-query top-1 scores':}")
    print(f"  {'Query':<50} {'Old':>8} {'New':>8}")
    for q_type, q_text in EVAL_QUERIES:
        ov = old["per_query"].get(q_text, {}).get("top1_score", 0.0)
        nv = new["per_query"].get(q_text, {}).get("top1_score", 0.0)
        short_q = q_text[:48]
        print(f"  [{q_type[:4]}] {short_q:<46} {ov:>8.4f} {nv:>8.4f}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval quality between two indexes.")
    parser.add_argument("--old-index", default=str(ROOT / "data" / "vector_store"),
                        help="Path to the old (active) index.")
    parser.add_argument("--new-index", default=str(ROOT / "data" / "vector_store_v1"),
                        help="Path to the new (candidate) index.")
    parser.add_argument("--top-k",    type=int, default=5)
    args = parser.parse_args()

    old_dir = Path(args.old_index)
    new_dir = Path(args.new_index)

    for d in (old_dir, new_dir):
        if not (d / "faiss.index").exists():
            print(f"ERROR: faiss.index not found in {d}")
            sys.exit(1)

    print("Loading indexes (embedding model loads once)...")
    old_vs = load_vs(old_dir)
    new_vs = load_vs(new_dir)

    old_res = evaluate_index(old_vs, f"old ({old_dir.name})", EVAL_QUERIES, args.top_k)
    new_res = evaluate_index(new_vs, f"new ({new_dir.name})", EVAL_QUERIES, args.top_k)

    print_comparison(old_res, new_res)

    output = {"old": old_res, "new": new_res}
    with open(RESULTS_FILE, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
