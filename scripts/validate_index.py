"""
scripts/validate_index.py
--------------------------
Validate a FAISS index directory before activation.

Checks performed (all 10 must pass for PASS verdict):
  1.  All required files exist: faiss.index, documents.pkl, config.pkl.
  2.  FAISS vector count == len(documents.pkl).
  3.  FAISS dimension matches config.pkl['dimension'].
  4.  Embedding model name in config matches --model argument.
  5.  No empty content in any chunk.
  6.  No HTML tags in any chunk (<tag> pattern).
  7.  Content hash uniqueness across all chunks.
  8.  All chunks have a non-empty doc_id.
  9.  Source distribution is non-empty and printed.
  10. version.json is present and readable (warning if absent).
  11. build_status.json shows status=complete (warning if absent or failed).
  12. End-to-end load test: VectorStore.load() succeeds and returns results
      for 3 representative medical queries (score > 0.3).

Exit code: 0 if all error-level checks pass, 1 if any error exists.

Usage:
    python scripts/validate_index.py [--index-dir data/vector_store_v1]
    python scripts/validate_index.py --index-dir data/vector_store   # validate active
"""

import argparse
import hashlib
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


SPOT_QUERIES = [
    "diabetes symptoms",
    "hypertension treatment",
    "heart attack warning signs",
]
SPOT_MIN_SCORE = 0.30
REQUIRED_FILES = ["faiss.index", "documents.pkl", "config.pkl"]


def _load_raw_artifacts(index_dir: Path):
    """Load (faiss_index, documents, config) without loading the embedding model."""
    import faiss

    for fname in REQUIRED_FILES:
        fpath = index_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")

    faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "documents.pkl", "rb") as fh:
        documents = pickle.load(fh)
    with open(index_dir / "config.pkl", "rb") as fh:
        config = pickle.load(fh)
    return faiss_index, documents, config


def validate_index(
    index_dir: Path,
    expected_model: str = "BAAI/bge-base-en-v1.5",
) -> bool:
    """Run all validation checks. Returns True iff no error-level failures."""
    print(f"Validating index at: {index_dir}")
    print("-" * 60)
    errors   = []
    warnings = []

    # ── Load raw artifacts ────────────────────────────────────────────
    try:
        faiss_index, documents, config = _load_raw_artifacts(index_dir)
    except Exception as exc:
        print(f"  FAIL  Could not load artifacts: {exc}")
        return False

    n_vecs = faiss_index.ntotal
    n_docs = len(documents)
    dim    = faiss_index.d

    # ── Check 1: required files (already covered by _load_raw_artifacts) ──
    print(f"  PASS  All required files present")

    # ── Check 2: vector count == document count ───────────────────────
    if n_vecs != n_docs:
        errors.append(f"FAISS vector count ({n_vecs:,}) != document list ({n_docs:,})")
    else:
        print(f"  PASS  Vector count == document list: {n_vecs:,}")

    # ── Check 3: dimension matches config ─────────────────────────────
    cfg_dim = config.get("dimension", -1)
    if dim != cfg_dim:
        errors.append(f"FAISS dimension ({dim}) != config dimension ({cfg_dim})")
    else:
        print(f"  PASS  Dimension: {dim}")

    # ── Check 4: model name ───────────────────────────────────────────
    cfg_model = config.get("embedding_model_name", "")
    if cfg_model != expected_model:
        warnings.append(
            f"Config model '{cfg_model}' != expected '{expected_model}'"
        )
        print(f"  WARN  Embedding model: {cfg_model!r} (expected {expected_model!r})")
    else:
        print(f"  PASS  Embedding model: {cfg_model}")

    # ── Check 5: no empty content ─────────────────────────────────────
    empty = [d for d in documents if not d.get("content", "").strip()]
    if empty:
        errors.append(f"{len(empty)} chunk(s) with empty content")
    else:
        print(f"  PASS  No empty content chunks")

    # ── Check 6: no HTML tags ─────────────────────────────────────────
    html_pat = re.compile(r"<[a-zA-Z][^>]*>")
    html_chunks = [d for d in documents if html_pat.search(d.get("content", ""))]
    if html_chunks:
        errors.append(f"{len(html_chunks)} chunk(s) contain HTML tags")
        # Show a sample to help debug
        sample = html_chunks[0]["content"][:120]
        print(f"  FAIL  HTML tags found — sample: {sample!r}")
    else:
        print(f"  PASS  No HTML tags in content")

    # ── Check 7: content hash uniqueness ─────────────────────────────
    hashes = [hashlib.md5(d["content"].encode("utf-8", errors="replace")).hexdigest()
              for d in documents]
    n_unique = len(set(hashes))
    dup_count = len(hashes) - n_unique
    if dup_count:
        warnings.append(f"{dup_count} duplicate chunk content hash(es) detected")
        print(f"  WARN  {dup_count} duplicate chunk content hashes")
    else:
        print(f"  PASS  All {n_unique:,} chunk content hashes unique")

    # ── Check 8: all chunks have doc_id ──────────────────────────────
    missing_ids = [d for d in documents if not str(d.get("doc_id", "")).strip()]
    if missing_ids:
        warnings.append(f"{len(missing_ids)} chunk(s) have no doc_id")
        print(f"  WARN  {len(missing_ids)} chunks have no doc_id")
    else:
        print(f"  PASS  All chunks have doc_id")

    # ── Check 9: source distribution ─────────────────────────────────
    source_dist = Counter(d.get("source", "?") for d in documents)
    if not source_dist:
        errors.append("Source distribution is empty")
    else:
        print(f"  PASS  Source distribution ({len(source_dist)} source(s)):")
        for src, cnt in source_dist.most_common():
            pct = 100.0 * cnt / n_docs
            print(f"          {src}: {cnt:,} ({pct:.1f}%)")

    # ── Check 10: version.json ────────────────────────────────────────
    version_path = index_dir / "version.json"
    if version_path.exists():
        try:
            with open(version_path) as fh:
                version = json.load(fh)
            print(f"  PASS  version.json: version={version.get('version','?')}"
                  f"  created={version.get('created_at','?')[:19]}"
                  f"  mode={version.get('build_mode','?')}"
                  f"  corpus_hash={version.get('source_corpus_hash','?')}")
        except Exception as exc:
            warnings.append(f"version.json parse error: {exc}")
    else:
        warnings.append("version.json not present")

    # ── Check 11: build_status.json ───────────────────────────────────
    status_path = index_dir / "build_status.json"
    if status_path.exists():
        try:
            with open(status_path) as fh:
                bstatus = json.load(fh)
            status_val = bstatus.get("status", "")
            if status_val == "complete":
                print(f"  PASS  build_status.json: {status_val}")
            else:
                warnings.append(
                    f"build_status.json shows status='{status_val}' (expected 'complete')"
                    " — this may be a partial build."
                )
                print(f"  WARN  build_status.json status={status_val!r}")
        except Exception as exc:
            warnings.append(f"build_status.json parse error: {exc}")
    else:
        warnings.append("build_status.json not present")

    # ── Check 12: end-to-end load + search via VectorStore ───────────
    # Use the model stored in config.pkl for the actual search.
    # expected_model is only used for warning comparison above.
    actual_model = config.get("embedding_model_name") or expected_model
    print(f"  Running end-to-end search test ({len(SPOT_QUERIES)} queries, model={actual_model})...")
    try:
        from src.vector_store import VectorStore

        vs = VectorStore(
            embedding_model_name = actual_model,
            chunk_size           = config.get("chunk_size", 256),
            chunk_overlap        = config.get("chunk_overlap", 40),
            use_bge_prefix       = config.get("use_bge_prefix", True),
        )
        vs.load(str(index_dir))

        n_found = 0
        for q in SPOT_QUERIES:
            results = vs.search(q, top_k=5, score_threshold=0.0)
            if results and results[0][1] >= SPOT_MIN_SCORE:
                n_found += 1
                top = results[0]
                print(f"          q={q!r:40s}  top1_score={top[1]:.4f}"
                      f"  src={top[0].get('source','?')[:30]}")
            else:
                print(f"          q={q!r:40s}  no result above {SPOT_MIN_SCORE}")

        if n_found == len(SPOT_QUERIES):
            print(f"  PASS  All {n_found}/{len(SPOT_QUERIES)} spot queries returned results")
        else:
            errors.append(
                f"Spot-check: only {n_found}/{len(SPOT_QUERIES)} queries "
                f"returned results above score {SPOT_MIN_SCORE}"
            )

    except Exception as exc:
        errors.append(f"End-to-end load/search test failed: {exc}")
        print(f"  FAIL  End-to-end test: {exc}")

    # ── URL / metadata survival spot-check ───────────────────────────
    url_count    = sum(1 for d in documents if d.get("url"))
    no_url_count = n_docs - url_count
    if no_url_count > 0:
        warnings.append(
            f"{no_url_count}/{n_docs} chunks have no URL "
            "(Patient Education Summary records expected to have none)"
        )
        print(f"  WARN  {no_url_count}/{n_docs} chunks have no URL")
    else:
        print(f"  PASS  All {n_docs:,} chunks have a URL")

    year_none = sum(1 for d in documents if not d.get("year"))
    pct_no_yr = 100.0 * year_none / n_docs if n_docs else 0
    print(f"  INFO  {year_none}/{n_docs} chunks have year=None ({pct_no_yr:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────
    print("-" * 60)
    if warnings:
        for w in warnings:
            print(f"  WARN  {w}")
    if errors:
        for e in errors:
            print(f"  FAIL  {e}")
        print(f"\nValidation FAILED  ({len(errors)} error(s), {len(warnings)} warning(s))")
        return False
    else:
        print(f"\nValidation PASSED  ({len(warnings)} warning(s))")
        print(f"  Total chunks     : {n_docs:,}")
        print(f"  Unique sources   : {len(source_dist)}")
        print(f"  Dimension        : {dim}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate a versioned FAISS index.")
    parser.add_argument(
        "--index-dir",
        default=str(ROOT / "data" / "vector_store_v1"),
        help="Path to the index directory to validate (default: data/vector_store_v1/).",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="Expected embedding model name (default: %(default)s).",
    )
    args = parser.parse_args()

    ok = validate_index(Path(args.index_dir), expected_model=args.model)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
