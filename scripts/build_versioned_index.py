"""
scripts/build_versioned_index.py
---------------------------------
Production-ready versioned FAISS index builder for MedSource.

Selected configuration (from benchmark results in
data/processed/chunk_benchmark_results.json):
  chunk_size    = 256 tokens   ← best top-1 cosine, lowest ctx cost, lowest latency
  chunk_overlap = 40  tokens
  model         = BAAI/bge-base-en-v1.5 (768-dim, BGE asymmetric prefixes)
  index type    = FAISS IndexFlatIP (exact inner-product == cosine on L2-norm vectors)
  normalisation = L2 (applied by SentenceTransformer encode, normalize_embeddings=True)

Benchmark evidence (10-query evaluation, top-5 retrieval):
  chunk_size=256: avg_top1=0.7540, avg_ctx_tokens=869,  latency=31.3ms
  chunk_size=400: avg_top1=0.7563, avg_ctx_tokens=1145, latency=40.0ms  (+31.8% ctx cost)
  chunk_size=500: avg_top1=0.7496, avg_ctx_tokens=1392, latency=41.7ms  (+60.2% ctx cost)
256 wins on context budget and latency; the 0.002 score difference vs 400 is within noise.

Safety guarantees:
  - Never writes to data/vector_store/ (the active index).
  - Writes to data/vector_store_v1/ (or --output-dir).
  - Exits immediately if the output dir already contains faiss.index.
  - Writes build_status.json with status=in_progress at start and
    status=complete at end, so partial builds are detectable.
  - Pre-save integrity assertions stop a corrupt index being saved.

Incremental updates:
  Pass --incremental to skip re-embedding chunks whose doc_id already
  appears in an existing build at --output-dir.  Only NEW doc_ids are
  embedded and appended.  Use --full-rebuild to force a clean build.
  WARNING: incremental mode is only safe when no existing doc has changed
  content.  If chunking config or embedding model changed, use --full-rebuild.

Usage (from project root):
    # Full build (recommended, ~15 min on CPU):
    python scripts/build_versioned_index.py

    # Incremental update (fast, only new docs):
    python scripts/build_versioned_index.py --incremental

    # Custom output directory:
    python scripts/build_versioned_index.py --output-dir data/vector_store_v2
"""

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_CORPUS  = ROOT / "data" / "processed" / "clean_corpus_v1.json"
DEFAULT_OUT   = ROOT / "data" / "vector_store_v1"

# Benchmark-selected configuration — do not change without re-running the benchmark.
DEFAULT_CHUNK_SIZE    = 256
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_MODEL         = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE    = 64
INDEX_TYPE            = "IndexFlatIP"
NORMALISATION         = "L2"
USE_BGE_PREFIX        = True


# ── helpers ───────────────────────────────────────────────────────────────────

def _corpus_hash(records: list[dict]) -> str:
    """Stable hash of the full corpus content for reproducibility tracking."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.get("doc_id", "")):
        h.update(r.get("doc_id", "").encode())
        h.update(r.get("content_hash", r.get("content", "")[:64]).encode())
    return h.hexdigest()[:16]


def _write_build_status(output_dir: Path, status: str, extra: dict | None = None) -> None:
    record = {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        record.update(extra)
    with open(output_dir / "build_status.json", "w") as fh:
        json.dump(record, fh, indent=2)


def load_clean_corpus(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Clean corpus not found: {path}\n"
            "Run scripts/clean_corpus.py first."
        )
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    print(f"Loaded {len(records):,} clean records from {path.name}")
    return records


# ── build ──────────────────────────────────────────────────────────────────────

def build_full(
    records: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    batch_size: int,
    output_dir: Path,
) -> dict:
    """Full (clean) build: embed all clean records and save to output_dir."""
    from src.data_ingestion import MedicalDocument
    from src.vector_store import VectorStore

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

    print(f"Building index: chunk_size={chunk_size} tok, "
          f"overlap={chunk_overlap} tok, model={model_name}")
    print(f"  Index type: {INDEX_TYPE}, normalisation: {NORMALISATION}, "
          f"BGE prefix: {USE_BGE_PREFIX}")

    vs = VectorStore(
        embedding_model_name = model_name,
        chunk_size           = chunk_size,
        chunk_overlap        = chunk_overlap,
        use_bge_prefix       = USE_BGE_PREFIX,
        embedding_batch_size = batch_size,
    )

    _write_build_status(output_dir, "in_progress",
                        {"started_at": datetime.now(timezone.utc).isoformat()})

    t0 = time.time()
    n_added = vs.add_documents(docs)
    elapsed = time.time() - t0
    print(f"Indexed {n_added:,} chunks in {elapsed:.1f}s")

    return _save_and_verify(vs, output_dir, chunk_size, chunk_overlap,
                            model_name, elapsed, records, mode="full")


def build_incremental(
    records: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    batch_size: int,
    output_dir: Path,
) -> dict:
    """Incremental build: load existing index, embed only new doc_ids.

    Only safe when:
     - embedding model is unchanged
     - chunk_size and chunk_overlap are unchanged
     - no existing doc has changed content (content-only changes require a full rebuild)
    """
    from src.data_ingestion import MedicalDocument
    from src.vector_store import VectorStore
    import pickle, faiss

    # Verify the existing index is compatible
    cfg_path = output_dir / "config.pkl"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No existing config.pkl in {output_dir}. "
            "Run a full build first, or use --full-rebuild."
        )
    with open(cfg_path, "rb") as fh:
        cfg = pickle.load(fh)

    if cfg.get("embedding_model_name") != model_name:
        raise ValueError(
            f"Existing index used model '{cfg['embedding_model_name']}' "
            f"but current model is '{model_name}'. Use --full-rebuild."
        )
    if cfg.get("chunk_size") != chunk_size or cfg.get("chunk_overlap") != chunk_overlap:
        raise ValueError(
            f"Existing index used chunk_size={cfg.get('chunk_size')} / "
            f"overlap={cfg.get('chunk_overlap')}, "
            f"but current config is {chunk_size}/{chunk_overlap}. Use --full-rebuild."
        )

    # Load existing VectorStore
    vs = VectorStore(
        embedding_model_name = model_name,
        chunk_size           = chunk_size,
        chunk_overlap        = chunk_overlap,
        use_bge_prefix       = USE_BGE_PREFIX,
        embedding_batch_size = batch_size,
    )
    vs.load(str(output_dir))
    existing_ids = {d["doc_id"] for d in vs.documents}
    print(f"Loaded existing index: {len(vs.documents):,} chunks, "
          f"{len(existing_ids):,} unique doc_ids")

    new_records = [r for r in records if r.get("doc_id", "") not in existing_ids]
    if not new_records:
        print("No new documents found. Index is already up to date.")
        return {
            "total_chunks":    len(vs.documents),
            "unique_doc_ids":  len(existing_ids),
            "new_chunks":      0,
            "dimension":       vs.dimension,
            "build_time_s":    0.0,
            "source_distribution": dict(Counter(d["source"] for d in vs.documents)),
            "mode": "incremental_no_op",
        }

    print(f"Found {len(new_records):,} new records to embed "
          f"({len(records) - len(new_records):,} already indexed).")

    new_docs = [
        MedicalDocument(
            content  = r["content"],
            title    = r.get("title", ""),
            source   = r.get("source", ""),
            doc_id   = r.get("doc_id", ""),
            year     = r.get("year"),
            url      = r.get("url"),
            metadata = r.get("metadata", {}),
        )
        for r in new_records
    ]

    _write_build_status(output_dir, "in_progress_incremental",
                        {"started_at": datetime.now(timezone.utc).isoformat()})

    t0 = time.time()
    n_added = vs.add_documents(new_docs)
    elapsed = time.time() - t0
    print(f"Added {n_added:,} new chunks in {elapsed:.1f}s")

    return _save_and_verify(vs, output_dir, chunk_size, chunk_overlap,
                            model_name, elapsed, records, mode="incremental")


def _save_and_verify(
    vs,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
    elapsed: float,
    source_records: list[dict],
    mode: str,
) -> dict:
    """Post-build integrity checks, save, and write provenance metadata."""
    import faiss

    # Pre-save integrity assertions
    if vs.index.ntotal != len(vs.documents):
        _write_build_status(output_dir, "failed_integrity",
                            {"error": "FAISS ntotal != documents list length"})
        raise AssertionError(
            f"FAISS vector count ({vs.index.ntotal}) != document list "
            f"({len(vs.documents)}) — aborting before save."
        )
    if vs.index.d != vs.dimension:
        _write_build_status(output_dir, "failed_integrity",
                            {"error": "FAISS dimension != model dimension"})
        raise AssertionError(
            f"FAISS dimension ({vs.index.d}) != model dimension ({vs.dimension})"
        )

    source_dist = dict(Counter(d["source"] for d in vs.documents))
    unique_docs  = len(set(d["doc_id"] for d in vs.documents))
    avg_chars    = sum(len(d["content"]) for d in vs.documents) // len(vs.documents)
    corpus_hash  = _corpus_hash(source_records)

    # Save FAISS + documents + config (uses VectorStore.save which writes config.pkl)
    vs.save(str(output_dir))

    # Write version.json with full provenance
    version_info = {
        "version":              "v1",
        "build_mode":           mode,
        "created_at":           datetime.now(timezone.utc).isoformat(),
        "embedding_model":      model_name,
        "embedding_dimension":  vs.dimension,
        "index_type":           INDEX_TYPE,
        "normalisation":        NORMALISATION,
        "use_bge_prefix":       USE_BGE_PREFIX,
        "chunk_size_tokens":    chunk_size,
        "chunk_overlap_tokens": chunk_overlap,
        "source_corpus_file":   str(CLEAN_CORPUS),
        "source_corpus_hash":   corpus_hash,
        "total_chunks":         len(vs.documents),
        "unique_doc_ids":       unique_docs,
        "avg_chunk_chars":      avg_chars,
        "source_distribution":  source_dist,
        "build_time_s":         round(elapsed, 1),
    }
    with open(output_dir / "version.json", "w") as fh:
        json.dump(version_info, fh, indent=2)

    _write_build_status(output_dir, "complete",
                        {"completed_at": datetime.now(timezone.utc).isoformat()})

    print(f"Saved index to {output_dir}")

    stats = {
        "total_chunks":      len(vs.documents),
        "unique_doc_ids":    unique_docs,
        "dimension":         vs.dimension,
        "avg_chunk_chars":   avg_chars,
        "build_time_s":      round(elapsed, 1),
        "source_distribution": source_dist,
        "corpus_hash":       corpus_hash,
        "mode":              mode,
    }
    return stats


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a versioned FAISS index from clean_corpus_v1.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark-selected defaults (do not change without re-running the benchmark):
  --chunk-size 256  --chunk-overlap 40  --model BAAI/bge-base-en-v1.5
""",
    )
    parser.add_argument("--chunk-size",    type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Max chunk size in tokens (default: %(default)s)")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help="Chunk overlap in tokens (default: %(default)s)")
    parser.add_argument("--model",         type=str, default=DEFAULT_MODEL,
                        help="Sentence-transformers model (default: %(default)s)")
    parser.add_argument("--batch-size",    type=int, default=DEFAULT_BATCH_SIZE,
                        help="Embedding batch size (default: %(default)s)")
    parser.add_argument("--output-dir",    type=str, default=str(DEFAULT_OUT),
                        help="Output directory (default: data/vector_store_v1/)")
    parser.add_argument("--incremental",   action="store_true",
                        help="Add only new doc_ids to an existing index at --output-dir. "
                             "Requires the existing index to use the same model/chunk config.")
    parser.add_argument("--full-rebuild",  action="store_true",
                        help="Force a clean build even if --output-dir already exists "
                             "(renames the existing directory to <dir>_old_<timestamp>).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Safety guards
    if args.incremental and args.full_rebuild:
        parser.error("--incremental and --full-rebuild are mutually exclusive.")

    if (output_dir / "faiss.index").exists():
        if args.full_rebuild:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            renamed = output_dir.parent / f"{output_dir.name}_old_{ts}"
            output_dir.rename(renamed)
            print(f"Renamed existing {output_dir.name} to {renamed.name}")
        elif not args.incremental:
            print(f"\nIndex already exists at {output_dir}")
            print("Options:")
            print("  --incremental   : add only new documents to the existing index")
            print("  --full-rebuild  : rename old index and build from scratch")
            sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_clean_corpus(CLEAN_CORPUS)

    if args.incremental:
        stats = build_incremental(
            records, args.chunk_size, args.chunk_overlap,
            args.model, args.batch_size, output_dir,
        )
    else:
        stats = build_full(
            records, args.chunk_size, args.chunk_overlap,
            args.model, args.batch_size, output_dir,
        )

    print("\n" + "=" * 60)
    print("Index build complete")
    print(f"  Mode           : {stats.get('mode', 'full')}")
    print(f"  Total chunks   : {stats['total_chunks']:,}")
    print(f"  Unique doc_ids : {stats['unique_doc_ids']:,}")
    print(f"  Dimension      : {stats['dimension']}")
    print(f"  Avg chunk      : {stats['avg_chunk_chars']} chars")
    print(f"  Build time     : {stats['build_time_s']}s")
    print(f"  Output dir     : {output_dir}")
    print(f"  Corpus hash    : {stats.get('corpus_hash', 'n/a')}")
    print("=" * 60)
    print(f"\nNext step: validate the new index:")
    print(f"  python scripts/validate_index.py --index-dir {output_dir}")
    print(f"Then activate (with automatic backup):")
    print(f"  python scripts/swap_index.py --new-index {output_dir}")
    print(f"Rollback if needed:")
    print(f"  python scripts/swap_index.py --rollback")


if __name__ == "__main__":
    main()
