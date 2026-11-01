"""
scripts/benchmark_chunk_sizes.py
---------------------------------
Offline benchmark for chunk sizes 256, 400, and 500 tokens.
Uses the clean corpus (data/processed/clean_corpus_v1.json) and the
existing BGE embedding model.  No Gemini calls are made.

Metrics measured for each configuration:
  - chunk count and average chunk length (chars, tokens)
  - retrieval relevance: cosine similarity scores for a fixed set of
    representative medical queries (top-5 scores, avg, min)
  - context coverage: fraction of top-5 chunks that are NOT from the
    same source document (diversity)
  - context token cost: total tokens in the top-5 context window

The script prints a comparison table and writes
data/processed/chunk_benchmark_results.json.

Usage (from project root):
    python scripts/benchmark_chunk_sizes.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLEAN_CORPUS = ROOT / "data" / "processed" / "clean_corpus_v1.json"
RESULTS_FILE = ROOT / "data" / "processed" / "chunk_benchmark_results.json"

# Representative medical queries covering different complexity levels
BENCHMARK_QUERIES = [
    "What are the symptoms of type 2 diabetes?",
    "How does metformin work as a diabetes medication?",
    "What are the risk factors for hypertension?",
    "What are common side effects of lisinopril?",
    "How is anemia diagnosed and treated?",
    "What are the warning signs of a heart attack?",
    "What vaccines are recommended for adults?",
    "How is asthma managed in clinical practice?",
    "What are the symptoms and treatment of depression?",
    "What is the mechanism of action of aspirin?",
]

CHUNK_SIZES    = [256, 400, 500]
CHUNK_OVERLAP  = 40          # constant across all configs
TOP_K          = 5
BGE_MODEL      = "BAAI/bge-base-en-v1.5"

_BGE_QUERY_PREFIX   = "Represent this sentence for searching relevant passages: "
_BGE_PASSAGE_PREFIX = "Represent this passage: "


def load_clean_corpus() -> list[dict]:
    if not CLEAN_CORPUS.exists():
        raise FileNotFoundError(
            f"Clean corpus not found: {CLEAN_CORPUS}\n"
            "Run scripts/clean_corpus.py first."
        )
    with open(CLEAN_CORPUS, encoding="utf-8") as fh:
        return json.load(fh)


def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {BGE_MODEL} ...")
    model = SentenceTransformer(BGE_MODEL)
    print(f"  Dimension: {model.get_embedding_dimension()}")
    return model


def make_vector_store_for_config(records: list[dict], chunk_size: int, model) -> Any:
    """Build an in-memory VectorStore using the given chunk_size."""
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

    vs = VectorStore(
        embedding_model_name = BGE_MODEL,
        chunk_size           = chunk_size,
        chunk_overlap        = CHUNK_OVERLAP,
        use_bge_prefix       = True,
        embedding_batch_size = 64,
    )
    # Patch the already-loaded model in to avoid re-loading it per config
    vs.embedding_model = model

    t0 = time.time()
    n_added = vs.add_documents(docs)
    elapsed = time.time() - t0

    return vs, n_added, elapsed


def count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))
    except ImportError:
        return len(text.split())


def evaluate_config(
    vs, chunk_size: int, queries: list[str], top_k: int = TOP_K
) -> dict:
    """Run all benchmark queries against the vector store and aggregate metrics."""
    all_top_scores = []
    all_avg_scores = []
    all_min_scores = []
    all_diversity  = []
    all_ctx_tokens = []
    all_latencies  = []

    for q in queries:
        t0 = time.perf_counter()
        results = vs.search(q, top_k=top_k, score_threshold=0.0)
        latency_ms = (time.perf_counter() - t0) * 1000

        if not results:
            continue

        scores = [s for _, s in results]
        all_top_scores.append(scores[0])
        all_avg_scores.append(sum(scores) / len(scores))
        all_min_scores.append(scores[-1])
        all_latencies.append(latency_ms)

        # Diversity: unique doc_ids / number of returned results
        doc_ids   = [d["doc_id"] for d, _ in results]
        diversity = len(set(doc_ids)) / len(doc_ids)
        all_diversity.append(diversity)

        # Context token cost: sum of tokens for all top-k chunks
        ctx_tokens = sum(count_tokens(d["content"]) for d, _ in results)
        all_ctx_tokens.append(ctx_tokens)

    def _avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Chunk length stats
    chunk_lens_chars  = [len(d["content"]) for d in vs.documents]
    chunk_lens_tokens = [count_tokens(d["content"]) for d in vs.documents]

    return {
        "chunk_size_cfg":       chunk_size,
        "total_chunks":         len(vs.documents),
        "avg_chunk_chars":      int(sum(chunk_lens_chars) / len(chunk_lens_chars)) if chunk_lens_chars else 0,
        "avg_chunk_tokens":     int(sum(chunk_lens_tokens) / len(chunk_lens_tokens)) if chunk_lens_tokens else 0,
        "min_chunk_tokens":     min(chunk_lens_tokens) if chunk_lens_tokens else 0,
        "max_chunk_tokens":     max(chunk_lens_tokens) if chunk_lens_tokens else 0,
        "avg_top1_score":       _avg(all_top_scores),
        "avg_avg_score":        _avg(all_avg_scores),
        "avg_min_score":        _avg(all_min_scores),
        "avg_diversity":        _avg(all_diversity),
        "avg_ctx_tokens_top5":  int(_avg(all_ctx_tokens)),
        "avg_latency_ms":       round(_avg(all_latencies), 1),
    }


def main():
    records = load_clean_corpus()
    print(f"Loaded {len(records):,} clean documents.")

    model   = load_embedding_model()
    results = {}

    for cs in CHUNK_SIZES:
        print(f"\n--- Chunk size: {cs} tokens (overlap={CHUNK_OVERLAP}) ---")
        vs, n_added, build_time = make_vector_store_for_config(records, cs, model)
        print(f"  Added {n_added:,} chunks  (build time: {build_time:.1f}s)")

        metrics = evaluate_config(vs, cs, BENCHMARK_QUERIES)
        metrics["build_time_s"] = round(build_time, 1)
        results[str(cs)] = metrics

        print(f"  avg_top1_score={metrics['avg_top1_score']:.4f}  "
              f"avg_avg_score={metrics['avg_avg_score']:.4f}  "
              f"avg_diversity={metrics['avg_diversity']:.3f}  "
              f"avg_ctx_tokens={metrics['avg_ctx_tokens_top5']:,}  "
              f"avg_latency={metrics['avg_latency_ms']:.1f}ms")

    # Print comparison table
    print("\n" + "=" * 75)
    print("CHUNK SIZE BENCHMARK RESULTS")
    print("-" * 75)
    header = f"{'Metric':<30} {'256 tok':>10} {'400 tok':>10} {'500 tok':>10}"
    print(header)
    print("-" * 75)

    rows = [
        ("Total chunks",           "total_chunks",         "d"),
        ("Avg chunk (chars)",       "avg_chunk_chars",      "d"),
        ("Avg chunk (tokens)",      "avg_chunk_tokens",     "d"),
        ("Avg top-1 cosine",        "avg_top1_score",       ".4f"),
        ("Avg top-5 cosine (mean)", "avg_avg_score",        ".4f"),
        ("Avg top-5 cosine (min)",  "avg_min_score",        ".4f"),
        ("Avg src diversity (0-1)", "avg_diversity",        ".3f"),
        ("Avg ctx tokens (top-5)",  "avg_ctx_tokens_top5",  "d"),
        ("Avg retrieval latency",   "avg_latency_ms",       ".1f"),
        ("Index build time (s)",    "build_time_s",         ".1f"),
    ]

    for label, key, fmt in rows:
        vals = [results[str(cs)].get(key, "n/a") for cs in CHUNK_SIZES]
        formatted = [
            (f"{v:{fmt}}" if isinstance(v, (int, float)) else str(v))
            for v in vals
        ]
        print(f"  {label:<28} {formatted[0]:>10} {formatted[1]:>10} {formatted[2]:>10}")

    print("=" * 75)

    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
