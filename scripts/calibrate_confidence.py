"""
Measure the real BGE cosine score distribution against the live vector store and
print calibration recommendations for _calculate_confidence() in rag_pipeline.py.

Usage (from project root, venv active):
    python scripts/calibrate_confidence.py

What it does
------------
1. Loads the live vector store (same model + settings as the running app).
2. Runs 40 representative medical queries — a mix of well-covered topics
   (where scores should be high) and off-topic queries (where they should be low).
3. For each query, records the top-5 cosine scores returned by FAISS.
4. Prints the score distribution: p25, p50, p75, p90, p95, p99, max.
5. Recommends updated values for:
     confidence_threshold  (set to ~p25 of on-topic scores)
     BGE_PRACTICAL_MAX     (set to ~p95 of on-topic scores)

These values can then be copy-pasted into src/rag_pipeline.py and .env /
render.yaml. Re-run whenever the corpus or embedding model changes.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.config import get_settings
from src.vector_store import VectorStore

# ── query sets ───────────────────────────────────────────────────────────────

# On-topic: the store should have strong matches for these
ON_TOPIC = [
    "symptoms of anemia",
    "iron deficiency anemia treatment",
    "type 2 diabetes management",
    "hypertension risk factors",
    "asthma inhaler corticosteroids",
    "vitamin D deficiency symptoms",
    "tuberculosis first-line treatment",
    "HIV antiretroviral therapy",
    "heart attack warning signs",
    "metformin side effects diabetes",
    "community-acquired pneumonia antibiotics",
    "ibuprofen contraindications NSAID",
    "hypothyroidism levothyroxine",
    "chronic kidney disease causes",
    "dengue fever clinical presentation",
    "COVID-19 symptoms treatment",
    "scurvy vitamin C deficiency",
    "sepsis early recognition management",
    "malaria Plasmodium treatment",
    "migraine headache triggers treatment",
]

# Off-topic: the store should return low scores for these
OFF_TOPIC = [
    "capital of France",
    "best stock to buy today",
    "recipe for chocolate cake",
    "history of the Roman Empire",
    "how to fix a bicycle flat tyre",
    "python programming tutorial",
    "weather forecast tomorrow",
    "football world cup winners",
    "Shakespeare sonnets analysis",
    "how to bake sourdough bread",
]


def load_store() -> VectorStore:
    settings = get_settings()
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    vs.load(settings.vector_store_path)
    print(f"Loaded {len(vs.documents)} chunks  |  model: {vs.embedding_model_name}\n")
    return vs, settings


def collect_scores(vs: VectorStore, queries: list, label: str) -> list:
    scores = []
    for q in queries:
        results = vs.search(q, top_k=5, score_threshold=0.0)
        for _, s in results:
            scores.append(s)
    scores.sort()
    return scores


def percentile(data: list, p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def print_distribution(scores: list, label: str):
    if not scores:
        print(f"  {label}: no scores collected")
        return
    print(f"  {label}  (n={len(scores)} scores from {len(scores)//5} queries × top-5):")
    for p in (5, 25, 50, 75, 90, 95, 99):
        print(f"    p{p:>2}: {percentile(scores, p):.4f}")
    print(f"    max: {max(scores):.4f}")
    print(f"    min: {min(scores):.4f}")


def main():
    vs, settings = load_store()

    print("Collecting scores for ON-TOPIC queries...")
    on_scores  = collect_scores(vs, ON_TOPIC,  "on-topic")
    print("Collecting scores for OFF-TOPIC queries...")
    off_scores = collect_scores(vs, OFF_TOPIC, "off-topic")

    print("\n" + "=" * 60)
    print("SCORE DISTRIBUTION")
    print("=" * 60)
    print_distribution(on_scores,  "On-topic ")
    print()
    print_distribution(off_scores, "Off-topic")

    print("\n" + "=" * 60)
    print("CALIBRATION RECOMMENDATIONS")
    print("=" * 60)

    # Threshold: reject off-topic noise — use p75 of off-topic as the floor,
    # but never go below p25 of on-topic (don't reject valid matches).
    off_p75   = percentile(off_scores, 75)
    on_p25    = percentile(on_scores,  25)
    on_p95    = percentile(on_scores,  95)
    threshold = max(off_p75, on_p25)

    current_threshold = settings.retrieval_confidence_threshold

    print(f"\n  Current  confidence_threshold : {current_threshold:.3f}")
    print(f"  Suggested confidence_threshold : {threshold:.3f}  "
          f"(max of off-topic p75={off_p75:.3f}, on-topic p25={on_p25:.3f})")
    print(f"\n  Current  BGE_PRACTICAL_MAX     : 0.850  (hardcoded in rag_pipeline.py)")
    print(f"  Suggested BGE_PRACTICAL_MAX    : {on_p95:.3f}  (on-topic p95)")

    print("""
To apply:
  1. In src/rag_pipeline.py, update _calculate_confidence():
       BGE_PRACTICAL_MAX = {p95:.3f}
  2. In .env and render.yaml, update:
       RETRIEVAL_CONFIDENCE_THRESHOLD={thr:.2f}
  3. Re-run this script after any corpus or model change.
""".format(p95=on_p95, thr=threshold))


if __name__ == "__main__":
    main()
