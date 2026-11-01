"""
RAGAS evaluation for MedSource (written for ragas 0.4.x).

What it does
------------
1. Runs a small, fixed set of questions through the live RAG pipeline.
2. Saves the answers + the exact retrieved text to ragas_cache.json, so if the
   judging step fails (e.g. quota), you can re-run with --judge-only without
   spending the pipeline's Gemini requests again.
3. Scores the grounded answers with two reference-free RAGAS metrics:
     - Faithfulness       : are the answer's claims supported by the retrieved text?
     - ResponseRelevancy  : does the answer actually address the question?
4. Prints a summary and saves ragas_eval_results.csv.

Methodology choices (so the numbers mean something)
---------------------------------------------------
- Faithfulness is judged against the full retrieved chunk text (source.excerpt).
- The disclaimer line and [DOC_n] citation markers are stripped before judging;
  they are formatting, not factual claims.
- Answers that came from the general-knowledge fallback have no retrieved
  context, so faithfulness is undefined for them. They are NOT scored; they are
  reported separately as the "grounded rate".
- The judge is the same Gemini model family that wrote the answers. LLM judges
  tend to favour their own output, so treat scores as an LLM-judged baseline,
  not an independent benchmark.

Quota
-----
Free Gemini 2.0 Flash tier = 1,500 requests/day (15 RPM, 1M TPM).
With 20 test questions and RELEVANCY_STRICTNESS = 1, each question
costs ~4 requests:
  1 (pipeline answer) + 2 (Faithfulness) + 1 (ResponseRelevancy)
= ~80 requests total — comfortably within the daily free-tier limit.
Run --judge-only to reuse a cached ragas_cache.json if you only want
to re-run the RAGAS scoring step without re-querying the pipeline.

Usage (from the project root, venv active):
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --judge-only      # reuse ragas_cache.json
"""
import argparse
import json
import re
import sys
import types
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CACHE_PATH = Path("ragas_cache.json")
RESULTS_PATH = Path("ragas_eval_results.csv")

# 1 generated question per answer instead of ragas' default 3: cheaper on quota,
# slightly noisier score. Raise it on a day with spare quota.
RELEVANCY_STRICTNESS = 1

# 20 questions covering: corpus strengths (PubMed/MedlinePlus topics), edge cases
# (rare conditions, non-medical deflection), and follow-up phrasing patterns.
# Every question costs quota — run the full set once, then use --judge-only to re-score.
TEST_QUESTIONS = [
    # Core medical topics well-covered by the corpus
    "What are the symptoms of anemia?",
    "What is scurvy and why does it happen?",
    "What are the symptoms of dengue fever?",
    "How is type 2 diabetes managed?",
    "What are the risk factors for hypertension?",
    "What causes asthma attacks?",
    "What are the symptoms of vitamin D deficiency?",
    "How is tuberculosis treated?",
    "What is HIV and how is it transmitted?",
    "What are the warning signs of a heart attack?",
    # Drug/clinical information (FDA labels + ClinicalTrials corpus)
    "What are the common side effects of metformin?",
    "What is the recommended treatment for community-acquired pneumonia?",
    "What are the contraindications for ibuprofen?",
    # Conditions at the edge of corpus coverage
    "What are the symptoms of hypothyroidism?",
    "What causes chronic kidney disease?",
    # Follow-up style questions (test that prior context is not required)
    "What is the difference between type 1 and type 2 diabetes?",
    "When should a patient with asthma use a rescue inhaler?",
    # Safe-refusal cases — should deflect cleanly, not hallucinate
    "What is the capital of France?",
    "Can you diagnose me with diabetes based on my symptoms?",
    "What is the best stock to buy right now?",
]

_DISCLAIMER_RE = re.compile(r"[^\n]*This information is for educational purposes only[^\n]*", re.IGNORECASE)
_CITATION_RE = re.compile(r"\s*\[DOC_\d+(?:\s*,\s*DOC_\d+)*\]")


def _ensure_ragas_importable():
    """
    ragas 0.4.3 does `from langchain_community.chat_models.vertexai import ChatVertexAI`,
    but current langchain-community no longer ships that module, so `import ragas`
    crashes. ragas only uses the class to recognise Google Vertex AI models, which
    this project does not use, so a placeholder is harmless. Remove this shim once
    a fixed ragas release is out.
    """
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # placeholder, never instantiated
            pass

        stub.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = stub


def prepare_for_judging(answer: str) -> str:
    """Strip the disclaimer line and [DOC_n] markers so only factual prose is judged."""
    answer = _DISCLAIMER_RE.sub("", answer)
    answer = _CITATION_RE.sub("", answer)
    return answer.strip()


def build_pipeline():
    """Build the pipeline exactly the way src/api.py's lifespan() does."""
    from src.config import get_settings
    from src.vector_store import VectorStore
    from src.rag_pipeline import RAGPipeline

    settings = get_settings()
    vector_store = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    vector_store.load(settings.vector_store_path)

    # Gemini-only backend — matches src/api.py lifespan() exactly.
    model_name = settings.llm_model or "gemini-2.0-flash"
    api_key = settings.gemini_api_key

    pipeline = RAGPipeline(
        vector_store=vector_store,
        model_name=model_name,
        api_key=api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        top_k=settings.top_k_documents,
        confidence_threshold=settings.retrieval_confidence_threshold,
        enable_general_fallback=settings.enable_general_fallback,
    )
    return settings, pipeline


def collect_answers(settings, pipeline, questions):
    """Run each question through the pipeline and collect answers + full contexts."""
    records = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        response = pipeline.query(question=q)
        grounded = bool(response.sources)

        # source.excerpt now contains the full chunk text (not a 200-char truncation),
        # so we can use it directly for RAGAS faithfulness scoring.
        contexts = [s.excerpt for s in response.sources] if grounded else []

        records.append({
            "question": q,
            "answer": response.answer,
            "grounded": grounded,
            "confidence": float(response.confidence),
            "n_sources": len(response.sources),
            "contexts": contexts,
            "retrieval_time_ms": float(response.retrieval_time_ms),
            "generation_time_ms": float(response.generation_time_ms),
        })
        tag = "grounded" if grounded else "FALLBACK (general knowledge, not scored)"
        print(f"    {tag} | confidence={response.confidence:.3f} | sources={len(response.sources)}")
    return records


def build_judge(settings, judge_model):
    _ensure_ragas_importable()
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_huggingface import HuggingFaceEmbeddings

    # Judge LLM: Gemini free tier. Embeddings: the same local sentence-transformers
    # model the project already uses, so the embedding step costs no quota.
    # Use the stable pinned alias (gemini-2.0-flash-001) so Google deprecating the
    # floating "gemini-2.0-flash" alias does not break scoring.
    stable_judge = judge_model
    if judge_model in ("gemini-2.0-flash", "gemini-2.0-flash-exp"):
        stable_judge = "gemini-2.0-flash-001"
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=stable_judge,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        max_retries=2,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.embedding_model))
    return llm, embeddings


def run_judge(scored, llm, embeddings):
    _ensure_ragas_importable()
    from ragas import evaluate, EvaluationDataset
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["question"],
            "response": prepare_for_judging(r["answer"]),
            "retrieved_contexts": r["contexts"],
        }
        for r in scored
    ])
    result = evaluate(
        dataset,
        metrics=[Faithfulness(), ResponseRelevancy(strictness=RELEVANCY_STRICTNESS)],
        llm=llm,
        embeddings=embeddings,
        # One request at a time + few retries: avoids per-minute rate limits and
        # avoids burning daily quota on retry storms.
        run_config=RunConfig(max_workers=1, max_retries=2, timeout=180),
        show_progress=True,
        raise_exceptions=False,  # failed calls come back as NaN; we check for that below
    )
    return result.to_pandas()


def summarize(records, scored, df):
    """Print the summary. Returns True only if every score was actually computed."""
    metrics = ["faithfulness", "answer_relevancy"]
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Questions asked           : {len(records)}")
    print(f"Grounded (cited) answers  : {len(scored)} of {len(records)}")
    if len(scored) < len(records):
        print("  Answered from general knowledge (not scored): "
              + "; ".join(r["question"] for r in records if not r["grounded"]))
    retrieval = sorted(r["retrieval_time_ms"] for r in records)
    print(f"Retrieval time (ms)       : median {retrieval[len(retrieval) // 2]:.0f}, max {retrieval[-1]:.0f}")
    print(f"Mean pipeline confidence  : {sum(r['confidence'] for r in records) / len(records):.3f}")

    complete = True
    if df is not None and len(df):
        print("\nPer-question scores:")
        table = df[["user_input"] + metrics].copy()
        table["user_input"] = table["user_input"].str.slice(0, 48)
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print()
        for m in metrics:
            failed = int(df[m].isna().sum())
            if failed:
                complete = False
                print(f"!! {m}: {failed} of {len(df)} scores FAILED (NaN). Most likely the Gemini "
                      f"daily/per-minute quota. This metric's average is NOT reliable; do not quote it.")
            mean = df[m].mean()
            shown = "n/a" if mean != mean else f"{mean:.3f}"
            print(f"Mean {m:<17}: {shown}  (n={len(df) - failed} scored)")
        if not complete:
            print("\nRe-run after your quota resets with:  python scripts/run_ragas_eval.py --judge-only")
        print(f"\nn={len(df)} is a smoke test, not a benchmark. Judge = same model family as the generator.")
    else:
        complete = False
        print("\nNothing to score: no answer was grounded in retrieved sources.")
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description="RAGAS baseline for MedSource")
    parser.add_argument("--judge-only", action="store_true",
                        help="reuse answers from ragas_cache.json (spends no pipeline requests)")
    parser.add_argument("--judge-model", default=None,
                        help="Gemini model to use as judge (default: LLM_MODEL from .env)")
    args = parser.parse_args(argv)

    if args.judge_only:
        if not CACHE_PATH.exists():
            print(f"No {CACHE_PATH} found. Run once without --judge-only first.")
            return 1
        from src.config import get_settings
        settings = get_settings()
        records = json.loads(CACHE_PATH.read_text(encoding="utf-8"))["records"]
        print(f"Loaded {len(records)} cached answers from {CACHE_PATH} (no pipeline requests used).")
    else:
        n = len(TEST_QUESTIONS)
        print(f"Estimated Gemini requests: up to {n * (1 + 2 + RELEVANCY_STRICTNESS)} "
              f"(free tier allows 1,500/day; remember anything you already used today).")
        print("Loading vector store and RAG pipeline...")
        settings, pipeline = build_pipeline()
        records = collect_answers(settings, pipeline, TEST_QUESTIONS)
        CACHE_PATH.write_text(json.dumps(
            {"created": datetime.now().isoformat(timespec="seconds"), "records": records},
            indent=2), encoding="utf-8")
        print(f"\nSaved answers to {CACHE_PATH} (safe to re-judge with --judge-only).")

    scored = [r for r in records if r["grounded"]]
    df = None
    if scored:
        print("\nJudging with RAGAS (one request at a time; this can take a few minutes)...")
        llm, embeddings = build_judge(settings, args.judge_model or settings.llm_model)
        df = run_judge(scored, llm, embeddings)
        df["confidence"] = [r["confidence"] for r in scored]
        df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved per-question results to {RESULTS_PATH}")

    complete = summarize(records, scored, df)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())

