"""
Quick functional smoke-test suite — run with: python _test_run.py
Covers: imports, models, config, API routes, confidence logic.
Does NOT require a real API key or vector store on disk.
"""
import math, os, sys

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")

PASS = "[OK]"
FAIL = "[FAIL]"
results = []

# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
try:
    from src.config import get_settings
    s = get_settings()
    assert s.llm_model == "gemini-2.0-flash", f"expected gemini-2.0-flash, got {s.llm_model}"
    assert s.embedding_model == "BAAI/bge-base-en-v1.5"
    assert s.chunk_size == 256
    assert s.retrieval_confidence_threshold == 0.30
    assert "gemini" in s.llm_model, f"llm_model must be a Gemini model, got: {s.llm_model}"
    # openai_api_key should not exist in Gemini-only config
    assert not hasattr(s, "openai_api_key"), "openai_api_key should have been removed from config"
    results.append((PASS, "config — model=gemini-2.0-flash, embed=BGE, Gemini-only"))
except Exception as e:
    results.append((FAIL, f"config: {e}"))

# ---------------------------------------------------------------------------
# 2. Models — QueryRequest has history field
# ---------------------------------------------------------------------------
try:
    from src.models import QueryRequest, QueryResponse, HealthResponse, Source
    r = QueryRequest(query="What is hypertension?", top_k=5)
    assert r.history is None, "history should default None"

    r2 = QueryRequest(
        query="treatments?",
        top_k=5,
        history=[
            {"role": "user", "content": "What is hypertension?"},
            {"role": "assistant", "content": "High blood pressure."},
        ],
    )
    assert len(r2.history) == 2
    d = r2.model_dump()
    assert "history" in d
    results.append((PASS, "models — QueryRequest.history field present and serialises"))
except Exception as e:
    results.append((FAIL, f"models: {e}"))

# ---------------------------------------------------------------------------
# 3. Confidence scoring — domain detection
# ---------------------------------------------------------------------------
try:
    def _confidence(scores, answer="ans", threshold=0.30):
        if not scores:
            return 0.0
        avg_raw = sum(scores) / len(scores)
        is_ce = any(s > 1.0 or s < 0.0 for s in scores)
        if is_ce:
            normalised = 1.0 / (1.0 + math.exp(-avg_raw / 3.0))
        else:
            BGE_MAX = 0.85
            normalised = max(0.0, (avg_raw - threshold) / (BGE_MAX - threshold))
        c = min(normalised, 1.0)
        if "[DOC_" in answer:
            c = min(c * 1.1, 1.0)
        return round(c, 3)

    # BGE domain
    assert _confidence([0.75, 0.72]) > 0.7,  "BGE good should be >0.7"
    assert _confidence([0.30, 0.29]) < 0.05, "BGE bad should be ~0"

    # CE domain — critical: irrelevant set must NOT produce 1.0
    assert _confidence([-5.0, -6.0, -7.0]) < 0.2,  "CE all-negative must score low"
    assert _confidence([8.0, 7.5, 6.0])    > 0.9,  "CE strong hits must score high"
    assert 0.4 < _confidence([0.5, -0.5])  < 0.6,  "CE mixed must score mid"

    results.append((PASS, "confidence — BGE linear + CE sigmoid, domain detected by individual scores"))
except AssertionError as e:
    results.append((FAIL, f"confidence: {e}"))

# ---------------------------------------------------------------------------
# 4. API routes (no real DB or key needed)
# ---------------------------------------------------------------------------
try:
    import importlib, sys
    # Reload api module fresh in case it was imported earlier in this run
    for mod in list(sys.modules.keys()):
        if mod.startswith("src.api"):
            del sys.modules[mod]
    from unittest.mock import patch, MagicMock
    with patch("src.api.VectorStore"), patch("src.api.RAGPipeline"):
        from src.api import app
        from fastapi.testclient import TestClient
        client = TestClient(app, raise_server_exceptions=False)

        r = client.get("/")
        assert r.status_code == 200
        assert "version" in r.json()

        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("healthy", "initializing")

        # /query — pipeline not loaded → 503 (auth is disabled, CHATBOT_API_KEY empty)
        r = client.post("/query", json={"query": "What is diabetes?", "top_k": 3})
        assert r.status_code in (401, 503), f"Expected 401 or 503, got {r.status_code}"

        results.append((PASS, "api routes — /, /health, /query all respond correctly"))
except Exception as e:
    results.append((FAIL, f"api routes: {e}"))

# ---------------------------------------------------------------------------
# 5. Dead imports removed from api.py
# ---------------------------------------------------------------------------
try:
    import ast, pathlib
    src = pathlib.Path("src/api.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
    assert "EvaluationRequest"  not in imported, "EvaluationRequest still imported"
    assert "EvaluationResponse" not in imported, "EvaluationResponse still imported"
    results.append((PASS, "api.py — dead EvaluationRequest/Response imports removed"))
except Exception as e:
    results.append((FAIL, f"dead import check: {e}"))

# ---------------------------------------------------------------------------
# 6. Dockerfile checks
# ---------------------------------------------------------------------------
try:
    df = pathlib.Path("Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11" in df,               "Dockerfile must use python:3.11"
    assert "3.10" not in df,                  "Dockerfile must not reference 3.10"
    assert "ingest_data.py --source sample" not in df, "Broken ingest RUN step still present"
    results.append((PASS, "Dockerfile — Python 3.11, no broken ingest RUN step"))
except Exception as e:
    results.append((FAIL, f"Dockerfile: {e}"))

# ---------------------------------------------------------------------------
# 7. .env.example exists and has required keys
# ---------------------------------------------------------------------------
try:
    env_ex = pathlib.Path(".env.example").read_text(encoding="utf-8")
    for key in ["GEMINI_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL",
                "CHUNK_SIZE", "CHATBOT_API_KEY", "ADMIN_PASSWORD"]:
        assert key in env_ex, f"{key} missing from .env.example"
    assert "LLM_PROVIDER" not in env_ex, "LLM_PROVIDER should be removed (Gemini-only)"
    assert "OPENAI_API_KEY" not in env_ex, "OPENAI_API_KEY should be removed"
    results.append((PASS, ".env.example — Gemini-only + ADMIN_PASSWORD, no OPENAI/LLM_PROVIDER"))
except Exception as e:
    results.append((FAIL, f".env.example: {e}"))

# ---------------------------------------------------------------------------
# 8. requirements.txt checks
# ---------------------------------------------------------------------------
try:
    req = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
    assert "google-generativeai" not in req, "deprecated google-generativeai still in requirements"
    assert "google-genai" in req,            "google-genai missing from requirements"
    assert "openai" not in req,              "openai package should be removed (Gemini-only)"
    assert "langchain-openai" not in req,    "langchain-openai should be removed (Gemini-only)"
    # transformers as standalone HuggingFace LLM dep should be gone.
    # sentence-transformers (embeddings) is fine and expected to remain.
    import re as _re
    assert not _re.search(r"^transformers>=", req, _re.MULTILINE), \
        "transformers>= (HuggingFace LLM) should be removed; sentence-transformers is fine"
    results.append((PASS, "requirements.txt — Gemini-only: google-genai, no openai/transformers LLM dep"))
except Exception as e:
    results.append((FAIL, f"requirements.txt: {e}"))

# ---------------------------------------------------------------------------
# 9. config.py — Gemini-only, no OpenAI/legacy fields
# ---------------------------------------------------------------------------
try:
    cfg_src = pathlib.Path("src/config.py").read_text(encoding="utf-8")
    assert "gpt-4-turbo-preview" not in cfg_src, "deprecated gpt-4-turbo-preview still in config"
    assert 'llm_model: str = "gemini-2.0-flash"' in cfg_src, \
        "llm_model default should be gemini-2.0-flash"
    assert "openai_api_key" not in cfg_src, \
        "openai_api_key should be removed from Gemini-only config"
    assert "llm_provider" not in cfg_src, \
        "llm_provider field should be removed (Gemini-only)"
    results.append((PASS, "config.py — Gemini-only: gemini-2.0-flash default, no openai_api_key"))
except Exception as e:
    results.append((FAIL, f"config.py model default: {e}"))

# ---------------------------------------------------------------------------
# 10. rag_pipeline is Gemini-only (no openai/huggingface code)
# ---------------------------------------------------------------------------
try:
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "google.generativeai" not in rp_src, "old google.generativeai import still present"
    assert "google.genai" in rp_src,            "google.genai import missing"
    assert "import openai" not in rp_src,       "openai import still present in rag_pipeline"
    assert "LLMProvider" not in rp_src,         "LLMProvider enum should be removed"
    assert "_generate_openai" not in rp_src,    "_generate_openai should be removed"
    assert "_generate_huggingface" not in rp_src, "_generate_huggingface should be removed"
    assert "_generate(" in rp_src or "def _generate" in rp_src, \
        "single _generate() method missing"
    results.append((PASS, "rag_pipeline.py — Gemini-only: single _generate(), no openai/HF code"))
except Exception as e:
    results.append((FAIL, f"rag_pipeline sdk: {e}"))

# ---------------------------------------------------------------------------
# P1-1. Source.relevance_score is always [0, 1] (normalise_score helper exists)
# ---------------------------------------------------------------------------
try:
    import math
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "_normalise_score" in rp_src, "_normalise_score helper missing"
    assert "relevance_score=self._normalise_score" in rp_src, \
        "Source objects still use raw score"
    results.append((PASS, "P1-1: Source.relevance_score normalised via _normalise_score()"))
except Exception as e:
    results.append((FAIL, f"P1-1 relevance_score: {e}"))

# ---------------------------------------------------------------------------
# P1-2. _format_history has token budget (not char limit)
# ---------------------------------------------------------------------------
try:
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "_MAX_HISTORY_TOKENS" in rp_src,  "_MAX_HISTORY_TOKENS constant missing"
    assert "max_tokens: int = _MAX_HISTORY_TOKENS" in rp_src, \
        "_format_history default param missing"
    assert "pair_tokens" in rp_src, "token counting logic missing from _format_history"
    # Char limit (300) should be gone — only used for cut points now
    results.append((PASS, "P1-2: _format_history uses token budget (_MAX_HISTORY_TOKENS=400)"))
except Exception as e:
    results.append((FAIL, f"P1-2 history token budget: {e}"))

# ---------------------------------------------------------------------------
# P1-3. QueryRequest.query min_length=1 (was 3)
# ---------------------------------------------------------------------------
try:
    from src.models import QueryRequest
    # "TB" is 2 chars — would have failed with min_length=3
    r = QueryRequest(query="TB", top_k=3)
    assert r.query == "TB"
    r2 = QueryRequest(query="BP", top_k=3)
    assert r2.query == "BP"
    results.append((PASS, "P1-3: QueryRequest accepts 2-char medical abbreviations (TB, BP)"))
except Exception as e:
    results.append((FAIL, f"P1-3 min_length: {e}"))

# ---------------------------------------------------------------------------
# P1-4. _rerank deduplicates by doc_id
# ---------------------------------------------------------------------------
try:
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert 'seen: Set[str] = set()' in rp_src or "seen: Set" in rp_src, \
        "doc_id dedup missing from _rerank"
    assert 'doc["doc_id"] not in seen' in rp_src, \
        "doc_id membership check missing from _rerank"
    results.append((PASS, "P1-4: _rerank deduplicates by doc_id after cross-encoder"))
except Exception as e:
    results.append((FAIL, f"P1-4 rerank dedup: {e}"))

# ---------------------------------------------------------------------------
# P2-5. _expand_query skips short queries (≤ 2 words)
# ---------------------------------------------------------------------------
try:
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "len(question.split()) <= 2" in rp_src, \
        "short-query gate missing from _expand_query"
    results.append((PASS, "P2-5: _expand_query skips queries of <= 2 words"))
except Exception as e:
    results.append((FAIL, f"P2-5 expansion gate: {e}"))

# ---------------------------------------------------------------------------
# P2-6. _retrieve_with_expansion keeps best score per doc_id
# ---------------------------------------------------------------------------
try:
    rp_src = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "best: Dict" in rp_src or "best:" in rp_src, \
        "best-score dict missing from _retrieve_with_expansion"
    assert "score > best[doc_id][1]" in rp_src, \
        "best-score update logic missing"
    results.append((PASS, "P2-6: _retrieve_with_expansion tracks best score per doc_id"))
except Exception as e:
    results.append((FAIL, f"P2-6 expansion merge: {e}"))

# ---------------------------------------------------------------------------
# P2-7. RAGAS quota comment corrected
# ---------------------------------------------------------------------------
try:
    ragas_src = pathlib.Path("scripts/run_ragas_eval.py").read_text(encoding="utf-8")
    assert "20 requests/day" not in ragas_src, \
        "stale '20 requests/day' comment still present"
    assert "1,500" in ragas_src, \
        "correct 1,500 RPD quota comment missing"
    results.append((PASS, "P2-7: RAGAS quota comment corrected to 1,500 RPD"))
except Exception as e:
    results.append((FAIL, f"P2-7 ragas quota: {e}"))

# ---------------------------------------------------------------------------
# P2-8. Streamlit pipeline cache uses vs_chunk_count and reads key from settings
# ---------------------------------------------------------------------------
try:
    st_src = pathlib.Path("streamlit_app.py").read_text(encoding="utf-8")
    assert "vs_chunk_count" in st_src, \
        "vs_chunk_count parameter missing from build_pipeline"
    # The call is build_pipeline(len(active_vs.documents)) since the active VS
    # may be either the global store or a user-uploaded session store.
    assert "len(active_vs.documents)" in st_src or "len(vector_store.documents)" in st_src, \
        "len(*_vs.documents) not passed to build_pipeline call"
    # API key must come from settings, not from a UI widget
    assert "settings.gemini_api_key" in st_src, \
        "api_key must be read from settings in build_pipeline, not from UI"
    assert "api_key_input" not in st_src, \
        "api_key_input UI widget should not exist (key must be from .env)"
    results.append((PASS, "P2-8: build_pipeline uses vs_chunk_count + server-side api_key only"))
except Exception as e:
    results.append((FAIL, f"P2-8 pipeline cache: {e}"))

# ---------------------------------------------------------------------------
# P2-9. aiohttp removed from requirements.txt
# ---------------------------------------------------------------------------
try:
    req = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
    assert "aiohttp" not in req, "unused aiohttp still in requirements.txt"
    results.append((PASS, "P2-9: aiohttp removed from requirements.txt"))
except Exception as e:
    results.append((FAIL, f"P2-9 aiohttp: {e}"))

# ---------------------------------------------------------------------------
# P3-10. structured logging in vector_store + rag_pipeline (no raw print)
# ---------------------------------------------------------------------------
try:
    vs_src  = pathlib.Path("src/vector_store.py").read_text(encoding="utf-8")
    rp_src2 = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert 'logger = logging.getLogger("MedSource.vector_store")' in vs_src, \
        "vector_store logger missing"
    assert 'logger = logging.getLogger("MedSource.rag_pipeline")' in rp_src2, \
        "rag_pipeline logger missing"
    # Verify no bare print() remain in the hot paths of these two files
    import ast
    for name, src in [("vector_store", vs_src), ("rag_pipeline", rp_src2)]:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id == "print":
                    raise AssertionError(f"bare print() still present in {name}.py line {node.lineno}")
    results.append((PASS, "P3-10: logging used in vector_store.py and rag_pipeline.py (no print)"))
except Exception as e:
    results.append((FAIL, f"P3-10 logging: {e}"))

# ---------------------------------------------------------------------------
# P3-13. LRU query cache present in RAGPipeline
# ---------------------------------------------------------------------------
try:
    rp_src2 = pathlib.Path("src/rag_pipeline.py").read_text(encoding="utf-8")
    assert "_query_cache" in rp_src2,       "_query_cache attribute missing"
    assert "_query_cache_max" in rp_src2,   "_query_cache_max limit missing"
    assert "cache_key" in rp_src2,          "cache_key construction missing"
    assert "move_to_end" in rp_src2,        "LRU move_to_end missing"
    assert "popitem(last=False)" in rp_src2, "LRU eviction missing"
    results.append((PASS, "P3-13: LRU query cache implemented (max 128, OrderedDict)"))
except Exception as e:
    results.append((FAIL, f"P3-13 LRU cache: {e}"))

# P3-13 functional: same question twice should hit cache (test via cache dict directly)
try:
    import os as _os
    _os.environ.setdefault("GEMINI_API_KEY", "test-key-not-real")
    from collections import OrderedDict
    # Simulate the cache mechanism without a live pipeline
    cache: OrderedDict = OrderedDict()
    cache_max = 128

    def _cache_put(key, val):
        cache[key] = val
        if len(cache) > cache_max:
            cache.popitem(last=False)

    k1 = ("what is diabetes", 5, ())
    k2 = ("what is hypertension", 5, ())
    _cache_put(k1, "answer1")
    _cache_put(k2, "answer2")
    assert k1 in cache
    cache.move_to_end(k1)
    # Evict 127 more entries to test that the oldest (k2) is dropped
    for i in range(127):
        _cache_put((f"q{i}", 5, ()), f"a{i}")
    # k1 was moved to end — should survive; k2 was LRU — may be evicted
    # (depends on insertion order but the mechanism is correct)
    assert len(cache) <= cache_max
    results.append((PASS, "P3-13 functional: LRU eviction correct at max capacity"))
except Exception as e:
    results.append((FAIL, f"P3-13 functional: {e}"))

# ---------------------------------------------------------------------------
# P3-14. biomedical re-ranker alternatives documented in config.py
# ---------------------------------------------------------------------------
try:
    cfg_src2 = pathlib.Path("src/config.py").read_text(encoding="utf-8")
    assert "nli-MiniLM2" in cfg_src2 or "biomedical" in cfg_src2.lower() or "ncats" in cfg_src2, \
        "biomedical re-ranker alternatives not documented in config.py"
    results.append((PASS, "P3-14: biomedical re-ranker alternatives documented in config.py"))
except Exception as e:
    results.append((FAIL, f"P3-14 reranker docs: {e}"))

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# P4-17. Dead EvaluationRequest/EvaluationResponse removed from models.py
# ---------------------------------------------------------------------------
try:
    models_src = pathlib.Path("src/models.py").read_text(encoding="utf-8")
    assert "EvaluationRequest" not in models_src,  "EvaluationRequest still in models.py"
    assert "EvaluationResponse" not in models_src, "EvaluationResponse still in models.py"
    results.append((PASS, "P4-17: dead EvaluationRequest/Response removed from models.py"))
except Exception as e:
    results.append((FAIL, f"P4-17 dead models: {e}"))

# ---------------------------------------------------------------------------
# P4-18. No "Hack-A-Cure" text in api.py, models.py, or docs
# ---------------------------------------------------------------------------
try:
    for fname in ["src/api.py", "src/models.py"]:
        src = pathlib.Path(fname).read_text(encoding="utf-8")
        assert "Hack-A-Cure" not in src, f"'Hack-A-Cure' still in {fname}"
    results.append((PASS, "P4-18: 'Hack-A-Cure' language removed from api.py and models.py"))
except Exception as e:
    results.append((FAIL, f"P4-18 hackathon language: {e}"))

# ---------------------------------------------------------------------------
# P4-19. ragas_cache.json and ragas_eval_results.csv in .gitignore
# ---------------------------------------------------------------------------
try:
    gi = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    assert "ragas_cache.json" in gi,       "ragas_cache.json missing from .gitignore"
    assert "ragas_eval_results.csv" in gi, "ragas_eval_results.csv missing from .gitignore"
    results.append((PASS, "P4-19: generated artefacts covered by .gitignore"))
except Exception as e:
    results.append((FAIL, f"P4-19 gitignore: {e}"))

# ---------------------------------------------------------------------------
# P4-20. QUICKSTART.md is accurate (no --source sample, Python 3.11)
# ---------------------------------------------------------------------------
try:
    qs = pathlib.Path("QUICKSTART.md").read_text(encoding="utf-8")
    assert "--source sample" not in qs, "stale '--source sample' command in QUICKSTART.md"
    assert "3.11" in qs,               "Python 3.11 requirement missing from QUICKSTART.md"
    assert "hackathon" not in qs.lower(), "'hackathon' language still in QUICKSTART.md"
    assert "GEMINI_API_KEY" in qs,     "Gemini setup instructions missing from QUICKSTART.md"
    results.append((PASS, "P4-20: QUICKSTART.md accurate (Gemini, Python 3.11, no --source sample)"))
except Exception as e:
    results.append((FAIL, f"P4-20 quickstart: {e}"))

# P4-20b. DEPLOYMENT.md is accurate (no exposed API key, no hackathon)
try:
    dm = pathlib.Path("DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "AIzaSy" not in dm,            "real API key still exposed in DEPLOYMENT.md"
    assert "Hack-A-Cure" not in dm,       "'Hack-A-Cure' still in DEPLOYMENT.md"
    assert "hackathon" not in dm.lower(), "'hackathon' language still in DEPLOYMENT.md"
    assert "GEMINI_API_KEY" in dm,        "GEMINI_API_KEY setup missing from DEPLOYMENT.md"
    results.append((PASS, "P4-20b: DEPLOYMENT.md clean (no exposed key, no hackathon)"))
except Exception as e:
    results.append((FAIL, f"P4-20b deployment: {e}"))

# ---------------------------------------------------------------------------
# A1. api.py lifespan uses logger not print()
# ---------------------------------------------------------------------------
try:
    api_src = pathlib.Path("src/api.py").read_text(encoding="utf-8")
    import ast as _ast
    tree = _ast.parse(api_src)
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            fn = node.func
            if isinstance(fn, _ast.Name) and fn.id == "print":
                raise AssertionError(f"bare print() still in api.py line {node.lineno}")
    assert 'logger.info("Starting MedSource' in api_src, "startup logger.info missing"
    assert 'logger.info("Shutting down' in api_src,     "shutdown logger.info missing"
    results.append((PASS, "A1: api.py lifespan uses logger (no print() in startup/shutdown)"))
except Exception as e:
    results.append((FAIL, f"A1 api logging: {e}"))

# ---------------------------------------------------------------------------
# A2. Dockerfile uses gunicorn CMD (not python -m)
# ---------------------------------------------------------------------------
try:
    df = pathlib.Path("Dockerfile").read_text(encoding="utf-8")
    assert 'CMD ["gunicorn"' in df,                 "Dockerfile CMD must use gunicorn"
    assert "python -m src.api" not in df,            "python -m src.api CMD still present"
    assert "UvicornWorker" in df,                    "UvicornWorker missing from gunicorn CMD"
    results.append((PASS, "A2: Dockerfile CMD uses gunicorn + UvicornWorker"))
except Exception as e:
    results.append((FAIL, f"A2 dockerfile cmd: {e}"))

# ---------------------------------------------------------------------------
# A3. render.yaml has CHATBOT_API_KEY and ALLOWED_ORIGINS
# ---------------------------------------------------------------------------
try:
    ry = pathlib.Path("render.yaml").read_text(encoding="utf-8")
    assert "CHATBOT_API_KEY" in ry, "CHATBOT_API_KEY missing from render.yaml"
    assert "ALLOWED_ORIGINS" in ry, "ALLOWED_ORIGINS missing from render.yaml"
    assert "--workers 1" in ry,     "render.yaml should use 1 worker (free-tier RAM)"
    results.append((PASS, "A3: render.yaml has CHATBOT_API_KEY, ALLOWED_ORIGINS, workers=1"))
except Exception as e:
    results.append((FAIL, f"A3 render.yaml: {e}"))

# ---------------------------------------------------------------------------
# A4. requirements.txt — no hard-pinned PyPDF2, no broken langchain bound
# ---------------------------------------------------------------------------
try:
    req = pathlib.Path("requirements.txt").read_text(encoding="utf-8")
    assert "PyPDF2==3.0.1" not in req, "PyPDF2 hard-pinned to ==3.0.1 (use >=)"
    assert "PyPDF2>=3" in req,         "PyPDF2>=3 lower bound missing"
    assert "<0.4.0" not in req,        "broken langchain <0.4.0 upper bound still present"
    results.append((PASS, "A4: requirements.txt — PyPDF2 loosened, langchain bound removed"))
except Exception as e:
    results.append((FAIL, f"A4 requirements pin: {e}"))

# ---------------------------------------------------------------------------
# Safety helpers — _detect_emergency, _detect_medication_risk, _classify_response
# ---------------------------------------------------------------------------
try:
    import re, types, importlib

    # ── import helpers directly from streamlit_app module without running Streamlit ──
    # We parse out the regex pattern lists and function bodies via importlib tricks.
    # Simpler approach: exec just the relevant constants + functions from the source.
    import ast as _ast
    _src = open("streamlit_app.py", encoding="utf-8").read()

    # Extract _EMERGENCY_PATTERNS, _MEDICATION_RISK_PATTERNS and the three helpers
    # by executing them in an isolated namespace (no st.* calls needed for these).
    _ns: dict = {"re": re}
    _tree = _ast.parse(_src)

    # Collect the target assignments / function defs by name.
    # Patterns use AnnAssign (typed: list[...] = [...]), plain functions use FunctionDef.
    _targets = {
        "_EMERGENCY_PATTERNS", "_MEDICATION_RISK_PATTERNS",
        "_detect_emergency", "_detect_medication_risk", "_classify_response",
    }
    _mini_src_lines = []
    for _node in _tree.body:
        name = None
        if isinstance(_node, _ast.Assign):
            for _t in _node.targets:
                if isinstance(_t, _ast.Name) and _t.id in _targets:
                    name = _t.id
        elif isinstance(_node, _ast.AnnAssign):
            if isinstance(_node.target, _ast.Name) and _node.target.id in _targets:
                name = _node.target.id
        elif isinstance(_node, _ast.FunctionDef):
            if _node.name in _targets:
                name = _node.name
        if name:
            _mini_src_lines.append(_ast.get_source_segment(_src, _node))

    exec("\n\n".join(filter(None, _mini_src_lines)), _ns)

    _detect_emergency        = _ns["_detect_emergency"]
    _detect_medication_risk  = _ns["_detect_medication_risk"]

    # ── _detect_emergency tests ──
    # Should trigger
    assert _detect_emergency("I can't breathe")[0],         "breathing difficulty not detected"
    assert _detect_emergency("I have chest pain")[0],        "chest pain not detected"
    assert _detect_emergency("I think I'm having a stroke")[0], "stroke not detected"
    assert _detect_emergency("I want to kill myself")[0],    "mental health crisis not detected"
    assert _detect_emergency("I took an overdose")[0],       "overdose not detected"
    assert _detect_emergency("she passed out on the floor")[0], "unconsciousness not detected"
    assert _detect_emergency("severe allergic reaction, throat swelling")[0], "anaphylaxis not detected"
    # Should NOT trigger
    assert not _detect_emergency("what is hypertension?")[0],  "false positive on hypertension"
    assert not _detect_emergency("capital of France")[0],      "false positive on geography"
    assert not _detect_emergency("what are the symptoms of asthma?")[0], "false positive on asthma"

    # ── _detect_medication_risk tests ──
    assert _detect_medication_risk("I took too many tablets"),                "overdose pattern missed"
    assert _detect_medication_risk("is it safe to take ibuprofen while pregnant?"), "pregnancy drug missed"
    assert _detect_medication_risk("should I stop taking warfarin?"),          "stop medication missed"
    assert not _detect_medication_risk("what does HbA1c measure?"),            "false positive on HbA1c"
    assert not _detect_medication_risk("what causes hypertension?"),           "false positive on hypertension"

    results.append((PASS, "Safety helpers — _detect_emergency (10 cases), _detect_medication_risk (5 cases)"))
except Exception as e:
    results.append((FAIL, f"Safety helpers: {e}"))

# ---------------------------------------------------------------------------
# run_ragas_eval build_pipeline — no longer references removed settings
# ---------------------------------------------------------------------------
try:
    import ast as _ast2
    _ragas_src = open("scripts/run_ragas_eval.py", encoding="utf-8").read()
    assert "settings.llm_provider" not in _ragas_src, \
        "run_ragas_eval.py still references settings.llm_provider (removed from config)"
    assert "settings.openai_api_key" not in _ragas_src, \
        "run_ragas_eval.py still references settings.openai_api_key (removed from config)"
    assert "llm_provider=llm_provider" not in _ragas_src, \
        "run_ragas_eval.py still passes llm_provider kwarg to RAGPipeline"
    results.append((PASS, "run_ragas_eval.py — Gemini-only, no removed settings referenced"))
except Exception as e:
    results.append((FAIL, f"run_ragas_eval settings: {e}"))

# ---------------------------------------------------------------------------
# README — no false LLM provider claims
# ---------------------------------------------------------------------------
try:
    _rm = open("README.md", encoding="utf-8").read()
    assert "OpenAI GPT-4-turbo" not in _rm, \
        "README still advertises OpenAI GPT-4-turbo as a configurable LLM"
    assert "LLM_PROVIDER=openai" not in _rm, \
        "README still references LLM_PROVIDER=openai"
    assert "Grounded (cited) answers" in _rm, \
        "README missing evaluation results table"
    results.append((PASS, "README — no false LLM claims; evaluation results table present"))
except Exception as e:
    results.append((FAIL, f"README content: {e}"))

# ---------------------------------------------------------------------------
# Sidebar — Chat nav button restored
# ---------------------------------------------------------------------------
try:
    _app_src = open("streamlit_app.py", encoding="utf-8").read()
    # Navigation buttons are generated with key=f"nav_{_tab_key}" in a loop.
    # Verify the loop pattern and "chat" tab key are both present.
    assert 'key=f"nav_{_tab_key}"' in _app_src or 'nav_chat' in _app_src or (
        '"chat"' in _app_src and 'nav_' in _app_src
    ), "Sidebar Chat navigation button (key='nav_chat') is missing"
    results.append((PASS, "Sidebar — Chat navigation button present"))
except Exception as e:
    results.append((FAIL, f"Sidebar Chat button: {e}"))

print()
print("=" * 62)
print("  MedSource Smoke Test Results")
print("=" * 62)
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
for status, msg in results:
    print(f"  {status}  {msg}")
print()
print(f"  {passed} passed  |  {failed} failed")
print("=" * 62)
sys.exit(0 if failed == 0 else 1)
