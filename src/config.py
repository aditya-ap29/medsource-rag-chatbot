"""
Configuration management for MedSource RAG Chatbot (Gemini-only)
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ------------------------------------------------------------------ #
    # LLM — Google Gemini (free tier: 1,500 req/day, no billing needed)  #
    # Get a key at: https://aistudio.google.com/apikey                   #
    # ------------------------------------------------------------------ #
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_temperature: float = 0.1
    max_tokens: int = 1000

    # ------------------------------------------------------------------ #
    # App auth — key callers send in X-API-Key header.                   #
    # Leave blank to disable auth (local dev only).                      #
    # ------------------------------------------------------------------ #
    chatbot_api_key: str = ""

    # Comma-separated list of frontend origins allowed to call this API (CORS).
    allowed_origins: str = "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501"

    # ------------------------------------------------------------------ #
    # Embedding model — do NOT change unless you re-embed the store      #
    # ------------------------------------------------------------------ #
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    # ------------------------------------------------------------------ #
    # Retrieval                                                           #
    # ------------------------------------------------------------------ #
    top_k_documents: int = 5
    retrieval_confidence_threshold: float = 0.30

    # When True: low-confidence queries get a general-knowledge answer (clearly labeled).
    # When False: low-confidence queries always return a hard refusal.
    enable_general_fallback: bool = True

    # When True: each query is first expanded into 3 rephrased variants via one LLM
    # call, then all variants are searched and results are merged. Improves recall for
    # short or ambiguous queries at the cost of one extra Gemini call per question.
    enable_query_expansion: bool = False

    # When True: retrieved candidates are re-ranked with a cross-encoder before the
    # top-k are sent to Gemini. More accurate than cosine similarity at the cost of
    # ~30 ms CPU per query. Model is lazy-loaded on first use (~23 MB).
    enable_reranker: bool = False

    # Cross-encoder model used when enable_reranker=True.
    # Default: MS MARCO MiniLM — fast, general-purpose (~23 MB).
    # Biomedical alternative: "cross-encoder/nli-MiniLM2-L6-H768" (NLI-trained,
    # better on clinical terminology) or "ncats/biobert-reranker" for PubMed queries.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------ #
    # API server                                                          #
    # ------------------------------------------------------------------ #
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ------------------------------------------------------------------ #
    # Vector store / chunking — do NOT change after first ingest         #
    # ------------------------------------------------------------------ #
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 256        # tokens (fits BGE's 512-token window)
    chunk_overlap: int = 40      # tokens
    embedding_batch_size: int = 32

    # ------------------------------------------------------------------ #
    # Semantic domain classifier (NEW — disabled by default)             #
    # When True: replaces the keyword gate as primary classifier.        #
    # Requires: trained model in semantic_classifier_path.               #
    # The keyword gate remains as emergency bypass and fallback.         #
    # ------------------------------------------------------------------ #
    enable_semantic_classifier: bool = False

    # Directory where the trained SemanticClassifier is persisted.
    # Created by: python scripts/train_classifier.py
    semantic_classifier_path: str = "./data/semantic_classifier_v1"

    # Ambiguous classification handling:
    #   "pass"   — let ambiguous questions through to the medical pipeline
    #   "warn"   — pass but add an uncertainty warning
    #   "refuse" — return a clarification prompt (most conservative)
    ambiguous_handling: str = "warn"

    # ------------------------------------------------------------------ #
    # Hybrid retrieval (NEW — disabled by default)                       #
    # When True: fuses BGE/FAISS + BM25 with Reciprocal Rank Fusion.    #
    # Does NOT rebuild the FAISS index.                                  #
    # ------------------------------------------------------------------ #
    enable_hybrid_retrieval: bool = False

    # RRF smoothing constant (Cormack 2009 default = 60)
    rrf_k: int = 60

    # Dense retrieval weight in RRF (relative to sparse)
    rrf_dense_weight: float = 1.0

    # Sparse (BM25) retrieval weight in RRF
    rrf_sparse_weight: float = 1.0

    # BM25 term frequency saturation parameter
    bm25_k1: float = 1.5

    # BM25 length normalisation parameter
    bm25_b: float = 0.75

    # ------------------------------------------------------------------ #
    # Evidence-sufficiency assessment (NEW — disabled by default)        #
    # When True: gates answer generation on evidence quality signals     #
    # beyond the existing confidence threshold.                          #
    # ------------------------------------------------------------------ #
    enable_evidence_assessment: bool = False

    # ------------------------------------------------------------------ #
    # App metadata                                                        #
    # ------------------------------------------------------------------ #
    app_name: str = "MedSource - Medical RAG Chatbot"
    app_version: str = "1.0.0"

    # PubMed ingestion email — sent with NCBI API requests (usage etiquette).
    pubmed_email: str = ""

    # ------------------------------------------------------------------ #
    # Live PubMed retrieval (optional augmentation)                       #
    # When True: queries that fail the confidence gate also search PubMed #
    # live and merge results with local FAISS hits.                       #
    # ------------------------------------------------------------------ #
    enable_live_pubmed: bool = False

    # NCBI API key — raises rate limit from 3 req/s to 10 req/s.
    # Get a free key at: https://www.ncbi.nlm.nih.gov/account/
    ncbi_api_key: str = ""

    # Hard timeout in seconds for live PubMed HTTP calls.
    live_pubmed_timeout_s: int = 5

    # Maximum abstracts to fetch per live query (1–10 recommended to stay fast).
    live_pubmed_max_results: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()