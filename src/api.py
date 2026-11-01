"""
FastAPI backend for MedSource Medical RAG Chatbot (Gemini-only)
"""
import os
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.config import get_settings
from src.models import (
    QueryRequest,
    QueryResponse,
    HealthResponse,
)
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

# Global variables
logger = logging.getLogger("MedSource.api")
logging.basicConfig(level=logging.INFO)

vector_store: Optional[VectorStore] = None
rag_pipeline: Optional[RAGPipeline] = None
settings = get_settings()

# --- API key auth ---
# Clients must send this app's key in the X-API-Key header.
# This is separate from OPENAI_API_KEY / GEMINI_API_KEY, which are for
# talking to the LLM provider, not for authenticating callers of this API.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(provided_key: Optional[str] = Security(api_key_header)):
    """Reject requests without a valid X-API-Key header.

    If CHATBOT_API_KEY is unset in .env, auth is skipped (local dev only).
    """
    if not settings.chatbot_api_key:
        return  # auth disabled — only acceptable for local development
    if not provided_key or provided_key != settings.chatbot_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Send it in the X-API-Key header.",
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global vector_store, rag_pipeline

    logger.info("Starting MedSource Medical RAG Chatbot")

    try:
        # Initialize vector store
        logger.info("Loading vector store...")
        vector_store = VectorStore(
            embedding_model_name=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            embedding_batch_size=settings.embedding_batch_size
        )

        # Try to load existing vector store
        if os.path.exists(settings.vector_store_path):
            vector_store.load(settings.vector_store_path)
            stats = vector_store.get_stats()
            logger.info(
                "Vector store loaded: %d chunks, %d unique documents",
                stats["total_chunks"], stats["unique_documents"],
            )
        else:
            logger.warning(
                "No existing vector store found at %s. "
                "Please run the ingestion script first.",
                settings.vector_store_path,
            )

        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")

        model_name = settings.llm_model or "gemini-2.0-flash"
        api_key = settings.gemini_api_key
        logger.info("LLM: Google Gemini (%s)", model_name)

        rag_pipeline = RAGPipeline(
            vector_store=vector_store,
            model_name=model_name,
            api_key=api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.max_tokens,
            top_k=settings.top_k_documents,
            confidence_threshold=settings.retrieval_confidence_threshold,
            enable_general_fallback=settings.enable_general_fallback,
            enable_query_expansion=settings.enable_query_expansion,
            enable_reranker=settings.enable_reranker,
            reranker_model=settings.reranker_model,
            enable_live_pubmed=settings.enable_live_pubmed,
            pubmed_email=settings.pubmed_email,
            ncbi_api_key=settings.ncbi_api_key,
            live_pubmed_timeout_s=settings.live_pubmed_timeout_s,
            live_pubmed_max_results=settings.live_pubmed_max_results,
        )

        logger.info(
            "RAG pipeline ready — API serving at http://%s:%s",
            settings.api_host, settings.api_port,
        )

    except Exception as e:
        logger.exception("Error during initialization: %s — API will start but may not function correctly.", e)

    yield

    # Cleanup
    logger.info("Shutting down MedSource...")

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A medical Q&A system using RAG that provides accurate, explainable, and citation-backed answers",
    lifespan=lifespan
)

# Rate limiting — per client IP; tune via settings if needed.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — set ALLOWED_ORIGINS in .env as comma-separated frontend URLs.
allowed_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MedSource Medical RAG Chatbot API",
        "version": settings.app_version,
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if vector_store and rag_pipeline else "initializing",
        app_name=settings.app_name,
        version=settings.app_version,
        vector_store_loaded=vector_store is not None and len(vector_store.documents) > 0
    )

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def query_endpoint(request: Request, query_request: QueryRequest):
    """
    Answer a medical question using Retrieval-Augmented Generation.

    Retrieves the most relevant passages from the indexed document corpus,
    re-ranks them (if enabled), and generates a grounded answer with inline
    `[DOC_X]` citations. When no verified source passes the confidence
    threshold the answer is clearly labeled as general knowledge.

    **Request fields**
    - `query`: the medical question (required, min 1 character)
    - `top_k`: number of context passages to retrieve (1–20, default 5)
    - `history`: optional prior conversation turns for multi-turn dialogue

    **Response fields**
    - `answer`: generated answer with inline `[DOC_X]` citations
    - `contexts`: retrieved passage texts (plain strings, for RAGAS evaluation)
    - `sources`: structured source metadata with relevance scores
    - `confidence`: overall retrieval confidence (0–1)
    - `warning`: medical disclaimer or low-confidence notice
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not initialized. Please check server logs."
        )
    
    if not vector_store or len(vector_store.documents) == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Please run data ingestion first."
        )
    
    try:
        # Support both 'query' (competition format) and 'question' (legacy)
        question = query_request.query if query_request.query else query_request.question
        
        # Call RAG pipeline
        response = rag_pipeline.query(
            question=question,
            top_k=query_request.top_k,
            history=query_request.history,
        )
        
        # Extract plain text contexts from sources (full chunk text for RAGAS faithfulness)
        contexts = [source.excerpt for source in response.sources[:query_request.top_k]] if response.sources else []
        
        # Return competition-compliant format with optional extended fields
        return QueryResponse(
            answer=response.answer,
            contexts=contexts,
            question=question,
            sources=response.sources if query_request.include_sources else None,
            confidence=response.confidence,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=response.total_time_ms,
            warning=response.warning
        )
    
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing query. Please try again or contact support if this persists."
        )

@app.get("/stats", response_model=dict, dependencies=[Depends(verify_api_key)])
async def stats_endpoint():
    """Get vector store statistics"""
    if not vector_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized."
        )
    
    try:
        stats = vector_store.get_stats()
        return {
            "vector_store": stats,
            "rag_config": {
                "llm_model": rag_pipeline.model_name if rag_pipeline else "N/A",
                "llm_provider": "gemini",
                "top_k": settings.top_k_documents,
                "confidence_threshold": settings.retrieval_confidence_threshold,
                "temperature": settings.llm_temperature
            }
        }
    except Exception as e:
        logger.exception("Error retrieving stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving stats. Please try again or contact support if this persists."
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again or contact support if this persists."}
    )

def start_server():
    """Start the FastAPI server"""
    # Use PORT from environment (Render) or fallback to settings
    port = int(os.getenv("PORT", settings.api_port))
    
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=port,
        reload=False,
        workers=1  # Use 1 worker for development, increase for production
    )

if __name__ == "__main__":
    start_server()