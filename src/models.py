"""
Pydantic models for request/response validation
"""
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Model for citation sources"""
    doc_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title or source")
    year: Optional[str] = Field(None, description="Publication year")
    url: Optional[str] = Field(None, description="URL to source")
    relevance_score: float = Field(..., description="Retrieval confidence score")
    excerpt: str = Field(..., description="Relevant excerpt from document")
    provenance: Optional[str] = Field(
        None,
        description="Source of the retrieved passage: 'indexed' (from FAISS) or 'live_pubmed' (fetched at query time)",
    )


class QueryRequest(BaseModel):
    """Model for incoming query requests"""
    # min_length=1 to allow common 2-character medical abbreviations (TB, BP, MI, etc.)
    query: str = Field(..., min_length=1, description="User's medical question")
    top_k: int = Field(5, ge=1, le=20, description="Number of context snippets to retrieve")

    # Conversation history — list of prior turns in chronological order.
    # Each entry must have "role" ("user" or "assistant") and "content".
    history: Optional[List[Dict[str, str]]] = Field(
        None, description="Prior conversation turns for multi-turn dialogue"
    )

    # Legacy support
    question: Optional[str] = Field(None, min_length=1, description="User's medical question (legacy)")
    include_sources: Optional[bool] = Field(True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Model for query responses"""
    answer: str = Field(..., description="Generated answer with inline citations")
    contexts: List[str] = Field(default_factory=list, description="Retrieved context snippets as plain strings")
    
    # Optional extended fields for debugging/monitoring
    question: Optional[str] = Field(None, description="Original question")
    sources: Optional[List[Source]] = Field(None, description="Retrieved sources with metadata")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Overall confidence score")
    retrieval_time_ms: Optional[float] = Field(None, description="Time taken for retrieval in milliseconds")
    generation_time_ms: Optional[float] = Field(None, description="Time taken for generation in milliseconds")
    total_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    warning: Optional[str] = Field(None, description="Warning or disclaimer message")


class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    vector_store_loaded: bool = Field(..., description="Whether vector store is initialized")


