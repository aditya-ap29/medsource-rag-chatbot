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


class QueryRequest(BaseModel):
    """Model for incoming query requests - Hack-A-Cure competition format"""
    query: str = Field(..., min_length=3, description="User's medical question")
    top_k: int = Field(5, ge=1, le=20, description="Number of context snippets to retrieve")
    
    # Legacy support
    question: Optional[str] = Field(None, min_length=3, description="User's medical question (legacy)")
    include_sources: Optional[bool] = Field(True, description="Include source documents in response")


class QueryResponse(BaseModel):
    """Model for query responses - Hack-A-Cure competition format"""
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


class EvaluationRequest(BaseModel):
    """Model for RAGAS evaluation requests"""
    questions: List[str] = Field(..., description="List of questions to evaluate")
    ground_truths: Optional[List[str]] = Field(None, description="Optional ground truth answers")


class EvaluationResponse(BaseModel):
    """Model for RAGAS evaluation results"""
    faithfulness: float = Field(..., description="Faithfulness score (0-1)")
    context_precision: float = Field(..., description="Context precision score (0-1)")
    answer_relevance: float = Field(..., description="Answer relevance score (0-1)")
    context_recall: Optional[float] = Field(None, description="Context recall score (0-1)")
    overall_score: float = Field(..., description="Average RAGAS score")
    details: List[Dict] = Field(default_factory=list, description="Per-question evaluation details")
