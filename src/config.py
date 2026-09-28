"""
Configuration management for MedSource RAG Chatbot
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Key required to call this app's own API (NOT an LLM provider key).
    # Set this in .env as CHATBOT_API_KEY. If left blank, auth is disabled
    # (fine for local dev, NOT fine if this server is exposed publicly).
    chatbot_api_key: str = ""

    # Comma-separated list of frontend origins allowed to call this API from
    # a browser (CORS). Defaults to local Streamlit/dev ports only.
    allowed_origins: str = "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501"
    
    # Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4-turbo-preview"
    llm_provider: str = "gemini"  # gemini, openai, mistral, qwen, huggingface
    llm_temperature: float = 0.1
    max_tokens: int = 1000
    
    # Retrieval Configuration
    top_k_documents: int = 5
    retrieval_confidence_threshold: float = 0.75

    # If True, when no retrieved document passes the confidence threshold,
    # the bot answers from the LLM's general knowledge instead of refusing
    # outright -- clearly labeled as unverified/unsourced. If False, it
    # always refuses on low-confidence retrieval (the original behavior).
    enable_general_fallback: bool = True
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    
    # Vector Store Configuration
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Application Metadata
    app_name: str = "MedSource - Medical RAG Chatbot"
    app_version: str = "1.0.0"

    # PubMed ingestion — contact email sent with API requests per NCBI's
    # usage etiquette (not an account/login, no verification happens).
    pubmed_email: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # don't crash if .env has keys not defined here


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()