"""
MedSource RAG Chatbot - Source package
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Don't import everything at package level to avoid dependency issues
# Import only when needed in individual modules

__all__ = [
    "get_settings",
    "VectorStore",
    "RAGPipeline",
    "MedicalDocument",
    "MedicalCorpusBuilder",
]
