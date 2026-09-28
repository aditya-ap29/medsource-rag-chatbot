"""
Minimal example of using MedSource directly (without API)
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.config import get_settings

# Load environment
load_dotenv()
settings = get_settings()


def main():
    print("Loading MedSource RAG Pipeline...")
    
    # Load vector store
    vector_store = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    if os.path.exists(settings.vector_store_path):
        vector_store.load(settings.vector_store_path)
        print(f"✅ Loaded {len(vector_store.documents)} document chunks\n")
    else:
        print("❌ Vector store not found. Run: python scripts/ingest_data.py")
        return
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_provider="openai",  # or "mistral", "qwen"
        model_name=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        top_k=settings.top_k_documents,
        confidence_threshold=settings.retrieval_confidence_threshold
    )
    
    print("🤖 MedSource Ready!\n")
    
    # Example query
    question = "What are the symptoms of vitamin D deficiency?"
    
    print(f"Q: {question}\n")
    
    response = rag_pipeline.query(question)
    
    print(f"A: {response.answer}\n")
    print(f"Confidence: {response.confidence:.2%}")
    print(f"Sources: {len(response.sources)}")


if __name__ == "__main__":
    main()
