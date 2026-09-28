"""
Diagnostic script: bypasses the RAG pipeline and API entirely to check
raw FAISS retrieval for a test query. Run from the project root with the
venv active: python diagnose_retrieval.py
"""
from src.vector_store import VectorStore
from src.config import get_settings

settings = get_settings()

print(f"Loading vector store from: {settings.vector_store_path}")
vs = VectorStore(
    embedding_model_name=settings.embedding_model,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
vs.load(settings.vector_store_path)

print(f"\nLoaded {len(vs.documents)} chunks total.\n")

query = "I am feeling headache and little bit fever"
print(f"Query: {query!r}\n")

# Search with NO threshold, to see the raw top scores regardless of cutoff
results = vs.search(query, top_k=10, score_threshold=0.0)

print(f"Raw top {len(results)} results (no threshold applied):\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. score={score:.4f} | title={doc.get('title', 'N/A')[:80]}")

print(f"\nCurrent RETRIEVAL_CONFIDENCE_THRESHOLD in settings: {settings.retrieval_confidence_threshold}")
print(f"Current TOP_K_DOCUMENTS in settings: {settings.top_k_documents}")