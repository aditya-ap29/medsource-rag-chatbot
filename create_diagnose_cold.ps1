$content = @'
from src.vector_store import VectorStore
from src.config import get_settings

settings = get_settings()
vs = VectorStore(
    embedding_model_name=settings.embedding_model,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)
vs.load(settings.vector_store_path)

query = "What are the symptoms of the common cold?"
results = vs.search(query, top_k=10, score_threshold=0.0)

print(f"\nQuery: {query!r}\n")
print("Top 10 raw results:\n")
for i, (doc, score) in enumerate(results, 1):
    print(f"{i}. score={score:.4f} | title={doc.get('title', 'N/A')[:80]}")

print(f"\nCurrent threshold: {settings.retrieval_confidence_threshold}")

'@
Set-Content -Path 'diagnose_cold.py' -Value $content -Encoding utf8
Write-Host 'diagnose_cold.py has been created.'