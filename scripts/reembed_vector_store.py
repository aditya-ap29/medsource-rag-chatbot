"""
Re-embeds an existing vector store with a NEW embedding model, without
re-fetching from PubMed and without re-chunking (the chunks you already
have are reused as-is — only their vectors are recomputed).

Supports the BGE asymmetric instruction prefix scheme used by
BAAI/bge-base-en-v1.5 and similar models: each chunk is prefixed with
"Represent this passage: " before encoding.  The flag is stored in
config.pkl so the live VectorStore reads it back correctly.

Usage (from project root, venv active):
    python scripts/reembed_vector_store.py

Reads EMBEDDING_MODEL from .env for the NEW model, and VECTOR_STORE_PATH
for where to load the existing chunks from / save the upgraded store to.
Makes a backup of the old store first, just in case.
"""
import sys
import shutil
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.config import get_settings

settings = get_settings()
store_path = Path(settings.vector_store_path)

new_model_name = settings.embedding_model
print(f"New embedding model (from .env): {new_model_name}")
print(f"Vector store path:               {store_path}\n")

backup_path = store_path.parent / (store_path.name + "_backup_pre_reembed")
if store_path.exists() and not backup_path.exists():
    print(f"Backing up existing store to: {backup_path}")
    shutil.copytree(store_path, backup_path)
else:
    print(f"Backup already exists at {backup_path}, skipping.")

docs_path = store_path / "documents.pkl"
if not docs_path.exists():
    print(f"ERROR: {docs_path} not found. Nothing to re-embed.")
    sys.exit(1)

with open(docs_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} existing chunks to re-embed.\n")

print(f"Loading new embedding model: {new_model_name}...")
model = SentenceTransformer(new_model_name)
dimension = model.get_embedding_dimension()
print(f"New embedding dimension: {dimension}\n")

_BGE_MODEL_PREFIXES = ("BAAI/bge",)
_BGE_PASSAGE_PREFIX = "Represent this passage: "

use_bge_prefix = any(new_model_name.startswith(p) for p in _BGE_MODEL_PREFIXES)
if use_bge_prefix:
    print(f"BGE model detected — applying passage prefix.")
else:
    print("Non-BGE model — no instruction prefix applied.")

texts = [c["content"] for c in chunks]
if use_bge_prefix:
    texts = [_BGE_PASSAGE_PREFIX + t for t in texts]

print(f"\nGenerating new embeddings for {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=settings.embedding_batch_size,
)

print("\nBuilding new FAISS index...")
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype("float32"))

store_path.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(store_path / "faiss.index"))

with open(store_path / "documents.pkl", "wb") as f:
    pickle.dump(chunks, f)

config = {
    "embedding_model_name": new_model_name,
    "use_bge_prefix": use_bge_prefix,
    "chunk_size": settings.chunk_size,
    "chunk_overlap": settings.chunk_overlap,
    "embedding_batch_size": settings.embedding_batch_size,
    "dimension": dimension,
    "num_documents": len(chunks),
}
with open(store_path / "config.pkl", "wb") as f:
    pickle.dump(config, f)

print(f"\n✅ Re-embedding complete!")
print(f"   {len(chunks)} chunks re-embedded with {new_model_name}")
print(f"   New dimension:  {dimension}")
print(f"   BGE prefix:     {'on' if use_bge_prefix else 'off'}")
print(f"   Saved to:       {store_path}")
print(f"   Old store backed up at: {backup_path}")
print(f"\nRestart the API / Streamlit app so they load the new embeddings.")
