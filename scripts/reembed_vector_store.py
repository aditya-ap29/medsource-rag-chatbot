"""
Re-embeds an existing vector store with a NEW embedding model, without
re-fetching from PubMed and without re-chunking (the chunks you already
have are reused as-is -- only their vectors are recomputed).

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

print(f"New embedding model (from .env): {settings.embedding_model}")
print(f"Vector store path: {store_path}\n")

# --- Step 1: back up the existing store, just in case ---
backup_path = store_path.parent / (store_path.name + "_backup_pre_reembed")
if store_path.exists() and not backup_path.exists():
    print(f"Backing up existing store to: {backup_path}")
    shutil.copytree(store_path, backup_path)
elif backup_path.exists():
    print(f"Backup already exists at {backup_path}, skipping backup step.")

# --- Step 2: load existing chunks (content + metadata), NOT the old FAISS index ---
docs_path = store_path / "documents.pkl"
if not docs_path.exists():
    print(f"ERROR: {docs_path} not found. Nothing to re-embed.")
    sys.exit(1)

with open(docs_path, "rb") as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} existing chunks to re-embed.\n")

# --- Step 3: load the NEW embedding model ---
print(f"Loading new embedding model: {settings.embedding_model} (this may take a minute)...")
model = SentenceTransformer(settings.embedding_model)
dimension = model.get_sentence_embedding_dimension()
print(f"New embedding dimension: {dimension}\n")

# --- Step 4: re-embed all chunk texts with the new model ---
texts = [c["content"] for c in chunks]
print(f"Generating new embeddings for {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
    batch_size=64,
)

# --- Step 5: build a fresh FAISS index with the new vectors ---
print("\nBuilding new FAISS index...")
index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype("float32"))

# --- Step 6: save the upgraded store (overwrites in place) ---
store_path.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(store_path / "faiss.index"))

with open(store_path / "documents.pkl", "wb") as f:
    pickle.dump(chunks, f)

config = {
    "embedding_model_name": settings.embedding_model,
    "chunk_size": settings.chunk_size,
    "chunk_overlap": settings.chunk_overlap,
    "dimension": dimension,
    "num_documents": len(chunks),
}
with open(store_path / "config.pkl", "wb") as f:
    pickle.dump(config, f)

print(f"\n✅ Re-embedding complete!")
print(f"   {len(chunks)} chunks re-embedded with {settings.embedding_model}")
print(f"   New dimension: {dimension}")
print(f"   Saved to: {store_path}")
print(f"   Old store backed up at: {backup_path}")