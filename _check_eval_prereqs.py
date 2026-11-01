import os, sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

key = os.getenv("GEMINI_API_KEY", "")
placeholder = key in ("", "your-gemini-key-here", "your-key-here")
print(f"GEMINI_API_KEY set: {bool(key) and not placeholder}")
if placeholder:
    print("  -> Key is missing or still set to the placeholder value")

vs_path = Path("data/vector_store")
if vs_path.exists():
    files = list(vs_path.iterdir())
    print(f"Vector store files: {[f.name for f in files]}")
    idx = vs_path / "faiss.index"
    docs = vs_path / "documents.pkl"
    if idx.exists():
        print(f"  faiss.index size : {idx.stat().st_size / 1024:.0f} KB")
    if docs.exists():
        print(f"  documents.pkl size: {docs.stat().st_size / 1024:.0f} KB")
else:
    print("Vector store: NOT FOUND — run ingest first")

try:
    import ragas
    print(f"ragas version: {ragas.__version__}")
except ImportError:
    print("ragas: NOT INSTALLED (run: pip install ragas)")
