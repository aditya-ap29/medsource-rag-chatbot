"""
Vector store implementation using FAISS with sentence-transformers.

Key design choices
------------------
Embedding model — BAAI/bge-base-en-v1.5 (default)
  768-dimensional, MTEB SOTA for retrieval tasks. BGE models are trained with
  an asymmetric instruction scheme: passage embeddings are prefixed with
  "Represent this passage: " and query embeddings are prefixed with
  "Represent this sentence for searching relevant passages: ". These prefixes
  are applied automatically based on whether we are indexing or searching.
  The model is symmetric-compatible (prefixes can be disabled via use_bge_prefix=False)
  so existing MiniLM-based stores continue to work without re-embedding.

Chunking — token-aware RecursiveCharacterTextSplitter
  chunk_size and chunk_overlap are measured in **tokens** (via tiktoken cl100k_base),
  not raw characters. This prevents sentences from being cut mid-word when a
  document mixes short and long sentences. Default 256 tokens / 40-token overlap
  fits inside BGE's 512-token context window with room for the passage prefix.

  Sentence boundary separators are ordered from coarsest to finest so the splitter
  prefers natural breaks:
      paragraph → sentence end (. ! ?) → clause (; ,) → word → character

FAISS index — IndexFlatIP
  Exact inner-product search on L2-normalised vectors equals cosine similarity,
  with no approximation loss. Suitable for corpora up to ~500 K chunks on CPU.
"""
import logging
import os
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_ingestion import MedicalDocument

logger = logging.getLogger("MedSource.vector_store")

# BGE asymmetric instruction prefixes (prepended only at encode time, not stored)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_BGE_PASSAGE_PREFIX = "Represent this passage: "

# Substring-matched so bge-base, bge-large, etc. all trigger prefix mode.
_BGE_MODEL_PREFIXES = ("BAAI/bge",)


def _make_token_length_fn():
    """Token counter using tiktoken cl100k_base; falls back to len() if not installed."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def _len(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))
        return _len
    except ImportError:
        return len  # graceful fallback


# Ordered coarsest → finest so the splitter prefers natural sentence breaks.
_MEDICAL_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class VectorStore:
    """FAISS-based vector store for document retrieval.

    Parameters
    ----------
    embedding_model_name : str
        HuggingFace model id or local path.  Defaults to BAAI/bge-base-en-v1.5.
    chunk_size : int
        Maximum chunk size measured in **tokens** (not characters).
    chunk_overlap : int
        Overlap between consecutive chunks, in tokens.
    use_bge_prefix : bool or None
        Whether to apply BGE asymmetric instruction prefixes.
        None (default) = auto-detect from model name.
        Set to False explicitly when loading a store built without prefixes.
    embedding_batch_size : int
        Batch size passed to SentenceTransformer.encode().
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-base-en-v1.5",
        chunk_size: int = 256,
        chunk_overlap: int = 40,
        use_bge_prefix: Optional[bool] = None,
        embedding_batch_size: int = 32,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_batch_size = embedding_batch_size
        self.dimension = self.embedding_model.get_embedding_dimension()

        # Auto-detect BGE prefix mode from the model name
        if use_bge_prefix is None:
            self.use_bge_prefix = any(
                embedding_model_name.startswith(p) for p in _BGE_MODEL_PREFIXES
            )
        else:
            self.use_bge_prefix = use_bge_prefix

        # Build a token-aware text splitter
        token_len = _make_token_length_fn()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_len,
            separators=_MEDICAL_SEPARATORS,
            keep_separator=False,
        )

        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[Dict[str, Any]] = []

    def _chunk_documents(self, documents: List[MedicalDocument]) -> List[Dict[str, Any]]:
        """Split documents into token-sized chunks with metadata."""
        chunked_docs = []
        for doc in documents:
            if not doc.content or not doc.content.strip():
                continue

            chunks = self.text_splitter.split_text(doc.content)
            chunks = [c.strip() for c in chunks if c.strip()]

            for i, chunk in enumerate(chunks):
                chunked_docs.append({
                    "doc_id": doc.doc_id,
                    "chunk_id": f"{doc.doc_id}_chunk_{i}",
                    "content": chunk,
                    "title": doc.title,
                    "source": doc.source,
                    "year": doc.year,
                    "url": doc.url,
                    "metadata": doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })
        return chunked_docs

    def _encode_passages(self, texts: List[str]) -> np.ndarray:
        """Encode passage texts (indexing time) with optional BGE prefix."""
        if self.use_bge_prefix:
            texts = [_BGE_PASSAGE_PREFIX + t for t in texts]
        return self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=self.embedding_batch_size,
        ).astype("float32")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string (search time) with optional BGE prefix."""
        text = (_BGE_QUERY_PREFIX + query) if self.use_bge_prefix else query
        return self.embedding_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=1,
        ).astype("float32")

    def add_documents(self, documents: List[MedicalDocument]) -> int:
        """Chunk, embed, and index a list of MedicalDocuments.

        Deduplication: chunks whose doc_id already exists in the store are
        silently skipped so re-ingesting the same source does not double the
        index or distort retrieval scores.

        Returns the number of *new* chunks actually added (not total submitted).
        """
        logger.info("Chunking %d document(s)...", len(documents))
        chunked_docs = self._chunk_documents(documents)
        if not chunked_docs:
            logger.warning("No usable chunks produced from %d document(s).", len(documents))
            return 0

        # Deduplicate: skip any doc_id already present in the store.
        existing_doc_ids: set = {d["doc_id"] for d in self.documents}
        new_chunks = [c for c in chunked_docs if c["doc_id"] not in existing_doc_ids]
        skipped = len(chunked_docs) - len(new_chunks)
        if skipped:
            logger.info("Skipped %d duplicate chunk(s) (doc_id already indexed).", skipped)
        if not new_chunks:
            logger.warning("All %d document(s) already indexed; nothing added.", len(documents))
            return 0

        avg_chars = sum(len(c["content"]) for c in new_chunks) // len(new_chunks)
        logger.info("Created %d new chunks (avg %d chars/chunk)", len(new_chunks), avg_chars)

        logger.info("Generating embeddings...")
        texts = [doc["content"] for doc in new_chunks]
        embeddings = self._encode_passages(texts)

        logger.info("Adding %d vectors to FAISS index...", len(new_chunks))
        self.index.add(embeddings)
        self.documents.extend(new_chunks)

        logger.info("Vector store now contains %d chunks", len(self.documents))
        return len(new_chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return up to top_k chunks whose cosine similarity >= score_threshold."""
        if not self.documents:
            return []

        query_embedding = self._encode_query(query)
        k_request = min(max(top_k * 3, top_k + 10), len(self.documents))
        scores, indices = self.index.search(query_embedding, k_request)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            if score < score_threshold:
                continue
            results.append((self.documents[idx], float(score)))
            if len(results) >= top_k:
                break

        return results

    def save(self, save_dir: str):
        """Persist the FAISS index, document list, and config to disk."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_path / "faiss.index"))

        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)

        config = {
            "embedding_model_name": self.embedding_model_name,
            "use_bge_prefix": self.use_bge_prefix,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_batch_size": self.embedding_batch_size,
            "dimension": self.dimension,
            "num_documents": len(self.documents),
        }
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)

        logger.info("Vector store saved to %s", save_path)

    def load(self, load_dir: str):
        """Load a persisted vector store from disk."""
        load_path = Path(load_dir)
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store directory not found: {load_dir}")

        self.index = faiss.read_index(str(load_path / "faiss.index"))

        with open(load_path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)

        config_path = load_path / "config.pkl"
        if config_path.exists():
            with open(config_path, "rb") as f:
                cfg = pickle.load(f)
            stored_name = cfg.get("embedding_model_name", "")
            if isinstance(stored_name, int):
                # Old stores saved dimension as int; validate dimension only.
                stored_dim = stored_name
                if stored_dim != self.dimension:
                    logger.warning(
                        "Stored vector dimension (%d) does not match current model "
                        "dimension (%d). The loaded index is incompatible — run "
                        "scripts/reembed_vector_store.py.",
                        stored_dim, self.dimension,
                    )
            elif stored_name and stored_name != self.embedding_model_name:
                logger.warning(
                    "Vector store was built with '%s' but current model is '%s'. "
                    "Similarity scores will be wrong. Run scripts/reembed_vector_store.py "
                    "to rebuild with the current model.",
                    stored_name, self.embedding_model_name,
                )
            if "use_bge_prefix" in cfg:
                self.use_bge_prefix = cfg["use_bge_prefix"]

        logger.info("Vector store loaded from %s (%d chunks)", load_path, len(self.documents))

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics about the current vector store."""
        unique_docs = len(set(doc["doc_id"] for doc in self.documents))
        avg_chunk_len = (
            sum(len(d["content"]) for d in self.documents) // len(self.documents)
            if self.documents else 0
        )
        return {
            "total_chunks": len(self.documents),
            "unique_documents": unique_docs,
            "dimension": self.dimension,
            "embedding_model": self.embedding_model_name,
            "use_bge_prefix": self.use_bge_prefix,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "avg_chunk_chars": avg_chunk_len,
        }
