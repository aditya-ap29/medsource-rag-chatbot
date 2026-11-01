"""
src/hybrid_retriever.py
-----------------------
Hybrid retrieval: BGE/FAISS dense + BM25 sparse, fused with Reciprocal
Rank Fusion (RRF).

Architecture
------------
1.  Dense retrieval  — VectorStore.search() returns (doc, cosine_score) pairs.
2.  Sparse retrieval — BM25Retriever.search() returns (doc, bm25_score) pairs.
3.  RRF fusion       — Combines ranked lists by chunk_id using the formula:
                            RRF(d) = Σ_r  1 / (k + rank_r(d))
                       where k=60 is the RRF smoothing constant (Cormack 2009).
4.  Deduplication    — Results are keyed on chunk_id; the highest-scored
                       occurrence of each chunk is retained.
5.  Cross-encoder    — Optional post-fusion reranking (same as current pipeline).

Configuration
-------------
All behaviour is controlled via ``HybridConfig`` flags.  With
``enable_hybrid=False`` the retriever falls back to pure FAISS, making the
integration fully transparent to the rest of the pipeline.

Design constraints
------------------
- Does NOT rebuild or replace the FAISS index.
- Does NOT modify VectorStore.documents.
- Handles empty results, missing metadata, and individual retriever failures
  gracefully (logs a warning, returns whatever the other retriever produced).
- Source provenance, URLs, and citation metadata pass through unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("MedSource.hybrid_retriever")

# Default RRF smoothing constant (Cormack & Lynam, SIGIR 2009)
_RRF_K = 60


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class HybridConfig:
    """Runtime configuration for the hybrid retriever.

    All fields have safe defaults so the retriever works out-of-the-box.
    """
    # Master switch — set to True to enable BM25+RRF fusion.
    # When False, falls back to pure FAISS (existing behaviour).
    enable_hybrid: bool = False

    # Candidate pool multiplier: each retriever fetches top_k * pool_multiplier
    # candidates before RRF merges and truncates to top_k.
    pool_multiplier: int = 4

    # RRF smoothing constant (higher = less weight on top ranks)
    rrf_k: int = _RRF_K

    # Weight to apply to dense retrieval ranks (higher = more influence)
    dense_weight: float = 1.0

    # Weight to apply to sparse (BM25) retrieval ranks (higher = more influence)
    sparse_weight: float = 1.0

    # BM25 parameters — passed to BM25Retriever on construction
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


# ---------------------------------------------------------------------------
# RRF implementation
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, Any]]],
    weights: Optional[List[float]] = None,
    k: int = _RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of list of (chunk_id, doc_or_score) tuples
        Each inner list is one retriever's result, ordered best-first.
        Only the *order* matters — scores are ignored.
    weights : list of floats, optional
        Per-retriever weight (default: 1.0 for all).  Controls how much each
        retriever contributes to the final score.
    k : int
        RRF smoothing constant.  Default 60.

    Returns
    -------
    list of (chunk_id, rrf_score) sorted by rrf_score descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    rrf_scores: Dict[str, float] = {}
    for ranked_list, w in zip(ranked_lists, weights):
        for rank, (chunk_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + w / (k + rank)

    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_ids


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Fuses FAISS dense retrieval with BM25 sparse retrieval via RRF.

    Parameters
    ----------
    vector_store : VectorStore
        The loaded FAISS vector store (never modified).
    config : HybridConfig
        Runtime configuration flags.
    bm25_retriever : BM25Retriever, optional
        If None, one is created lazily on first hybrid search call.
    """

    def __init__(
        self,
        vector_store: Any,          # VectorStore — avoid circular import
        config: Optional[HybridConfig] = None,
        bm25_retriever: Optional[Any] = None,  # BM25Retriever
    ):
        self.vector_store = vector_store
        self.config = config or HybridConfig()
        self._bm25: Optional[Any] = bm25_retriever
        self._bm25_ready = False

    # ------------------------------------------------------------------
    # BM25 lazy init
    # ------------------------------------------------------------------

    def _get_bm25(self) -> Any:
        """Return the BM25 retriever, building it if needed."""
        from src.bm25_retriever import BM25Retriever

        if self._bm25 is None:
            self._bm25 = BM25Retriever(
                self.vector_store.documents,
                k1=self.config.bm25_k1,
                b=self.config.bm25_b,
            )
            self._bm25_ready = False

        if not self._bm25_ready:
            self._bm25.build()
            self._bm25_ready = True

        return self._bm25

    # ------------------------------------------------------------------
    # Core search interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top-k chunks.

        When ``config.enable_hybrid`` is False, delegates to the pure FAISS
        path so behaviour is identical to the original pipeline.

        Returns
        -------
        list of (doc_dict, score) pairs sorted by relevance descending.
        For dense-only: score = FAISS cosine similarity.
        For hybrid: score = RRF score (comparable across hybrid results only).
        """
        if not self.config.enable_hybrid:
            return self.vector_store.search(
                query, top_k=top_k, score_threshold=score_threshold
            )

        return self._hybrid_search(query, top_k=top_k, score_threshold=score_threshold)

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        score_threshold: float,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Perform RRF-fused hybrid retrieval."""
        pool = top_k * max(self.config.pool_multiplier, 1)

        # --- Dense retrieval ---
        dense_results: List[Tuple[Dict[str, Any], float]] = []
        try:
            dense_results = self.vector_store.search(
                query, top_k=pool, score_threshold=score_threshold
            )
        except Exception as exc:
            logger.warning("Dense retrieval failed: %s", exc)

        # --- Sparse retrieval ---
        sparse_results: List[Tuple[Dict[str, Any], float]] = []
        try:
            bm25 = self._get_bm25()
            sparse_results = bm25.search(query, top_k=pool)
        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)

        # If both failed, return empty
        if not dense_results and not sparse_results:
            logger.warning("Both dense and sparse retrieval returned nothing.")
            return []

        # If only one succeeded, fall back to that list
        if not dense_results:
            logger.debug("Dense empty — returning BM25 results only.")
            seen: Set[str] = set()
            results = []
            for doc, score in sparse_results:
                cid = doc.get("chunk_id", doc.get("doc_id", ""))
                if cid not in seen:
                    seen.add(cid)
                    results.append((doc, score))
            return results[:top_k]

        if not sparse_results:
            logger.debug("BM25 empty — returning dense results only.")
            return dense_results[:top_k]

        # --- Build chunk_id-indexed lookups ---
        chunk_id_to_doc: Dict[str, Dict[str, Any]] = {}

        dense_ranked: List[Tuple[str, float]] = []
        for doc, score in dense_results:
            cid = doc.get("chunk_id") or doc.get("doc_id", "")
            if cid:
                chunk_id_to_doc[cid] = doc
                dense_ranked.append((cid, score))

        sparse_ranked: List[Tuple[str, float]] = []
        for doc, score in sparse_results:
            cid = doc.get("chunk_id") or doc.get("doc_id", "")
            if cid:
                chunk_id_to_doc.setdefault(cid, doc)
                sparse_ranked.append((cid, score))

        # --- RRF fusion ---
        fused = reciprocal_rank_fusion(
            [dense_ranked, sparse_ranked],
            weights=[self.config.dense_weight, self.config.sparse_weight],
            k=self.config.rrf_k,
        )

        # --- Reconstruct (doc, rrf_score) pairs ---
        results = []
        seen_chunk_ids: Set[str] = set()
        for cid, rrf_score in fused:
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            doc = chunk_id_to_doc.get(cid)
            if doc is None:
                continue
            results.append((doc, rrf_score))
            if len(results) >= top_k:
                break

        logger.debug(
            "Hybrid search: dense=%d sparse=%d fused=%d returned=%d",
            len(dense_ranked), len(sparse_ranked), len(fused), len(results),
        )
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about both retrievers."""
        stats: Dict[str, Any] = {
            "enable_hybrid": self.config.enable_hybrid,
            "rrf_k": self.config.rrf_k,
            "dense_weight": self.config.dense_weight,
            "sparse_weight": self.config.sparse_weight,
        }
        if self._bm25 is not None:
            stats["bm25"] = self._bm25.get_stats()
        return stats
