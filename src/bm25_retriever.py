"""
src/bm25_retriever.py
---------------------
BM25 lexical retriever over the same document chunks used by FAISS.

Design notes
------------
1.  Built on top of the existing ``VectorStore.documents`` list — the same
    in-memory list of chunk dicts that backs the FAISS index.  No separate
    corpus file; no index rebuild.

2.  Stable chunk IDs: every chunk dict already has a ``chunk_id`` field
    (e.g. ``"medline_12345_chunk_3"``).  BM25 document positions are mapped
    1-to-1 to the ``documents`` list so provenance, URLs and metadata are
    always retrievable from the same dict.

3.  Text normalisation preserves medically meaningful content:
    - Lowercased.
    - Drug names, numeric values, abbreviations and hyphenated terms are kept.
    - Only pure punctuation tokens are removed; not alphanumeric tokens.

4.  Lazy build: the BM25 index is built on first ``search()`` call or
    explicitly via ``build()``.  Building takes < 1 s for 6,718 chunks.

5.  ``rank_bm25`` is an optional dependency so the module can be imported
    safely even if it is not installed; an ImportError with a clear message
    is raised only when search() or build() is actually called.

Dependencies
------------
    pip install rank-bm25
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("MedSource.bm25_retriever")

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Characters to strip when splitting into tokens (keep hyphens inside words,
# keep digits, keep forward slashes in dosage like "500mg/day").
_PUNCT_STRIP_RE = re.compile(r"[^\w\s\-/.]")
# Collapse runs of whitespace
_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> List[str]:
    """Normalise a string into a token list suitable for BM25.

    Preserves:
    - Alphanumeric tokens (drug names, numbers, abbreviations)
    - Intra-word hyphens (e.g. "beta-blocker")
    - Forward slashes in dosage strings (e.g. "500mg/day")
    - Dotted abbreviations (e.g. "U.S.A") — dots in otherwise-word context

    Removes:
    - Purely punctuation tokens
    - Tokens shorter than 1 character
    """
    text = text.lower()
    text = _PUNCT_STRIP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    tokens = [t for t in text.split() if t and re.search(r"\w", t)]
    return tokens


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 retriever that operates on the VectorStore document list.

    Parameters
    ----------
    documents : list of chunk dicts
        The ``VectorStore.documents`` list.  Passed by reference so updates
        to the vector store are automatically reflected after a ``build()``
        call.
    k1 : float
        BM25 term frequency saturation parameter (default 1.5).
    b : float
        BM25 length normalisation parameter (default 0.75).
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self._bm25 = None          # rank_bm25.BM25Okapi instance
        self._index_doc_count = 0  # number of docs indexed in the current BM25

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Build (or rebuild) the BM25 index from the current document list.

        Safe to call multiple times — rebuilds from scratch each time.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install with: pip install rank-bm25"
            ) from exc

        if not self.documents:
            logger.warning("BM25 build: document list is empty.")
            self._bm25 = None
            self._index_doc_count = 0
            return

        logger.info("Building BM25 index over %d chunks...", len(self.documents))
        tokenised = [normalise_text(doc["content"]) for doc in self.documents]
        self._bm25 = BM25Okapi(tokenised, k1=self.k1, b=self.b)
        self._index_doc_count = len(self.documents)
        logger.info("BM25 index built.")

    def _ensure_built(self) -> bool:
        """Build index lazily on first use. Returns False if index is empty."""
        if self._bm25 is None or self._index_doc_count != len(self.documents):
            self.build()
        return self._bm25 is not None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return up to top_k (doc_dict, bm25_score) pairs, sorted descending.

        Returns an empty list if the index is empty or the query has no tokens.

        Parameters
        ----------
        query : str
            The raw user question.
        top_k : int
            Maximum number of results to return.

        Notes
        -----
        BM25 scores are *not* normalised to [0, 1] — they are raw BM25Okapi
        values and should not be compared directly with FAISS cosine scores.
        Fusion (RRF) uses ranks, not raw scores, so this is safe.
        """
        if not self._ensure_built():
            return []

        tokens = normalise_text(query)
        if not tokens:
            logger.debug("BM25 search: empty token list after normalisation.")
            return []

        try:
            raw_scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            logger.warning("BM25 scoring failed: %s", exc)
            return []

        # Stable partial sort: get top_k indices
        n = len(self.documents)
        top_k_actual = min(top_k, n)
        # Use argpartition for efficiency, then sort the partition
        if top_k_actual < n:
            partition_idx = np.argpartition(raw_scores, -top_k_actual)[-top_k_actual:]
        else:
            partition_idx = np.arange(n)

        sorted_idx = partition_idx[np.argsort(raw_scores[partition_idx])[::-1]]

        results = []
        for idx in sorted_idx:
            score = float(raw_scores[idx])
            if score <= 0.0:
                break  # BM25 scores are non-negative; zero means no match
            if idx < len(self.documents):
                results.append((self.documents[idx], score))

        return results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """Return basic stats about the current BM25 index."""
        return {
            "indexed_chunks": self._index_doc_count,
            "k1": self.k1,
            "b": self.b,
            "is_built": self._bm25 is not None,
        }
