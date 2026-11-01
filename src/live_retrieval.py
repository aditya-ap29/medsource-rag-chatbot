"""
Live PubMed retrieval — ephemeral augmentation for the RAG pipeline.

At query time, when local FAISS retrieval falls below the confidence threshold
(or when ENABLE_LIVE_PUBMED=true unconditionally), this module fetches fresh
PubMed abstracts via NCBI E-utilities and returns them as in-memory
MedicalDocument objects that can be merged with FAISS candidates.

Key design decisions
--------------------
- Results are NEVER written back to the FAISS index; they are ephemeral per query.
  This keeps the shared index clean and avoids stale live data accumulating.
- Deduplication is by PMID (exact match).  Any PMID already present in the FAISS
  store is excluded from live results to prevent the same passage appearing twice.
- Provenance is recorded in MedicalDocument.metadata["provenance"] = "live_pubmed".
- A hard timeout (default 5 s) prevents slow NCBI calls from blocking the response.
- Rate-limit delays: 0.35 s between calls without an API key, 0.12 s with one.
  Both are well under NCBI's stated limits (3 req/s / 10 req/s).

Usage (inside RAGPipeline)
--------------------------
    from src.live_retrieval import PubMedLiveRetriever
    retriever = PubMedLiveRetriever(
        email=settings.pubmed_email,
        api_key=settings.ncbi_api_key,
        timeout_s=settings.live_pubmed_timeout_s,
    )
    live_docs = retriever.search_and_fetch(question, max_results=5, existing_pmids=indexed_pmids)
"""
import hashlib
import logging
import time
from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

from src.data_ingestion import MedicalDocument

logger = logging.getLogger("MedSource.live_retrieval")

# NCBI E-utilities base URL
_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

# Polite sleep between NCBI calls (seconds)
_DELAY_NO_KEY = 0.35   # 3 req/s limit without API key
_DELAY_WITH_KEY = 0.12  # 10 req/s limit with API key


class PubMedLiveRetriever:
    """Fetch fresh PubMed abstracts at query time via NCBI E-utilities.

    Parameters
    ----------
    email : str
        Required by NCBI usage policy — any valid email address.
    api_key : str, optional
        NCBI API key — raises rate limit from 3 to 10 req/s.
        Get a free key at https://www.ncbi.nlm.nih.gov/account/
    timeout_s : int
        Hard HTTP timeout for each NCBI call (default 5 s).
    """

    def __init__(
        self,
        email: str = "medsource@example.com",
        api_key: str = "",
        timeout_s: int = 5,
    ):
        self.email = email or "medsource@example.com"
        self.api_key = api_key or ""
        self.timeout_s = timeout_s
        self._delay = _DELAY_WITH_KEY if self.api_key else _DELAY_NO_KEY

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 5,
        existing_pmids: Optional[Set[str]] = None,
    ) -> List[MedicalDocument]:
        """Search PubMed and return up to max_results new abstracts.

        Parameters
        ----------
        query : str
            Free-text medical query (same phrasing used for the user's question).
        max_results : int
            Maximum number of abstracts to return (capped at 10 to stay fast).
        existing_pmids : set of str, optional
            PMIDs already in the FAISS index.  Matching PMIDs are excluded from
            live results so identical text never appears as two separate sources.

        Returns
        -------
        List[MedicalDocument]
            Empty list on any network or parse error (fail-safe).
        """
        max_results = min(max_results, 10)
        existing = existing_pmids or set()

        try:
            pmids = self._esearch(query, max_results * 2)  # fetch 2× to allow dedup headroom
        except Exception as exc:
            logger.warning("Live PubMed esearch failed for '%s': %s", query[:60], exc)
            return []

        # Filter out PMIDs already indexed
        new_pmids = [p for p in pmids if p not in existing]
        if not new_pmids:
            logger.debug("All live PubMed results for '%s' already indexed.", query[:60])
            return []

        new_pmids = new_pmids[:max_results]
        try:
            docs = self._efetch(new_pmids)
        except Exception as exc:
            logger.warning("Live PubMed efetch failed for '%s': %s", query[:60], exc)
            return []

        logger.info("Live PubMed: fetched %d new abstract(s) for '%s'", len(docs), query[:60])
        return docs

    def get_indexed_pmids(self, documents: list) -> Set[str]:
        """Extract the set of PMIDs already present in a VectorStore document list.

        Looks for doc_ids formatted as 'PMID_<number>' (set by PubMedLoader).
        """
        pmids: Set[str] = set()
        for doc in documents:
            doc_id = doc.get("doc_id", "")
            if doc_id.startswith("PMID_"):
                pmids.add(doc_id[5:])  # strip "PMID_" prefix
            # Also check metadata["pmid"] stored by PubMedLoader
            meta = doc.get("metadata", {})
            if isinstance(meta, dict) and "pmid" in meta:
                pmids.add(str(meta["pmid"]))
        return pmids

    # ------------------------------------------------------------------
    # Internal NCBI helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> dict:
        """Common query params sent with every NCBI request."""
        p: dict = {"email": self.email}
        if self.api_key:
            p["api_key"] = self.api_key
        return p

    def _esearch(self, query: str, max_results: int) -> List[str]:
        """Return a list of PMIDs matching query (newest-first)."""
        params = self._base_params()
        params.update({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        })
        resp = requests.get(
            f"{_EUTILS_BASE}esearch.fcgi",
            params=params,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        time.sleep(self._delay)
        return resp.json().get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: List[str]) -> List[MedicalDocument]:
        """Fetch full abstracts for a list of PMIDs and return MedicalDocuments."""
        params = self._base_params()
        params.update({
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        })
        resp = requests.get(
            f"{_EUTILS_BASE}efetch.fcgi",
            params=params,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        time.sleep(self._delay)

        soup = BeautifulSoup(resp.content, "xml")
        docs: List[MedicalDocument] = []

        for article in soup.find_all("PubmedArticle"):
            try:
                pmid_elem = article.find("PMID")
                if not pmid_elem:
                    continue
                pmid = pmid_elem.text.strip()

                title_elem = article.find("ArticleTitle")
                title = title_elem.get_text(strip=True) if title_elem else f"PubMed {pmid}"

                abstract_elem = article.find("AbstractText")
                abstract = abstract_elem.get_text(separator=" ", strip=True) if abstract_elem else ""
                if not abstract or len(abstract) < 50:
                    continue

                year: Optional[str] = None
                year_elem = article.find("PubDate")
                if year_elem and year_elem.find("Year"):
                    year = year_elem.find("Year").text.strip()

                docs.append(MedicalDocument(
                    content=abstract,
                    title=title,
                    source="PubMed (live)",
                    doc_id=f"PMID_{pmid}",
                    year=year,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    metadata={
                        "pmid": pmid,
                        "provenance": "live_pubmed",
                    },
                ))
            except Exception as exc:
                logger.debug("Skipping malformed PubMed article: %s", exc)
                continue

        return docs
