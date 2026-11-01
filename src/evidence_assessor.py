"""
src/evidence_assessor.py
------------------------
Evidence-sufficiency assessment for the MedSource RAG pipeline.

Purpose
-------
Domain classification (domain_gate.py / semantic_classifier.py) answers:
"Is this question medical?"

Evidence assessment answers: "Do the retrieved passages actually support a
safe, grounded answer?"  These are separate concerns.

A question classified as medical may still have insufficient evidence if:
  - Too few passages were retrieved above the score threshold.
  - Retrieved passages score below a credibility floor (low-quality chunks).
  - Coverage across the question's key topics is sparse.
  - The only matching source has low authority (no URL, no year, etc.).
  - Top passages come from a single document (low diversity = single-source risk).
  - Conflicting signals appear across passages.

Assessment logic
----------------
The assessor evaluates a set of configurable policy signals and aggregates
them into one of three verdicts:

  SUFFICIENT   — proceed with full cited answer
  INSUFFICIENT — abstain or use general-knowledge fallback (existing behaviour)
  UNCERTAIN    — surface a warning but allow generation (conservative pass)

Each signal contributes a signed integer score.  The final verdict is
determined by comparing the total to two configurable thresholds.

All thresholds are class-level constants that can be overridden via
``EvidenceAssessorConfig``.

Integration
-----------
The assessor is called from ``RAGPipeline.query()`` after retrieval and
domain classification, but before answer generation, when
``enable_evidence_assessment=True`` (a new config flag).  When disabled,
the existing confidence threshold logic handles this role unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MedSource.evidence_assessor")


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERDICT_SUFFICIENT = "sufficient"
VERDICT_INSUFFICIENT = "insufficient"
VERDICT_UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvidenceAssessorConfig:
    """Configurable thresholds for evidence-sufficiency assessment.

    Positive signal scores push toward SUFFICIENT; negative toward INSUFFICIENT.
    The final verdict is determined by comparing the sum to
    ``sufficient_threshold`` and ``insufficient_threshold``.
    """

    # Minimum BGE cosine score for a chunk to be considered credible.
    credibility_floor: float = 0.55

    # Minimum number of chunks that must clear the credibility floor.
    min_credible_chunks: int = 2

    # Cosine score above which a chunk is considered strongly relevant.
    strong_relevance_threshold: float = 0.70

    # Minimum number of strongly relevant chunks for a "sufficient" signal.
    min_strong_chunks: int = 1

    # Maximum fraction of chunks that may come from a single doc_id.
    # If all chunks share one doc_id, diversity is 0.0 (single-source).
    min_diversity: float = 0.4

    # Score sum >= sufficient_threshold → SUFFICIENT
    sufficient_threshold: int = 3

    # Score sum <= insufficient_threshold → INSUFFICIENT
    insufficient_threshold: int = -1

    # Penalty when a source has no URL (low traceability).
    no_url_penalty: int = -1

    # Penalty when a source has no year (unknown currency / authority).
    no_year_penalty: int = -1

    # Bonus when the top chunk has a known URL (traceable authority).
    top_url_bonus: int = 1


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvidenceAssessment:
    """Result of assessing evidence sufficiency.

    Attributes
    ----------
    verdict : str
        One of VERDICT_SUFFICIENT, VERDICT_INSUFFICIENT, VERDICT_UNCERTAIN.
    score : int
        Aggregated signal score.  Higher = stronger evidence.
    signals : dict
        Named signal values recorded during assessment (for logging / reports).
    reason : str
        Human-readable summary of the most important signal(s).
    """
    verdict: str
    score: int
    signals: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------

class EvidenceAssessor:
    """Assess whether retrieved passages provide sufficient evidence.

    Parameters
    ----------
    config : EvidenceAssessorConfig, optional
        Policy thresholds.  Uses conservative defaults when None.
    """

    def __init__(self, config: Optional[EvidenceAssessorConfig] = None):
        self.config = config or EvidenceAssessorConfig()

    def assess(
        self,
        question: str,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        confidence_threshold: float = 0.30,
    ) -> EvidenceAssessment:
        """Assess evidence sufficiency for a retrieved document set.

        Parameters
        ----------
        question : str
            The user's question (used only for future NLI-based signals).
        retrieved_docs : list of (doc_dict, score)
            Retrieved chunks from the hybrid / FAISS retriever.
        confidence_threshold : float
            The pipeline's configured confidence threshold (used as a lower
            bound check).

        Returns
        -------
        EvidenceAssessment
        """
        cfg = self.config
        signals: Dict[str, Any] = {}
        score = 0
        reasons: List[str] = []

        if not retrieved_docs:
            return EvidenceAssessment(
                verdict=VERDICT_INSUFFICIENT,
                score=-10,
                signals={"n_docs": 0},
                reason="no documents retrieved",
            )

        scores = [s for _, s in retrieved_docs]
        docs = [d for d, _ in retrieved_docs]
        n = len(retrieved_docs)

        # --- Signal 1: Credible chunk count ---
        # Chunks are "credible" if their score clears the floor.
        # For RRF scores (hybrid), we use a relative heuristic:
        # the top chunk's score normalises the others.
        is_rrf = any(s < 0.1 for s in scores if s > 0)  # RRF scores are tiny
        if is_rrf:
            # For RRF: require at least half the pool to be non-zero
            credible_chunks = sum(1 for s in scores if s > 0)
        else:
            credible_chunks = sum(1 for s in scores if s >= cfg.credibility_floor)

        signals["credible_chunks"] = credible_chunks
        if credible_chunks >= cfg.min_credible_chunks:
            score += 2
        elif credible_chunks == 1:
            score += 1
        else:
            score -= 2
            reasons.append(f"only {credible_chunks} credible chunk(s)")

        # --- Signal 2: Strong relevance ---
        if is_rrf:
            top_score = max(scores)
            strong_chunks = n  # can't threshold RRF scores meaningfully
        else:
            top_score = max(scores)
            strong_chunks = sum(1 for s in scores if s >= cfg.strong_relevance_threshold)

        signals["strong_chunks"] = strong_chunks
        signals["top_score"] = round(top_score, 4)

        if not is_rrf:
            if strong_chunks >= cfg.min_strong_chunks:
                score += 2
                reasons.append(f"top score={top_score:.3f}")
            elif top_score >= confidence_threshold:
                score += 1
            else:
                score -= 2
                reasons.append(f"max score={top_score:.3f} below threshold")

        # --- Signal 3: Source diversity ---
        unique_doc_ids = len(set(d.get("doc_id", "") for d in docs))
        diversity = unique_doc_ids / max(n, 1)
        signals["source_diversity"] = round(diversity, 3)
        signals["unique_doc_ids"] = unique_doc_ids

        if diversity >= cfg.min_diversity and unique_doc_ids >= 2:
            score += 1
        elif unique_doc_ids == 1:
            score -= 1
            reasons.append("all chunks from single source")

        # --- Signal 4: URL / authority traceability ---
        n_with_url = sum(1 for d in docs if d.get("url"))
        n_with_year = sum(1 for d in docs if d.get("year"))
        signals["n_with_url"] = n_with_url
        signals["n_with_year"] = n_with_year

        # Top doc URL is a proxy for citable authority
        if docs and docs[0].get("url"):
            score += cfg.top_url_bonus
        else:
            score += cfg.no_url_penalty
            reasons.append("top source has no URL")

        if n_with_year == 0:
            score += cfg.no_year_penalty

        # --- Signal 5: HTML contamination check ---
        import re
        html_re = re.compile(r"<[a-zA-Z][^>]{0,60}>")
        n_html = sum(1 for d in docs if html_re.search(d.get("content", "")))
        signals["html_chunks"] = n_html
        if n_html > 0:
            score -= 1
            reasons.append(f"{n_html} chunk(s) contain HTML")

        # --- Final verdict ---
        signals["total_score"] = score

        if score >= cfg.sufficient_threshold:
            verdict = VERDICT_SUFFICIENT
            summary = f"score={score}: {'; '.join(reasons) or 'strong evidence'}"
        elif score <= cfg.insufficient_threshold:
            verdict = VERDICT_INSUFFICIENT
            summary = f"score={score}: {'; '.join(reasons) or 'weak evidence'}"
        else:
            verdict = VERDICT_UNCERTAIN
            summary = f"score={score}: uncertain — {'; '.join(reasons) or 'borderline'}"

        return EvidenceAssessment(
            verdict=verdict,
            score=score,
            signals=signals,
            reason=summary,
        )
