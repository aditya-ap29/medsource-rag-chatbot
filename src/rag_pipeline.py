"""
RAG Pipeline — Google Gemini backend only.
"""
import logging
import re
import threading
import time
import random
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple, Set
import google.genai as genai

logger = logging.getLogger("MedSource.rag_pipeline")

from src.vector_store import VectorStore
from src.models import Source, QueryResponse
from src.domain_gate import classify_domain, is_emergency, OUT_OF_SCOPE_RESPONSE


DISCLAIMER = "⚠️ This information is for educational purposes only and is not a substitute for professional medical advice."


# Maximum tokens of context to send to the LLM before truncating low-scoring chunks.
# System prompt (~300 tok) + question (~60 tok) + answer budget (max_tokens) must all
# fit inside the model's context window. 1500 tokens of context is safe.
_MAX_CONTEXT_TOKENS = 1500

# Maximum tokens reserved for conversation history in the prompt.
# Keeps history from crowding out retrieved context when prior turns are long.
_MAX_HISTORY_TOKENS = 400

# Prompt used to expand a short user query into multiple search phrasings.
# We ask for exactly 3 alternatives so the cost is predictable (one small LLM call).
_EXPANSION_PROMPT = """You are a medical search assistant. Rewrite the following medical question into 3 different search queries that would help retrieve relevant medical documents. Make each query use different vocabulary (e.g. clinical terms, layman terms, symptom-focused). Output only the 3 queries, one per line, with no numbering, labels, or extra text.

Question: {question}"""


class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline (Gemini backend)."""

    SYSTEM_PROMPT = """You are MedSource, a medical AI assistant providing fact-based, cited answers.
Answer ONLY from the retrieved documents below. Never invent or infer facts.

RULES:
1. Cite every factual claim as [DOC_X] (X = the document number in the context).
2. Do NOT add a "Sources:" list or a disclaimer — both are added automatically.
3. Do NOT diagnose or prescribe. Inform only.
4. If the documents do not answer the question, reply exactly:
   "I'm sorry, I don't have enough verified information to answer that safely."

OUTPUT FORMAT — use these exact headings, in order:
**Direct Answer:** One to three sentences answering the question directly, with citations.
**Explanation:** Two to five bullet points or short paragraphs with supporting detail and citations. Omit if the Direct Answer is already complete.
**Practical Guidance:** Relevant next steps, warning signs, or when to seek care — only when the question calls for it. Omit otherwise.

Adapt to question type:
- Symptoms: list by system/severity, cite each.
- Medications: include mechanism, common side-effects, contraindications if in the docs.
- Emergency keywords (chest pain, can't breathe, stroke, overdose, unconscious): open with urgent action guidance before detail.
- Insufficient evidence: use the exact refusal phrase from Rule 4.

{history}Retrieved Context:
{context}

User Query: {question}

Answer (follow the OUTPUT FORMAT above):"""

    # Used ONLY when retrieval finds nothing above the confidence threshold.
    GENERAL_KNOWLEDGE_PROMPT = """You are MedSource, a medical information assistant.
No verified documents were found for this question. Answer from general medical knowledge.

RULES:
1. Begin with: "Note: This answer is based on general medical knowledge, not verified documents."
2. Do NOT diagnose or prescribe. Do NOT add a closing disclaimer.
3. Be cautious; if unsure, say so.

OUTPUT FORMAT:
**Direct Answer:** One to three sentences.
**Explanation:** Bullet points if helpful. Omit if not needed.
**Practical Guidance:** When to seek care, only if relevant. Omit otherwise.

User Query: {question}

Answer:"""

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        top_k: int = 5,
        confidence_threshold: float = 0.75,
        enable_general_fallback: bool = True,
        max_history_turns: int = 3,
        enable_query_expansion: bool = False,
        enable_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        # Live PubMed retrieval
        enable_live_pubmed: bool = False,
        pubmed_email: str = "medsource@example.com",
        ncbi_api_key: str = "",
        live_pubmed_timeout_s: int = 5,
        live_pubmed_max_results: int = 5,
        # Legacy parameter — accepted but ignored (always Gemini now)
        llm_provider: str = "gemini",
        # --- NEW: Semantic classifier ---
        enable_semantic_classifier: bool = False,
        semantic_classifier_path: str = "./data/semantic_classifier_v1",
        ambiguous_handling: str = "warn",
        # --- NEW: Hybrid retrieval ---
        enable_hybrid_retrieval: bool = False,
        rrf_k: int = 60,
        rrf_dense_weight: float = 1.0,
        rrf_sparse_weight: float = 1.0,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        # --- NEW: Evidence assessment ---
        enable_evidence_assessment: bool = False,
    ):
        self.vector_store = vector_store
        self.model_name = model_name or "gemini-2.0-flash"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.enable_general_fallback = enable_general_fallback
        self.max_history_turns = max_history_turns
        self.enable_query_expansion = enable_query_expansion
        self.enable_reranker = enable_reranker
        self.reranker_model_name = reranker_model
        self._reranker = None  # lazy-loaded on first use

        # Live PubMed retrieval (optional, ephemeral — never writes to FAISS)
        self.enable_live_pubmed = enable_live_pubmed
        self._live_retriever = None  # lazy-initialised on first use
        self._live_pubmed_email = pubmed_email
        self._ncbi_api_key = ncbi_api_key
        self._live_pubmed_timeout_s = live_pubmed_timeout_s
        self._live_pubmed_max_results = live_pubmed_max_results

        # --- Semantic classifier (NEW) ---
        self.enable_semantic_classifier = enable_semantic_classifier
        self.semantic_classifier_path = semantic_classifier_path
        self.ambiguous_handling = ambiguous_handling
        self._semantic_clf = None  # lazy-loaded on first use

        # --- Hybrid retrieval (NEW) ---
        self.enable_hybrid_retrieval = enable_hybrid_retrieval
        self._hybrid_retriever = None  # lazy-initialised on first use
        self._rrf_k = rrf_k
        self._rrf_dense_weight = rrf_dense_weight
        self._rrf_sparse_weight = rrf_sparse_weight
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

        # --- Evidence assessment (NEW) ---
        self.enable_evidence_assessment = enable_evidence_assessment
        self._evidence_assessor = None  # lazy-created on first use

        # Bounded LRU response cache — avoids re-hitting FAISS + Gemini for
        # repeated identical questions. Key: (question, top_k, history_tuple).
        # Max 128 entries; oldest entry is evicted when the limit is reached.
        # The lock guards concurrent access from multiple Streamlit sessions
        # sharing the same @st.cache_resource pipeline instance.
        self._query_cache: OrderedDict = OrderedDict()
        self._query_cache_max = 128
        self._cache_lock = threading.Lock()

        self._client = genai.Client(api_key=api_key)
        logger.info("Gemini model initialised: %s", self.model_name)

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _expand_query(self, question: str) -> List[str]:
        """Return up to 3 alternative search phrasings for the question.

        Makes one cheap LLM call (low temperature, short output). Returns an
        empty list on any error so the caller falls back to the original query.
        Only called when enable_query_expansion=True.

        Short queries (≤ 2 words) are skipped — the LLM cannot meaningfully
        rephrase a single keyword like "flu" or "TB" into 3 variants.
        """
        if len(question.split()) <= 2:
            return []
        try:
            prompt = _EXPANSION_PROMPT.format(question=question)
            raw = self._generate(prompt)
            variants = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            variants = [v for v in variants if len(v) < 300 and not v.lower().startswith("error")]
            return variants[:3]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Lazy helpers for new components
    # ------------------------------------------------------------------

    def _get_hybrid_retriever(self):
        """Return a lazily-created HybridRetriever for the current vector store."""
        if self._hybrid_retriever is None:
            from src.hybrid_retriever import HybridRetriever, HybridConfig
            cfg = HybridConfig(
                enable_hybrid=self.enable_hybrid_retrieval,
                rrf_k=self._rrf_k,
                dense_weight=self._rrf_dense_weight,
                sparse_weight=self._rrf_sparse_weight,
                bm25_k1=self._bm25_k1,
                bm25_b=self._bm25_b,
            )
            self._hybrid_retriever = HybridRetriever(
                vector_store=self.vector_store,
                config=cfg,
            )
            logger.info(
                "HybridRetriever initialised (enable_hybrid=%s)",
                self.enable_hybrid_retrieval,
            )
        return self._hybrid_retriever

    def _get_semantic_clf(self):
        """Return the lazily-loaded SemanticClassifier, or None if unavailable."""
        if self._semantic_clf is not None:
            return self._semantic_clf

        from pathlib import Path
        from src.semantic_classifier import SemanticClassifier

        clf_path = Path(self.semantic_classifier_path)
        if not clf_path.exists():
            logger.warning(
                "Semantic classifier not found at '%s'. "
                "Run: python scripts/train_classifier.py to train it. "
                "Falling back to keyword gate.",
                clf_path,
            )
            return None

        try:
            clf = SemanticClassifier(
                embedding_model=self.vector_store.embedding_model,
                embedding_model_name=self.vector_store.embedding_model_name,
            )
            clf.load(str(clf_path))
            self._semantic_clf = clf
            logger.info("Semantic classifier loaded from %s", clf_path)
        except Exception as exc:
            logger.warning(
                "Failed to load semantic classifier (%s). "
                "Falling back to keyword gate.",
                exc,
            )
            return None

        return self._semantic_clf

    def _get_evidence_assessor(self):
        """Return the lazily-created EvidenceAssessor."""
        if self._evidence_assessor is None:
            from src.evidence_assessor import EvidenceAssessor
            self._evidence_assessor = EvidenceAssessor()
            logger.debug("EvidenceAssessor created.")
        return self._evidence_assessor

    # ------------------------------------------------------------------
    # Live PubMed retrieval
    # ------------------------------------------------------------------

    def _fetch_live_pubmed(
        self,
        question: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fetch fresh PubMed abstracts and return them as (doc_dict, score) pairs.

        Assigns a synthetic relevance score of `confidence_threshold` so live
        results pass the confidence gate without inflating the displayed score.
        They are marked with provenance="live_pubmed" in metadata.

        Returns an empty list on any error — calling code always checks for empty.
        """
        from src.live_retrieval import PubMedLiveRetriever

        if self._live_retriever is None:
            self._live_retriever = PubMedLiveRetriever(
                email=self._live_pubmed_email,
                api_key=self._ncbi_api_key,
                timeout_s=self._live_pubmed_timeout_s,
            )

        # Collect PMIDs already in the FAISS store to avoid duplicate passages.
        indexed_pmids = self._live_retriever.get_indexed_pmids(self.vector_store.documents)

        try:
            live_docs = self._live_retriever.search_and_fetch(
                question,
                max_results=min(k, self._live_pubmed_max_results),
                existing_pmids=indexed_pmids,
            )
        except Exception as exc:
            logger.warning("Live PubMed retrieval error: %s", exc)
            return []

        # Convert MedicalDocument objects to the same dict shape used by VectorStore
        results: List[Tuple[Dict[str, Any], float]] = []
        for live_doc in live_docs:
            doc_dict: Dict[str, Any] = {
                "doc_id": live_doc.doc_id,
                "chunk_id": f"{live_doc.doc_id}_live",
                "content": live_doc.content,
                "title": live_doc.title,
                "source": live_doc.source,
                "year": live_doc.year,
                "url": live_doc.url,
                "metadata": {**(live_doc.metadata or {}), "provenance": "live_pubmed"},
                "chunk_index": 0,
                "total_chunks": 1,
            }
            # Assign the confidence threshold as a floor score so the gate passes.
            results.append((doc_dict, self.confidence_threshold))

        return results

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _retrieve_with_expansion(
        self,
        question: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Retrieve top-k unique docs across the original query + expanded variants.

        Merges results by keeping the **best score** seen for each doc_id across
        all query variants (not just the first-found score).
        """
        queries = [question] + self._expand_query(question)

        best: Dict[str, Tuple[Dict[str, Any], float]] = {}
        for q in queries:
            candidates = self.vector_store.search(
                q,
                top_k=max(k * 4, 20),
                score_threshold=self.confidence_threshold,
            )
            for doc, score in candidates:
                doc_id = doc["doc_id"]
                if doc_id not in best or score > best[doc_id][1]:
                    best[doc_id] = (doc, score)

        merged = list(best.values())
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:k]

    def _rerank(
        self,
        question: str,
        candidates: List[Tuple[Dict[str, Any], float]],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Re-rank candidates with a cross-encoder and return the top_k results.

        Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — 23 MB, ~30 ms / 20 pairs CPU.
        Falls back to the original ranked list on any import or inference error.
        """
        if not candidates:
            return candidates
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                logger.info("Loading re-ranker: %s", self.reranker_model_name)
                self._reranker = CrossEncoder(self.reranker_model_name, max_length=512)

            pairs = [(question, doc["content"]) for doc, _ in candidates]
            ce_scores = self._reranker.predict(pairs)

            reranked = sorted(
                zip([doc for doc, _ in candidates], ce_scores.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            # Deduplicate by doc_id — keep only the highest-scored chunk per source.
            seen: Set[str] = set()
            deduped = []
            for doc, s in reranked:
                if doc["doc_id"] not in seen:
                    seen.add(doc["doc_id"])
                    deduped.append((doc, round(s, 4)))
                if len(deduped) >= top_k:
                    break
            return deduped
        except Exception as exc:
            logger.warning("Re-ranker failed (%s), using original ranking.", exc)
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _token_count(text: str) -> int:
        """Approximate token count via tiktoken; falls back to word-count heuristic."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text, disallowed_special=()))
        except ImportError:
            return len(text.split())

    def _format_context(
        self,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        max_tokens: int = _MAX_CONTEXT_TOKENS,
    ) -> str:
        """Format retrieved docs as numbered context blocks, capped at max_tokens."""
        context_parts = []
        used = 0
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            block = (
                f"[DOC_{i}] {doc['title']} ({doc['source']}, {doc.get('year', 'N/A')})\n"
                f"Content: {doc['content']}\n"
                f"Relevance Score: {score:.3f}\n"
            )
            block_tokens = self._token_count(block)
            if used + block_tokens > max_tokens:
                break
            context_parts.append(block)
            used += block_tokens
        return "\n".join(context_parts)

    @staticmethod
    def _format_history(
        history: Optional[List[Dict[str, str]]],
        max_turns: int,
        max_tokens: int = _MAX_HISTORY_TOKENS,
    ) -> str:
        """Render the last max_turns Q&A pairs as a compact conversation block.

        Token-budget aware: turns are added most-recent-first until the total
        token count would exceed max_tokens, then truncated.
        """
        if not history:
            return ""

        turns: List[tuple] = []
        i = len(history) - 1
        while i >= 0 and len(turns) < max_turns:
            if history[i]["role"] == "assistant" and i > 0 and history[i - 1]["role"] == "user":
                turns.append((history[i - 1]["content"], history[i]["content"]))
                i -= 2
            else:
                i -= 1

        if not turns:
            return ""

        header = "Prior conversation (most recent last):"
        budget = max_tokens - RAGPipeline._token_count(header) - 2

        lines = [header]
        used = 0
        for user_msg, asst_msg in reversed(turns):
            user_line = f"User: {user_msg}"
            asst_line = f"Assistant: {asst_msg}"
            pair_tokens = RAGPipeline._token_count(user_line + "\n" + asst_line)
            if used + pair_tokens > budget:
                for cut in (200, 100, 50):
                    preview = asst_msg[:cut].rsplit(" ", 1)[0] + "…"
                    asst_line = f"Assistant: {preview}"
                    pair_tokens = RAGPipeline._token_count(user_line + "\n" + asst_line)
                    if used + pair_tokens <= budget:
                        break
                else:
                    break
            lines.append(user_line)
            lines.append(asst_line)
            used += pair_tokens

        if len(lines) == 1:
            return ""

        lines.append("")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _clean_answer(answer: str, add_disclaimer: bool = True) -> str:
        """Strip model-added sources/disclaimers and append exactly one disclaimer."""
        if not answer:
            return answer

        answer = re.sub(r"\n+Sources:.*$", "", answer, flags=re.DOTALL | re.IGNORECASE).strip()
        disclaimer_pattern = re.escape(
            "This information is for educational purposes only and is not a substitute for professional medical advice."
        )
        answer = re.sub(r"⚠️?\s*" + disclaimer_pattern + r"\.?", "", answer, flags=re.IGNORECASE).strip()
        answer = re.sub(r"(?m)^\*\*\*", "* **", answer)
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        if add_disclaimer:
            answer = f"{answer}\n\n{DISCLAIMER}"
        return answer

    # ------------------------------------------------------------------
    # Gemini generation
    # ------------------------------------------------------------------

    # Retry configuration — used by _generate() for transient failures.
    # At most _RETRY_MAX attempts after the initial try (total = _RETRY_MAX + 1).
    # Delay doubles each attempt with ±25 % jitter, capped at _RETRY_CAP_S.
    _RETRY_MAX   = 3
    _RETRY_BASE_S = 1.0   # initial wait in seconds
    _RETRY_CAP_S  = 16.0  # maximum wait per attempt
    # HTTP / SDK error strings that indicate a transient failure worth retrying.
    _TRANSIENT_SIGNALS = (
        "429", "quota", "resource_exhausted",
        "503", "internal", "unavailable",
        "timeout", "deadline",
        "server error",
    )

    def _is_transient(self, exc: Exception) -> bool:
        """Return True when *exc* looks like a retryable Gemini / network error."""
        msg = str(exc).lower()
        return any(sig in msg for sig in self._TRANSIENT_SIGNALS)

    def _generate(self, prompt: str) -> str:
        """Generate text with the configured Gemini model.

        Retries up to _RETRY_MAX times for transient errors (429, 503, timeout)
        using exponential backoff with ±25 % jitter, capped at _RETRY_CAP_S.

        Raises:
            RuntimeError: On any Gemini API or SDK error after exhausting retries,
                          so callers can distinguish a failed generation from a
                          valid (empty) answer.  The message is human-readable and
                          does not expose internal SDK details.
        """
        last_exc: Exception | None = None
        for attempt in range(self._RETRY_MAX + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                last_exc = e
                if attempt < self._RETRY_MAX and self._is_transient(e):
                    delay = min(
                        self._RETRY_BASE_S * (2 ** attempt),
                        self._RETRY_CAP_S,
                    ) * (0.75 + random.random() * 0.5)  # ±25 % jitter
                    logger.warning(
                        "Gemini transient error (attempt %d/%d, retry in %.1f s): %s",
                        attempt + 1, self._RETRY_MAX + 1, delay, e,
                    )
                    time.sleep(delay)
                else:
                    break

        # Produce a safe human-readable message — never expose raw SDK internals.
        raw = str(last_exc)
        if "429" in raw or "quota" in raw.lower() or "resource_exhausted" in raw.lower():
            user_msg = (
                "The AI service is temporarily rate-limited. "
                "Please wait a moment and try again."
            )
        elif "timeout" in raw.lower() or "deadline" in raw.lower():
            user_msg = (
                "The AI service did not respond in time. "
                "Please try again in a few seconds."
            )
        elif "503" in raw or "unavailable" in raw.lower() or "internal" in raw.lower():
            user_msg = (
                "The AI service is temporarily unavailable. "
                "Please try again shortly."
            )
        else:
            user_msg = "The AI service returned an error. Please try again."

        raise RuntimeError(f"LLM generation failed: {user_msg}") from last_exc

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_score(raw: float, is_ce_domain: bool, confidence_threshold: float) -> float:
        """Map a single raw retrieval score to a [0, 1] display value.

        BGE cosine: linear scale — threshold → 0.0, 0.85 (ceiling) → 1.0.
        CE score  : sigmoid centred at 0 — score 0 → 0.5, +6 → ~1.0, -6 → ~0.0.
        """
        import math
        if is_ce_domain:
            return round(min(1.0 / (1.0 + math.exp(-raw / 3.0)), 1.0), 4)
        else:
            BGE_PRACTICAL_MAX = 0.85
            return round(min(max(0.0, (raw - confidence_threshold) /
                               (BGE_PRACTICAL_MAX - confidence_threshold)), 1.0), 4)

    def _calculate_confidence(
        self,
        retrieved_docs: List[Tuple[Dict[str, Any], float]],
        answer: str,
    ) -> float:
        """Calculate confidence score based on retrieval scores and answer quality."""
        if not retrieved_docs:
            return 0.0

        scores = [score for _, score in retrieved_docs]
        avg_raw = sum(scores) / len(scores)
        is_ce_domain = any(s > 1.0 or s < 0.0 for s in scores)
        normalised = self._normalise_score(avg_raw, is_ce_domain, self.confidence_threshold)
        confidence = normalised

        has_citations = "[DOC_" in answer
        uncertainty_phrases = [
            "don't have enough", "cannot answer",
            "insufficient information", "not enough verified"
        ]
        admits_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

        if has_citations:
            confidence = min(confidence * 1.1, 1.0)
        if admits_uncertainty:
            confidence *= 0.5

        return round(confidence, 3)

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryResponse:
        """Process a query through the RAG pipeline.

        Args:
            question: The user's medical question.
            top_k: Override the default number of retrieved documents.
            history: Optional list of prior chat messages in the format
                     [{"role": "user"|"assistant", "content": "..."}].

        Repeated identical (question, top_k, history) combinations are served
        from an in-process LRU cache (max 128 entries).
        """
        k = top_k or self.top_k

        # Build a hashable cache key using only role+content from each history turn.
        # History entries are full assistant message dicts that include list-valued
        # fields (sources, suggestions) which would make the key unhashable.
        history_key: tuple = tuple(
            (turn.get("role", ""), turn.get("content", ""))
            for turn in (history or [])
        )
        cache_key = (question.strip().lower(), k, history_key)
        with self._cache_lock:
            if cache_key in self._query_cache:
                self._query_cache.move_to_end(cache_key)
                logger.debug("Cache hit for question: %.60s", question)
                return self._query_cache[cache_key]

        # Retrieval phase
        # When enable_hybrid_retrieval=True, the HybridRetriever fuses FAISS+BM25.
        # When False, it delegates to vector_store.search() unchanged.
        retrieval_start = time.time()
        retriever = self._get_hybrid_retriever()

        if self.enable_query_expansion:
            raw_pool = self._retrieve_with_expansion(question, max(k * 4, 20))
        else:
            # When hybrid retrieval is enabled, RRF scores are on a different
            # scale (~0.01–0.05) than FAISS cosine scores (~0.3–0.9). Passing
            # the confidence_threshold as a score_threshold would filter out
            # all hybrid results. Pass 0.0 so the hybrid retriever returns all
            # candidates; the evidence assessor or confidence gate below handles
            # quality filtering. For pure FAISS mode (enable_hybrid=False), the
            # existing score_threshold is passed through unchanged.
            _score_threshold = 0.0 if self.enable_hybrid_retrieval else self.confidence_threshold
            raw_candidates = retriever.search(
                question,
                top_k=max(k * 4, 20),
                score_threshold=_score_threshold,
            )
            seen_doc_ids: Set[str] = set()
            raw_pool = []
            for doc, score in raw_candidates:
                if doc["doc_id"] in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc["doc_id"])
                raw_pool.append((doc, score))

        # Live PubMed augmentation:
        #   - Always augment when enable_live_pubmed=True and the local pool is below k.
        #   - Deduplicates by doc_id against the local pool before merging.
        if self.enable_live_pubmed:
            local_doc_ids: Set[str] = {doc["doc_id"] for doc, _ in raw_pool}
            live_pool = self._fetch_live_pubmed(question, k)
            for live_doc, live_score in live_pool:
                if live_doc["doc_id"] not in local_doc_ids:
                    raw_pool.append((live_doc, live_score))
                    local_doc_ids.add(live_doc["doc_id"])
            # Re-sort by score after merge so best results float to top
            raw_pool.sort(key=lambda x: x[1], reverse=True)

        if self.enable_reranker and raw_pool:
            retrieved_docs = self._rerank(question, raw_pool, k)
        else:
            retrieved_docs = raw_pool[:k]

        retrieval_time = (time.time() - retrieval_start) * 1000

        # ------------------------------------------------------------------
        # Domain-relevance gate
        # Emergency bypass is always applied first via is_emergency() regardless
        # of which classifier is active — this is a separate safety layer.
        #
        # When enable_semantic_classifier=True AND the model is loaded:
        #   - Uses SemanticClassifier for medical/out_of_scope/ambiguous.
        #   - Falls back to keyword gate if the model is unavailable.
        # When enable_semantic_classifier=False:
        #   - Uses existing keyword gate (unchanged baseline behaviour).
        # ------------------------------------------------------------------
        # Emergency check is always first and never depends on the classifier
        if is_emergency(question):
            domain_label, domain_reason = "emergency", "emergency keyword matched"
        elif self.enable_semantic_classifier:
            clf = self._get_semantic_clf()
            if clf is not None:
                sem_label, sem_conf, sem_reason = clf.predict(question)
                logger.debug("Semantic classifier: %s (conf=%.3f, %s)", sem_label, sem_conf, sem_reason)
                if sem_label == "out_of_scope" and sem_conf >= 0.7:
                    domain_label, domain_reason = "out_of_scope", sem_reason
                elif sem_label == "ambiguous":
                    domain_label, domain_reason = "ambiguous", sem_reason
                else:
                    # medical or low-confidence → keyword gate as safety net
                    kw_label, kw_reason = classify_domain(question, retrieved_docs)
                    if kw_label == "out_of_scope":
                        domain_label, domain_reason = "out_of_scope", kw_reason
                    else:
                        domain_label, domain_reason = "medical", sem_reason
            else:
                # Classifier not available — fall back to keyword gate
                domain_label, domain_reason = classify_domain(question, retrieved_docs)
        else:
            domain_label, domain_reason = classify_domain(question, retrieved_docs)

        logger.debug("Domain gate: %s (%s)", domain_label, domain_reason)

        # Handle ambiguous classification
        if domain_label == "ambiguous":
            if self.ambiguous_handling == "refuse":
                result = QueryResponse(
                    question=question,
                    answer=(
                        "Your question is a bit ambiguous. Could you clarify whether you're "
                        "asking about a medical condition, symptom, or treatment? "
                        "I'm happy to help once I understand your medical question better."
                        f"\n\n{DISCLAIMER}"
                    ),
                    sources=[],
                    confidence=0.0,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=0.0,
                    total_time_ms=retrieval_time,
                    warning="⚠️ Question intent is ambiguous.",
                )
                with self._cache_lock:
                    self._query_cache[cache_key] = result
                    if len(self._query_cache) > self._query_cache_max:
                        self._query_cache.popitem(last=False)
                return result
            # "pass" or "warn" — continue as medical; warning added later
            domain_label = "medical"

        if domain_label == "out_of_scope":
            result = QueryResponse(
                question=question,
                answer=OUT_OF_SCOPE_RESPONSE,
                sources=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=0.0,
                total_time_ms=retrieval_time,
                warning="⚠️ This question is outside MedSource's medical scope.",
            )
            with self._cache_lock:
                self._query_cache[cache_key] = result
                if len(self._query_cache) > self._query_cache_max:
                    self._query_cache.popitem(last=False)
            return result

        # ------------------------------------------------------------------
        # Evidence-sufficiency assessment (NEW — runs when enabled)
        # Separate from domain classification: a medical question may still
        # have insufficient evidence to answer safely.
        # When disabled, the existing confidence threshold is the sole gate.
        # ------------------------------------------------------------------
        evidence_warning: Optional[str] = None
        if self.enable_evidence_assessment and retrieved_docs:
            assessor = self._get_evidence_assessor()
            assessment = assessor.assess(
                question=question,
                retrieved_docs=retrieved_docs,
                confidence_threshold=self.confidence_threshold,
            )
            logger.debug(
                "Evidence assessment: verdict=%s score=%d reason=%s",
                assessment.verdict, assessment.score, assessment.reason,
            )
            from src.evidence_assessor import VERDICT_INSUFFICIENT, VERDICT_UNCERTAIN
            if assessment.verdict == VERDICT_INSUFFICIENT:
                # Treat as empty retrieval — fall through to general fallback logic
                retrieved_docs = []
            elif assessment.verdict == VERDICT_UNCERTAIN:
                evidence_warning = (
                    "⚠️ Retrieved evidence is limited. The answer below may be incomplete. "
                    "Please verify with a healthcare professional."
                )

        # Low-confidence / empty retrieval path
        if not retrieved_docs or all(score < self.confidence_threshold for _, score in retrieved_docs):
            if not self.enable_general_fallback:
                result = QueryResponse(
                    question=question,
                    answer=self._clean_answer(
                        "I'm sorry, I don't have enough verified information to answer that safely. "
                        "Please consult with a healthcare professional for accurate medical advice."
                    ),
                    sources=[],
                    confidence=0.0,
                    retrieval_time_ms=retrieval_time,
                    generation_time_ms=0.0,
                    total_time_ms=retrieval_time,
                    warning="⚠️ Insufficient verified information available. Please consult a healthcare professional.",
                )
                with self._cache_lock:
                    self._query_cache[cache_key] = result
                    if len(self._query_cache) > self._query_cache_max:
                        self._query_cache.popitem(last=False)
                return result

            gen_start = time.time()
            try:
                answer = self._generate(self.GENERAL_KNOWLEDGE_PROMPT.format(question=question))
            except RuntimeError as gen_err:
                generation_time = (time.time() - gen_start) * 1000
                raise RuntimeError(str(gen_err)) from gen_err
            generation_time = (time.time() - gen_start) * 1000

            result = QueryResponse(
                question=question,
                answer=self._clean_answer(answer),
                sources=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time,
                warning="⚠️ No verified source in our document database matched this question. "
                        "The answer above is general knowledge, not a cited/sourced answer. "
                        "Please consult a healthcare professional.",
            )
            with self._cache_lock:
                self._query_cache[cache_key] = result
                if len(self._query_cache) > self._query_cache_max:
                    self._query_cache.popitem(last=False)
            return result

        history_block = self._format_history(history, self.max_history_turns)
        context = self._format_context(retrieved_docs)
        prompt = self.SYSTEM_PROMPT.format(
            history=history_block, context=context, question=question
        )

        # Generation phase
        generation_start = time.time()
        try:
            answer = self._generate(prompt)
        except RuntimeError as gen_err:
            generation_time = (time.time() - generation_start) * 1000
            raise RuntimeError(str(gen_err)) from gen_err
        generation_time = (time.time() - generation_start) * 1000

        uncertainty_phrases = [
            "don't have enough", "cannot answer",
            "insufficient information", "not enough verified"
        ]
        model_admits_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)

        if model_admits_uncertainty and self.enable_general_fallback:
            result = QueryResponse(
                question=question,
                answer=self._clean_answer(answer),
                sources=[],
                confidence=0.0,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                total_time_ms=retrieval_time + generation_time,
                warning="⚠️ Retrieved documents were topically related but did not directly answer "
                        "this question. The answer above is general knowledge, not a cited/sourced "
                        "answer. Please consult a healthcare professional.",
            )
            with self._cache_lock:
                self._query_cache[cache_key] = result
                if len(self._query_cache) > self._query_cache_max:
                    self._query_cache.popitem(last=False)
            return result

        confidence = self._calculate_confidence(retrieved_docs, answer)

        all_scores = [s for _, s in retrieved_docs]
        is_ce = any(s > 1.0 or s < 0.0 for s in all_scores)

        # Build source objects only for [DOC_X] citations actually present in the answer.
        # This prevents misaligned citations when the LLM skips or re-orders doc indices.
        cited_indices: Set[int] = set()
        for m in re.finditer(r"\[DOC_(\d+)\]", answer):
            idx = int(m.group(1))
            if 1 <= idx <= len(retrieved_docs):
                cited_indices.add(idx)

        sources = []
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            if cited_indices and i not in cited_indices:
                # Omit docs the model did not actually cite to avoid a misleading source list.
                # If the model cited nothing at all we fall back to including all retrieved docs.
                continue
            meta = doc.get("metadata") or {}
            prov = meta.get("provenance", "indexed")
            source = Source(
                doc_id=f"DOC_{i}",
                title=doc["title"],
                year=doc.get("year"),
                url=doc.get("url"),
                relevance_score=self._normalise_score(score, is_ce, self.confidence_threshold),
                excerpt=doc["content"],
                provenance=prov,
            )
            sources.append(source)

        # Compose final warning: combine ambiguous_handling=warn + evidence warning
        base_warning = "Always consult with qualified healthcare professionals for medical decisions."
        extra_parts = []
        if domain_reason and "ambiguous" in domain_reason.lower() and self.ambiguous_handling == "warn":
            extra_parts.append("⚠️ This question had ambiguous intent; answer may be general.")
        if evidence_warning:
            extra_parts.append(evidence_warning)
        final_warning = (
            " ".join(extra_parts) + " " + base_warning if extra_parts else base_warning
        ).strip()

        result = QueryResponse(
            question=question,
            answer=self._clean_answer(answer),
            sources=sources,
            confidence=confidence,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=retrieval_time + generation_time,
            warning=final_warning,
        )

        with self._cache_lock:
            self._query_cache[cache_key] = result
            if len(self._query_cache) > self._query_cache_max:
                self._query_cache.popitem(last=False)

        return result
