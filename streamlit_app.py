"""
MedSource — Streamlit UI

Security model:
  • All API keys read from .env only — never from the UI.
  • Admin features (data ingestion) hidden behind ADMIN_PASSWORD in .env.
  • Users can upload personal documents for session-scoped Q&A (not stored).

Run with:  streamlit run streamlit_app.py
"""
import html as _html
import os
import re
import sys
import time
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from src.config import get_settings
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from src.feedback_store import FeedbackStore
from src.data_ingestion import (
    MedicalDocument,
    MedicalCorpusBuilder,
    DocumentLoader,
    PubMedLoader,
    WHOGHOLoader,
    MedlinePlusLoader,
    OpenFDADrugLoader,
    ClinicalTrialsLoader,
)

settings = get_settings()
_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSource — Medical AI Assistant",
    page_icon=":material/ecg_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Theme detection — re-evaluated every rerun so switching works live
# ─────────────────────────────────────────────────────────────────────────────
try:
    _theme_type = st.context.theme.type or "light"
except Exception:
    _theme_type = "light"

_IS_DARK = _theme_type == "dark"

# Design tokens — mirrored from config.toml so custom HTML components stay in sync
if _IS_DARK:
    _T = {
        "bg":           "#111318",
        "bg2":          "#1c1f26",
        "primary":      "#5aac7d",
        "text":         "#e4e7ed",
        "text2":        "#9ca3af",
        "border":       "#2e3340",
        "border2":      "#3d4251",
        "ok_bg":        "#1a3328",  "ok_fg":    "#7bc99a",
        "warn_bg":      "#2d1e06",  "warn_fg":  "#fbbf24",
        "err_bg":       "#2a0e0e",  "err_fg":   "#f87171",
        "emerg_title":  "#f87171",
        "caution_body": "#fbbf24",
    }
else:
    _T = {
        "bg":           "#ffffff",
        "bg2":          "#f5f6f7",
        "primary":      "#3d7a5c",
        "text":         "#1a1f2e",
        "text2":        "#6b7280",
        "border":       "#e1e4e8",
        "border2":      "#c9cdd3",
        "ok_bg":        "#e8f5ee",  "ok_fg":    "#2a5c42",
        "warn_bg":      "#fef9ec",  "warn_fg":  "#92400e",
        "err_bg":       "#fef2f2",  "err_fg":   "#991b1b",
        "emerg_title":  "#991b1b",
        "caution_body": "#78350f",
    }

# ─────────────────────────────────────────────────────────────────────────────
# CSS — scoped overrides; theme colours live in config.toml
# Active nav state uses st-key-nav_<tab> classes (Streamlit auto-generates
# .st-key-<key> for every widget that has key=).
# ─────────────────────────────────────────────────────────────────────────────
_active_tab_css = st.session_state.get("active_tab", "chat")

# Build one rule per nav key so only the current page is highlighted
_nav_active_rules = "\n".join(
    f".st-key-nav_{tab} button {{ background: {_T['bg2']} !important; "
    f"color: {_T['primary']} !important; font-weight: 500 !important; "
    f"border-left: 3px solid {_T['primary']} !important; padding-left: 7px !important; }}"
    for tab in ["chat", "upload", "symptom", "insights", "admin"]
    if tab == _active_tab_css
)

_CSS = f"""
<style>
/* ── Layout ─────────────────────────────────────────────────────── */
.block-container {{
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}}

/* ── Sidebar nav — ghost base + active highlight via key class ─── */
[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    height: 36px !important;
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0 10px !important;
    color: {_T["text"]} !important;
    transition: background 0.1s;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: {_T["border"]} !important;
}}
{_nav_active_rules}

/* ── New chat button ─────────────────────────────────────────────── */
.st-key-new_chat_btn button {{
    background: {_T["bg2"]} !important;
    color: {_T["primary"]} !important;
    border: 1px solid {_T["border"]} !important;
    border-radius: 6px !important;
    font-size: 13.5px !important;
    font-weight: 600 !important;
    height: 40px !important;
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0 10px !important;
    margin-bottom: 4px !important;
}}
.st-key-new_chat_btn button:hover {{
    border-color: {_T["primary"]} !important;
}}

/* ── Sidebar section labels ──────────────────────────────────────── */
.sb-label {{
    font-size: 11px;
    font-weight: 600;
    color: {_T["text2"]};
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 12px 2px 4px;
    display: block;
}}

/* ── Chat input container ────────────────────────────────────────── */
[data-testid="stChatInputContainer"] {{
    max-width: 780px !important;
    margin: 0 auto !important;
}}

/* ── Chat messages ───────────────────────────────────────────────── */
[data-testid="stChatMessage"] {{
    max-width: 780px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}}

/* ── Welcome suggestion cards ────────────────────────────────────── */
.st-key-card_0 button, .st-key-card_1 button,
.st-key-card_2 button, .st-key-card_3 button {{
    height: auto !important;
    min-height: 60px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    line-height: 1.4 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    overflow-wrap: anywhere !important;
}}

/* ── Medical safety notices ──────────────────────────────────────── */
.mi-emergency {{
    background: {_T["err_bg"]};
    border: 1.5px solid {_T["err_fg"]};
    border-left: 5px solid {_T["err_fg"]};
    border-radius: 8px; padding: 14px 16px; margin-bottom: 12px;
}}
.mi-emergency-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {_T["err_fg"]}; margin-bottom: 4px;
}}
.mi-emergency-title {{
    font-size: 15px; font-weight: 600;
    color: {_T["emerg_title"]}; margin-bottom: 8px;
}}
.mi-emergency-body {{ font-size: 14px; color: {_T["emerg_title"]}; line-height: 1.6; }}
.mi-emergency-number {{
    display: inline-block; background: {_T["err_fg"]}; color: #fff;
    font-weight: 700; font-size: 15px; border-radius: 5px;
    padding: 2px 10px; margin: 6px 4px 6px 0;
}}
.mi-caution {{
    background: {_T["warn_bg"]};
    border: 1.5px solid {_T["warn_fg"]};
    border-left: 4px solid {_T["warn_fg"]};
    border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
}}
.mi-caution-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {_T["warn_fg"]}; margin-bottom: 4px;
}}
.mi-caution-body {{ font-size: 14px; color: {_T["caution_body"]}; line-height: 1.6; }}
.mi-uncertain {{
    background: {_T["bg2"]};
    border: 1px solid {_T["border2"]};
    border-left: 4px solid {_T["text2"]};
    border-radius: 8px; padding: 10px 16px; margin-bottom: 10px;
}}
.mi-uncertain-label {{
    font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: {_T["text2"]}; margin-bottom: 4px;
}}
.mi-uncertain-body {{ font-size: 13px; color: {_T["text2"]}; line-height: 1.6; }}

/* ── Answer key-takeaway card ────────────────────────────────────── */
.mi-answer-eyebrow {{
    font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: {_T["primary"]}; margin-bottom: 4px;
}}
.mi-answer-key {{
    font-size: 15px; font-weight: 600; color: {_T["text"]}; line-height: 1.55;
    background: {_T["ok_bg"]}; border-left: 3px solid {_T["primary"]};
    border-radius: 0 6px 6px 0; padding: 8px 12px; margin-bottom: 12px;
}}

/* ── Source cards ────────────────────────────────────────────────── */
.mi-src {{
    background: {_T["bg2"]}; border: 1px solid {_T["border"]};
    border-left: 3px solid {_T["primary"]};
    border-radius: 6px; padding: 8px 12px; margin-bottom: 6px;
}}
.mi-src-title {{ font-weight: 600; color: {_T["text"]}; font-size: 13px; line-height: 1.4; }}
.mi-src-meta {{
    color: {_T["text2"]}; font-size: 12px; margin: 3px 0 4px;
    display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
}}
.mi-src-meta a {{ color: {_T["primary"]} !important; text-decoration: none; }}
.mi-src-meta a:hover {{ text-decoration: underline; }}
.mi-src-rel {{ font-weight: 600; color: {_T["primary"]}; font-size: 11px; }}
.mi-bar-bg  {{ background: {_T["border"]}; border-radius: 2px; height: 3px; width: 100%; margin-top: 4px; }}
.mi-bar-fill{{ background: {_T["primary"]}; border-radius: 2px; height: 3px; }}
.mi-live-badge {{
    display: inline-block; background: {_T["ok_bg"]}; color: {_T["ok_fg"]};
    font-size: 10px; font-weight: 600; padding: 1px 6px;
    border-radius: 999px; letter-spacing: .03em; vertical-align: middle;
}}

/* ── Confidence badges ───────────────────────────────────────────── */
.mi-badge {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 999px; font-size: 11.5px; font-weight: 500;
}}
.mi-badge-high {{ background: {_T["ok_bg"]};  color: {_T["ok_fg"]};  }}
.mi-badge-med  {{ background: {_T["warn_bg"]}; color: {_T["warn_fg"]}; }}
.mi-badge-low  {{ background: {_T["err_bg"]};  color: {_T["err_fg"]};  }}
.mi-badge-gen  {{ background: {_T["bg2"]};     color: {_T["text2"]};  }}

/* ── Per-response disclaimer ─────────────────────────────────────── */
.mi-response-footer {{
    font-size: 12px; color: {_T["text2"]};
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid {_T["border"]}; line-height: 1.5;
}}

/* ── Error state ─────────────────────────────────────────────────── */
.mi-error-state {{
    background: {_T["bg2"]}; border: 1px solid {_T["border2"]};
    border-radius: 8px; padding: 12px 16px; font-size: 14px; color: {_T["text2"]};
}}

/* ── Upload hint ─────────────────────────────────────────────────── */
.mi-upload-hint {{
    background: {_T["ok_bg"]}; border: 1px dashed {_T["primary"]};
    border-radius: 8px; padding: 12px 16px; font-size: 14px;
    color: {_T["text"]}; margin-bottom: 12px;
}}
.mi-upload-hint b {{ color: {_T["primary"]}; }}

/* ── Insights metric cards ───────────────────────────────────────── */
.mi-metric {{
    background: {_T["bg"]}; border-radius: 8px;
    border: 1px solid {_T["border"]}; border-top: 2px solid {_T["primary"]};
    padding: 12px 10px; text-align: center;
}}
.mi-metric h3 {{ font-size: 22px; font-weight: 700; color: {_T["text"]}; margin: 0 0 4px; }}
.mi-metric p  {{ font-size: 12px; color: {_T["text2"]}; margin: 0; }}

/* ── Severity chips ──────────────────────────────────────────────── */
.sev-mild     {{ background:{_T["ok_bg"]};   color:{_T["ok_fg"]};   border:1px solid; border-radius:5px; padding:2px 8px; font-size:12px; font-weight:500; }}
.sev-moderate {{ background:{_T["warn_bg"]};  color:{_T["warn_fg"]};  border:1px solid; border-radius:5px; padding:2px 8px; font-size:12px; font-weight:500; }}
.sev-severe   {{ background:{_T["err_bg"]};   color:{_T["err_fg"]};   border:1px solid; border-radius:5px; padding:2px 8px; font-size:12px; font-weight:500; }}

/* ── Bottom disclaimer ───────────────────────────────────────────── */
.ms-disclaimer {{
    text-align: center; font-size: 12px; color: {_T["text2"]};
    padding: 10px 16px 14px; line-height: 1.5; max-width: 780px; margin: 0 auto;
}}

/* ── Reduced motion ──────────────────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {{
    * {{ transition: none !important; animation: none !important; }}
}}
</style>
"""
st.html(_CSS)


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up suggestion lookup
# ─────────────────────────────────────────────────────────────────────────────
_FOLLOW_UPS = {
    "diabetes":     ["What are the symptoms of low blood sugar?", "How is type 1 different from type 2?", "What foods should a diabetic avoid?"],
    "hypertension": ["What lifestyle changes lower blood pressure?", "What are hypertension medications?", "Can hypertension cause a stroke?"],
    "asthma":       ["What triggers an asthma attack?", "How does a rescue inhaler work?", "Is asthma hereditary?"],
    "anemia":       ["What foods increase iron levels?", "What are the types of anemia?", "Can anemia cause fatigue?"],
    "heart":        ["What are the warning signs of a heart attack?", "How is heart failure treated?", "What is a normal heart rate?"],
    "cancer":       ["What are common cancer screening tests?", "What is chemotherapy?", "What are early warning signs of cancer?"],
    "depression":   ["What are treatments for depression?", "How is depression diagnosed?", "Depression vs grief — what's different?"],
    "kidney":       ["What foods should someone with CKD avoid?", "What are the stages of kidney disease?", "Can kidney disease be reversed?"],
    "thyroid":      ["What are symptoms of hypothyroidism?", "What is TSH and what does it measure?", "How is thyroid disease treated?"],
}


def _get_suggestions(question: str, answer: str) -> list[str]:
    combined = (question + " " + answer).lower()
    for kw, sugs in _FOLLOW_UPS.items():
        if kw in combined:
            return sugs
    return [
        "What are the risk factors for this condition?",
        "When should I see a doctor?",
        "Are there lifestyle changes that help?",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Answer parser
# ─────────────────────────────────────────────────────────────────────────────
def _parse_answer(answer: str) -> tuple[str, str]:
    """Split answer text into (key_takeaway, body)."""
    disclaimer_re = re.compile(r"\n*⚠️.*educational purposes only.*$", re.DOTALL | re.IGNORECASE)
    clean = disclaimer_re.sub("", answer).strip()
    if not clean:
        return "", answer

    da_re = re.compile(
        r"\*\*Direct Answer:\*\*\s*(.+?)(?=\n\s*\*\*|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    m = da_re.search(clean)
    if m:
        takeaway = m.group(1).strip()
        body = clean[m.end():].strip()
        sent_end = re.search(r"(?<=[.!?])\s", takeaway)
        if sent_end and sent_end.start() > 20:
            takeaway = takeaway[: sent_end.start()].strip()
        return takeaway, body if body else ""

    parts = clean.split("\n\n", 1)
    if len(parts) == 2 and len(parts[0]) < len(parts[1]) * 0.6:
        return parts[0].strip(), parts[1].strip()

    return "", clean


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base…")
def load_vector_store():
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        embedding_batch_size=settings.embedding_batch_size,
    )
    if os.path.exists(settings.vector_store_path):
        try:
            vs.load(settings.vector_store_path)
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
    return vs


@st.cache_resource(show_spinner="Starting AI pipeline…")
def build_pipeline(vs_chunk_count: int):
    """Build and cache the RAG pipeline (keyed on chunk count to invalidate on new data)."""
    vs = st.session_state.vector_store
    return RAGPipeline(
        vector_store=vs,
        model_name=settings.llm_model,
        api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        top_k=settings.top_k_documents,
        confidence_threshold=settings.retrieval_confidence_threshold,
        enable_general_fallback=settings.enable_general_fallback,
        enable_query_expansion=settings.enable_query_expansion,
        enable_reranker=settings.enable_reranker,
        reranker_model=settings.reranker_model,
        enable_live_pubmed=settings.enable_live_pubmed,
        pubmed_email=settings.pubmed_email,
        ncbi_api_key=settings.ncbi_api_key,
        live_pubmed_timeout_s=settings.live_pubmed_timeout_s,
        live_pubmed_max_results=settings.live_pubmed_max_results,
    )


def _build_doc_pipeline(doc_vs: VectorStore) -> RAGPipeline:
    """Build a one-off pipeline scoped to an uploaded-document vector store."""
    return RAGPipeline(
        vector_store=doc_vs,
        model_name=settings.llm_model,
        api_key=settings.gemini_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.max_tokens,
        top_k=min(settings.top_k_documents, 3),
        confidence_threshold=0.20,
        enable_general_fallback=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core query execution — single authoritative path used by chat, follow-ups,
# retry, and regenerate.  Returns a dict ready to push onto messages, or
# raises an exception that the caller should surface.
# ─────────────────────────────────────────────────────────────────────────────
def _execute_query(question: str, *, bypass_cache: bool = False) -> dict:
    """Run RAG + LLM for *question* and return a complete assistant message dict.

    Args:
        question: The user's medical question.
        bypass_cache: When True the pipeline's LRU cache is bypassed so
            regeneration always produces a fresh response.

    Raises RuntimeError with a human-readable description on failure so the
    caller can display it without exposing stack traces.
    """
    active_vs = st.session_state.user_doc_vs or st.session_state.vector_store
    if not active_vs.documents:
        raise RuntimeError(
            "The knowledge base has no indexed documents. "
            "Upload a document via **Upload & Ask**, or ask an administrator to add sources."
        )

    try:
        pipe = (
            _build_doc_pipeline(active_vs)
            if st.session_state.user_doc_vs
            else build_pipeline(len(active_vs.documents))
        )

        # Bypass the pipeline's LRU cache for regeneration so a fresh answer
        # is generated rather than returning the identical cached response.
        if bypass_cache:
            # Mirror the cache-key logic in RAGPipeline.query(): use only role+content
            # so the key is hashable even when history entries contain list-valued fields
            # (sources, suggestions) from a prior assistant message.
            history_key: tuple = tuple(
                (t.get("role", ""), t.get("content", ""))
                for t in (st.session_state.messages or [])
            )
            stale_key = (question.strip().lower(), settings.top_k_documents, history_key)
            with pipe._cache_lock:
                pipe._query_cache.pop(stale_key, None)

        response = pipe.query(
            question,
            top_k=settings.top_k_documents,
            history=st.session_state.messages,
        )
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc

    if response is None:
        raise RuntimeError("The pipeline returned no response. Please try again.")

    is_emergency, emergency_label = _detect_emergency(question)
    is_med_risk = _detect_medication_risk(question)
    classification = _classify_response(response)
    summary, body = _parse_answer(response.answer)

    st.session_state.msg_counter += 1
    msg_id = st.session_state.msg_counter

    suggestions: list[str] = []
    if not is_emergency:
        suggestions = _get_suggestions(question, response.answer)

    return {
        "role": "assistant",
        "content": response.answer,
        "summary": summary,
        "body": body,
        "sources": response.sources,
        "badge_html": confidence_badge(response.confidence, classification == "general"),
        "classification": classification,
        "emergency_label": emergency_label if is_emergency else "",
        "medication_risk": is_med_risk,
        "suggestions": suggestions,
        "msg_id": msg_id,
        "retrieval_ms": response.retrieval_time_ms,
        "generation_ms": response.generation_time_ms,
        # Store the question so regenerate can replay it
        "_question": question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Medical safety logic
# ─────────────────────────────────────────────────────────────────────────────
_EMERGENCY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(
        r"\b(can'?t|cannot|unable to|difficulty|trouble|struggling to)\s+(breathe|breathing)\b"
        r"|\bsevere\s+(breathing|respiratory)\b|\bno\s+air\b|\bbreath\s*less(ness)?\b",
        re.I), "breathing difficulty"),
    (re.compile(
        r"\b(chest\s+pain|chest\s+tightness|crushing\s+chest|pressure\s+(on|in)\s+(my\s+)?chest)\b"
        r"|\bheart\s+attack\b",
        re.I), "chest pain"),
    (re.compile(
        r"\b(stroke|face\s+drooping|arm\s+weak(ness)?\b|sudden\s+speech\s+(loss|problem|difficulty))\b"
        r"|\b(numb(ness)?\s+(on\s+one|on\s+my)\s+side|sudden\s+confusion)\b",
        re.I), "stroke symptoms"),
    (re.compile(r"\b(unconscious|unresponsive|passed\s+out|fainted|collapse(d)?)\b", re.I), "unconsciousness"),
    (re.compile(r"\b(suicide|suicidal|kill\s+(my)?self|end\s+my\s+life|want\s+to\s+die)\b", re.I), "mental health crisis"),
    (re.compile(
        r"\b(overdose|took\s+too\s+many\s+(pills|tablets)|swallowed\s+(too\s+much|the\s+whole\s+bottle))\b",
        re.I), "possible overdose"),
    (re.compile(r"\b(severe\s+allergic|anaphylaxis|anaphylactic|throat\s+swelling|can'?t\s+swallow)\b", re.I), "severe allergic reaction"),
    (re.compile(r"\b(heavy\s+bleeding|blood\s+loss|won'?t\s+stop\s+bleeding|bleeding\s+heavily)\b", re.I), "severe bleeding"),
]

_MEDICATION_RISK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(overdose|took\s+too\s+much|too\s+many\s+(pills|tablets|capsules))\b", re.I),
    re.compile(r"\b(pregnant|pregnancy)\s+.{0,40}(medication|drug|dose|tablet|pill)\b", re.I),
    re.compile(r"\b(stop|stopping|quit)\s+(taking|my)\s+[a-z]+\b", re.I),
    re.compile(r"\b(can\s+I\s+take|safe\s+to\s+take)\s+.{0,30}(during\s+pregnancy|while\s+pregnant|with\s+[a-z]+)\b", re.I),
    re.compile(r"\b(dose|dosage|how\s+much|how\s+many)\s+.{0,30}(metformin|lisinopril|warfarin|insulin|lithium|digoxin|phenytoin|methotrexate)\b", re.I),
]


def _detect_emergency(text: str) -> tuple[bool, str]:
    for pattern, label in _EMERGENCY_PATTERNS:
        if pattern.search(text):
            return True, label
    return False, ""


def _detect_medication_risk(text: str) -> bool:
    return any(p.search(text) for p in _MEDICATION_RISK_PATTERNS)


def _classify_response(response) -> str:
    if not response.sources:
        return "general"
    conf = response.confidence or 0.0
    if conf < 0.35:
        return "uncertain"
    conflict_signals = re.compile(
        r"\b(conflict|disagree|contradict|debated|inconsistent|varies\s+by|some\s+studies\s+suggest|"
        r"evidence\s+is\s+mixed|no\s+consensus)\b",
        re.I,
    )
    if conflict_signals.search(response.answer):
        return "conflicting"
    return "grounded"


# ─────────────────────────────────────────────────────────────────────────────
# UI renderers
# ─────────────────────────────────────────────────────────────────────────────
def confidence_badge(score: float | None, is_general: bool = False) -> str:
    if is_general:
        return '<span class="mi-badge mi-badge-gen">General knowledge</span>'
    if score is None:
        return '<span class="mi-badge mi-badge-gen">—</span>'
    if score >= 0.75:
        cls, label = "mi-badge-high", "High confidence"
    elif score >= 0.45:
        cls, label = "mi-badge-med",  "Medium confidence"
    else:
        cls, label = "mi-badge-low",  "Low confidence"
    return f'<span class="mi-badge {cls}">{label} · {score:.2f}</span>'


def _render_answer(summary: str, body: str) -> None:
    with st.container(border=True):
        if summary:
            st.html(
                '<div class="mi-answer-eyebrow">Key takeaway</div>'
                f'<div class="mi-answer-key">{summary}</div>'
            )
        if body:
            st.markdown(body)


def _render_sources(sources: list | None, expanded: bool = False) -> None:
    if not sources:
        return
    label = f":material/menu_book: {len(sources)} source{'s' if len(sources) != 1 else ''}"
    with st.expander(label, expanded=expanded):
        for src in sources:
            rel = src.relevance_score or 0.0
            bar = int(rel * 100)
            url_html = (
                f'<a href="{src.url}" target="_blank" rel="noopener noreferrer">View ↗</a>'
                if src.url else ""
            )
            prov = getattr(src, "provenance", None) or "indexed"
            prov_badge = '<span class="mi-live-badge">LIVE</span> ' if prov == "live_pubmed" else ""
            st.html(
                f'<div class="mi-src">'
                f'<div class="mi-src-title">{prov_badge}{src.doc_id} — {src.title}</div>'
                f'<div class="mi-src-meta">'
                f'<span>{src.year or "n/a"}</span>'
                f'<span class="mi-src-rel">{bar}% relevance</span>'
                f'{url_html}'
                f'</div>'
                f'<div class="mi-bar-bg"><div class="mi-bar-fill" style="width:{bar}%"></div></div>'
                f'</div>'
            )
            if src.excerpt:
                with st.expander("Show excerpt", expanded=False):
                    st.caption(src.excerpt[:600])


def _render_followups(suggestions: list[str], msg_id: int) -> None:
    """Follow-up suggestion buttons.

    Uses plain st.button (no persistent widget state) so replaying history
    never causes stale selections or widget-key conflicts.  Each button has a
    unique key composed of msg_id + index.
    """
    if not suggestions:
        return
    st.caption("You might also ask:")
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        with cols[i]:
            if st.button(
                sug,
                key=f"followup_{msg_id}_{i}",
                help=sug,
                use_container_width=True,
            ):
                # Set pending and rerun — picked up at the top of the chat page
                st.session_state.pending_question = sug
                st.rerun()


def _render_emergency_notice(symptom_label: str) -> None:
    st.html(
        '<div class="mi-emergency" role="alert">'
        '<div class="mi-emergency-label">Emergency guidance</div>'
        f'<div class="mi-emergency-title">Your message mentions {symptom_label}. '
        'If this is a medical emergency, please seek help immediately.</div>'
        '<div class="mi-emergency-body">'
        'Call or have someone call emergency services now.<br>'
        '<span class="mi-emergency-number">India: 112</span>&nbsp;&nbsp;'
        '<span class="mi-emergency-number">Ambulance: 108</span><br><br>'
        'Do not wait for an AI response to decide whether to call for help. '
        'If you are unsure, treat it as an emergency and call.'
        '</div>'
        '</div>'
    )


def _render_uncertainty_notice(classification: str) -> None:
    if classification == "general":
        st.html(
            '<div class="mi-uncertain" role="note">'
            '<div class="mi-uncertain-label">Not verified from documents</div>'
            '<div class="mi-uncertain-body">'
            'This answer is based on general model knowledge, not retrieved medical documents. '
            'Consult a qualified healthcare professional before acting on this information.'
            '</div></div>'
        )
    elif classification == "uncertain":
        st.html(
            '<div class="mi-uncertain" role="note">'
            '<div class="mi-uncertain-label">Limited source support</div>'
            '<div class="mi-uncertain-body">'
            'The retrieved documents did not strongly support an answer to this question. '
            'Please verify with a healthcare provider.'
            '</div></div>'
        )
    elif classification == "conflicting":
        st.html(
            '<div class="mi-uncertain" role="note">'
            '<div class="mi-uncertain-label">Sources may conflict</div>'
            '<div class="mi-uncertain-body">'
            'Available sources present differing information on this topic. '
            'Professional interpretation is recommended.'
            '</div></div>'
        )


def _render_medication_caution() -> None:
    st.html(
        '<div class="mi-caution" role="note">'
        '<div class="mi-caution-label">Medication caution</div>'
        '<div class="mi-caution-body">'
        'MedSource can provide general, source-supported information about medications, '
        'but cannot prescribe, recommend dose changes, or advise stopping prescription medicines. '
        'For dosage questions, interactions, pregnancy, or a possible adverse reaction, '
        'contact your pharmacist or prescribing clinician.'
        '</div></div>'
    )


def _render_response_disclaimer() -> None:
    st.html(
        '<div class="mi-response-footer" role="note">'
        'This response is for educational purposes only and does not constitute medical advice, '
        'diagnosis, or treatment. Always consult a qualified healthcare professional.'
        '</div>'
    )


def _render_assistant_message(msg: dict, is_last: bool = False) -> None:
    """Render a single stored assistant message dict.

    is_last — if True, renders follow-up suggestions so they appear only under
    the most recent answer (avoids widget-key clutter and duplicate buttons).
    """
    if msg.get("emergency_label"):
        _render_emergency_notice(msg["emergency_label"])
    if msg.get("medication_risk"):
        _render_medication_caution()
    if msg.get("classification") in ("general", "uncertain", "conflicting"):
        _render_uncertainty_notice(msg["classification"])

    summary = msg.get("summary", "")
    body    = msg.get("body", msg.get("content", ""))
    badge   = msg.get("badge_html", "")

    if summary or body:
        _render_answer(summary, body)
    else:
        st.markdown(msg.get("content", ""))

    if badge:
        st.html(badge)

    _render_sources(msg.get("sources", []), expanded=False)
    _render_response_disclaimer()

    msg_id   = msg.get("msg_id", 0)
    act_key  = f"act_{msg_id}"
    _render_actions(
        msg.get("content", ""), act_key, msg.get("_question", ""),
        msg_id=msg_id, is_last=is_last,
    )

    if is_last and msg.get("suggestions"):
        _render_followups(msg["suggestions"], msg_id)


def _render_actions(
    answer_text: str,
    action_key: str,
    original_question: str,
    msg_id: int = 0,
    is_last: bool = False,
) -> None:
    """Copy, feedback (with SQLite persistence), and (for the last message) regenerate.

    Feedback is stored in session state for fast reads and persisted to SQLite
    so it survives app restarts.  msg_id is used as the stable database key.
    """
    fb_key = f"fb_{action_key}"

    # Initialise session-state feedback from the persistent store on first render.
    if fb_key not in st.session_state:
        stored = st.session_state.feedback_store.load(msg_id) if msg_id else None
        st.session_state[fb_key] = stored

    fb_val = st.session_state[fb_key]

    with st.container(horizontal=True):
        with st.popover(":material/content_copy: Copy", help="Copy answer text"):
            st.code(answer_text, language=None, wrap_lines=True)

        up_type = "primary" if fb_val == "up" else "tertiary"
        if st.button(
            "Helpful" if fb_val != "up" else "✓ Helpful",
            key=f"{action_key}_up",
            type=up_type,
            icon=":material/thumb_up:",
            help="Mark as helpful",
        ):
            new_val = "up" if fb_val != "up" else None
            st.session_state[fb_key] = new_val
            if msg_id:
                st.session_state.feedback_store.save(msg_id, new_val)
            st.rerun()

        dn_type = "primary" if fb_val == "down" else "tertiary"
        if st.button(
            "Not helpful" if fb_val != "down" else "✓ Not helpful",
            key=f"{action_key}_dn",
            type=dn_type,
            icon=":material/thumb_down:",
            help="Mark as not helpful",
        ):
            new_val = "down" if fb_val != "down" else None
            st.session_state[fb_key] = new_val
            if msg_id:
                st.session_state.feedback_store.save(msg_id, new_val)
            st.rerun()

        # Regenerate only on the last assistant message and only when we have the question
        if is_last and original_question:
            if st.button(
                "Regenerate",
                key=f"{action_key}_regen",
                type="tertiary",
                icon=":material/refresh:",
                help="Regenerate this response",
            ):
                st.session_state.regenerate_question = original_question
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Session state — initialise all keys before any widget renders
# ─────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "user_doc_vs" not in st.session_state:
    st.session_state.user_doc_vs = None
if "user_doc_name" not in st.session_state:
    st.session_state.user_doc_name = None
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
if "regenerate_question" not in st.session_state:
    st.session_state.regenerate_question = ""
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"
if "msg_counter" not in st.session_state:
    st.session_state.msg_counter = 0
# Persistent feedback store — one instance per session, backed by SQLite.
if "feedback_store" not in st.session_state:
    st.session_state.feedback_store = FeedbackStore()
# Guard against double-submission: stores the question currently being processed.
if "_submitting" not in st.session_state:
    st.session_state._submitting = ""

vector_store = st.session_state.vector_store

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### :material/ecg_heart: MedSource")
    st.caption("Medical AI assistant · RAG-powered")

    if vector_store.documents:
        kb_stats = vector_store.get_stats()
        st.caption(f":material/database: {kb_stats['total_chunks']:,} chunks · {kb_stats['unique_documents']} docs")
    else:
        st.caption(":material/database: Knowledge base empty")
    if st.session_state.user_doc_name:
        st.caption(f":material/description: {st.session_state.user_doc_name}")

    st.divider()

    if st.button("New chat", key="new_chat_btn", icon=":material/add:"):
        st.session_state.messages = []
        st.session_state.msg_counter = 0
        st.session_state.pending_question = ""
        st.session_state.regenerate_question = ""
        st.rerun()

    st.html('<span class="sb-label">Navigation</span>')

    _active = st.session_state.active_tab
    for _tab_key, _tab_label in [
        ("chat",     ":material/chat: Chat"),
        ("upload",   ":material/attach_file: Upload & Ask"),
        ("symptom",  ":material/healing: Symptom checker"),
        ("insights", ":material/bar_chart: Insights"),
    ]:
        if st.button(_tab_label, key=f"nav_{_tab_key}"):
            st.session_state.active_tab = _tab_key
            st.rerun()

    st.divider()
    st.html('<span class="sb-label">Document</span>')

    _doc_active = st.session_state.user_doc_vs is not None
    if st.button(
        "Remove document",
        disabled=not _doc_active,
        key="remove_doc_btn",
        icon=":material/delete:",
        help="Remove the currently active uploaded document",
    ):
        st.session_state.user_doc_vs = None
        st.session_state.user_doc_name = None
        st.rerun()

    if st.button("Clear chat history", key="clear_chat_btn", icon=":material/cleaning_services:"):
        st.session_state.messages = []
        st.session_state.msg_counter = 0
        st.session_state.pending_question = ""
        st.session_state.regenerate_question = ""
        st.rerun()

    # Recent questions
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if user_msgs:
        st.divider()
        st.html('<span class="sb-label">Recent</span>')
        for _hi, m in enumerate(reversed(user_msgs[-5:])):
            q = m["content"]
            if q.startswith("[Symptom Checker]"):
                q = q[17:].strip()
            label = (q[:38] + "…") if len(q) > 38 else q
            if st.button(label, key=f"hist_{_hi}", help=m["content"]):
                st.session_state.pending_question = (
                    m["content"].lstrip("[Doc] ").lstrip("[Symptom Checker] ").strip()
                )
                st.session_state.active_tab = "chat"
                st.rerun()

    # Admin gate
    if _ADMIN_PASSWORD:
        st.divider()
        st.html('<span class="sb-label">Admin</span>')
        if not st.session_state.admin_authenticated:
            with st.expander("Admin login", icon=":material/lock:"):
                pw = st.text_input("Password", type="password", key="admin_pw_input", label_visibility="collapsed")
                if st.button("Sign in", key="admin_login_btn", type="primary", icon=":material/login:"):
                    if pw == _ADMIN_PASSWORD:
                        st.session_state.admin_authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password.")
        else:
            st.success("Admin active", icon=":material/verified:")
            if st.button("Sign out", key="admin_logout_btn", icon=":material/logout:"):
                st.session_state.admin_authenticated = False
                st.rerun()
            if st.button("Reload index", key="reload_index_btn", icon=":material/refresh:"):
                st.cache_resource.clear()
                st.session_state.vector_store = load_vector_store()
                st.rerun()
            if st.button(":material/database: Data sources", key="nav_admin"):
                st.session_state.active_tab = "admin"
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Page routing
# ─────────────────────────────────────────────────────────────────────────────
_current_page = st.session_state.active_tab

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: CHAT
# ═════════════════════════════════════════════════════════════════════════════
if _current_page == "chat":

    # ── Resolve the question for this run ────────────────────────────────────
    # Priority: typed chat_input > pending_question (follow-up / card click) > regenerate
    # We snapshot pending/regenerate BEFORE rendering widgets so they're stable.
    _pending_val     = st.session_state.pending_question
    _regen_question  = st.session_state.regenerate_question

    # ── Welcome / empty state ─────────────────────────────────────────────────
    if not st.session_state.messages:
        _wcol = st.columns([1, 4, 1])[1]
        with _wcol:
            st.title(":material/ecg_heart: How can I help you today?", anchor=False)
            st.write(
                "Ask medical questions and get citation-backed answers from "
                "PubMed, WHO, FDA, and your uploaded documents."
            )
            with st.container(horizontal=True):
                st.badge("Gemini AI", icon=":material/smart_toy:", color="green")
                st.badge("PubMed",    icon=":material/science:",    color="blue")
                st.badge("WHO",       icon=":material/public:",     color="gray")
                st.badge("FDA",       icon=":material/medication:", color="orange")

        st.space("small")

        _CARD_QUESTIONS = [
            (":material/medication: Drug interactions & side effects", "What are the side effects of metformin?"),
            (":material/biotech: Lab results explained",               "What does a high HbA1c level mean?"),
            (":material/monitor_heart: Chronic condition management",  "How is type 2 diabetes managed?"),
            (":material/stethoscope: Symptom assessment",              "What are warning signs of a heart attack?"),
        ]
        _gc1, _gc2 = st.columns(2)
        _card_cols = [_gc1, _gc2, _gc1, _gc2]
        for _i, (_label, _q) in enumerate(_CARD_QUESTIONS):
            with _card_cols[_i]:
                if st.button(_label, key=f"card_{_i}", help=_q):
                    st.session_state.pending_question = _q
                    st.rerun()

        if not vector_store.documents and not st.session_state.user_doc_vs:
            st.info(
                "**Knowledge base is empty.** "
                "Upload a document via **Upload & Ask**, or ask an administrator to add sources.",
                icon=":material/info:",
            )

    else:
        # ── Chat history replay ───────────────────────────────────────────────
        # Determine the last assistant message index for follow-up placement
        _last_asst_idx = max(
            (i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"),
            default=-1,
        )
        for _idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    _render_assistant_message(msg, is_last=(_idx == _last_asst_idx))
                else:
                    st.markdown(msg["content"])

    # ── Chat input (always rendered, even during history replay) ─────────────
    _typed = st.chat_input("Ask a medical question…")

    # ── Determine which question to process this run ──────────────────────────
    question: str = ""
    _is_regen = False

    if _typed:
        # Typed input — clear any stale pending/regen
        question = _typed
        st.session_state.pending_question = ""
        st.session_state.regenerate_question = ""
    elif _pending_val:
        question = _pending_val
        st.session_state.pending_question = ""
        st.session_state.regenerate_question = ""
    elif _regen_question:
        question = _regen_question
        _is_regen = True
        st.session_state.regenerate_question = ""

    # ── Process the question ──────────────────────────────────────────────────
    if question:
        # ── Duplicate-submission guard ────────────────────────────────────────
        # Prevent the same question being processed twice in rapid succession
        # (e.g. double-click of a follow-up button or fast page reloads).
        # We track the question currently in flight; if it matches, skip.
        if st.session_state._submitting == question and not _is_regen:
            # Already processing — render history and wait for next run.
            pass
        else:
            st.session_state._submitting = question

            is_emergency, emergency_label = _detect_emergency(question)
            is_med_risk = _detect_medication_risk(question)

            # _saved_asst_msg holds the removed regen target so it can be restored on failure.
            _saved_asst_msg = None
            if _is_regen:
                # Remove the previous assistant message for this question so we replace it.
                # Walk backwards so we only remove the MOST RECENT matching response.
                msgs = st.session_state.messages
                for _ri in range(len(msgs) - 1, -1, -1):
                    if msgs[_ri]["role"] == "assistant" and msgs[_ri].get("_question") == question:
                        _saved_asst_msg = msgs[_ri]
                        msgs.pop(_ri)
                        break
            else:
                # Append user message only for new questions, not regenerations.
                # Check it wasn't already appended in a prior run for the same question.
                already_added = (
                    st.session_state.messages
                    and st.session_state.messages[-1]["role"] == "user"
                    and st.session_state.messages[-1]["content"] == question
                )
                if not already_added:
                    st.session_state.messages.append({"role": "user", "content": question})

            # Show the user message bubble (for new questions the history will show it
            # on re-render; we also render it inline for immediate feedback)
            if not _is_regen:
                with st.chat_message("user"):
                    st.markdown(question)

            # Show safety notices before the spinner
            with st.chat_message("assistant"):
                if is_emergency:
                    _render_emergency_notice(emergency_label)
                if is_med_risk:
                    _render_medication_caution()

                with st.status("Searching…", expanded=False) as _status:
                    st.write("Retrieving relevant passages…")
                    try:
                        asst_msg = _execute_query(question, bypass_cache=_is_regen)
                        _status.update(label="Done", state="complete")
                    except RuntimeError as _exc:
                        _status.update(label="Error", state="error")
                        asst_msg = None
                        _err_text = str(_exc)

                if asst_msg:
                    # Attach safety metadata from pre-flight checks
                    asst_msg["emergency_label"] = emergency_label if is_emergency else ""
                    asst_msg["medication_risk"]  = is_med_risk

                    classification = asst_msg["classification"]
                    if classification in ("general", "uncertain", "conflicting"):
                        _render_uncertainty_notice(classification)

                    summary = asst_msg["summary"]
                    body    = asst_msg["body"]
                    if summary or body:
                        _render_answer(summary, body)
                    else:
                        st.markdown(asst_msg["content"])

                    st.html(asst_msg["badge_html"])
                    _render_sources(asst_msg["sources"], expanded=True)
                    _render_response_disclaimer()
                    _render_actions(
                        asst_msg["content"],
                        f"act_{asst_msg['msg_id']}",
                        question,
                        msg_id=asst_msg["msg_id"],
                        is_last=True,
                    )
                    if not is_emergency and asst_msg.get("suggestions"):
                        _render_followups(asst_msg["suggestions"], asst_msg["msg_id"])

                    st.session_state.messages.append(asst_msg)
                    # Clear submission guard now that processing is complete
                    st.session_state._submitting = ""

                else:
                    # Error path — restore state so retry is clean.
                    if not _is_regen:
                        # Remove the user message we appended so retry re-adds it once.
                        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                            st.session_state.messages.pop()
                    else:
                        # Regeneration failed — restore the original assistant message
                        # so the conversation is not left without an answer.
                        if _saved_asst_msg is not None:
                            st.session_state.messages.append(_saved_asst_msg)
                    # Clear submission guard so retry can re-enter
                    st.session_state._submitting = ""

                    # Escape error text before inserting into HTML to prevent injection.
                    _safe_err = _html.escape(_err_text)
                    st.html(
                        f'<div class="mi-error-state" role="alert">'
                        f'We could not generate a response: {_safe_err}'
                        f'</div>'
                    )
                    if st.button(
                        "Retry",
                        key=f"retry_{len(st.session_state.messages)}",
                        type="primary",
                        icon=":material/refresh:",
                    ):
                        # Set pending so the question re-runs through the full
                        # pipeline on the next rerun.
                        st.session_state.pending_question = question
                        st.rerun()

    # ── Bottom disclaimer ─────────────────────────────────────────────────────
    st.html(
        '<div class="ms-disclaimer">'
        'Not a diagnostic tool · For educational purposes only · '
        'Always consult a qualified healthcare professional for medical decisions.'
        '</div>'
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD & ASK
# ═════════════════════════════════════════════════════════════════════════════
elif _current_page == "upload":
    st.header(":material/attach_file: Upload & Ask", anchor=False)

    st.html(
        '<div class="mi-upload-hint">'
        'Upload a <b>PDF, DOCX, or TXT</b> — e.g. a discharge summary, lab report, or medical article.<br>'
        'Processed <b>for this session only</b> and never saved to the shared knowledge base.'
        '</div>'
    )

    uploaded = st.file_uploader(
        "Drop or browse a file",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        key="user_upload",
    )

    # 10 MB hard limit for session-scoped uploads.
    _UPLOAD_MAX_BYTES = 10 * 1024 * 1024

    if uploaded:
        suffix = Path(uploaded.name).suffix.lower()

        # Enforce size limit before touching the file.
        if uploaded.size > _UPLOAD_MAX_BYTES:
            st.error(
                f"**File too large.** The maximum allowed size is "
                f"{_UPLOAD_MAX_BYTES // (1024 * 1024)} MB. "
                f"Your file is {uploaded.size / (1024 * 1024):.1f} MB.",
                icon=":material/error:",
            )
        else:
            _ci, _cb = st.columns([3, 1])
            with _ci:
                st.success(f":material/description: **{uploaded.name}** ({uploaded.size // 1024} KB)")
            with _cb:
                process = st.button("Process", type="primary", key="upload_process_btn", icon=":material/bolt:")

            if process:
                with st.spinner("Reading and indexing your document…"):
                    _parse_error: str = ""
                    raw_docs: list = []
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name
                        if suffix == ".pdf":
                            for i, pt in enumerate(DocumentLoader.load_pdf(tmp_path)):
                                if pt.strip():
                                    raw_docs.append(MedicalDocument(
                                        content=pt,
                                        title=f"{Path(uploaded.name).stem} — p.{i+1}",
                                        source=uploaded.name,
                                        metadata={"page": i + 1},
                                    ))
                        elif suffix == ".docx":
                            c = DocumentLoader.load_docx(tmp_path)
                            if c.strip():
                                raw_docs.append(MedicalDocument(content=c, title=Path(uploaded.name).stem, source=uploaded.name))
                        elif suffix == ".txt":
                            c = DocumentLoader.load_txt(tmp_path)
                            if c.strip():
                                raw_docs.append(MedicalDocument(content=c, title=Path(uploaded.name).stem, source=uploaded.name))
                        os.unlink(tmp_path)
                    except Exception as e:
                        _parse_error = str(e)

                if _parse_error:
                    # Escape before inserting into HTML to prevent injection.
                    _safe_parse_err = _html.escape(_parse_error)
                    st.html(f'<div class="mi-error-state"><strong>Could not read this file.</strong><br>{_safe_parse_err}</div>')
                    _ec1, _ec2 = st.columns(2)
                    with _ec1:
                        if st.button("Retry", type="primary", key="upload_retry", icon=":material/refresh:"):
                            st.rerun()
                    with _ec2:
                        if st.button("Upload another file", key="upload_other", icon=":material/upload_file:"):
                            st.rerun()
                elif not raw_docs:
                    st.html(
                        '<div class="mi-error-state">No readable text could be extracted from this file. '
                        'Check that the file is not password-protected, image-only, or empty.</div>'
                    )
                    if st.button("Upload another file", key="upload_other_empty", icon=":material/upload_file:"):
                        st.rerun()
                else:
                    doc_vs = VectorStore(
                        embedding_model_name=settings.embedding_model,
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        embedding_batch_size=settings.embedding_batch_size,
                    )
                    doc_vs.add_documents(raw_docs)
                    st.session_state.user_doc_vs  = doc_vs
                    st.session_state.user_doc_name = uploaded.name
                    st.success(
                        f"Indexed **{uploaded.name}** into {len(doc_vs.documents)} chunk(s). "
                        "Switch to Chat to ask questions.",
                        icon=":material/check_circle:",
                    )

    if st.session_state.user_doc_vs:
        st.divider()
        st.subheader(":material/description: Active document", anchor=False)
        st.caption(
            f"`{st.session_state.user_doc_name}` — "
            f"{len(st.session_state.user_doc_vs.documents)} chunks · session-scoped"
        )

        doc_q = st.text_input(
            "Ask a question about this document",
            placeholder="e.g. What medications were prescribed? What is the diagnosis?",
            key="doc_question_input",
        )
        _qa, _qr = st.columns([2, 1])
        with _qa:
            ask_btn = st.button("Ask", type="primary", key="doc_ask_btn", icon=":material/search:")
        with _qr:
            if st.button("Remove document", key="doc_remove_btn", icon=":material/delete:"):
                st.session_state.user_doc_vs  = None
                st.session_state.user_doc_name = None
                st.rerun()

        if ask_btn and doc_q:
            _doc_resp = None
            _doc_err  = ""
            with st.spinner("Searching document…"):
                try:
                    _pipe     = _build_doc_pipeline(st.session_state.user_doc_vs)
                    _doc_resp = _pipe.query(doc_q)
                except Exception as _de:
                    _doc_err = str(_de)

            if _doc_resp:
                _s, _b   = _parse_answer(_doc_resp.answer)
                _doc_cls = _classify_response(_doc_resp)
                if _doc_cls in ("general", "uncertain", "conflicting"):
                    _render_uncertainty_notice(_doc_cls)
                if _s or _b:
                    _render_answer(_s, _b)
                else:
                    st.markdown(_doc_resp.answer)
                _render_sources(_doc_resp.sources, expanded=False)
                _render_response_disclaimer()

                st.session_state.messages.append({"role": "user", "content": f"[Doc] {doc_q}"})
                st.session_state.msg_counter += 1
                _doc_msg_id = st.session_state.msg_counter
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": _doc_resp.answer,
                    "summary": _s, "body": _b,
                    "sources": _doc_resp.sources,
                    "badge_html": confidence_badge(_doc_resp.confidence),
                    "classification": _doc_cls,
                    "msg_id": _doc_msg_id,
                    "retrieval_ms": _doc_resp.retrieval_time_ms,
                    "generation_ms": _doc_resp.generation_time_ms,
                    "_question": doc_q,
                })
            else:
                err_text = _doc_err or "We could not generate a response. Please try again."
                st.html(f'<div class="mi-error-state">Could not generate a response: {err_text}</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SYMPTOM CHECKER
# ═════════════════════════════════════════════════════════════════════════════
elif _current_page == "symptom":
    st.header(":material/healing: Symptom checker", anchor=False)
    st.info(
        "Fill in the form below to get an AI-generated educational overview of possible conditions. "
        "**This is not a diagnosis.** Always consult a doctor for medical advice.",
        icon=":material/info:",
    )

    with st.form("symptom_form", clear_on_submit=False):
        _c1, _c2 = st.columns(2)
        with _c1:
            body_system = st.selectbox("Body system affected", [
                "— Select —", "Heart / Cardiovascular", "Lungs / Respiratory",
                "Digestive / GI", "Brain / Neurological", "Joints / Musculoskeletal",
                "Skin", "Kidneys / Urinary", "Reproductive / Hormonal",
                "Eyes / Ears / Nose / Throat", "Mental health", "General / Whole body",
            ])
            duration = st.selectbox("Duration of symptoms", [
                "Less than 24 hours", "1–3 days", "4–7 days", "1–4 weeks", "More than a month",
            ])
        with _c2:
            severity = st.select_slider("Severity", options=[
                "Mild (noticeable but manageable)",
                "Moderate (affecting daily life)",
                "Severe (debilitating or alarming)",
            ])
            age_group = st.selectbox("Age group", ["Child (0–12)", "Teen (13–17)", "Adult (18–64)", "Senior (65+)"])

        symptoms = st.text_area(
            "Describe your symptoms",
            placeholder="e.g. Persistent dry cough for 3 days, mild fever (38 °C), shortness of breath on exertion.",
            height=100,
        )
        other_context = st.text_input(
            "Additional context (optional)",
            placeholder="e.g. recent travel, known allergies, current medications",
        )
        submitted = st.form_submit_button("Analyse symptoms", type="primary", icon=":material/search:")

    if submitted:
        if body_system == "— Select —" or not symptoms.strip():
            st.warning("Please select a body system and describe your symptoms.", icon=":material/warning:")
        else:
            _sc_emergency, _sc_emerg_label = _detect_emergency(symptoms)
            if _sc_emergency:
                _render_emergency_notice(_sc_emerg_label)

            structured_q = (
                f"Patient profile: {age_group}, {body_system} symptoms, "
                f"duration: {duration}, severity: {severity}.\n"
                f"Symptoms: {symptoms.strip()}.\n"
                f"{'Additional context: ' + other_context.strip() + '.' if other_context.strip() else ''}\n"
                "What medical conditions could explain these symptoms? What are the typical treatments or "
                "next steps? When should the patient seek emergency care?"
            )

            with st.spinner("Analysing symptoms…"):
                _avs = st.session_state.user_doc_vs or vector_store
                response = None
                _sc_err  = ""
                if _avs.documents:
                    try:
                        _pipe    = _build_doc_pipeline(st.session_state.user_doc_vs) if st.session_state.user_doc_vs else build_pipeline(len(_avs.documents))
                        response = _pipe.query(structured_q)
                    except Exception as _se:
                        _sc_err = str(_se)

            if response:
                _s, _b   = _parse_answer(response.answer)
                _sc_cls  = _classify_response(response)
                st.subheader("Assessment", anchor=False)

                if "Severe" in severity:
                    st.html(
                        '<div class="mi-caution"><div class="mi-caution-label">Severity: severe</div>'
                        '<div class="mi-caution-body">You described severe symptoms. If this is a medical emergency, '
                        'call 112 (India) or your local emergency number immediately. '
                        'Otherwise, please consult a healthcare provider as soon as possible.</div></div>'
                    )
                elif "Moderate" in severity:
                    st.html(
                        '<div class="mi-caution"><div class="mi-caution-label">Severity: moderate</div>'
                        '<div class="mi-caution-body">You described moderate symptoms. '
                        'Consider consulting a healthcare provider soon.</div></div>'
                    )

                if _sc_cls in ("general", "uncertain", "conflicting"):
                    _render_uncertainty_notice(_sc_cls)
                if _s or _b:
                    _render_answer(_s, _b)
                else:
                    st.markdown(response.answer)

                st.html(
                    '<div class="mi-uncertain"><div class="mi-uncertain-label">Important limitation</div>'
                    '<div class="mi-uncertain-body">This assessment presents possible explanations, not a confirmed diagnosis. '
                    'A qualified healthcare professional must evaluate your symptoms in person.</div></div>'
                )
                _badge = confidence_badge(response.confidence, _sc_cls == "general")
                st.html(_badge)
                _render_sources(response.sources, expanded=False)
                _render_response_disclaimer()

                label = f"[Symptom Checker] {body_system} — {symptoms[:60]}…"
                st.session_state.messages.append({"role": "user", "content": label})
                st.session_state.msg_counter += 1
                _sc_msg_id = st.session_state.msg_counter
                st.session_state.messages.append({
                    "role": "assistant", "content": response.answer,
                    "summary": _s, "body": _b,
                    "sources": response.sources, "badge_html": _badge,
                    "classification": _sc_cls,
                    "emergency_label": _sc_emerg_label if _sc_emergency else "",
                    "msg_id": _sc_msg_id,
                    "retrieval_ms": response.retrieval_time_ms,
                    "generation_ms": response.generation_time_ms,
                    "_question": structured_q,
                })
            else:
                err = _sc_err or "The knowledge base may be empty, or a temporary error occurred."
                st.html(f'<div class="mi-error-state">We could not generate an assessment: {err}</div>')


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS
# ═════════════════════════════════════════════════════════════════════════════
elif _current_page == "insights":
    st.header(":material/bar_chart: Insights", anchor=False)

    if not vector_store.documents:
        st.info("No documents indexed yet. Add sources via the Data sources tab (admin).", icon=":material/info:")
    else:
        st.subheader("Knowledge base overview", anchor=False)
        kb = vector_store.get_stats()
        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.html(f'<div class="mi-metric"><h3>{kb["total_chunks"]}</h3><p>Chunks</p></div>')
        _m2.html(f'<div class="mi-metric"><h3>{kb["unique_documents"]}</h3><p>Documents</p></div>')
        _m3.html(f'<div class="mi-metric"><h3>{kb["dimension"]}d</h3><p>Embed dim</p></div>')
        _m4.html(f'<div class="mi-metric"><h3>{kb["chunk_size"]} tok</h3><p>Chunk size</p></div>')

        st.caption(
            f"Model: `{kb.get('embedding_model', '—')}` · "
            f"BGE prefix: {'✓' if kb.get('use_bge_prefix') else '✗'} · "
            f"Avg chunk: {kb.get('avg_chunk_chars', '?')} chars"
        )
        st.subheader("Indexed documents", anchor=False)
        seen: dict = {}
        for d in vector_store.documents:
            seen.setdefault(d["doc_id"], {"title": d["title"], "source": d["source"], "year": d.get("year"), "chunks": 0})
            seen[d["doc_id"]]["chunks"] += 1
        import pandas as pd
        df = pd.DataFrame([
            {"Title": v["title"], "Source": v["source"], "Year": v["year"], "Chunks": v["chunks"]}
            for v in seen.values()
        ])
        st.dataframe(df, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA SOURCES (Admin only)
# ═════════════════════════════════════════════════════════════════════════════
elif _current_page == "admin" and st.session_state.admin_authenticated:
    st.header(":material/database: Data sources", anchor=False)
    st.caption("Everything added here becomes retrievable evidence the chatbot can cite.")

    corpus_builder = MedicalCorpusBuilder()
    WHO_FACT_SHEETS = {
        "Diabetes": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "Hypertension": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "Obesity and overweight": "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
        "Tuberculosis": "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
        "Malaria": "https://www.who.int/news-room/fact-sheets/detail/malaria",
        "HIV": "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
        "Cardiovascular diseases": "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
        "Mental disorders": "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
        "Asthma": "https://www.who.int/news-room/fact-sheets/detail/asthma",
        "Anaemia": "https://www.who.int/news-room/fact-sheets/detail/anaemia",
    }

    (src_tab_files, src_tab_pubmed, src_tab_gho, src_tab_url,
     src_tab_medlineplus, src_tab_fda, src_tab_trials, src_tab_text) = st.tabs([
        ":material/description: Files", ":material/science: PubMed",
        ":material/public: WHO GHO",    ":material/link: URL",
        ":material/menu_book: MedlinePlus", ":material/medication: FDA labels",
        ":material/biotech: Trials",    ":material/edit_note: Paste",
    ])

    with src_tab_files:
        st.write("Upload PDF, DOCX, TXT, or JSON.")
        uploaded_files = st.file_uploader("Choose file(s)", type=["pdf", "docx", "txt", "json"], accept_multiple_files=True, key="admin_file_upload")
        if st.button("Ingest uploaded files", disabled=not uploaded_files, key="admin_ingest_files", type="primary", icon=":material/upload:"):
            new_docs: list = []
            with st.spinner("Parsing…"):
                for f in uploaded_files:
                    sfx = Path(f.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=sfx) as tmp:
                        tmp.write(f.read()); tmp_path = tmp.name
                    try:
                        if sfx == ".pdf":
                            for i, pt in enumerate(DocumentLoader.load_pdf(tmp_path)):
                                if pt.strip():
                                    new_docs.append(MedicalDocument(content=pt, title=f"{Path(f.name).stem} p.{i+1}", source=f.name, metadata={"page": i+1}))
                        elif sfx == ".docx":
                            c = DocumentLoader.load_docx(tmp_path)
                            if c.strip(): new_docs.append(MedicalDocument(content=c, title=Path(f.name).stem, source=f.name))
                        elif sfx == ".txt":
                            c = DocumentLoader.load_txt(tmp_path)
                            if c.strip(): new_docs.append(MedicalDocument(content=c, title=Path(f.name).stem, source=f.name))
                        elif sfx == ".json":
                            for item in DocumentLoader.load_json(tmp_path):
                                if isinstance(item, dict) and item.get("content"):
                                    new_docs.append(MedicalDocument(content=item["content"], title=item.get("title", Path(f.name).stem), source=item.get("source", f.name), year=item.get("year"), url=item.get("url")))
                    finally:
                        os.unlink(tmp_path)
            if new_docs:
                with st.spinner(f"Embedding {len(new_docs)} section(s)…"):
                    vector_store.add_documents(new_docs)
                    vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus(new_docs, f"upload_{int(time.time())}.json")
                st.success(f"Added {len(new_docs)} section(s)", icon=":material/check_circle:"); st.rerun()
            else:
                st.warning("No readable content found.", icon=":material/warning:")

    with src_tab_pubmed:
        pubmed_query = st.text_input("PubMed query", placeholder="e.g. type 2 diabetes management", key="admin_pubmed_query")
        pubmed_email = st.text_input("Contact email (NCBI policy)", value=settings.pubmed_email, key="admin_pubmed_email")
        max_results  = st.slider("Max abstracts", 5, 100, 20, key="admin_pubmed_max")
        if st.button("Fetch PubMed", disabled=not (pubmed_query and pubmed_email), key="admin_fetch_pubmed", type="primary", icon=":material/download:"):
            with st.spinner("Fetching…"):
                loader = PubMedLoader(email=pubmed_email)
                pmids  = loader.search_pubmed(pubmed_query, max_results=max_results)
                docs   = loader.fetch_abstracts(pmids) if pmids else []
            if docs:
                vector_store.add_documents(docs); vector_store.save(settings.vector_store_path)
                corpus_builder.save_corpus(docs, f"pubmed_{int(time.time())}.json")
                st.success(f"Added {len(docs)} abstract(s)", icon=":material/check_circle:"); st.rerun()
            else:
                st.warning("No abstracts found.", icon=":material/warning:")

    with src_tab_gho:
        _gho = WHOGHOLoader()
        indicator_label = st.selectbox("Indicator", list(_gho.COMMON_INDICATORS.values()), key="admin_gho_indicator")
        indicator_code  = [k for k, v in _gho.COMMON_INDICATORS.items() if v == indicator_label][0]
        country_iso3    = st.text_input("Country ISO3 (optional)", placeholder="e.g. IND", key="admin_gho_country")
        latest_n        = st.slider("Max data points", 10, 200, 50, key="admin_gho_max")
        if st.button("Fetch WHO GHO", key="admin_fetch_gho", type="primary", icon=":material/download:"):
            docs = _gho.to_documents(indicator_code, indicator_label, country_iso3 or None, latest_n)
            if docs:
                vector_store.add_documents(docs); vector_store.save(settings.vector_store_path)
                corpus_builder.save_corpus(docs, f"who_gho_{int(time.time())}.json")
                st.success(f"Added {len(docs)} doc(s)", icon=":material/check_circle:"); st.rerun()
            else:
                st.warning("No data returned.", icon=":material/warning:")

    with src_tab_url:
        import requests as _req
        from bs4 import BeautifulSoup as _BS
        quick_picks = st.multiselect("WHO fact sheets", list(WHO_FACT_SHEETS.keys()), key="admin_url_quickpicks")
        if st.button("Fetch fact sheets", disabled=not quick_picks, key="admin_fetch_factsheets", type="primary", icon=":material/download:"):
            added = 0
            for topic in quick_picks:
                try:
                    r = _req.get(WHO_FACT_SHEETS[topic], timeout=15, headers={"User-Agent": "MedSourceBot/1.0"})
                    soup = _BS(r.content, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
                    text = " ".join(soup.get_text(separator=" ").split())
                    if text and len(text) > 200:
                        doc = MedicalDocument(content=text[:20000], title=f"WHO: {topic}", source="WHO", url=WHO_FACT_SHEETS[topic])
                        vector_store.add_documents([doc])
                        corpus_builder.save_corpus([doc], f"who_{int(time.time())}.json")
                        added += 1
                except Exception as e:
                    st.warning(f"{topic}: {e}", icon=":material/warning:")
            if added:
                vector_store.save(settings.vector_store_path)
                st.success(f"Added {added} fact sheet(s)", icon=":material/check_circle:"); st.rerun()

        page_url = st.text_input("Or any URL", key="admin_url_input")
        if st.button("Fetch URL", disabled=not page_url, key="admin_fetch_url", type="primary", icon=":material/download:"):
            try:
                r = _req.get(page_url, timeout=15, headers={"User-Agent": "MedSourceBot/1.0"})
                soup = _BS(r.content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]): tag.decompose()
                title = soup.title.string.strip() if soup.title and soup.title.string else page_url
                text  = " ".join(soup.get_text(separator=" ").split())
                if text and len(text) > 200:
                    doc = MedicalDocument(content=text[:20000], title=title, source=page_url.split("/")[2] if "//" in page_url else page_url, url=page_url)
                    vector_store.add_documents([doc]); vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus([doc], f"url_{int(time.time())}.json")
                    st.success(f"Added '{title}'", icon=":material/check_circle:"); st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}", icon=":material/error:")

    with src_tab_medlineplus:
        ml_queries = st.text_area("Queries (one per line)", value="\n".join(["diabetes", "hypertension", "anemia", "asthma", "depression"]), height=120, key="admin_ml_queries")
        max_ml     = st.slider("Max per query", 1, 10, 5, key="admin_ml_max")
        if st.button("Fetch MedlinePlus", key="admin_fetch_ml", type="primary", icon=":material/download:"):
            loader   = MedlinePlusLoader(); all_docs: list = []
            for q in [x.strip() for x in ml_queries.splitlines() if x.strip()]:
                all_docs.extend(loader.search(q, max_results=max_ml))
            if all_docs:
                vector_store.add_documents(all_docs); vector_store.save(settings.vector_store_path)
                corpus_builder.save_corpus(all_docs, f"medlineplus_{int(time.time())}.json")
                st.success(f"Added {len(all_docs)}", icon=":material/check_circle:"); st.rerun()

    with src_tab_fda:
        fda_drugs = st.text_area("Drug names (one per line)", value="\n".join(["metformin", "lisinopril", "ibuprofen", "sertraline", "albuterol"]), height=130, key="admin_fda_drugs")
        fda_max   = st.slider("Max records per drug", 1, 5, 2, key="admin_fda_max")
        if st.button("Fetch FDA labels", key="admin_fetch_fda", type="primary", icon=":material/download:"):
            loader   = OpenFDADrugLoader(); all_docs = []
            for drug in [d.strip() for d in fda_drugs.splitlines() if d.strip()]:
                all_docs.extend(loader.search_drug(drug, max_results=fda_max))
            if all_docs:
                vector_store.add_documents(all_docs); vector_store.save(settings.vector_store_path)
                corpus_builder.save_corpus(all_docs, f"fda_{int(time.time())}.json")
                st.success(f"Added {len(all_docs)}", icon=":material/check_circle:"); st.rerun()

    with src_tab_trials:
        ct_queries = st.text_area("Conditions (one per line)", value="\n".join(["type 2 diabetes", "hypertension", "asthma"]), height=100, key="admin_ct_queries")
        ct_max     = st.slider("Max per query", 1, 20, 5, key="admin_ct_max")
        ct_status  = st.selectbox("Status", ["COMPLETED", "RECRUITING", "(any)"], key="admin_ct_status")
        if st.button("Fetch trials", key="admin_fetch_ct", type="primary", icon=":material/download:"):
            loader   = ClinicalTrialsLoader(); all_docs = []
            sf = None if ct_status == "(any)" else ct_status
            for q in [x.strip() for x in ct_queries.splitlines() if x.strip()]:
                all_docs.extend(loader.search(q, max_results=ct_max, status=sf))
            if all_docs:
                vector_store.add_documents(all_docs); vector_store.save(settings.vector_store_path)
                corpus_builder.save_corpus(all_docs, f"trials_{int(time.time())}.json")
                st.success(f"Added {len(all_docs)}", icon=":material/check_circle:"); st.rerun()

    with src_tab_text:
        with st.form("paste_text"):
            t_title   = st.text_input("Title",           key="admin_paste_title")
            t_source  = st.text_input("Source",          key="admin_paste_source")
            t_year    = st.text_input("Year (optional)", key="admin_paste_year")
            t_url     = st.text_input("URL (optional)",  key="admin_paste_url")
            t_content = st.text_area("Content", height=180, key="admin_paste_content")
            if st.form_submit_button("Ingest text", type="primary", icon=":material/upload:"):
                if t_title and t_content:
                    doc = MedicalDocument(content=t_content, title=t_title, source=t_source or "Manual", year=t_year or None, url=t_url or None)
                    vector_store.add_documents([doc]); vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus([doc], f"manual_{int(time.time())}.json")
                    st.success(f"Added '{t_title}'", icon=":material/check_circle:"); st.rerun()

elif _current_page == "admin" and not st.session_state.admin_authenticated:
    st.warning("Admin access required. Please log in via the sidebar.", icon=":material/lock:")
