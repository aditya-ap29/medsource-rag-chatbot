"""
MedSource — Streamlit UI
A polished chat + data-ingestion front end for the existing MedSource RAG pipeline.

Run with:
    streamlit run streamlit_app.py
"""
import os
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
from src.data_ingestion import (
    MedicalDocument,
    MedicalCorpusBuilder,
    DocumentLoader,
    PubMedLoader,
    WHOGHOLoader,
)

# Curated WHO fact-sheet URLs for one-click ingestion (narrative text, not raw stats)
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

# --------------------------------------------------------------------------
# Page config + styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MedSource — Trustworthy Medical AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --mi-primary: #0f766e;
    --mi-primary-dark: #115e59;
    --mi-accent: #2dd4bf;
    --mi-bg-soft: #f0fdfa;
}
.stApp { background: linear-gradient(180deg, #f8fafc 0%, #f0fdfa 100%); }

.mi-hero {
    background: linear-gradient(120deg, #0f766e 0%, #0891b2 100%);
    padding: 1.6rem 2rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
}
.mi-hero h1 { margin: 0; font-size: 1.9rem; }
.mi-hero p { margin: 0.35rem 0 0 0; opacity: 0.92; font-size: 0.95rem; }

.mi-disclaimer {
    background: #fff7ed;
    border: 1px solid #fdba74;
    color: #9a3412;
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.85rem;
    margin-bottom: 1rem;
}

.mi-source-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-left: 4px solid var(--mi-primary);
    border-radius: 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.55rem;
}
.mi-source-title { font-weight: 600; color: #0f172a; font-size: 0.92rem; }
.mi-source-meta { color: #64748b; font-size: 0.78rem; margin: 0.1rem 0 0.35rem 0; }
.mi-source-excerpt { color: #334155; font-size: 0.85rem; font-style: italic; }

.mi-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.mi-badge-high { background: #dcfce7; color: #166534; }
.mi-badge-med { background: #fef9c3; color: #854d0e; }
.mi-badge-low { background: #fee2e2; color: #991b1b; }

.mi-metric-card {
    background: white;
    border-radius: 12px;
    padding: 0.9rem;
    border: 1px solid #e2e8f0;
    text-align: center;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

settings = get_settings()

# --------------------------------------------------------------------------
# Cached resource loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model + vector store...")
def load_vector_store():
    vs = VectorStore(
        embedding_model_name=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if os.path.exists(settings.vector_store_path):
        try:
            vs.load(settings.vector_store_path)
        except Exception as e:
            st.warning(f"Could not load existing vector store: {e}")
    return vs


def build_pipeline(vector_store, provider, model_name, api_key, top_k, threshold, temperature):
    return RAGPipeline(
        vector_store=vector_store,
        llm_provider=provider,
        model_name=model_name or None,
        api_key=api_key or None,
        temperature=temperature,
        max_tokens=settings.max_tokens,
        top_k=top_k,
        confidence_threshold=threshold,
        enable_general_fallback=settings.enable_general_fallback,
    )


def confidence_badge(score: float) -> str:
    if score >= 0.8:
        cls, label = "mi-badge-high", "High confidence"
    elif score >= 0.5:
        cls, label = "mi-badge-med", "Medium confidence"
    else:
        cls, label = "mi-badge-low", "Low confidence"
    return f'<span class="mi-badge {cls}">{label} · {score:.2f}</span>'


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "messages" not in st.session_state:
    st.session_state.messages = []

vector_store = st.session_state.vector_store

# --------------------------------------------------------------------------
# Sidebar — model + retrieval settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Model settings")

    provider = st.selectbox(
        "LLM provider",
        options=["gemini", "openai", "mistral", "qwen", "huggingface"],
        index=["gemini", "openai", "mistral", "qwen", "huggingface"].index(settings.llm_provider)
        if settings.llm_provider in ["gemini", "openai", "mistral", "qwen", "huggingface"] else 0,
    )

    default_models = {
        "gemini": "gemini-2.0-flash",
        "openai": settings.llm_model or "gpt-4-turbo-preview",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "qwen": "Qwen/Qwen1.5-7B-Chat",
        "huggingface": "",
    }
    model_name = st.text_input("Model name", value=default_models[provider])

    env_key = settings.gemini_api_key if provider == "gemini" else settings.openai_api_key
    api_key_input = st.text_input(
        f"{provider.upper()} API key",
        value="",
        type="password",
        placeholder="Uses .env value if left blank",
    )
    api_key = api_key_input or env_key

    st.markdown("### 🔍 Retrieval settings")
    top_k = st.slider("Documents to retrieve (top_k)", 1, 10, settings.top_k_documents)
    threshold = st.slider(
        "Confidence threshold", 0.0, 1.0, settings.retrieval_confidence_threshold, 0.05
    )
    temperature = st.slider("Temperature", 0.0, 1.0, settings.llm_temperature, 0.05)

    st.markdown("### 📊 Vector store")
    stats = vector_store.get_stats() if vector_store.documents else None
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("Chunks", stats["total_chunks"])
        c2.metric("Documents", stats["unique_documents"])
    else:
        st.info("No documents indexed yet — add sources in the **📥 Data Sources** tab.")

    if st.button("🔄 Reload index from disk", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.vector_store = load_vector_store()
        st.rerun()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------
# Hero header
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="mi-hero">
        <h1>🩺 MedSource</h1>
        <p>A trustworthy, citation-backed medical Q&A assistant — powered by Retrieval-Augmented Generation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="mi-disclaimer">
        ⚠️ <b>Not a diagnostic tool.</b> MedSource is for educational purposes only and does not replace
        professional medical advice. Always consult a qualified healthcare provider.
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_data, tab_insights = st.tabs(["💬 Ask MedSource", "📥 Data Sources", "📊 Insights"])

# --------------------------------------------------------------------------
# TAB 1 — Chat
# --------------------------------------------------------------------------
with tab_chat:
    if not vector_store.documents:
        st.warning(
            "Your vector store is empty. Go to **📥 Data Sources** to add medical documents "
            "(files, PubMed, or a trusted URL) before asking questions."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 {len(msg['sources'])} source(s) · " + msg.get("badge_html", ""), expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"""
                            <div class="mi-source-card">
                                <div class="mi-source-title">{src.doc_id}: {src.title}</div>
                                <div class="mi-source-meta">{src.year or "n/a"} · relevance {src.relevance_score:.2f}
                                {" · <a href='" + src.url + "' target='_blank'>source link</a>" if src.url else ""}</div>
                                <div class="mi-source-excerpt">{src.excerpt}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                st.caption(
                    f"Retrieval {msg.get('retrieval_ms', 0):.0f} ms · "
                    f"Generation {msg.get('generation_ms', 0):.0f} ms"
                )

    question = st.chat_input("Ask a medical question, e.g. 'What are the symptoms of anemia?'")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            if not vector_store.documents:
                answer = "I don't have any indexed medical documents yet, so I can't answer safely. Please add data sources first."
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                with st.spinner("Retrieving evidence and generating a grounded answer..."):
                    try:
                        pipeline = build_pipeline(
                            vector_store, provider, model_name, api_key, top_k, threshold, temperature
                        )
                        response = pipeline.query(question, top_k=top_k)
                    except Exception as e:
                        st.error(f"Error running the RAG pipeline: {e}")
                        response = None

                if response:
                    st.markdown(response.answer)
                    badge_html = confidence_badge(response.confidence)
                    st.markdown(badge_html, unsafe_allow_html=True)

                    if response.sources:
                        with st.expander(f"📚 {len(response.sources)} source(s)", expanded=True):
                            for src in response.sources:
                                st.markdown(
                                    f"""
                                    <div class="mi-source-card">
                                        <div class="mi-source-title">{src.doc_id}: {src.title}</div>
                                        <div class="mi-source-meta">{src.year or "n/a"} · relevance {src.relevance_score:.2f}
                                        {" · <a href='" + src.url + "' target='_blank'>source link</a>" if src.url else ""}</div>
                                        <div class="mi-source-excerpt">{src.excerpt}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    st.caption(
                        f"Retrieval {response.retrieval_time_ms:.0f} ms · "
                        f"Generation {response.generation_time_ms:.0f} ms · "
                        f"Total {response.total_time_ms:.0f} ms"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "badge_html": badge_html,
                            "retrieval_ms": response.retrieval_time_ms,
                            "generation_ms": response.generation_time_ms,
                        }
                    )

# --------------------------------------------------------------------------
# TAB 2 — Data sources
# --------------------------------------------------------------------------
with tab_data:
    st.markdown("#### Add medical knowledge to the vector store")
    st.caption(
        "Every source added here becomes retrievable evidence the chatbot can cite. "
        "Nothing is added to the index until you click the ingest button in each section."
    )

    corpus_builder = MedicalCorpusBuilder()

    src_tab_files, src_tab_pubmed, src_tab_gho, src_tab_url, src_tab_text = st.tabs(
        [
            "📄 Upload files (books/PDFs)",
            "🔬 PubMed API",
            "🌍 WHO GHO Indicators",
            "🌐 WHO / CDC / NIH URL",
            "✍️ Paste text",
        ]
    )

    # ---- Upload files (this covers "books" as PDF/DOCX/TXT) ----
    with src_tab_files:
        st.write(
            "Upload medical textbook chapters, guideline PDFs, DOCX, TXT, or a structured "
            "JSON file (list of `{title, content, source, year, url}` objects)."
        )
        uploaded_files = st.file_uploader(
            "Choose file(s)",
            type=["pdf", "docx", "txt", "json"],
            accept_multiple_files=True,
        )
        if st.button("📥 Ingest uploaded files", disabled=not uploaded_files):
            new_docs = []
            with st.spinner("Parsing files..."):
                for f in uploaded_files:
                    suffix = Path(f.name).suffix.lower()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.read())
                        tmp_path = tmp.name
                    try:
                        if suffix == ".pdf":
                            pages = DocumentLoader.load_pdf(tmp_path)
                            for i, page_text in enumerate(pages):
                                if page_text.strip():
                                    new_docs.append(
                                        MedicalDocument(
                                            content=page_text,
                                            title=f"{Path(f.name).stem} — Page {i + 1}",
                                            source=f.name,
                                            metadata={"uploaded_as": f.name, "page": i + 1},
                                        )
                                    )
                        elif suffix == ".docx":
                            content = DocumentLoader.load_docx(tmp_path)
                            if content.strip():
                                new_docs.append(
                                    MedicalDocument(content=content, title=Path(f.name).stem, source=f.name)
                                )
                        elif suffix == ".txt":
                            content = DocumentLoader.load_txt(tmp_path)
                            if content.strip():
                                new_docs.append(
                                    MedicalDocument(content=content, title=Path(f.name).stem, source=f.name)
                                )
                        elif suffix == ".json":
                            items = DocumentLoader.load_json(tmp_path)
                            for item in items:
                                if isinstance(item, dict) and item.get("content"):
                                    new_docs.append(
                                        MedicalDocument(
                                            content=item["content"],
                                            title=item.get("title", Path(f.name).stem),
                                            source=item.get("source", f.name),
                                            year=item.get("year"),
                                            url=item.get("url"),
                                            metadata=item.get("metadata", {}),
                                        )
                                    )
                    finally:
                        os.unlink(tmp_path)

            if new_docs:
                with st.spinner(f"Embedding {len(new_docs)} chunks and updating the index..."):
                    vector_store.add_documents(new_docs)
                    vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus(new_docs, f"upload_{int(time.time())}.json")
                st.success(f"Added {len(new_docs)} document section(s) to the vector store ✅")
                st.rerun()
            else:
                st.warning("No readable content found in the uploaded file(s).")

    # ---- PubMed ----
    with src_tab_pubmed:
        st.write("Fetch peer-reviewed abstracts directly from PubMed/NCBI's public E-utilities API.")
        pubmed_query = st.text_input("PubMed search query", placeholder="e.g. type 2 diabetes management")
        pubmed_email = st.text_input(
            "Contact email (required by NCBI's usage policy)",
            value=os.getenv("PUBMED_EMAIL", ""),
        )
        max_results = st.slider("Max abstracts to fetch", 5, 100, 20)
        if st.button("📥 Fetch & ingest from PubMed", disabled=not (pubmed_query and pubmed_email)):
            with st.spinner(f"Searching PubMed for '{pubmed_query}'..."):
                loader = PubMedLoader(email=pubmed_email)
                pmids = loader.search_pubmed(pubmed_query, max_results=max_results)
                docs = loader.fetch_abstracts(pmids) if pmids else []
            if docs:
                with st.spinner(f"Embedding {len(docs)} abstracts and updating the index..."):
                    vector_store.add_documents(docs)
                    vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus(docs, f"pubmed_{int(time.time())}.json")
                st.success(f"Added {len(docs)} PubMed abstract(s) to the vector store ✅")
                st.rerun()
            else:
                st.warning("No abstracts found for that query.")

    # ---- WHO GHO official indicator API (authentic, structured WHO data) ----
    with src_tab_gho:
        st.write(
            "Pull authentic, official statistics straight from WHO's **Global Health Observatory "
            "(GHO) OData API** — no API key required. Raw numbers are converted into short, "
            "citeable sentences per country/year before indexing."
        )
        loader = WHOGHOLoader()
        indicator_label = st.selectbox("Indicator", options=list(loader.COMMON_INDICATORS.values()))
        indicator_code = [k for k, v in loader.COMMON_INDICATORS.items() if v == indicator_label][0]
        st.caption(f"Indicator code: `{indicator_code}` · source: ghoapi.azureedge.net (official WHO API)")
        country_iso3 = st.text_input(
            "Country ISO3 code (optional — leave blank for all countries)",
            placeholder="e.g. IND, USA, GBR",
        )
        latest_n = st.slider("Max data points to fetch", 10, 200, 50)
        if st.button("📥 Fetch & ingest WHO GHO data"):
            with st.spinner(f"Querying WHO GHO API for {indicator_code}..."):
                docs = loader.to_documents(
                    indicator_code, indicator_label, country_iso3 or None, latest_n
                )
            if docs:
                with st.spinner(f"Embedding {len(docs)} country profile(s) and updating the index..."):
                    vector_store.add_documents(docs)
                    vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus(docs, f"who_gho_{indicator_code}_{int(time.time())}.json")
                st.success(f"Added {len(docs)} WHO GHO document(s) for '{indicator_label}' ✅")
                st.rerun()
            else:
                st.warning("No data returned for that indicator/country combination.")

    # ---- WHO / CDC / NIH URL ----
    with src_tab_url:
        st.write(
            "Paste a public fact-sheet URL (e.g. a WHO health-topic page, CDC page, NIH/MedlinePlus "
            "article) and MedSource will extract the readable text as a new source."
        )

        st.markdown("**Quick-add official WHO fact sheets:**")
        quick_picks = st.multiselect("Choose topic(s)", options=list(WHO_FACT_SHEETS.keys()))
        if st.button("📥 Fetch & ingest selected fact sheets", disabled=not quick_picks):
            import requests
            from bs4 import BeautifulSoup

            added = 0
            with st.spinner("Fetching WHO fact sheets..."):
                for topic in quick_picks:
                    url = WHO_FACT_SHEETS[topic]
                    try:
                        resp = requests.get(url, timeout=15, headers={"User-Agent": "MedSourceBot/1.0"})
                        resp.raise_for_status()
                        soup = BeautifulSoup(resp.content, "html.parser")
                        for tag in soup(["script", "style", "nav", "footer", "header"]):
                            tag.decompose()
                        text = " ".join(soup.get_text(separator=" ").split())
                        if text and len(text) > 200:
                            doc = MedicalDocument(
                                content=text[:20000],
                                title=f"WHO Fact Sheet: {topic}",
                                source="World Health Organization",
                                url=url,
                            )
                            vector_store.add_documents([doc])
                            corpus_builder.save_corpus([doc], f"who_factsheet_{topic}_{int(time.time())}.json")
                            added += 1
                    except Exception as e:
                        st.warning(f"Could not fetch {topic}: {e}")
            if added:
                vector_store.save(settings.vector_store_path)
                st.success(f"Added {added} WHO fact sheet(s) ✅")
                st.rerun()

        st.markdown("**Or enter any URL:**")
        page_url = st.text_input("Page URL", placeholder="https://www.who.int/news-room/fact-sheets/detail/diabetes")
        if st.button("📥 Fetch & ingest URL", disabled=not page_url):
            import requests
            from bs4 import BeautifulSoup

            try:
                with st.spinner(f"Fetching {page_url} ..."):
                    resp = requests.get(page_url, timeout=15, headers={"User-Agent": "MedSourceBot/1.0"})
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.content, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header"]):
                        tag.decompose()
                    title = soup.title.string.strip() if soup.title and soup.title.string else page_url
                    text = " ".join(soup.get_text(separator=" ").split())
                if text and len(text) > 200:
                    doc = MedicalDocument(
                        content=text[:20000],
                        title=title,
                        source=page_url.split("/")[2] if "//" in page_url else page_url,
                        url=page_url,
                    )
                    with st.spinner("Embedding page content and updating the index..."):
                        vector_store.add_documents([doc])
                        vector_store.save(settings.vector_store_path)
                        corpus_builder.save_corpus([doc], f"webpage_{int(time.time())}.json")
                    st.success(f"Added '{title}' to the vector store ✅")
                    st.rerun()
                else:
                    st.warning("Could not extract enough readable text from that page.")
            except Exception as e:
                st.error(f"Failed to fetch page: {e}")

    # ---- Paste text ----
    with src_tab_text:
        st.write("Paste in a guideline excerpt, clinical note, or any trusted medical text directly.")
        with st.form("paste_text_form"):
            t_title = st.text_input("Title")
            t_source = st.text_input("Source name", placeholder="e.g. Harrison's Internal Medicine, 21st ed.")
            t_year = st.text_input("Year (optional)")
            t_url = st.text_input("URL (optional)")
            t_content = st.text_area("Content", height=200)
            submitted = st.form_submit_button("📥 Ingest text")
        if submitted:
            if t_title and t_content:
                doc = MedicalDocument(
                    content=t_content, title=t_title, source=t_source or "Manual entry",
                    year=t_year or None, url=t_url or None,
                )
                with st.spinner("Embedding and updating the index..."):
                    vector_store.add_documents([doc])
                    vector_store.save(settings.vector_store_path)
                    corpus_builder.save_corpus([doc], f"manual_{int(time.time())}.json")
                st.success(f"Added '{t_title}' to the vector store ✅")
                st.rerun()
            else:
                st.warning("Title and content are required.")

# --------------------------------------------------------------------------
# TAB 3 — Insights
# --------------------------------------------------------------------------
with tab_insights:
    st.markdown("#### Vector store overview")
    if not vector_store.documents:
        st.info("No documents indexed yet.")
    else:
        stats = vector_store.get_stats()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="mi-metric-card"><h3>{stats["total_chunks"]}</h3>Chunks</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="mi-metric-card"><h3>{stats["unique_documents"]}</h3>Documents</div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="mi-metric-card"><h3>{stats["dimension"]}</h3>Embedding dim</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="mi-metric-card"><h3>{stats["chunk_size"]}</h3>Chunk size</div>', unsafe_allow_html=True)

        st.markdown("#### Indexed documents")
        seen = {}
        for d in vector_store.documents:
            seen.setdefault(d["doc_id"], {"title": d["title"], "source": d["source"], "year": d.get("year"), "chunks": 0})
            seen[d["doc_id"]]["chunks"] += 1
        import pandas as pd

        df = pd.DataFrame(
            [{"Title": v["title"], "Source": v["source"], "Year": v["year"], "Chunks": v["chunks"]} for v in seen.values()]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)