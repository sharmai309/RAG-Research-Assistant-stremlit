"""
RAG System UI - Streamlit App
Portfolio project for OpenAI Data Scientist application
"""

import streamlit as st
import json
import time
from rag_pipeline import RAGSystem, ArxivFetcher

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    .main { background: #0a0a0f; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h2 { color: #e94560; margin: 0; font-size: 2rem; }
    .metric-card p { color: #8892b0; margin: 0; font-size: 0.85rem; }

    .flag-badge {
        display: inline-block;
        background: #e9456022;
        border: 1px solid #e94560;
        color: #e94560;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-family: 'Space Mono', monospace;
        margin: 2px;
    }
    .flag-badge.ok {
        background: #00ff8822;
        border-color: #00ff88;
        color: #00ff88;
    }

    .chunk-card {
        background: #111122;
        border-left: 3px solid #0f3460;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .chunk-card.high { border-left-color: #00ff88; }
    .chunk-card.medium { border-left-color: #ffd700; }
    .chunk-card.low { border-left-color: #e94560; }

    .answer-box {
        background: #111122;
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.5rem;
        line-height: 1.7;
    }

    .health-bar-bg {
        background: #1a1a2e;
        border-radius: 8px;
        height: 8px;
        margin-top: 4px;
    }
    .health-bar {
        border-radius: 8px;
        height: 8px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

if "rag" not in st.session_state:
    st.session_state.rag = None
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 RAG Research Assistant")
    st.markdown("*Powered by arXiv + OpenAI*")
    st.divider()

    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("**Search Topic**")
    topic = st.text_input("arXiv query", value="large language models")

    num_papers = st.slider("Number of papers", 5, 50, 20)

    if st.button("🚀 Build Knowledge Base", use_container_width=True, type="primary"):
        if not api_key:
            st.error("Please enter your OpenAI API key")
        else:
            with st.spinner("Fetching papers from arXiv..."):
                try:
                    fetcher = ArxivFetcher()
                    papers = fetcher.fetch_papers(query=topic, max_results=num_papers)
                    st.success(f"Fetched {len(papers)} papers!")

                    rag = RAGSystem(api_key=api_key)
                    progress = st.progress(0, text="Embedding papers...")

                    def update_progress(current, total):
                        progress.progress(current / total, text=f"Embedding paper {current}/{total}...")

                    total_chunks = rag.ingest_papers(papers, progress_callback=update_progress)
                    progress.empty()

                    st.session_state.rag = rag
                    st.session_state.ingested = True
                    st.success(f"✅ {total_chunks} chunks indexed!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.ingested:
        st.divider()
        st.markdown("**Knowledge Base**")
        rag = st.session_state.rag
        st.metric("Papers", len(rag.papers))
        st.metric("Chunks indexed", rag.vector_store.size())

        st.divider()
        if st.button("📊 View Eval Summary", use_container_width=True):
            st.session_state.show_eval = True


# ─────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────

st.markdown("# RAG Research Assistant")
st.markdown("*Ask questions about the latest AI papers. Failure analysis included.*")
st.divider()

if not st.session_state.ingested:
    st.info("👈 Enter your API key and build the knowledge base to get started.")

    st.markdown("### What this system does:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **📥 Ingestion**
        - Fetches papers from arXiv
        - Chunks text with overlap
        - Embeds with OpenAI
        """)
    with col2:
        st.markdown("""
        **🔍 Retrieval**
        - Cosine similarity search
        - Top-K chunk retrieval
        - Source attribution
        """)
    with col3:
        st.markdown("""
        **⚠️ Failure Analysis**
        - Low confidence detection
        - Hallucination flags
        - Source diversity checks
        """)

else:
    rag = st.session_state.rag

    # ── Query Input ──
    query = st.text_input(
        "Ask a question about the papers",
        placeholder="e.g. What are the main challenges in training large language models?",
        key="query_input"
    )

    col_a, col_b = st.columns([1, 4])
    with col_a:
        top_k = st.selectbox("Retrieve top-K", [3, 5, 7, 10], index=1)
    with col_b:
        st.markdown("")  # spacing

    if st.button("🔍 Search & Analyze", type="primary", use_container_width=False):
        if not query.strip():
            st.warning("Please enter a question")
        else:
            with st.spinner("Retrieving and generating..."):
                result = rag.query(query, top_k=top_k)
                st.session_state.history.insert(0, result)

    # ── Results ──
    if st.session_state.history:
        result = st.session_state.history[0]

        st.divider()
        st.markdown("### 💬 Answer")

        # Health score
        health = result.failure_details.get("health_score", 100)
        health_color = "#00ff88" if health >= 80 else "#ffd700" if health >= 60 else "#e94560"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <h2 style="color:{health_color}">{health}</h2>
                <p>Health Score</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h2>{result.latency_ms:.0f}ms</h2>
                <p>Latency</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            top_sim = result.failure_details.get("top_similarity", 0)
            st.markdown(f"""<div class="metric-card">
                <h2>{top_sim:.2f}</h2>
                <p>Top Similarity</p>
            </div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card">
                <h2>{result.failure_details.get('unique_papers_retrieved', 0)}</h2>
                <p>Sources Used</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Failure flags
        if result.failure_flags:
            st.markdown("**⚠️ Failure Flags:**")
            flags_html = " ".join([f'<span class="flag-badge">{f}</span>' for f in result.failure_flags])
            st.markdown(flags_html, unsafe_allow_html=True)
        else:
            st.markdown('<span class="flag-badge ok">✓ NO FAILURES DETECTED</span>', unsafe_allow_html=True)

        st.markdown("")

        # Answer
        st.markdown(f'<div class="answer-box">{result.answer}</div>', unsafe_allow_html=True)

        # Retrieved chunks
        st.markdown("### 📄 Retrieved Chunks")
        for i, chunk in enumerate(result.retrieved_chunks):
            score = chunk.similarity_score
            quality = "high" if score >= 0.85 else "medium" if score >= 0.70 else "low"
            with st.expander(f"[{score:.3f}] {chunk.paper_title[:80]}..."):
                st.markdown(f'<div class="chunk-card {quality}">{chunk.text}</div>', unsafe_allow_html=True)
                st.markdown(f"🔗 [View paper](https://arxiv.org/abs/{chunk.arxiv_id}) · Chunk #{chunk.chunk_index}")

        # Failure details JSON
        with st.expander("🔎 Full Failure Analysis Details"):
            st.json(result.failure_details)

        # Query history
        if len(st.session_state.history) > 1:
            st.divider()
            st.markdown("### 📜 Query History")
            for old_result in st.session_state.history[1:]:
                flags_str = ", ".join(old_result.failure_flags) if old_result.failure_flags else "✓ Clean"
                health_old = old_result.failure_details.get("health_score", 100)
                st.markdown(f"**{old_result.query[:60]}...** — Health: {health_old} | Flags: {flags_str}")

    # ── Eval Summary ──
    if st.session_state.get("show_eval") and st.session_state.history:
        st.divider()
        st.markdown("### 📊 Evaluation Summary")
        summary = rag.get_eval_summary()
        if summary:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Queries", summary["total_queries"])
            c2.metric("Clean Queries", summary["clean_queries"])
            c3.metric("Flagged Queries", summary["flagged_queries"])
            c4.metric("Avg Health Score", f"{summary['avg_health_score']:.1f}")
            st.markdown("**Flag Distribution:**")
            st.bar_chart(summary["flag_distribution"])
            st.session_state.show_eval = False
        else:
            st.info("Run some queries first to see the eval summary.")
