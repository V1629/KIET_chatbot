# ══════════════════════════════════════════════════════
# app.py — Streamlit UI (entry point)
#
# Run:  streamlit run app.py
#
# This file ONLY handles UI rendering.
# All logic lives in pipeline/orchestrator.py
# ══════════════════════════════════════════════════════

import os
import glob
import threading

import streamlit as st
from openai import OpenAI
import cohere
from dotenv import load_dotenv

load_dotenv()  # Loads keys from .env file automatically

from config.settings import DATA_FOLDER, PERSIST_DIR
from storage.store   import data_changed, indexes_exist
from indexing.indexer import build_indexes, load_indexes_from_disk
from pipeline.orchestrator import (
    run_pipeline, run_faithfulness, cache_clear, cache_size
)


# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Advanced RAG Chatbot",
    page_icon  = "🧠",
    layout     = "wide",
)

st.markdown("""
<style>
.route-badge {
    display: inline-block; padding: 3px 12px;
    border-radius: 12px; font-size: 12px; font-weight: bold; margin: 4px 0;
}
.badge-TEXT       { background:#DBEAFE; color:#1D4ED8; }
.badge-ANALYTICAL { background:#D1FAE5; color:#065F46; }
.badge-GENERAL    { background:#EDE9FE; color:#5B21B6; }
.step-box { font-size:13px; color:#6B7280; padding: 4px 0; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Advanced RAG Chatbot")
st.caption("Hybrid Search · HyDE · Cohere Rerank · SQL Agent · Persistent Storage · Faithfulness Check")


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ API Keys")
    openai_key = st.text_input(
        "OpenAI API Key", type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
    )
    cohere_key = st.text_input(
        "Cohere API Key", type="password",
        value=os.environ.get("COHERE_API_KEY", ""),
        help="Free key at cohere.com",
    )

    st.divider()
    st.subheader("📁 Data Folder")
    st.code(f"./{DATA_FOLDER}/", language=None)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    json_files = glob.glob(os.path.join(DATA_FOLDER, "*.json"))
    if json_files:
        st.success(f"✅ {len(json_files)} JSON file(s) found")
        for f in json_files:
            kb = os.path.getsize(f) / 1024
            st.caption(f"  • {os.path.basename(f)}  ({kb:.0f} KB)")
    else:
        st.warning("No .json files in data/ folder")

    st.divider()
    st.subheader("💾 Persistent Storage")
    if indexes_exist():
        if data_changed():
            st.warning("⚠️ Data changed — re-index needed")
        else:
            st.success("✅ Indexes up to date")
        st.caption(f"Location: {PERSIST_DIR}/")
    else:
        st.info("No index yet — will build on first run")

    col1, col2 = st.columns(2)
    with col1:
        force_reindex = st.button("🔄 Re-index", help="Force rebuild all indexes")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption(f"💾 Cached queries: {cache_size()}/200")
    if st.button("🧹 Clear Cache"):
        cache_clear()
        st.rerun()

    st.divider()
    st.subheader("🔬 Pipeline")
    st.markdown("""
| Step | Method |
|------|--------|
| Route + HyDE | 1 GPT-4o-mini call |
| Retrieval | Vector + BM25 + RRF |
| Diversity | MMR |
| Reranking | Cohere v3 |
| Analytics | DuckDB SQL |
| Verify | Faithfulness (async) |
""")


# ══════════════════════════════════════════════════════
# GUARDS
# ══════════════════════════════════════════════════════

if not openai_key:
    st.info("👈 Enter your OpenAI API key in the sidebar.")
    st.stop()
if not cohere_key:
    st.info("👈 Enter your Cohere API key in the sidebar.")
    st.stop()
if not json_files:
    st.warning(f"Add .json files to `./{DATA_FOLDER}/` and refresh.")
    st.stop()


# ══════════════════════════════════════════════════════
# LOAD / BUILD INDEX (once per session or on data change)
# ══════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def get_indexes(_openai_key: str, _trigger: str):
    """
    Load from disk if index exists and data hasn't changed.
    Build from scratch only when necessary.
    _trigger forces cache invalidation on manual re-index.
    """
    if indexes_exist() and not data_changed():
        with st.spinner("⚡ Loading indexes from disk..."):
            return load_indexes_from_disk()
    else:
        progress_bar = st.progress(0, text="Building index for the first time...")
        def _cb(msg, pct):
            progress_bar.progress(pct, text=msg)
        idx = build_indexes(_openai_key, progress_callback=_cb)
        progress_bar.empty()
        return idx


# Trigger key: changes when user clicks Re-index
if "reindex_trigger" not in st.session_state:
    st.session_state.reindex_trigger = "initial"

if force_reindex:
    st.session_state.reindex_trigger = f"manual_{os.urandom(4).hex()}"
    st.cache_resource.clear()

try:
    idx = get_indexes(openai_key, st.session_state.reindex_trigger)
    idx["db_schemas"] = idx.get("db_schemas", {})
except Exception as e:
    st.error(f"❌ Indexing error: {e}")
    st.stop()

# Init OpenAI + Cohere clients
client    = OpenAI(api_key=openai_key)
co_client = cohere.Client(api_key=cohere_key)


# ══════════════════════════════════════════════════════
# CHAT HISTORY
# ══════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("route"):
            badge = msg["route"]
            st.markdown(
                f'<span class="route-badge badge-{badge}">{badge}</span>',
                unsafe_allow_html=True,
            )
        if msg.get("df") is not None:
            st.dataframe(msg["df"].head(50), use_container_width=True)
        if msg.get("sources"):
            with st.expander("📎 Sources & Scores", expanded=False):
                for s in msg["sources"]:
                    st.caption(f"• {s}")
        if msg.get("faithful") is False:
            st.warning("⚠️ Faithfulness check flagged this — please verify.")
        if msg.get("hyde_used"):
            st.caption("💡 HyDE was applied to improve recall")


# ══════════════════════════════════════════════════════
# CHAT INPUT → PIPELINE
# ══════════════════════════════════════════════════════

if prompt := st.chat_input("Ask anything about your data..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        step_ph  = st.empty()   # Live pipeline steps
        badge_ph = st.empty()   # Route badge

        try:
            # ── Run pipeline ───────────────────────────────────────────────
            with st.spinner("Thinking..."):
                pipeline_result = run_pipeline(idx, client, co_client, prompt)

            # Show steps
            step_ph.markdown(
                "\n\n".join(
                    f'<div class="step-box">{s}</div>'
                    for s in pipeline_result.steps
                ),
                unsafe_allow_html=True,
            )

            # Show route badge
            route = pipeline_result.route
            badge_ph.markdown(
                f'<span class="route-badge badge-{route}">{route}</span>',
                unsafe_allow_html=True,
            )

            # Show dataframe (ANALYTICAL)
            if pipeline_result.result_df is not None and not pipeline_result.result_df.empty:
                st.dataframe(pipeline_result.result_df.head(50), use_container_width=True)

            # ── Stream answer ──────────────────────────────────────────────
            if pipeline_result.from_cache and isinstance(pipeline_result.stream, str):
                step_ph.empty()
                st.markdown(pipeline_result.stream)
                response_text = pipeline_result.stream
            else:
                step_ph.empty()
                response_text = st.write_stream(pipeline_result.stream)

            # Store answer text for faithfulness check
            pipeline_result.answer_text = response_text

            # ── Faithfulness check (runs AFTER streaming — user sees no delay) ──
            faithful = True
            if route == "TEXT" and pipeline_result.chunks:
                faithful_result = {"value": True}

                def _faith_check():
                    faithful_result["value"] = run_faithfulness(client, pipeline_result)

                faith_thread = threading.Thread(target=_faith_check, daemon=True)
                faith_thread.start()
                faith_thread.join(timeout=5)  # Max 5s wait, then skip
                faithful = faithful_result["value"]

                if not faithful:
                    st.warning("⚠️ Faithfulness check flagged this answer — please verify.")

            # ── Sources ────────────────────────────────────────────────────
            sources = []
            if pipeline_result.chunks:
                seen = {}
                for c in pipeline_result.chunks:
                    src   = c["meta"]["source"]
                    score = c.get("rerank_score", 0)
                    if src not in seen or score > seen[src]:
                        seen[src] = score
                sources = [f"{s}  (relevance: {v:.2f})"
                           for s, v in sorted(seen.items(), key=lambda x: -x[1])]
                with st.expander("📎 Sources & Relevance Scores", expanded=False):
                    for s in sources:
                        st.caption(f"• {s}")

            if pipeline_result.hyde_used:
                st.caption("💡 HyDE was applied to improve recall")

            # ── Save to history ────────────────────────────────────────────
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   response_text,
                "route":     route,
                "sources":   sources,
                "df":        pipeline_result.result_df,
                "faithful":  faithful,
                "hyde_used": pipeline_result.hyde_used,
            })

        except Exception as e:
            err = f"❌ Error: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
