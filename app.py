import streamlit as st
from pathlib import Path
import tempfile
import pandas as pd

from config import ROLE_PROFILES
from rag_pipeline.loaders import load_contract
from rag_pipeline.risk_db import build_risk_vectorstore
from rag_pipeline.clause_extractor import extract_clauses
from rag_pipeline.risk_scorer import score_contract, classify_risk
from rag_pipeline.retriever import build_contract_vectorstore
from rag_pipeline.chatbot import (
    chat_once,
    summarize_contract_basic,
    summarize_contract_smart,
)
import os
import logging

os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
# Silence Chroma loggers
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

try:
    # Extra hard kill for telemetry capture if available
    import chromadb.telemetry as _ct

    if hasattr(_ct, "telemetry") and hasattr(_ct.telemetry, "_TelemetryClient"):
        _ct.telemetry._TelemetryClient.capture = staticmethod(lambda *a, **k: None)
except Exception:
    # If anything fails here, just ignore; it's only for log suppression
    pass

import functools

@functools.lru_cache(maxsize=1)
def ensure_risk_vectorstore():

# Build risk vector store at startup (from data/risk_db.csv)
    build_risk_vectorstore()

st.set_page_config(page_title="AIGenies-CRA", layout="wide")
st.title("🧾 AIGenies-Contract Risk Analysis with Chatbot")

ensure_risk_vectorstore()

# Sidebar: upload + role selection
st.sidebar.header("1️⃣ Upload contract document")
uploaded_file = st.sidebar.file_uploader(
    "Upload contract (PDF or DOCX only)",
    type=["pdf", "docx"],  # restricted to PDF + DOCX
    key="main_contract_uploader",
)

st.sidebar.header("2️⃣ Select role profile")
role_name = st.sidebar.selectbox("Risk appetite", list(ROLE_PROFILES.keys()))

# Session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_meta" not in st.session_state:
    st.session_state.analysis_meta = None
if "chat_history" not in st.session_state:
    # list of (role, text) with role = "user"/"assistant"
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "smart_summary" not in st.session_state:
    st.session_state.smart_summary = None   

# ---------- Helper functions ----------
def analyze_contract(file_bytes, filename: str):
    """
    Full RAG pipeline for a single uploaded contract:
      1. Save temp file
      2. Load contract (PDF/DOCX)
      3. Build contract vector store for RAG
      4. Extract clauses via LLM
      5. Score risk per clause + overall
      6. Return analysis + simple metadata
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # 1) Load contract into LangChain Documents
    docs = load_contract(tmp_path)

    # 2) Build contract vector store (RAG index) for this contract
    build_contract_vectorstore(docs)

    # 3) Aggregate text for clause extraction
    full_text = "\n\n".join(d.page_content for d in docs)

    # 4) Extract clauses with the LLM
    clauses = extract_clauses(full_text)

    # 5) Score clauses against risk DB
    analysis = score_contract(clauses)

    # 6) Metadata for status panel
    meta = {
        "file_name": filename,
        "num_docs": len(docs),       # for PDFs: pages; for DOCX: usually 1 big doc
        "num_clauses": len(clauses),
    }

    return analysis, meta


def build_clause_dataframe(analysis_result):
    """
    Flatten clause-level info into a pandas DataFrame
    for display and CSV/JSON download.
    """
    rows = []
    for clause in analysis_result["clauses"]:
        top_match = clause["matches"][0] if clause.get("matches") else {}
        rows.append({
            "clause_type": clause.get("clause_type"),
            "title": clause.get("title"),
            "risk_score": clause.get("risk_score"),
            "text": clause.get("text"),
            "top_match_risk_level": top_match.get("risk_level"),
            "top_match_similarity": top_match.get("similarity"),
            "top_match_weighted_score": top_match.get("weighted_score"),
            "top_match_notes": top_match.get("notes"),
        })
    return pd.DataFrame(rows)

# ---------- Layout ----------
col1, col2 = st.columns([1.4, 1])

# ===================== LEFT COLUMN: RISK ANALYSIS + SUMMARY =====================
with col1:
    st.subheader("Risk analysis")

    # Only operate on uploaded PDF/DOCX
    if uploaded_file is not None:
        if st.button("Run risk analysis", type="primary"):
            with st.spinner("Processing contract via RAG pipeline..."):
                analysis, meta = analyze_contract(
                    uploaded_file.read(), uploaded_file.name
                )
                st.session_state.analysis_result = analysis
                st.session_state.analysis_meta = meta
                # Reset chat history when a new contract is analyzed
                st.session_state.chat_history = []
                st.session_state.summary = None
                st.session_state.smart_summary = None

    if st.session_state.analysis_result:
        analysis = st.session_state.analysis_result
        meta = st.session_state.analysis_meta or {}

        # ---------- Status panel ----------
        st.markdown("### Status")

        c1, c2, c3 = st.columns(3)
        with c1:
            file_name = meta.get("file_name", "N/A")
            st.markdown(
                f"""
                <div style="
                    height: 60px; 
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <span style="font-size:16px; font-weight:600;">
                        File
                    </span>
                    <span style="
                        font-size:18px;
                        font-weight:400;
                        line-height:1.2;
                        word-wrap:break-word;
                        white-space:normal;
                        display:block;
                    ">
                        {file_name}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )   

        with c2:
            st.markdown(
                f"""
                <div style="height:60px; display:flex; align-items:center;">
                    <div>
                        <b>Pages / docs loaded</b><br>
                        <span style="font-size:24px;">{meta.get("num_docs", 0)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:
            st.markdown(
                f"""
                <div style="height:60px; display:flex; align-items:center;">
                    <div>
                        <b>Clauses extracted</b><br>
                        <span style="font-size:24px;">{meta.get("num_clauses", 0)}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        # ---------- Overall risk ----------
        overall = analysis["overall_risk_score"]
        overall_label = classify_risk(overall, role_name)

        st.metric(
            "Overall risk score (0 = low, 10 = high)",
            f"{overall:.2f}",
            help=f"Interpreted as {overall_label} risk for {role_name}",
        )

        st.markdown(f"**Role profile:** {role_name} → overall risk: **{overall_label}**")
        
        # ---------- Clause table + downloads ----------
        st.markdown("### Clause-level risk summary")

        clause_df = build_clause_dataframe(analysis)
        # Keep a copy of the full clause dataframe for downloads
        full_df = clause_df.copy()
        # Show only the top 1 highest-risk clause in the UI table
        display_df = clause_df.sort_values(by="risk_score", ascending=False).head(1)
        st.markdown("Showing the top 1 highest-risk clause (download to get full list)")
        st.dataframe(display_df, use_container_width=True, height=150)
        #st.dataframe(clause_df, height=250, width="stretch")
        
        # Build an expanded dataframe where each match is its own row
        rows_exp = []
        for clause in analysis["clauses"]:
            base = {
                "clause_type": clause.get("clause_type"),
                "title": clause.get("title"),
                "risk_score": clause.get("risk_score"),
                "text": clause.get("text"),
            }
            if clause.get("matches"):
                for m in clause["matches"]:
                    r = dict(base)
                    r.update({
                        "match_risk_level": m.get("risk_level"),
                        "match_similarity": m.get("similarity"),
                        "match_weighted": m.get("weighted_score"),
                        "match_text": m.get("text"),
                        "match_notes": m.get("notes"),
                    })
                    rows_exp.append(r)
            else:
                r = dict(base)
                r.update({
                    "match_risk_level": None,
                    "match_similarity": None,
                    "match_weighted": None,
                    "match_text": None,
                    "match_notes": None,
                })
                rows_exp.append(r)

        exp_df = pd.DataFrame(rows_exp)
        # Use the expanded dataframe for the primary downloads so they include all matches
        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        json_bytes = exp_df.to_json(orient="records", indent=2).encode("utf-8")
            
        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                label="⬇️ Download clauses as CSV",
                data=csv_bytes,
                file_name="contract_clause_risk.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                label="⬇️ Download clauses as JSON",
                data=json_bytes,
                file_name="contract_clause_risk.json",
                mime="application/json",
            )
        # (expanded-download buttons removed; primary download now contains expanded data)
            
        # ---------- Clause details ----------
        st.markdown("### Clause-level risk (details)")
        for clause in analysis["clauses"]:
            label = classify_risk(clause["risk_score"], role_name)
            title = clause.get("title") or clause["text"][:60] + "..."
            with st.expander(f"[{clause['clause_type']}] {title}"):
                st.write(f"**Risk score:** {clause['risk_score']:.2f} → **{label}**")
                st.write("**Clause text:**")
                st.write(clause["text"])
                st.write("**Top risk pattern matches:**")
                for m in clause["matches"]:
                    st.write(
                        f"- Risk level: {m['risk_level']} | "
                        f"sim: {m['similarity']:.2f} | "
                        f"weighted: {m['weighted_score']:.2f}"
                    )
                    st.write(f"  Pattern: {m['text']}")
                    if m.get("notes"):
                        st.write(f"  Notes: {m['notes']}")
                    st.write("---")

        # ---------- Contract summary ----------
        sum_col1, sum_col2 = st.columns(2)
        with sum_col1:
            if st.button("📝 Basic summary"):
                with st.spinner("Summarizing contract..."):
                    st.session_state.summary = summarize_contract_basic()
        with sum_col2:
            if st.button("🧠 Smart risk summary"):
                with st.spinner("Generating smart risk summary..."):
                    st.session_state.smart_summary = summarize_contract_smart()

        if st.session_state.summary:
            st.markdown("### Contract summary (overall)")
            st.markdown(st.session_state.summary)

        if st.session_state.smart_summary:
            st.markdown("### Smart risk summary (by risk theme)")
            st.markdown(st.session_state.smart_summary) 

with col2:
    # Small “popup-like” chat in an expander instead of full column
    st.markdown(
        "<div style='height: 40px'></div>",
        unsafe_allow_html=True,
    )  # some top spacing

    if st.session_state.analysis_result is None:
        st.info("Upload a PDF/DOCX and run risk analysis first.")
    else:
        # Collapsible “chat popup” in the right column
        chat_box = st.expander("💬 Chat with Contract Bot", expanded=False)

        with chat_box:
            st.markdown("Chat with the analyzed contract. Ask about risks, clauses, SLAs, etc.")
            st.markdown("---")

            # Render chat history inside the expander
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"**You:** {msg}")
                else:
                    st.markdown(f"**Bot:** {msg}")
                st.markdown("---")

            # Input + send button (no direct session_state mutation)
            user_input = st.text_input(
                "Type your question about this contract:",
                key="chat_input_box",
            )
            send = st.button("Send", key="chat_send_button")

            if send:
                # 🔹 Always read the latest value from session_state
                question = st.session_state.get("chat_input_box", "").strip()

                if question:
                    # 1) store user question
                    st.session_state.chat_history.append(("user", question))
                    lower_q = question.lower()

                    # 2) choose behavior based on question
                    if (
                    "smart summary" in lower_q
                    or "smart risk" in lower_q
                    or "red flag" in lower_q
                    ):
                        with st.spinner("Generating smart risk summary..."):
                            answer = summarize_contract_smart()
                    elif (
                        "summary" in lower_q
                        or "summarize" in lower_q
                        or "summarise" in lower_q
                    ):
                        with st.spinner("Summarizing contract..."):
                            answer = summarize_contract_basic()
                    else:
                        with st.spinner("Thinking..."):
                            answer = chat_once(question, st.session_state.chat_history)

                    # 3) store bot answer
                    st.session_state.chat_history.append(("assistant", answer))

                    # 4) trigger a rerun so the new messages appear immediately
                    if hasattr(st, "rerun"):
                        st.rerun()
                    elif hasattr(st, "experimental_rerun"):
                        st.experimental_rerun()