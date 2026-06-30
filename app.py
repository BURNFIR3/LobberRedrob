import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Lobber — Intelligent Resume Ranker",
    layout="wide",
    page_icon="🎯",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2rem; font-weight: 700; }
    h2 { font-size: 1.2rem; font-weight: 600; color: #555; margin-top: 0; }
    .stButton > button { font-weight: 600; }
    div[data-testid="stMetric"] {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }
    div[data-testid="stMetric"] div {
        color: #FFFFFF !important;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.4rem;
    }
    .reasoning-box {
        background: #f9fafb;
        border-left: 3px solid #4A90D9;
        padding: 0.6rem 0.9rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

os.makedirs("artifacts", exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Lobber — Intelligent Resume Ranker")
st.markdown(
    "<h2>Multi-stage ranking pipeline: hard filters → semantic embeddings → composite scoring</h2>",
    unsafe_allow_html=True,
)

st.divider()

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)

SAMPLE_PATH = "sample_data.jsonl"

data_source = st.radio(
    "Choose data source:",
    ["Use pre-loaded sample (100 candidates)", "Upload custom dataset"],
    horizontal=True,
    label_visibility="collapsed"
)

if data_source == "Upload custom dataset":
    uploaded_file = st.file_uploader(
        "Drag and drop a custom candidate file (.jsonl, .json, or .gz) here",
        type=["jsonl", "json", "gz"],
        label_visibility="visible",
    )
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.gz'):
            candidate_file_path = "uploaded_candidates.jsonl.gz"
        else:
            candidate_file_path = "uploaded_candidates.jsonl"
            
        with open(candidate_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.caption(f"Source: `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")
    else:
        st.info("Please upload a `.jsonl`, `.json`, or `.gz` file to proceed.")
        st.stop()
else:
    candidate_file_path = SAMPLE_PATH
    st.caption(f"Source: `{SAMPLE_PATH}` — 100 candidates from the competition dataset")

st.divider()
run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

if run_clicked:
    with st.spinner("Stage 1 — Hard Filters & Embeddings..."):
        r1 = subprocess.run(
            ["python", "precompute.py", "--candidates", candidate_file_path],
            capture_output=True, text=True,
        )
        if r1.returncode != 0:
            st.error("Stage 1 failed:")
            st.code(r1.stderr or r1.stdout, language="bash")
            st.stop()

    with st.spinner("Stage 2 — Scoring & Ranking..."):
        r2 = subprocess.run(["python", "rank.py"], capture_output=True, text=True)
        if r2.returncode != 0:
            st.error("Stage 2 failed:")
            st.code(r2.stderr or r2.stdout, language="bash")
            st.stop()

    if not os.path.exists("submission.csv"):
        st.error("submission.csv was not produced. Check your pipeline configuration.")
        st.stop()

    df = pd.read_csv("submission.csv")
    sig_df = pd.read_csv("signals.csv") if os.path.exists("signals.csv") else None

    st.divider()

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Candidates Ranked", len(df))
    m2.metric("Top Score", f"{df['score'].max():.4f}")
    m3.metric("Median Score", f"{df['score'].median():.4f}")

    st.divider()

    # ── Signal charts (3 specific) ────────────────────────────────────────────
    if sig_df is not None:
        st.markdown('<p class="section-label">Signal Breakdown by Rank</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            st.caption("Matched JD Skill Count")
            st.bar_chart(
                sig_df[["rank", "matched_skill_count"]].set_index("rank"),
                use_container_width=True,
            )

        with c2:
            st.caption("GitHub Activity Score (−1 = not linked)")
            # Treat -1 as NaN so it doesn't distort the chart
            gh = sig_df[["rank", "github_activity_score"]].copy()
            gh.loc[gh["github_activity_score"] < 0, "github_activity_score"] = None
            st.line_chart(gh.set_index("rank"), use_container_width=True)

        with c3:
            st.caption("Recruiter Response Rate (−1 = no data)")
            rr = sig_df[["rank", "recruiter_response_rate"]].copy()
            rr.loc[rr["recruiter_response_rate"] < 0, "recruiter_response_rate"] = None
            st.line_chart(rr.set_index("rank"), use_container_width=True)

        st.divider()

    # ── Ranked table + inline reasoning ──────────────────────────────────────
    st.markdown('<p class="section-label">Selected Candidates with Ranks and Reasoning</p>', unsafe_allow_html=True)

    for _, row in df.iterrows():
        with st.expander(
            f"**#{int(row['rank'])}** &nbsp; {row['candidate_id']} &nbsp;|&nbsp; Score: `{row['score']:.5f}`",
            expanded=int(row['rank']) <= 5,
        ):
            st.markdown(
                f'<div class="reasoning-box">{row["reasoning"]}</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download submission.csv",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
    )
