import streamlit as st
import subprocess
import os
import pandas as pd

st.set_page_config(
    page_title="Lobber — Intelligent Resume Ranker",
    layout="wide",
    page_icon="🎯",
)

# ── Custom CSS for a cleaner, more professional look ──────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 2rem; font-weight: 700; }
    h2 { font-size: 1.2rem; font-weight: 600; color: #555; margin-top: 0; }
    .stButton > button { font-weight: 600; }
    div[data-testid="stMetric"] {
        background: #f7f9fc;
        border: 1px solid #e0e4ea;
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

os.makedirs("artifacts", exist_ok=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Lobber — Intelligent Resume Ranker")
st.markdown(
    "<h2>Multi-stage ranking pipeline: hard filters → semantic embeddings → composite scoring → ranked output</h2>",
    unsafe_allow_html=True,
)

st.divider()

# ── Input section ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Input</p>', unsafe_allow_html=True)

col_check, col_upload = st.columns([1, 1], gap="large")

SAMPLE_PATH = "sample_data.jsonl"

with col_check:
    use_sample = st.checkbox(
        "Use pre-loaded dataset (100 candidates)",
        value=True,
        help="First 100 candidates extracted from the competition dataset. Runs in under 30 seconds.",
    )

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a custom candidate file (.jsonl / .json)",
        type=["jsonl", "json"],
        disabled=use_sample,
        label_visibility="collapsed" if use_sample else "visible",
    )

if use_sample:
    candidate_file_path = SAMPLE_PATH
    st.caption(f"Source: `{SAMPLE_PATH}` — 100 candidates from the competition dataset")
else:
    if uploaded_file is not None:
        candidate_file_path = "uploaded_candidates.jsonl"
        with open(candidate_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.caption(f"Source: `{uploaded_file.name}` ({uploaded_file.size:,} bytes)")
    else:
        st.info("Upload a `.jsonl` or `.json` file to proceed, or enable the pre-loaded dataset above.")
        st.stop()

st.divider()

run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)

if run_clicked:
    # ── Stage 1 ───────────────────────────────────────────────────────────────
    with st.spinner("Stage 1 — Hard Filters & Embeddings..."):
        result_pre = subprocess.run(
            ["python", "precompute.py", "--candidates", candidate_file_path],
            capture_output=True, text=True,
        )
        if result_pre.returncode != 0:
            st.error("Stage 1 failed. Error output:")
            st.code(result_pre.stderr or result_pre.stdout, language="bash")
            st.stop()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    with st.spinner("Stage 2 — Scoring & Ranking..."):
        result_rank = subprocess.run(
            ["python", "rank.py"],
            capture_output=True, text=True,
        )
        if result_rank.returncode != 0:
            st.error("Stage 2 failed. Error output:")
            st.code(result_rank.stderr or result_rank.stdout, language="bash")
            st.stop()

    # ── Results ───────────────────────────────────────────────────────────────
    if not os.path.exists("submission.csv"):
        st.error("submission.csv was not produced. Check your pipeline configuration.")
        st.stop()

    df = pd.read_csv("submission.csv")

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Summary</p>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates Ranked", len(df))
    m2.metric("Top Score", f"{df['score'].max():.4f}")
    m3.metric("Median Score", f"{df['score'].median():.4f}")
    m4.metric("Score Spread", f"{df['score'].max() - df['score'].min():.4f}")

    st.divider()

    # ── Visualisations ────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Score Distribution</p>', unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.caption("Score by Rank (top 100)")
        chart_df = df[["rank", "score"]].set_index("rank")
        st.line_chart(chart_df, use_container_width=True)

    with chart_col2:
        st.caption("Score histogram")
        import numpy as np
        hist_vals, bin_edges = np.histogram(df["score"], bins=20)
        bin_labels = [f"{e:.3f}" for e in bin_edges[:-1]]
        hist_df = pd.DataFrame({"Score bucket": bin_labels, "Count": hist_vals})
        st.bar_chart(hist_df.set_index("Score bucket"), use_container_width=True)

    st.divider()

    # ── Ranked table ──────────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Ranked Candidates</p>', unsafe_allow_html=True)

    # Split reasoning into its own expandable section to prevent horizontal overflow
    display_df = df[["rank", "candidate_id", "score"]].copy()
    st.dataframe(
        display_df,
        column_config={
            "rank":         st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "candidate_id": st.column_config.TextColumn("Candidate ID", width="medium"),
            "score":        st.column_config.NumberColumn("Score", format="%.5f", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        height=480,
    )

    st.divider()

    # ── Reasoning explorer ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Reasoning Explorer</p>', unsafe_allow_html=True)
    st.caption("Select a rank to read the full reasoning string for that candidate.")

    selected_rank = st.selectbox("Rank", options=df["rank"].tolist(), index=0)
    row = df[df["rank"] == selected_rank].iloc[0]
    st.markdown(f"**{row['candidate_id']}** — Score `{row['score']:.5f}`")
    st.info(row["reasoning"])

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
