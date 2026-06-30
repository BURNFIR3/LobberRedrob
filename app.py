import streamlit as st
import subprocess
import os
import io
import pandas as pd

st.set_page_config(page_title="Redrob AI Ranker", layout="wide", page_icon="🎯")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎯 Redrob AI Ranker Sandbox")
st.markdown(
    "End-to-end ranking pipeline — hard filters → semantic embeddings → scored & reasoned CSV output."
)

os.makedirs("artifacts", exist_ok=True)

# ── Data source (single page, no sidebar) ────────────────────────────────────
st.markdown("---")
st.subheader("📂 Candidate Data")

col_left, col_right = st.columns([1, 1])

with col_left:
    use_sample = st.checkbox("Use pre-loaded sample (first 100 candidates from dataset)", value=True)

with col_right:
    uploaded_file = st.file_uploader(
        "Or upload a custom candidates file (.jsonl / .json)",
        type=["jsonl", "json"],
        disabled=use_sample,
    )

# Resolve which file to use
SAMPLE_PATH = "sample_data.jsonl"

if use_sample:
    candidate_file_path = SAMPLE_PATH
    st.info(f"Using **{SAMPLE_PATH}** — 100 candidates pre-loaded from the competition dataset.")
else:
    if uploaded_file is not None:
        candidate_file_path = "uploaded_candidates.jsonl"
        with open(candidate_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
    else:
        st.warning("Upload a file above, or check **Use pre-loaded sample** to proceed.")
        st.stop()

st.markdown("---")

# ── Run button ────────────────────────────────────────────────────────────────
run_clicked = st.button("🚀 Run Ranking Pipeline", type="primary", use_container_width=True)

if run_clicked:
    # ── Step 1: precompute ───────────────────────────────────────────────────
    with st.status("⚙️ Stage 1: Hard Filters & Embeddings (precompute.py)...", expanded=True) as status1:
        result_pre = subprocess.run(
            ["python", "precompute.py", "--candidates", candidate_file_path],
            capture_output=True, text=True
        )
        log_pre = result_pre.stderr or result_pre.stdout
        st.code(log_pre, language="bash")
        if result_pre.returncode != 0:
            status1.update(label="❌ Precompute failed", state="error")
            st.stop()
        status1.update(label="✅ Stage 1 complete", state="complete")

    # ── Step 2: rank ─────────────────────────────────────────────────────────
    with st.status("📊 Stage 2: Scoring & Ranking (rank.py)...", expanded=True) as status2:
        result_rank = subprocess.run(
            ["python", "rank.py"],
            capture_output=True, text=True
        )
        log_rank = result_rank.stderr or result_rank.stdout
        st.code(log_rank, language="bash")
        if result_rank.returncode != 0:
            status2.update(label="❌ Ranking failed", state="error")
            st.stop()
        status2.update(label="✅ Stage 2 complete", state="complete")

    # ── Step 3: Results ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🏆 Ranked Candidates")

    if os.path.exists("submission.csv"):
        df = pd.read_csv("submission.csv")
        st.dataframe(
            df,
            column_config={
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="small"),
                "rank":         st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "score":        st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "reasoning":    st.column_config.TextColumn("Reasoning", width="large"),
            },
            hide_index=True,
            use_container_width=True,
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_bytes,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )
    else:
        st.error("submission.csv was not produced — check the logs above.")
