import streamlit as st
import subprocess
import os
import pandas as pd
import json

st.set_page_config(page_title="Redrob AI Ranker", layout="wide")

st.title("🎯 Redrob AI Ranker Sandbox")
st.markdown("This sandbox validates the end-to-end ranking pipeline. It runs the hard filters, computes semantic embeddings, and ranks candidates against the JD.")

# Ensure artifacts dir exists for the pipeline to write to
os.makedirs("artifacts", exist_ok=True)

# Sidebar
st.sidebar.header("Data Source")
data_source = st.sidebar.radio(
    "Select Candidate Source", 
    ["Pre-loaded Sample (100 candidates)", "Upload Custom JSONL"]
)

candidate_file_path = "sample_candidates.json" # Default bundled sample

if data_source == "Upload Custom JSONL":
    uploaded_file = st.sidebar.file_uploader("Upload candidates (.jsonl or .json)", type=["jsonl", "json"])
    if uploaded_file is not None:
        candidate_file_path = "uploaded_candidates.jsonl"
        with open(candidate_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Uploaded {uploaded_file.name}")
    else:
        st.info("Please upload a .jsonl file to continue, or use the pre-loaded sample.")
        st.stop()

if st.button("🚀 Run Ranking Pipeline", type="primary"):
    
    # --- Step 1: Precompute ---
    with st.spinner("Running Stage 1: Hard Filters & Embeddings (precompute.py)..."):
        try:
            result_pre = subprocess.run(
                ["python", "precompute.py", "--candidates", candidate_file_path],
                capture_output=True, text=True, check=True
            )
            st.success("Precompute completed!")
            with st.expander("View Precompute Logs"):
                st.code(result_pre.stderr or result_pre.stdout)
                
        except subprocess.CalledProcessError as e:
            st.error("Precompute failed!")
            st.code(e.stderr or e.stdout)
            st.stop()
            
    # --- Step 2: Rank ---
    with st.spinner("Running Stage 2: Scoring & Ranking (rank.py)..."):
        try:
            result_rank = subprocess.run(
                ["python", "rank.py"],
                capture_output=True, text=True, check=True
            )
            st.success("Ranking completed!")
            with st.expander("View Ranking Logs"):
                st.code(result_rank.stderr or result_rank.stdout)
                
        except subprocess.CalledProcessError as e:
            st.error("Ranking failed!")
            st.code(e.stderr or e.stdout)
            st.stop()
            
    # --- Step 3: Display Results ---
    st.markdown("### 🏆 Top Ranked Candidates")
    if os.path.exists("submission.csv"):
        df = pd.read_csv("submission.csv")
        
        # Display the dataframe with nice formatting
        st.dataframe(
            df,
            column_config={
                "candidate_id": st.column_config.TextColumn("Candidate ID", width="small"),
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "score": st.column_config.NumberColumn("Score", format="%.4f", width="small"),
                "reasoning": st.column_config.TextColumn("Reasoning", width="large")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Download button
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.error("Error: submission.csv was not generated.")
