---
title: Redrob AI Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Redrob Ranker Sandbox 🎯

This is a Hugging Face Space running a Streamlit dashboard that validates the end-to-end execution of our Redrob AI candidate ranking pipeline.

### Features
- **Reproducibility:** Runs the exact same `precompute.py` and `rank.py` logic as the local pipeline.
- **Explainability:** Displays the generated reasoning side-by-side with the candidate ranking.
- **Exportable:** Produces a strictly formatted `submission.csv` for download.

Upload a `.jsonl` file to test it, or use the pre-loaded 100-candidate sample!
