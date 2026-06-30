---
title: Redrob AI Ranker
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.32.0
python_version: "3.11"
app_file: app.py
pinned: false
---

# Lobber — Redrob AI Candidate Ranker

**Sandbox:** [huggingface.co/spaces/Burnfir3/LobberRedrob](https://huggingface.co/spaces/Burnfir3/LobberRedrob)

A multi-stage, fully explainable AI ranking pipeline built for the Redrob Hackathon. Given a pool of candidate profiles, the system outputs a ranked list of the top 100 candidates best suited for a senior NLP/IR engineering role, with a natural-language reasoning string for each candidate that cites specific facts from their profile.

---

## Architecture

The pipeline is split into two sequential stages, each implemented as a standalone Python script.

### Stage 1 — Hard Filtering (`precompute.py`)

This stage performs a deterministic, rule-based pass over the full candidate pool. Its goal is to eliminate structurally disqualified candidates before any expensive computation is performed.

Filters applied in order:

| Filter | Logic |
|---|---|
| **Honeypot detection** | Flags profiles with expert-level skills of zero duration, large unrecorded career gaps, or skill months that vastly exceed career months |
| **Hard title disqualification** | Rejects titles containing explicit disqualifying fragments (e.g. `qa engineer`, `frontend developer`) |
| **Location** | Requires India-based or explicit willingness to relocate |
| **Inactivity** | Rejects candidates inactive for more than 365 days who are not marked open to work |
| **Domain keyword presence** | Requires at least one JD-relevant keyword across the full profile text (titles, headline, summary, descriptions) |

Candidates surviving all filters are written to `artifacts/stage1_shortlist.jsonl`. Immediately after filtering, this stage encodes every surviving candidate's full profile text using `sentence-transformers/all-MiniLM-L6-v2` and saves the embedding matrix to `artifacts/embeddings.npy` alongside a candidate ID index at `artifacts/id_index.json`.

### Stage 2 — Scoring and Ranking (`rank.py`)

This stage loads the precomputed artifacts and assigns a composite score to each shortlisted candidate.

**Scoring components:**

| Component | Description |
|---|---|
| **Semantic similarity** | Cosine similarity between the candidate embedding and the JD embedding |
| **Keyword bonus** | Count of exact JD keyword hits across the profile, scaled linearly |
| **Skill depth bonus** | Bonus proportional to the number of matched JD skills (capped at 15) |
| **Semantic illusion penalty** | Penalises candidates with high semantic scores but very few keyword hits — catching profiles that "sound like" the JD without actually demonstrating the required skills |
| **Years of experience** | Penalty for candidates below the minimum threshold, graduated penalty for significant under-experience |
| **Job-hop penalty** | Penalty for average tenure below 18 months |
| **Notice period** | Tiered penalty: no penalty for preferred window, mild for manageable, significant for a lengthy notice |
| **GitHub activity bonus** | Secondary signal; capped at 0.06 (half the original weight) so it cannot dominate rank. Missing GitHub is treated as neutral |
| **Recruiter response rate** | Bonus/penalty based on historical response rate from Redrob signals |
| **Top-tier company bonus** | Small prestige bonus for FAANG and equivalent companies |
| **Junior mismatch penalty** | Penalises profiles claiming high YoE but with junior-level titles throughout |
| **Consulting-only penalty** | Mild penalty for candidates with an exclusively consulting/agency background |
| **Unendorsed expert skills** | Per-skill penalty for expert-level claims with zero endorsements |

**Reasoning generation:**

Each of the top 100 candidates receives a natural-language reasoning string. Every reasoning block opens with a concrete facts prefix:

```
Recent: {Title} @ {Company} ({Tenure}mo). Matched JD skills: {skill_1}, {skill_2}, ...
```

The remainder of the block selects from a template library to produce a coherent pro/con narrative, choosing language appropriate to the candidate's rank band.

---

## Repository Structure

```
.
├── app.py                      # Streamlit sandbox UI
├── precompute.py               # Stage 1: hard filters + embedding generation
├── rank.py                     # Stage 2: scoring, ranking, reasoning, CSV output
├── jd.py                       # Job description constants (keywords, thresholds, tiers)
├── validate_submission.py      # Submission format validator
├── requirements.txt            # Python dependencies
├── sample_data.jsonl           # First 100 candidates from the competition dataset (sandbox use)
├── submission.csv              # Latest ranked output
├── submission_metadata.yaml    # Submission metadata for the Redrob portal
├── Main/
│   └── rules.yaml              # Scoring rule configuration
├── artifacts/                  # Generated at runtime — not committed
│   ├── embeddings.npy
│   ├── id_index.json
│   └── stage1_shortlist.jsonl
└── problem_statements/         # Problem statement documents — not committed to git
```

---

## How to Run

### Requirements

- Python 3.11+
- ~4 GB RAM (for embedding 100k candidates)
- No GPU required; no network access required during ranking

### Install dependencies

```bash
pip install -r requirements.txt
```

### End-to-end execution

```bash
# Stage 1: filter candidates and generate embeddings
python precompute.py --candidates ./Main/candidates.jsonl

# Stage 2: score, rank, and produce submission.csv
python rank.py
```

The full 100k run completes in approximately 3–4 minutes on a modern CPU (8 cores, 16 GB RAM). The sandbox sample of 100 candidates completes in under 30 seconds.

### Validate the output

```bash
python validate_submission.py submission.csv
```

---

## Sandbox

A live, hosted version of this pipeline is available at:

**[https://huggingface.co/spaces/Burnfir3/LobberRedrob](https://huggingface.co/spaces/Burnfir3/LobberRedrob)**

The sandbox accepts either the pre-loaded 100-candidate sample or a custom `.jsonl` upload, runs the full two-stage pipeline, and produces a downloadable `submission.csv`. It runs on Hugging Face's free CPU tier and completes within the 5-minute compute budget.

---

## Reproduce Command

```bash
python precompute.py --candidates ./candidates.jsonl && python rank.py
```

Output is written to `submission.csv` in the working directory.
