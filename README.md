# Lobber — Intelligent Resume Ranker

A multi-stage, fully explainable AI ranking pipeline built for the Redrob Hackathon. Given a pool of candidate profiles in JSONL format, the system outputs a ranked list of the top 100 candidates best suited for a senior NLP/IR engineering role. Each candidate in the output receives a natural-language reasoning string that cites specific facts; employer, tenure, named skills, and signal values, directly from their profile.

The Resume Ranker is hosted at https://huggingface.co/spaces/Burnfir3/LobberRedrob

---

## Architecture

The pipeline is split into two sequential stages, each implemented as a standalone Python script with no inter-process state beyond files written to the `artifacts/` directory.

```text
candidates.jsonl
       │
       ▼
┌─────────────────────┐
│   precompute.py     │  Stage 1 — Deterministic hard filters + embedding generation
│                     │
│  • Honeypot check   │
│  • Title filter     │
│  • Location filter  │
│  • Inactivity gate  │
│  • Keyword gate     │
│  • Encode embeddings│
└──────────┬──────────┘
           │  artifacts/stage1_shortlist.jsonl
           │  artifacts/embeddings.npy
           │  artifacts/id_index.json
           ▼
┌─────────────────────┐
│     rank.py         │  Stage 2 — Composite scoring, ranking, reasoning
│                     │
│  • Semantic sim.    │
│  • Keyword bonus    │
│  • Skill depth      │
│  • YoE / tenure     │
│  • Notice period    │
│  • Redrob signals   │
│  • Reasoning gen.   │
└──────────┬──────────┘
           │
           ▼
     submission.csv  (top 100, ranked)
```

### Stage 1 — Hard Filtering (`precompute.py`)

Applies five deterministic gates in sequence. A candidate is eliminated at the first gate it fails; no subsequent gates are evaluated for that candidate.

| Filter | Logic |
|---|---|
| **Honeypot detection** | Flags profiles with expert-level skills of zero duration, large unrecorded career gaps (`>= HONEYPOT_UNRECORDED_GAP_PCT` of claimed YoE), or skill months that exceed career months by more than 4x |
| **Hard title disqualification** | Rejects titles containing explicit disqualifying fragments defined in `jd.py` (e.g. `qa engineer`, `frontend developer`, `android developer`) |
| **Location** | Requires the profile country or location field to contain "india", unless the candidate has explicitly indicated willingness to relocate |
| **Inactivity** | Rejects candidates whose last activity is more than `MAX_INACTIVE_DAYS_IF_NOT_OPEN` days ago and who are not flagged open to work |
| **Domain keyword presence** | Requires at least one term from `MUST_HAVE_SKILL_KEYWORDS` or `JD_RELEVANT_ROLE_KEYWORDS` to appear anywhere in the full profile text (all titles, headline, summary, descriptions) |

Survivors are written to `artifacts/stage1_shortlist.jsonl`. Their full profile text is then encoded using `sentence-transformers/all-MiniLM-L6-v2` (384-dimension dense vectors) and saved to `artifacts/embeddings.npy`, with a matching candidate ID index at `artifacts/id_index.json`.

### Stage 2 — Scoring and Ranking (`rank.py`)

Loads the precomputed artifacts and assigns each surviving candidate a composite floating-point score. The score is initialised from the semantic similarity (cosine) between the candidate embedding and the JD embedding, and then adjusted by additive bonuses and penalties.

**Scoring components:**

| Component | Direction | Description |
|---|---|---|
| **Semantic similarity** | Base | Cosine similarity between candidate and JD embeddings |
| **Keyword bonus** | + | Scaled count of exact JD keyword hits across the full profile text |
| **Skill depth bonus** | + | Proportional to matched JD skill count; capped at 15 matched skills |
| **Semantic illusion penalty** | − | Applied when semantic score is high but keyword hit count is low — catches profiles that "sound like" the JD without explicitly demonstrating the required skills |
| **YoE penalty** | − | Graduated penalty for candidates with fewer years of experience than the role minimum |
| **YoE gap penalty** | − | Additional penalty for a significant gap between claimed and verifiable experience |
| **Job-hop penalty** | − | Applied when average career tenure falls below 18 months |
| **Notice period** | − | Tiered: no penalty ≤ 30d, mild 31–60d, moderate 61–90d, significant > 90d |
| **GitHub activity bonus** | + | Secondary signal; bonus weight capped at 0.06. Missing GitHub (`github == -1`) is treated as neutral — no penalty applied |
| **Recruiter response rate** | +/− | Bonus for high response rate; penalty for low rate or missing signal |
| **Top-tier company bonus** | + | Additive bonus for FAANG/tier-1 company appearances in career history |
| **Junior mismatch penalty** | − | Penalises profiles claiming high YoE alongside exclusively junior-level titles |
| **Consulting-only penalty** | − | Mild penalty for an exclusively agency/consulting career history |
| **Unendorsed expert skills** | − | Per-skill penalty for expert-level self-assessments with zero endorsements |

**Reasoning generation:**

Each of the top 100 candidates receives a deterministically generated natural-language reasoning string. Every string opens with a facts prefix that names concrete profile data:

```text
Recent: {Title} @ {Company} ({Tenure}mo). Matched JD skills: {skill_1}, {skill_2}, ...
```

The remainder of the string selects from a template bank based on which scoring components dominated. Templates are differentiated by rank band (ranks 1–20 vs. 21–100) and reference specific signal values (semantic score, notice period days, YoE, company names) directly.

---

## Repository Structure

```text
.
├── app.py                      # Streamlit UI for hosted reproduction
├── precompute.py               # Stage 1: hard filters + embedding generation
├── rank.py                     # Stage 2: scoring, ranking, reasoning, CSV output
├── jd.py                       # Job description constants (keywords, thresholds, tier lists)
├── validate_submission.py      # Submission format validator per spec sections 2–3
├── requirements.txt            # Python dependencies
├── sample_data.jsonl           # 100-candidate extract for hosted reproduction
├── submission.csv              # Latest ranked output (top 100)
├── submission_metadata.yaml    # Submission metadata for the Redrob portal
├── Main/
│   ├── candidates.jsonl        # NOT included — competition dataset (place here manually)
│   └── rules.yaml
├── artifacts/                  # Runtime-generated; not committed
│   ├── embeddings.npy
│   ├── id_index.json
│   └── stage1_shortlist.jsonl
└── problem_statements/         # Competition documents; not committed
```

---

## Reproduction

### Requirements

- Python 3.11+
- ~4 GB RAM (for embedding the full 100k candidate pool)
- No GPU required
- No network access required during ranking

### Install

```powershell
pip install -r requirements.txt
```

### Data Setup

> **The `candidates.jsonl` dataset is not included in this repository** — it is the proprietary competition dataset provided by Redrob to registered participants only.

If you have access to the competition dataset, place it as follows:

```
LobberRedrob/
└── Main/
    └── candidates.jsonl      ← place the file here
```

If you want to run a quick test with the bundled 100-candidate sample instead (no download needed), skip the above and use `sample_data.jsonl` in the run command:

```powershell
# Quick test with the included 100-candidate sample
python precompute.py --candidates ./sample_data.jsonl
python rank.py
```

### Run end-to-end (full competition dataset)

Once `Main/candidates.jsonl` is in place:

```powershell
# Stage 1: apply hard filters and generate embeddings (~4-5 min on CPU)
python precompute.py --candidates ./Main/candidates.jsonl

# Stage 2: score, rank, and write submission.csv (~35 seconds)
python rank.py
```

The full 100k run completes in approximately **5–6 minutes** on a CPU (16 cores, 16 GB RAM).

### Validate output

```powershell
python validate_submission.py submission.csv
```

---

## Reproduce Command

```powershell
# PowerShell
python precompute.py --candidates .\Main\candidates.jsonl ; python rank.py
```

Output is written to `submission.csv` in the working directory.
