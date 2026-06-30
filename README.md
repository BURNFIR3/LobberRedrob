# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

> **Challenge:** Rank 100,000 candidates against a Senior AI Engineer JD and produce a top-100 submission CSV.

## System Overview

A 2-stage pipeline:

| Stage | Input | Output | Purpose |
|---|---|---|---|
| Preprocess | `candidates.jsonl` | `artifacts/` | Extract texts + features |
| Embeddings | `artifacts/texts.txt` | `artifacts/embeddings.npy` | Semantic vectors |
| Stage 1 | embeddings + BM25 + rules | `artifacts/stage1_out.csv` | 100k → 2,000 |
| Stage 2 | stage1 shortlist | `submission.csv` | 2,000 → top-100 |

## Setup

```bash
pip install -r requirements.txt
```

**Python 3.10+ required.** CPU-only — no GPU needed.

## Step 1 — Pre-computation (run once, results are cached)

```bash
# Extract canonical texts and numeric features from all 100k candidates
python preprocess.py --candidates Main/candidates.jsonl --out artifacts/

# Encode with sentence-transformers (all-MiniLM-L6-v2) — ~15-20 min on CPU
python embeddings.py --texts artifacts/texts.txt --jd jd_content.txt --out artifacts/
```

These steps are **idempotent** — if `artifacts/` already contains the outputs, they are skipped.

## Step 2 — Stage 1 prefilter (100k → 2,000)

```bash
python stage1.py \
  --candidates Main/candidates.jsonl \
  --jd jd_content.txt \
  --rules Main/rules.yaml \
  --artifacts artifacts/ \
  --out artifacts/stage1_out.csv
```

## Step 3 — Final ranking (2,000 → top-100) — **must complete in < 5 min**

```bash
python rank_final.py \
  --stage1 artifacts/stage1_out.csv \
  --candidates Main/candidates.jsonl \
  --artifacts artifacts/ \
  --jd jd_content.txt \
  --rules Main/rules.yaml \
  --out submission.csv
```

## One-command ranking (after pre-computation)

```bash
python stage1.py --candidates Main/candidates.jsonl --jd jd_content.txt --rules Main/rules.yaml --artifacts artifacts/ --out artifacts/stage1_out.csv && python rank_final.py --stage1 artifacts/stage1_out.csv --candidates Main/candidates.jsonl --artifacts artifacts/ --jd jd_content.txt --rules Main/rules.yaml --out submission.csv
```

## Validate submission

```bash
python validate_submission.py --submission submission.csv --candidates Main/candidates.jsonl
```

## Architecture

### Scoring signals

**Stage 1** (100k → 2,000) — broad prefilter:

| Signal | Weight | Method |
|---|---|---|
| Semantic similarity | 0.72 | `all-MiniLM-L6-v2` + FAISS HNSW |
| BM25 | 0.10 | Okapi BM25 over canonical texts |
| Availability | 0.08 | open_to_work, response_rate, recency |
| Location fit | 0.04 | India city + willing_to_relocate |
| Seniority fit | 0.03 | proximity to 7yr target |
| Production fit | 0.03 | deployment keywords in career descriptions |
| Rule adjustments | ±∞ | hard filters + soft penalties/boosts |

**Stage 2** (2,000 → 100) — precision rerank:

| Signal | Weight |
|---|---|
| Semantic similarity | 0.32 |
| JD keyword match | 0.18 |
| Availability composite | 0.10 |
| Production fit | 0.10 |
| BM25 | 0.08 |
| Seniority fit | 0.07 |
| Engagement score | 0.06 |
| Location fit | 0.05 |
| GitHub activity | 0.04 |

### Hard filters

- `years_of_experience < 4` → rejected
- Honeypot profiles → rejected (impossible experience, expert skills with 0 duration)
- Outside India AND not willing to relocate → rejected
- Not open to work AND inactive > 365 days → rejected
- Disqualifying titles (marketing manager, HR generalist, recruiter, sales) → penalized

### Honeypot detection

Three patterns detected:
1. **Zero-duration experts**: ≥8 "expert" skills with 0 months duration
2. **Impossible career gaps**: 8+ years experience with only 1 short job
3. **Duration overflow**: sum of skill durations > 4× total career length

### Canonical text

Each candidate text includes:
- Profile headline, summary, current title/industry/location
- Last 5 career history entries (title + description)
- Top 25 skills (name + proficiency + duration + endorsements)
- Education fields
- Redrob signal tokens (open_to_work, verified, responsive, active, GitHub)

### Rules

Rules are loaded from `Main/rules.yaml` (or `rules/rules.yaml`). The YAML controls all weights, thresholds, and keyword lists — nothing is hardcoded.

## File structure

```
IndiaRuns/
├── Main/
│   ├── candidates.jsonl       100k candidate profiles
│   └── rules.yaml             scoring rules
├── preprocess.py              text + feature extraction
├── embeddings.py              sentence-transformer encoding
├── stage1.py                  stage 1 prefilter
├── rank_final.py              stage 2 final ranker + reasoning
├── rules.py                   rule engine
├── utils.py                   shared utilities
├── requirements.txt
├── rules/
│   └── rules.yaml             copy of rules for the rules/ folder
├── jd_content.txt             job description
└── artifacts/                 (auto-created, gitignored)
    ├── candidate_ids.json
    ├── texts.txt
    ├── features.pkl
    ├── embeddings.npy         ~150 MB
    ├── jd_vec.npy
    └── stage1_out.csv
```

## Compute environment

- Python 3.11, CPU-only (16 GB RAM)
- Pre-computation (embeddings): ~15-20 min
- Ranking step (stage1 + stage2): < 5 min (after embeddings cached)
