# Redrob AI — Candidate Ranking Solution Report

---

## Slide 1 — Solution Overview

### What is the proposed solution?

A **fully deterministic, rules-based + semantic ranking engine** for the Senior AI Engineer role at Redrob AI. The system processes the full 100k-candidate pool through a two-stage pipeline:

1. **Stage 1 — Hard Filtering (precompute.py):** Eliminates structurally ineligible candidates using five binary gates (honeypot detection, location, inactivity, title relevance, and must-have keyword presence) applied in sequence over the raw JSONL stream — no ML model is invoked here.
2. **Stage 2 — Semantic Scoring + Multi-signal Ranking (rank.py):** The shortlisted candidates are embedded via all-MiniLM-L6-v2, scored against a JD embedding, then enriched with ten additional deterministic penalty/bonus components before a final sort. The top 100 are written to submission.csv with per-candidate natural-language reasoning.

### What differentiates this from traditional candidate matching systems?

| Traditional ATS / Keyword System | This Solution |
|---|---|
| Boolean keyword matching only | Semantic cosine similarity + exact keyword density combined |
| Single signal (skills or YoE) | 10+ independent scoring dimensions merged via additive formula |
| Opaque ranking, no explanations | Per-candidate reasoning derived from actual top-contributing components |
| No fraud/honeypot detection | Explicit honeypot detector for three distinct fabrication patterns |
| No career quality signals | Job-hopping penalty, avg tenure weighting, consulting-background penalty |
| Treats all companies equally | FAANG/top-tier company prestige bonus |
| Static, identical output format | Seeded-random sentence variation ensures every reasoning string is unique |

The core philosophy: **no LLM generates the reasoning or ranking** — every score component is a deterministic function of the candidate's raw profile fields and Redrob signals, making the system fully auditable and hallucination-free.

---

## Slide 2 — JD Understanding and Candidate Evaluation

### Key requirements extracted from the JD

**Hard requirements (enforced as binary filters):**
- Candidate must be based in India or willing to relocate
- Must have at least one NLP/IR/ML keyword anywhere in profile (embedding, retrieval, ranking, BERT, transformers, FAISS, etc.)
- Must have some JD-adjacent role in current or past titles (ML Engineer, AI Engineer, Search Engineer, NLP Engineer, Data Scientist, etc.)
- Must not be completely inactive (>365 days) without an open-to-work flag

**Target experience band:** 5-9 years (ideal: 7). Hard floor at 4 years.

**Must-have skills (from JD):**
- Embedding and dense retrieval: FAISS, Pinecone, Qdrant, Milvus, Weaviate, OpenSearch, Elasticsearch
- Semantic search, vector search, information retrieval
- Sentence-Transformers, BERT, Transformers
- Reranking, BM25, hybrid search, RAG (retrieval-augmented generation)
- Learning-to-rank, NDCG, MRR evaluation
- NLP pipelines, natural language processing

**Preferred skills (scoring bonuses):**
- LoRA, QLoRA, PEFT, fine-tuning
- XGBoost, LightGBM for ranking
- MLOps, model serving, inference optimization
- A/B testing, distributed systems

**Soft disqualifiers (penalized, not hard knock-outs):**
- Pure consulting/services background (TCS, Infosys, Wipro, Accenture, etc.)
- Entire career at LangChain/OpenAI-API tutorial level
- Computer vision or robotics only, with no NLP/IR signals

### Candidate signals used for relevance beyond keyword matching

The system evaluates 10 independent dimensions:

1. **Semantic cosine similarity** - Profile full-text vs. JD embedding (dampened to 50% weight)
2. **Exact keyword hit count** - Word-boundary regex over MUST_HAVE + PREFERRED skill keyword lists
3. **Core JD skill depth** - How many of 47 JD-core skills appear in the candidate skill list
4. **Experience band proximity** - Penalty for being outside the 5-9 YoE target band
5. **Career gap integrity** - Penalty when claimed YoE is not backed by recorded job history
6. **Job-hopping pattern** - Avg tenure across valid stints, weighted toward recent roles
7. **Notice period** - Soft penalty above 30 days, hard penalty above 90 days
8. **GitHub activity score** - Penalty for missing/low scores, bonus for high activity
9. **Recruiter response rate** - Penalty for non-responsive candidates, bonus for responsive
10. **Top-tier company prestige** - Bonus for FAANG/DeepMind/Anthropic/Cohere/Nvidia pedigree

---

## Slide 3 — Ranking Methodology

### How candidates are retrieved, scored, and ranked

**Step 1 — Stream and Filter (precompute.py)**
All ~100k candidates streamed from Main/candidates.jsonl. Five hard filters applied sequentially:
1. Honeypot detection - flags impossible/contradictory profiles (3 patterns)
2. Hard title check - rejects completely wrong domains (HR, Sales, Marketing, etc.)
3. Location gate - India or willing-to-relocate
4. Inactivity gate - not open-to-work AND inactive >365 days
5. Keyword gate - zero NLP/IR keywords anywhere in the profile

**Step 2 — Embed (precompute.py)**
Shortlisted candidates encoded with all-MiniLM-L6-v2 (384-dim). Saved to artifacts/embeddings.npy with ID map at artifacts/id_index.json.

**Step 3 — Score (rank.py)**
Composite score built additively:

  score = semantic_similarity x 0.5
        + keyword_density_bonus         (log-scaled, max 0.15)
        - semantic_illusion_penalty     (-0.05 if high semantic but <18 exact hits)
        + skill_depth_bonus             (graduated, max 0.08 for 10+ matched skills)
        - skill_depth_penalty           (up to -0.20 for zero JD skill matches)
        - yoe_penalty                   (0.08/yr below band, 0.015/yr above band)
        - yoe_gap_penalty               (-0.20 if <90% of claimed YoE is recorded)
        - job_hop_penalty               (-0.15 if avg tenure <18mo)
        - notice_penalty                (soft + hard tiers at 30/90 days)
        + github_bonus / - github_penalty
        + recruiter_bonus / - recruiter_penalty
        + top_tier_bonus                (max 0.03 for 2+ FAANG roles)
        - junior_mismatch_penalty       (0.04-0.10 for "Junior" title with 5+ YoE)
        - consulting_penalty            (0.03-0.06 for majority consulting career)
        - quality_penalty               (up to 0.05 for unendorsed expert skills)

**Step 4 — Sort and Top-100 Selection**
Sorted descending by score; ties broken by candidate_id (deterministic). Top 100 written to submission.csv.

### Models, algorithms, and heuristics used

| Component | Approach |
|---|---|
| Text embedding | sentence-transformers/all-MiniLM-L6-v2 (384-dim dense vectors) |
| Similarity | Cosine similarity (L2-normalised dot product) |
| Keyword scoring | Word-boundary regex over compiled pattern |
| Skill matching | Sorted-by-length keyword list to prevent partial matches |
| Job-hop detection | Weighted recent-tenure formula (recent roles x2 weight) |
| Honeypot detection | Three independent pattern-based rule checks |
| Top-tier prestige | Fuzzy substring match against elite company name list |

### How multiple signals combine into a final rank

Signals are weighted by domain expertise, not machine-learned weights. The semantic base contributes ~50% of possible score; the remaining ~50% comes from explicit profile quality signals. This split ensures:
- A candidate with perfect semantic alignment but zero verified skills does not beat one with strong exact-match skills and solid career quality
- The semantic_illusion_penalty guards against high-cosine-similarity profiles that lack actual keyword depth (generic AI buzzword profiles)

---

## Slide 4 — Explainability and Data Validation

### How ranking decisions are explained

Every candidate in the top 100 receives a natural-language reasoning string, generated entirely from their own verified profile data:

1. **Compute all score components first** - reasoning is generated *after* the final rank is known
2. **Identify top positive contributors** - sorts all bonus components by their score contribution, picks the top 2
3. **Identify top penalties** - sorts all penalty components similarly, picks the top 2
4. **Apply rank-aware tone** - Ranks 1-20 get positively-framed template with concerns footnoted; Ranks 21-100 get concerns-forward template that still acknowledges genuine strengths
5. **Seeded randomness for variation** - random.Random(candidate_id) seeds phrase selection, ensuring every reasoning string differs structurally while remaining deterministic

Every fact cited is extracted directly from the candidate's profile:
- Exact title from profile.current_title
- Exact YoE from profile.years_of_experience
- Most recent company from career_history[0].company
- Exact GitHub score from redrob_signals.github_activity_score
- Exact recruiter response rate from redrob_signals.recruiter_response_rate
- Named JD skills from skills[].name matched against the JD keyword list

### How hallucinations and unsupported justifications are prevented

- **Zero LLM in the loop** - no generative model produces any claim; every sentence is a template with slots filled from verified profile fields
- **Skill grounding** - matched skills derived by regex over the candidate's actual skills[] list; skills not in the profile cannot appear in reasoning
- **Company name extraction** - top-tier company names come directly from career_history[].company strings, never inferred
- **Numerical facts** - all numbers (YoE, GitHub score, notice period, recruiter rate, tenure) read directly from the data
- **Rank-tone consistency** - the reasoning template class (positive vs. critical) is determined by rank, so a rank-95 candidate physically cannot receive a glowing reasoning string

### How suspicious, low-quality, and inconsistent profiles are handled

**Three-tier defence:**

| Tier | Mechanism | Action |
|---|---|---|
| Hard eliminate | Honeypot patterns 1-3 | Removed in Stage 1; never reaches scoring |
| Score penalty | Career gap, job-hopping, consulting, low GitHub, poor recruiter rate | Score reduced; reflected in reasoning |
| Reasoning flag | Unverified expert skills, semantic illusion, title/YoE mismatch | Named explicitly in reasoning |

**Specific handling:**
- **Fabricated profiles (honeypot):** 5+ expert skills with 0 duration + no endorsements + no career mention = hard removed. 10%+ unrecorded YoE gap = hard removed at Stage 1; remaining gap triggers -0.20 penalty at Stage 2
- **Stale or misleading titles:** "Junior" title with 5+ YoE triggers 0.04-0.10 penalty (higher if career history doesn't corroborate YoE claim)
- **Consulting-only backgrounds:** Fraction of career months at known services firms computed; 50-80% = -0.03, >80% = -0.06
- **Unverified expertise:** Expert/advanced skills with 0 endorsements counted; penalty capped at 0.05
- **Semantic illusion:** Cosine similarity >0.35 but <18 exact keyword hits triggers -0.05 penalty; catches AI-buzzword-heavy summaries with no real technical depth

---

*Pipeline: precompute.py (filtering + embedding) -> rank.py (scoring + reasoning) -> submission.csv (output). Config lives in jd.py. No external APIs called during ranking.*
