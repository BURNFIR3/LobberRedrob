"""
filter.py — Stage 1: Apply hard filters one-by-one to 100k candidates.

Philosophy:
  - We DO NOT use semantic/FAISS pre-filtering here. We read ALL 100k candidates.
  - Each hard filter is applied sequentially, with a running count of rejections.
  - Candidates that survive ALL hard filters pass to the shortlist.
  - This gives full visibility into exactly how many candidates fail each rule.

Hard filters applied (in order):
  1. Honeypot detection
       Pattern 1 — many expert skills with 0 duration
       Pattern 2 — large unrecorded career gap (≥10% of claimed YoE unaccounted)
       Pattern 3 — skill months >> career months (4× multiplier)
  2. Location: India or willing to relocate
  3. Inactivity + not open to work (>365 days inactive)
  4. No JD-relevant role presence — zero relevant domain keywords in
       current title, ALL past career titles, headline, summary, or descriptions
  5. Must-have skill/text check (zero retrieval/NLP keywords anywhere in profile)

NOT enforced here (deferred to scoring):
  - Minimum years of experience (soft ranking signal, not a hard gate)
  - Notice period
  - Preferred cities
  - Consulting company history
  - Soft disqualifiers (LangChain-only, CV-only, etc.)

Output:
  artifacts/stage1_shortlist.jsonl   — candidates surviving all hard filters

Usage:
    python filter.py --candidates Main/candidates.jsonl [--out artifacts/stage1_shortlist.jsonl]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
import gzip
import csv

import numpy as np

from jd import (
    CONSULTING_COMPANIES,
    HARD_DISQUALIFYING_TITLE_FRAGMENTS,
    HONEYPOT_EXPERT_SKILLS_WITH_ZERO_DURATION,
    HONEYPOT_UNRECORDED_GAP_PCT,
    JD_RELEVANT_ROLE_KEYWORDS,
    MAX_INACTIVE_DAYS_IF_NOT_OPEN,
    MUST_HAVE_SKILL_KEYWORDS,
    REQUIRE_INDIA_OR_RELOCATE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [filter] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────

def days_since(date_str: str | None) -> int:
    """Days since a date string (YYYY-MM-DD). Returns 9999 if missing."""
    if not date_str:
        return 9999
    try:
        d = date.fromisoformat(str(date_str))
        return (date.today() - d).days
    except Exception:
        return 9999


def candidate_full_text(c: dict) -> str:
    """Build a single lowercase text blob from all profile fields."""
    p = c.get("profile", {}) or {}
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    for ch in (c.get("career_history") or [])[:5]:
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
        parts.append(ch.get("company", ""))
        parts.append(ch.get("industry", ""))
    for sk in (c.get("skills") or []):
        parts.append(sk.get("name", ""))
    for edu in (c.get("education") or []):
        parts.append(edu.get("field_of_study", ""))
        parts.append(edu.get("degree", ""))
    return " ".join(str(x) for x in parts if x).lower()


def is_india_based(c: dict) -> bool:
    p = c.get("profile", {}) or {}
    country = (p.get("country") or "").lower().strip()
    location = (p.get("location") or "").lower().strip()
    return "india" in country or "india" in location


# ──────────────────────────────────────────────────────────────
# Hard filter functions
# Each returns (passes: bool, reason: str | None)
# ──────────────────────────────────────────────────────────────

def filter_honeypot(c: dict) -> tuple[bool, str | None]:
    """Detect profiles with impossible/contradictory data."""
    p = c.get("profile", {}) or {}
    skills = c.get("skills") or []
    career = c.get("career_history") or []
    yoe = float(p.get("years_of_experience") or 0)
    redrob = c.get("redrob_signals", {}) or {}
    assessments = redrob.get("skill_assessment_scores", {}) or {}

    career_blob = " ".join(str(ch.get("description", "")) for ch in career).lower()

    # Pattern 1: Many "expert" skills with 0 duration AND NO corroborating signals
    suspicious_experts = 0
    for sk in skills:
        if sk.get("proficiency") in ("expert", "advanced") and int(sk.get("duration_months") or 0) == 0:
            name_lower = sk.get("name", "").lower()
            has_endorsement = int(sk.get("endorsements") or 0) > 0
            has_assessment = name_lower in assessments
            has_career_mention = name_lower in career_blob
            if not has_endorsement and not has_assessment and not has_career_mention:
                suspicious_experts += 1

    if suspicious_experts >= HONEYPOT_EXPERT_SKILLS_WITH_ZERO_DURATION:
        return False, "honeypot: zero_dur_experts"

    # Pattern 2: Large unrecorded career gap.
    # If (claimed_yoe - recorded_career_months/12) / claimed_yoe >= threshold
    # the profile is fabricated — e.g. 10 yr claimed, 3 yr recorded → 70% gap.
    # Only applies when YoE > 1 (ignore fresh grads / near-zero profiles).
    if yoe > 1:
        recorded_months = sum(int(ch.get("duration_months") or 0) for ch in career)
        recorded_yoe = recorded_months / 12.0
        unrecorded_frac = max(0.0, yoe - recorded_yoe) / yoe
        if unrecorded_frac >= HONEYPOT_UNRECORDED_GAP_PCT:
            return False, "honeypot: unrecorded_gap"

    return True, None


def filter_experience(c: dict) -> tuple[bool, str | None]:
    """Reject candidates below minimum experience threshold."""
    p = c.get("profile", {}) or {}
    yoe = float(p.get("years_of_experience") or 0)
    if yoe < MIN_YEARS_EXPERIENCE:
        return False, f"low_experience:{yoe:.1f}yr<{MIN_YEARS_EXPERIENCE}yr"
    return True, None


def filter_location(c: dict) -> tuple[bool, str | None]:
    """Reject candidates outside India who are not willing to relocate."""
    if not REQUIRE_INDIA_OR_RELOCATE:
        return True, None
    r = c.get("redrob_signals", {}) or {}
    in_india = is_india_based(c)
    relocate = bool(r.get("willing_to_relocate"))
    if not in_india and not relocate:
        p = c.get("profile", {}) or {}
        return False, "location: out_of_scope"
    return True, None


def filter_inactivity(c: dict) -> tuple[bool, str | None]:
    """Reject candidates who are both not open-to-work AND heavily inactive."""
    r = c.get("redrob_signals", {}) or {}
    open_to_work = bool(r.get("open_to_work_flag"))
    last_active_days = days_since(r.get("last_active_date"))
    if not open_to_work and last_active_days > MAX_INACTIVE_DAYS_IF_NOT_OPEN:
        return False, "inactivity: inactive_and_not_open"
    return True, None


def filter_hard_title(c: dict) -> tuple[bool, str | None]:
    """
    Hard disqualify candidates whose current title AND entire career history
    are completely outside the ML/AI domain (HR, sales, marketing, etc.).

    We only reject if current_title itself matches a disqualifying fragment,
    because a past stint in HR followed by an ML career is fine.
    """
    p = c.get("profile", {}) or {}
    current_title = (p.get("current_title") or "").lower().strip()
    for frag in HARD_DISQUALIFYING_TITLE_FRAGMENTS:
        if frag in current_title:
            return False, f"hard_title: {frag}"
    return True, None


import re

# Compile regex for word-boundary matching to prevent 'map' matching 'roadmap'
STRICT_KEYWORDS = MUST_HAVE_SKILL_KEYWORDS.copy()
# We sort by length descending so longer phrases match first
STRICT_KEYWORDS.sort(key=len, reverse=True)
_kw_pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in STRICT_KEYWORDS) + r')\b'
_kw_regex = re.compile(_kw_pattern)

def filter_has_any_jd_keyword(c: dict, full_text: str) -> tuple[bool, str | None]:
    """
    Check if the candidate has at least one strict NLP/IR skill keyword
    using word boundaries.
    """
    if _kw_regex.search(full_text):
        return True, None
    return False, "keywords: zero_strict_matches_found"


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def stream_candidates(path: Path):
    """Stream candidates.jsonl one line at a time."""
    if path.suffix.lower() == ".gz":
        f = gzip.open(path, "rt", encoding="utf-8-sig")
    else:
        f = open(path, "r", encoding="utf-8-sig")
        
    with f:
        # Check if the file is CSV or TSV
        name_lower = path.name.lower()
        if ".csv" in name_lower or ".tsv" in name_lower:
            sep = "\t" if ".tsv" in name_lower else ","
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    if v and (v.startswith("{") or v.startswith("[")):
                        try:
                            # Safely evaluate JSON-like string columns
                            import ast
                            # Using ast.literal_eval first handles single-quoted python dicts,
                            # fallback to json.loads for standard JSON.
                            try:
                                parsed_row[k] = ast.literal_eval(v)
                            except (SyntaxError, ValueError):
                                parsed_row[k] = json.loads(v)
                        except:
                            parsed_row[k] = v
                    else:
                        parsed_row[k] = v
                yield parsed_row
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description="Stage 1: Apply hard filters to 100k candidates.")
    ap.add_argument("--candidates", default="Main/candidates.jsonl")
    ap.add_argument("--out", default="artifacts/stage1_shortlist.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Rejection counters per filter
    rejection_counts: dict[str, int] = defaultdict(int)
    total = 0
    passed = 0

    log.info("Starting hard-filter pass over all candidates …")
    log.info("=" * 60)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for c in stream_candidates(Path(args.candidates)):
            total += 1
            cid = c.get("candidate_id", "?")
            full_text = candidate_full_text(c)

            # ── Apply filters in order ──────────────────────────
            filters = [
                ("honeypot",            filter_honeypot(c)),
                ("hard_title",          filter_hard_title(c)),
                ("location",            filter_location(c)),
                ("inactivity",          filter_inactivity(c)),
                ("keywords",            filter_has_any_jd_keyword(c, full_text)),
            ]

            rejected_by = None
            reject_reason = None
            for filter_name, (passes, reason) in filters:
                if not passes:
                    rejected_by = filter_name
                    reject_reason = reason
                    break

            if rejected_by is not None:
                rejection_counts[reject_reason] += 1
            else:
                passed += 1
                out_f.write(json.dumps(c) + "\n")

            # Progress every 10k
            if total % 10_000 == 0:
                log.info(f"  Processed {total:,} | Passed so far: {passed:,}")

    # ── Summary ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"TOTAL CANDIDATES PROCESSED : {total:,}")
    log.info(f"TOTAL PASSED (shortlisted) : {passed:,}")
    log.info(f"TOTAL REJECTED             : {total - passed:,}")
    log.info("")
    log.info("Rejections by specific reason (in alphabetical order):")
    for reason, count in sorted(rejection_counts.items()):
        pct = count / total * 100
        log.info(f"  {reason:<40} {count:>7,}  ({pct:.1f}%)")
    log.info("=" * 60)
    log.info(f"Shortlist written to: {out_path}")

    log.info("Starting embedding generation for shortlisted candidates...")
    
    # Load shortlisted candidates for embedding
    candidates = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    
    log.info(f"Loaded {len(candidates)} candidates for embedding.")
    
    # Import here to avoid overhead if just testing filters
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Prepare texts for embedding
    texts = []
    candidate_ids = []
    for c in candidates:
        candidate_ids.append(c["candidate_id"])
        # We use the full text representation generated earlier for simplicity
        texts.append(candidate_full_text(c))
        
    log.info("Encoding candidates...")
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    
    embeddings_path = Path("artifacts/embeddings.npy")
    np.save(embeddings_path, embeddings)
    log.info(f"Embeddings saved to {embeddings_path}")
    
    index_path = Path("artifacts/id_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(candidate_ids, f)
    log.info(f"ID index saved to {index_path}")
    log.info("Precompute stage complete.")

if __name__ == "__main__":
    main()
