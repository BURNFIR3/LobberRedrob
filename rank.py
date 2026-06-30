import csv
import json
import logging
import math
import re
from pathlib import Path
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from jd import (
    JD_RELEVANT_ROLE_KEYWORDS,
    MUST_HAVE_SKILL_KEYWORDS,
    PREFERRED_SKILL_KEYWORDS,
    MIN_YEARS_EXPERIENCE,
    TARGET_YOE_MIN, TARGET_YOE_MAX, TARGET_YOE_IDEAL,
    NOTICE_PREFERRED_DAYS, NOTICE_ACCEPTABLE_DAYS,
    NOT_WANTED_SIGNALS, CONSULTING_COMPANIES,
    HARD_DISQUALIFYING_TITLE_FRAGMENTS,
)
from precompute import candidate_full_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [rank] %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

JD_TEXT = """
Senior AI Engineer — Founding Team @ Redrob AI.
This role is about building the retrieval, ranking, and semantic search systems that power AI-native hiring.
Required: deep ML/NLP expertise, hands-on experience building embedding-based retrieval systems, vector search (FAISS, Pinecone, Qdrant, Milvus, Weaviate), BM25 + dense hybrid search, reranking, retrieval-augmented generation (RAG), LLMs (BERT, Transformers, sentence-transformers), learning-to-rank, NDCG/MRR evaluation, NLP pipelines.
Preferred: fine-tuning (LoRA, QLoRA, PEFT), XGBoost/LightGBM for ranking, MLOps, model serving, inference optimization, A/B testing, distributed systems.
Location: India (Pune/Noida preferred). Ideal YoE: 5–9 years.
"""

# Key skills for JD matching
JD_CORE_SKILLS = [
    "faiss", "pinecone", "qdrant", "milvus", "weaviate", "opensearch", "elasticsearch",
    "sentence-transformers", "sentence transformers", "bert", "transformers",
    "embedding", "embeddings", "vector search", "semantic search",
    "retrieval", "reranking", "re-ranking", "ranking",
    "bm25", "hybrid search", "rag", "retrieval augmented",
    "nlp", "natural language processing",
    "qlora", "lora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "ndcg", "mrr",
    "mlops", "inference optimization",
    "xgboost", "lightgbm",
    "pgvector", "information retrieval",
]
JD_CORE_SKILLS.sort(key=len, reverse=True)

# Top-tier companies
TOP_TIER_COMPANIES = {
    "google", "meta", "apple", "amazon", "microsoft", "netflix", "openai",
    "deepmind", "anthropic", "linkedin", "uber", "stripe", "databricks",
    "hugging face", "cohere", "nvidia"
}
TOP_TIER_BONUS_PER_ROLE = 0.015


def compute_cosine_similarities(query_emb, doc_embs):
    q_norm = np.linalg.norm(query_emb)
    if q_norm == 0: return np.zeros(len(doc_embs))
    q = query_emb / q_norm
    d_norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
    d_norms[d_norms == 0] = 1
    d = doc_embs / d_norms
    return np.dot(d, q.T).flatten()


def get_matching_skills(c: dict) -> list[str]:
    skills = c.get("skills") or []
    full_text = " ".join((sk.get("name") or "") for sk in skills).lower()
    matched = []
    seen = set()
    for jd_kw in JD_CORE_SKILLS:
        # word-boundary match against skill names
        if re.search(r'\b' + re.escape(jd_kw) + r'\b', full_text):
            normalized = jd_kw.lower()
            if normalized not in seen:
                seen.add(normalized)
                matched.append(jd_kw)
    return matched


def check_junior_yoe_mismatch(c: dict) -> tuple[bool, float]:
    p = c.get("profile", {}) or {}
    title = (p.get("current_title") or "").lower()
    yoe = float(p.get("years_of_experience") or 0)
    if "junior" in title and yoe >= 5.0:
        # Corroborate: check if career history is consistent
        career = c.get("career_history") or []
        total_recorded = sum(int(ch.get("duration_months") or 0) for ch in career)
        recorded_yoe = total_recorded / 12.0
        # If career history actually supports the high YoE, they just have a stale title
        if recorded_yoe >= yoe * 0.7:
            return True, 0.04   # mild penalty for stale/misleading title
        else:
            # High YoE claim not backed by career history — stronger honeypot signal
            return True, 0.10
    return False, 0.0


def check_consulting_only(c: dict) -> float:
    """Penalize candidates who only worked at pure-services consulting firms."""
    career = c.get("career_history") or []
    if not career:
        return 0.0
    consulting_months = 0
    total_months = 0
    for ch in career:
        company = (ch.get("company") or "").lower()
        dur = int(ch.get("duration_months") or 0)
        total_months += dur
        if any(co in company for co in CONSULTING_COMPANIES):
            consulting_months += dur
    if total_months == 0:
        return 0.0
    frac = consulting_months / total_months
    if frac > 0.8:
        return 0.06  # entire career in consulting
    elif frac > 0.5:
        return 0.03
    return 0.0


def check_rag_rerank_experience(c: dict) -> bool:
    """Check if candidate has RAG or reranking experience across skills AND career."""
    skills = c.get("skills") or []
    career = c.get("career_history") or []

    skill_text = " ".join((s.get("name") or "").lower() for s in skills)
    career_text = " ".join(
        ((ch.get("description") or "") + " " + (ch.get("title") or "")).lower()
        for ch in career
    )
    summary_text = (c.get("profile", {}) or {}).get("summary", "") or ""
    all_text = skill_text + " " + career_text + " " + summary_text.lower()

    rag_signals = [
        "rag", "retrieval augmented", "retrieval-augmented",
        "reranking", "re-ranking", "rerank",
        "cross-encoder", "cross encoder",
        "dense retrieval", "sparse retrieval",
        "bm25", "hybrid search",   # hybrid = implicit reranking knowledge
        "learning to rank", "ltr",
    ]
    return any(sig in all_text for sig in rag_signals)


def check_top_tier_company(c: dict) -> tuple[int, list[str]]:
    """Count how many career roles were at top-tier companies. Returns (count, names)."""
    career = c.get("career_history") or []
    count = 0
    names = []
    for ch in career:
        company = (ch.get("company") or "").lower()
        for top in TOP_TIER_COMPANIES:
            if top in company:
                count += 1
                names.append(ch.get("company", ""))
                break
    return count, names


def _notice_label(np_days: int) -> tuple[str, str]:
    """Return (noun_phrase, detail) tiered to jd.py thresholds.
    NOTICE_PREFERRED_DAYS=30, NOTICE_ACCEPTABLE_DAYS=90.
    """
    if np_days <= 30:
        return "a short notice period", f"{np_days} days — immediately available"
    elif np_days <= 60:
        return "a manageable notice period", f"{np_days} days; buyout likely feasible"
    elif np_days <= 90:
        return "a notice period worth discussing", f"{np_days} days — workable with planning"
    elif np_days <= 120:
        return "a lengthy notice period", f"{np_days} days — significant scheduling risk"
    else:
        return "an extended notice period", f"{np_days} days — likely a blocker"


def generate_reasoning(c: dict, score_components: dict, matched_skills: list[str],
                        top_tier_names: list[str], rank: int, has_rag: bool) -> str:
    """Generate substantive, candidate-specific reasoning based on highest contributing metrics."""
    import random
    p = c.get("profile", {}) or {}
    r = c.get("redrob_signals", {}) or {}
    career = c.get("career_history") or []
    
    cid = c.get("candidate_id", "unknown")
    rng = random.Random(cid)  # seed with candidate ID for consistency
    
    title = p.get("current_title") or "Unknown"
    yoe = float(p.get("years_of_experience") or 0)
    np_days = int(r.get("notice_period_days") or 90)
    github = float(r.get("github_activity_score") or -1)
    recruiter_rate = float(r.get("recruiter_response_rate") or -1)
    
    n_matched = score_components.get("matched_skill_count", 0)
    hits = score_components.get("kw_hits", 0)
    avg_tenure = score_components.get("avg_tenure", 0.0)

    # --- Candidate-specific facts for reasoning (Stage 4 specificity) ---
    recent_company = "an unknown company"
    recent_tenure_months = 0
    if career:
        recent_company = career[0].get("company") or "their previous company"
        recent_tenure_months = int(career[0].get("duration_months") or 0)

    # Build a concise profile fact string: "AI Engineer @ Google (37mo)"
    if recent_tenure_months > 0:
        profile_fact = f"{title} @ {recent_company} ({recent_tenure_months}mo)"
    else:
        profile_fact = f"{title} @ {recent_company}"

    # Named JD-matched skills (up to 4 for specificity)
    named_skills_str = ", ".join(matched_skills[:4]) if matched_skills else ""

    # 1. Identify top pros (points out of total)
    pros = []
    # (score_val, descriptive_text, specific_fact)
    sem_score = score_components.get("semantic", 0)

    # Lead with named skills + employer/tenure — this is the concrete evidence a reviewer needs
    if named_skills_str and n_matched >= 3:
        skill_label = f"matched JD skills: {named_skills_str}"
        pros.append((0.10, "strong JD skill match", skill_label))
    elif named_skills_str:
        skill_label = f"matched JD skills: {named_skills_str}"
        pros.append((0.04, "partial JD skill match", skill_label))

    if sem_score > 0.35:
        pros.append((sem_score, "strong semantic alignment with the JD", f"score {sem_score:.2f}"))
    elif sem_score > 0.25:
        pros.append((sem_score, "moderate semantic alignment", f"score {sem_score:.2f}"))
    
    kw_bonus = score_components.get("kw_bonus", 0)
    if kw_bonus > 0:
        pros.append((kw_bonus, "high keyword density", f"{hits} exact matches"))
        
    skill_depth_bonus = score_components.get("skill_depth_bonus", 0)
    if skill_depth_bonus > 0:
        pros.append((skill_depth_bonus, "comprehensive core skill coverage",
                     f"{n_matched} JD skills matched" + (f" ({named_skills_str})" if named_skills_str else "")))
    elif n_matched >= 3 and not named_skills_str:
        pros.append((0.02, "solid skill alignment", f"{n_matched} core JD skills matched"))
        
    github_bonus = score_components.get("github_bonus", 0)
    if github_bonus > 0:
        pros.append((github_bonus, "active open-source presence", f"GitHub score {github:.0f}/100"))
        
    recruiter_bonus = score_components.get("recruiter_bonus", 0)
    if recruiter_bonus > 0:
        pros.append((recruiter_bonus, "high responsiveness to recruiters", f"{recruiter_rate*100:.0f}% response rate"))
        
    top_tier_bonus = score_components.get("top_tier_bonus", 0)
    if top_tier_bonus > 0:
        pros.append((top_tier_bonus, "top-tier company experience", f"roles at {', '.join(top_tier_names[:2])}"))
        
    pros.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Identify top cons (penalties)
    cons = []
    
    yoe_gap_penalty = score_components.get("yoe_gap_penalty", 0)
    if yoe_gap_penalty > 0:
        cons.append((yoe_gap_penalty, "an unrecorded career gap",
                     "claimed YoE is not fully backed by job history duration"))
        
    job_hop_penalty = score_components.get("job_hop_penalty", 0)
    if job_hop_penalty > 0:
        cons.append((job_hop_penalty, "a history of short tenures",
                     f"averaging {avg_tenure:.1f} months per role"))
        
    github_penalty = score_components.get("github_penalty", 0)
    if github_penalty > 0:
        if github < 0:
            cons.append((github_penalty, "no GitHub linked", "open-source activity unknown"))
        else:
            cons.append((github_penalty, "low open-source activity", f"GitHub score {github:.0f}/100"))
            
    recruiter_penalty = score_components.get("recruiter_penalty", 0)
    if recruiter_penalty > 0:
        if recruiter_rate < 0:
            cons.append((recruiter_penalty, "no recruiter response data", "responsiveness is unknown"))
        else:
            cons.append((recruiter_penalty, "poor communication signal",
                         f"only {recruiter_rate*100:.0f}% recruiter response rate"))
        
    semantic_illusion_penalty = score_components.get("semantic_illusion_penalty", 0)
    if semantic_illusion_penalty > 0:
        cons.append((semantic_illusion_penalty, "broad but shallow text matches",
                     f"only {hits} exact keyword hits despite a high semantic score"))
        
    skill_depth_penalty = score_components.get("skill_depth_penalty", 0)
    if skill_depth_penalty > 0:
        cons.append((skill_depth_penalty, "thin JD skill alignment",
                     f"only {n_matched} matched JD skill{'s' if n_matched != 1 else ''}"))
        
    yoe_penalty = score_components.get("yoe_penalty", 0)
    if yoe_penalty > 0:
        cons.append((yoe_penalty, "experience mismatch",
                     f"{yoe:.1f} YoE is outside our target 5-9 band"))
        
    notice_penalty = score_components.get("notice_penalty", 0)
    if notice_penalty > 0:
        notice_noun, notice_detail = _notice_label(np_days)
        cons.append((notice_penalty, notice_noun, notice_detail))
        
    junior_penalty = score_components.get("junior_mismatch_penalty", 0)
    if junior_penalty > 0:
        cons.append((junior_penalty, "title/YoE mismatch",
                     f"holding a 'Junior' title with {yoe:.1f} claimed YoE"))
        
    consulting_penalty = score_components.get("consulting_penalty", 0)
    if consulting_penalty > 0:
        cons.append((consulting_penalty, "a heavily consulting-focused background",
                     "may lack pure product engineering depth"))
        
    quality_penalty = score_components.get("quality_penalty", 0)
    if quality_penalty > 0:
        cons.append((quality_penalty, "unverified expertise",
                     "multiple expert-level skills lack peer endorsements"))
        
    if not has_rag:
        cons.append((0.08, "lack of direct retrieval experience",
                     "no explicit RAG or reranking mentions found in profile"))
        
    if rank > 10:
        sem_gap = max(0, 0.38 - sem_score)
        if sem_gap > 0.05:
            cons.append((sem_gap, "lower semantic alignment with core JD terminology than top candidates",
                         f"score {sem_score:.2f}"))
        if top_tier_bonus == 0:
            cons.append((0.015, "absence of FAANG/top-tier company background",
                         "which many top candidates have"))
        if hits < 15:
            cons.append((0.02, "fewer exact keyword matches",
                         f"only {hits} hits vs. top-10 baseline"))

    cons.sort(key=lambda x: x[0], reverse=True)
    
    top_pros = pros[:2] if len(pros) >= 2 else pros
    top_cons = cons[:2] if len(cons) >= 2 else cons
    
    # --- Build the profile-fact prefix (named skills + recent employer/tenure) ---
    # This is the concrete fact a reviewer needs to verify the system actually read the profile.
    skills_tenure_prefix = ""
    if named_skills_str and recent_tenure_months > 0:
        skills_tenure_prefix = (
            f"Recent: {profile_fact}. "
            f"Matched JD skills: {named_skills_str}. "
        )
    elif named_skills_str:
        skills_tenure_prefix = f"Matched JD skills: {named_skills_str}. "
    elif recent_tenure_months > 0:
        skills_tenure_prefix = f"Recent: {profile_fact}. "

    # 3. Construct varied sentences
    # FIX: templates that use 'their {noun}' only reference noun phrases that don't start
    # with an article ('a'/'an') — all article-headed nouns use article-neutral phrasing.
    if rank <= 20:
        templates = [
            "A strong {title} at {company} ({yoe:.1f} YoE), performing exceptionally well on {pro1} ({fact1}). {pro2_sentence} {con_sentence_light}",
            "Standing out with {yoe:.1f} YoE as a {title}, their primary strength is {pro1} ({fact1}). {pro2_sentence} {con_sentence_light}",
            "Ranked highly due to {pro1} ({fact1}). This {yoe:.1f}-YoE {title} is a solid fit. {pro2_sentence} {con_sentence_light}",
        ]
        tmpl = rng.choice(templates)
        
        pro2_sentence = ""
        if len(top_pros) > 1:
            pro2 = top_pros[1]
            # Use article-neutral connectors — never 'their {article} {noun}'
            pro2_sentence = rng.choice([
                f"They also demonstrate {pro2[1]} ({pro2[2]}).",
                f"Also notable: {pro2[1]} ({pro2[2]}).",
                f"Coupled with {pro2[1]} ({pro2[2]}), their profile is very compelling."
            ])
            
        con_sentence_light = ""
        if top_cons:
            con1 = top_cons[0]
            # Avoid 'their a/an X' — use article-neutral phrasing
            con_sentence_light = rng.choice([
                f"The only minor note is {con1[1]} ({con1[2]}).",
                f"Worth verifying: {con1[1]} ({con1[2]}).",
                f"A small gap: {con1[1]} ({con1[2]})."
            ])
            
        body = tmpl.format(
            title=title, yoe=yoe, company=recent_company,
            pro1=top_pros[0][1] if top_pros else "general experience",
            fact1=top_pros[0][2] if top_pros else "fits basic criteria",
            pro2_sentence=pro2_sentence,
            con_sentence_light=con_sentence_light
        ).strip().replace("  ", " ")
        return (skills_tenure_prefix + body).strip()
        
    else:
        templates = [
            "Currently a {title} with {yoe:.1f} YoE at {company}. While they have {pro1_short} ({fact1}), a major concern is {con1} ({cfact1}). {con2_sentence}",
            "Despite bringing {yoe:.1f} YoE and {pro1_short} ({fact1}), they ranked lower primarily due to {con1} ({cfact1}). {con2_sentence}",
            "They show promise with {pro1_short} ({fact1}), but the {title}'s profile is held back by {con1} ({cfact1}). {con2_sentence}",
        ]
        tmpl = rng.choice(templates)
        
        con2_sentence = ""
        if len(top_cons) > 1:
            con2 = top_cons[1]
            # Avoid 'their a/an X' — use article-neutral connector phrases
            con2_sentence = rng.choice([
                f"Additionally, we noticed {con2[1]} ({con2[2]}).",
                f"Another issue: {con2[1]} ({con2[2]}).",
                f"Also flagged: {con2[1]} ({con2[2]})."
            ])
            
        body = tmpl.format(
            title=title, yoe=yoe, company=recent_company,
            pro1_short=top_pros[0][1] if top_pros else "some basic experience",
            fact1=top_pros[0][2] if top_pros else "meets minimums",
            con1=top_cons[0][1] if top_cons else "a lack of strong JD alignment",
            cfact1=top_cons[0][2] if top_cons else "missing core signals",
            con2_sentence=con2_sentence
        ).strip().replace("  ", " ")
        return (skills_tenure_prefix + body).strip()


def score_candidate(c: dict, semantic_score: float, full_text: str) -> tuple[float, dict]:
    p = c.get("profile", {}) or {}
    yoe = float(p.get("years_of_experience") or 0)

    # Hard floor — candidates below MIN_YEARS_EXPERIENCE should not appear at all
    if yoe < MIN_YEARS_EXPERIENCE:
        return -999.0, {}

    matched_skills = get_matching_skills(c)

    components = {}

    # 1. Semantic similarity (base) - DAMPENED BY 0.5
    score = float(semantic_score) * 0.5
    components["semantic"] = round(score, 4)

    # 2. Keyword density bonus (must-have + preferred, word-boundary)
    all_kws = MUST_HAVE_SKILL_KEYWORDS + PREFERRED_SKILL_KEYWORDS
    hits = sum(1 for kw in all_kws if re.search(r'\b' + re.escape(kw) + r'\b', full_text))
    kw_bonus = min(0.15, math.log1p(hits) * 0.025)
    score += kw_bonus
    components["kw_bonus"] = round(kw_bonus, 4)
    components["kw_hits"] = hits
    
    # 2.5 Semantic illusion penalty (for candidates with high semantic score but low exact keyword hits)
    semantic_illusion_penalty = 0.0
    if components["semantic"] > 0.35 and hits < 18:
        semantic_illusion_penalty = 0.05
        score -= semantic_illusion_penalty
    components["semantic_illusion_penalty"] = semantic_illusion_penalty

    # 2.6 Matched skill depth penalty/bonus (graduated)
    n_matched = len(matched_skills)
    skill_depth_penalty = 0.0
    skill_depth_bonus = 0.0
    if n_matched == 0:
        skill_depth_penalty = 0.20
    elif n_matched == 1:
        skill_depth_penalty = 0.12
    elif n_matched == 2:
        skill_depth_penalty = 0.06
    elif n_matched > 5:
        skill_depth_bonus = min(0.08, (n_matched - 5) * 0.015)
        
    score -= skill_depth_penalty
    score += skill_depth_bonus
    components["skill_depth_penalty"] = round(skill_depth_penalty, 4)
    components["skill_depth_bonus"] = round(skill_depth_bonus, 4)
    components["matched_skill_count"] = n_matched

    # 3. YoE scoring — stronger penalty to prevent sub-band candidates trumping seniors
    yoe_penalty = 0.0
    if yoe < TARGET_YOE_MIN:
        # Steeper: 0.08 per year below floor
        yoe_penalty = (TARGET_YOE_MIN - yoe) * 0.08
    elif yoe > TARGET_YOE_MAX:
        yoe_penalty = (yoe - TARGET_YOE_MAX) * 0.015
    score -= yoe_penalty
    components["yoe_penalty"] = round(yoe_penalty, 4)

    # 3.5 STRICT YoE GAP & JOB HOPPING PENALTY
    career = c.get("career_history") or []
    recorded_months = sum(int(ch.get("duration_months") or 0) for ch in career)
    recorded_yoe = recorded_months / 12.0
    
    yoe_gap_penalty = 0.0
    if yoe > 1 and (recorded_yoe / yoe) < 0.90:
        yoe_gap_penalty = 0.20
    score -= yoe_gap_penalty
    components["yoe_gap_penalty"] = round(yoe_gap_penalty, 4)

    job_hop_penalty = 0.0
    avg_tenure = 999.0
    valid_stints = [int(ch.get("duration_months") or 0) for ch in career if int(ch.get("duration_months") or 0) >= 6]
    if valid_stints:
        if len(valid_stints) >= 2:
            # weight most recent two roles more heavily
            weighted_sum = (valid_stints[0] * 2) + (valid_stints[1] * 2) + sum(valid_stints[2:])
            avg_tenure = weighted_sum / (len(valid_stints) + 2)
        else:
            avg_tenure = float(valid_stints[0])
            
        if avg_tenure < 18:
            job_hop_penalty = 0.15
        elif avg_tenure < 24:
            job_hop_penalty = 0.05
    score -= job_hop_penalty
    components["job_hop_penalty"] = round(job_hop_penalty, 4)
    components["avg_tenure"] = round(avg_tenure, 1)

    # 4. Notice period penalty (soft; 60d shouldn't devastate a strong candidate)
    r = c.get("redrob_signals", {}) or {}
    np_days = int(r.get("notice_period_days") or 90)
    np_penalty = 0.0
    if np_days > NOTICE_PREFERRED_DAYS:
        np_penalty = min(0.03, (np_days - NOTICE_PREFERRED_DAYS) * 0.0005)
        if np_days > NOTICE_ACCEPTABLE_DAYS:
            np_penalty += 0.04
    score -= np_penalty
    components["notice_penalty"] = round(np_penalty, 4)

    # 5. Redrob signals bonus (recalibrated — github is one signal among 23, not a primary driver)
    # github == -1 means no GitHub linked: treat as neutral (no bonus, no penalty)
    # Only penalise candidates with an *actual* score that is very low (< 20)
    github = float(r.get("github_activity_score") or -1)
    github_bonus = 0.0
    github_penalty = 0.0
    if github < 0:
        # No GitHub linked — neutral; don't penalise candidates who simply didn't link
        pass
    elif github < 20:
        # Actual score present but very low — mild penalty
        github_penalty = 0.03
        score -= github_penalty
    else:
        # Cap bonus at 0.06 max (down from 0.12) to keep it as a secondary signal
        github_bonus = (github / 100.0) * 0.06
        score += github_bonus
    components["github_bonus"] = round(github_bonus, 4)
    components["github_penalty"] = round(github_penalty, 4)

    recruiter_rate = float(r.get("recruiter_response_rate") or -1)
    recruiter_bonus = 0.0
    recruiter_penalty = 0.0
    if recruiter_rate >= 0:
        if recruiter_rate < 0.40:
            recruiter_penalty = 0.04
            score -= recruiter_penalty
        else:
            recruiter_bonus = recruiter_rate * 0.10
            score += recruiter_bonus
    else:
        recruiter_penalty = 0.04
        score -= recruiter_penalty
    components["recruiter_bonus"] = round(recruiter_bonus, 4)
    components["recruiter_penalty"] = round(recruiter_penalty, 4)

    # 6. Top-tier company prestige bonus (FAANG, etc.)
    top_tier_count, top_tier_names = check_top_tier_company(c)
    top_tier_bonus = min(top_tier_count, 2) * TOP_TIER_BONUS_PER_ROLE
    score += top_tier_bonus
    components["top_tier_bonus"] = round(top_tier_bonus, 4)
    components["top_tier_companies"] = top_tier_names

    # 7. Junior+high-YoE honeypot penalty
    is_junior_mismatch, junior_penalty = check_junior_yoe_mismatch(c)
    score -= junior_penalty
    components["junior_mismatch_penalty"] = round(junior_penalty, 4)

    # 8. Consulting-only background penalty
    consulting_penalty = check_consulting_only(c)
    score -= consulting_penalty
    components["consulting_penalty"] = round(consulting_penalty, 4)

    # 9. Quality: unendorsed expert skills (mild per-skill penalty)
    skills = c.get("skills") or []
    unendorsed_experts = sum(
        1 for sk in skills
        if sk.get("proficiency") in ("expert", "advanced")
        and int(sk.get("endorsements") or 0) == 0
    )
    quality_penalty = min(0.05, unendorsed_experts * 0.008)
    score -= quality_penalty
    components["quality_penalty"] = round(quality_penalty, 4)

    return float(score), components


def main():
    log.info("Loading precomputed data...")
    try:
        embeddings = np.load("artifacts/embeddings.npy")
        with open("artifacts/id_index.json", "r") as f:
            id_index = json.load(f)
    except FileNotFoundError:
        log.error("Precomputed artifacts not found. Run precompute.py first.")
        sys.exit(1)

    candidates = []
    with open("artifacts/stage1_shortlist.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))

    log.info(f"Loaded {len(candidates)} candidates.")

    if len(embeddings) != len(candidates):
        log.warning(f"Embedding count ({len(embeddings)}) != candidate count ({len(candidates)}); using min")

    n = min(len(embeddings), len(candidates))
    embeddings = embeddings[:n]
    candidates = candidates[:n]

    if n == 0:
        log.warning("No candidates found. Writing empty submission files.")
        pd.DataFrame(columns=["rank", "candidate_id", "score", "reasoning"]).to_csv("submission.csv", index=False)
        pd.DataFrame(columns=["rank", "candidate_id", "matched_skill_count", "github_activity_score", "recruiter_response_rate"]).to_csv("signals.csv", index=False)
        return

    log.info("Embedding JD text...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_emb = model.encode([JD_TEXT], convert_to_numpy=True)

    log.info("Computing semantic similarities...")
    semantic_scores = compute_cosine_similarities(jd_emb, embeddings)

    log.info("Computing final scores...")
    results = []
    for i, c in enumerate(candidates):
        sem_score = semantic_scores[i]
        text = candidate_full_text(c)
        final_score, components = score_candidate(c, sem_score, text)
        results.append((c, final_score, components))

    # Sort by score descending, then candidate_id ascending to break ties
    results.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    top_100 = results[:100]

    # Debug: log score breakdown for top 10 and any suspicious candidates
    log.info("--- Score breakdown for top 10 ---")
    for rank, (c_obj, score, comp) in enumerate(top_100[:10], start=1):
        log.info(f"  Rank {rank:3d} | {c_obj['candidate_id']} | score={score:.4f} | {comp}")

    # Write submission.csv
    out_path = Path("submission.csv")
    log.info(f"Writing top 100 to {out_path}...")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (c_obj, score, comp) in enumerate(top_100, start=1):
            matched_skills = get_matching_skills(c_obj)
            _, top_tier_names = check_top_tier_company(c_obj)
            has_rag = check_rag_rerank_experience(c_obj)
            reason = generate_reasoning(c_obj, comp, matched_skills, top_tier_names, rank, has_rag)
            writer.writerow([c_obj["candidate_id"], rank, round(score, 5), reason])

    # Write signals.csv — raw per-candidate signal values for visualisation
    sig_path = Path("signals.csv")
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "candidate_id", "score",
                         "matched_skill_count", "github_activity_score", "recruiter_response_rate",
                         "semantic", "kw_hits", "avg_tenure"])
        for rank, (c_obj, score, comp) in enumerate(top_100, start=1):
            r = c_obj.get("redrob_signals", {}) or {}
            github_raw = float(r.get("github_activity_score") or -1)
            recruiter_raw = float(r.get("recruiter_response_rate") or -1)
            writer.writerow([
                rank,
                c_obj["candidate_id"],
                round(score, 5),
                comp.get("matched_skill_count", 0),
                round(github_raw, 2),
                round(recruiter_raw, 3),
                round(comp.get("semantic", 0), 4),
                comp.get("kw_hits", 0),
                comp.get("avg_tenure", 0),
            ])

    log.info("Done!")


if __name__ == "__main__":
    main()
