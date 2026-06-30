"""
jd.py — Structured JD constants derived from job_description.docx.
This is the single source of truth for all JD-based filtering decisions.

Role: Senior AI Engineer — Founding Team @ Redrob AI
"""

# ──────────────────────────────────────────────────────────────
# EXPERIENCE
# ──────────────────────────────────────────────────────────────
# Hard floor — JD says "seriously consider candidates outside the band if other
# signals are strong", but also lists hard disqualifiers below.
# We use 4 years as absolute minimum (same reasoning in JD).
MIN_YEARS_EXPERIENCE = 4

# Target band (used for scoring proximity, not hard filtering)
TARGET_YOE_MIN = 5
TARGET_YOE_MAX = 9
TARGET_YOE_IDEAL = 7   # midpoint

# ──────────────────────────────────────────────────────────────
# LOCATION
# ──────────────────────────────────────────────────────────────
# JD says: Pune/Noida preferred, open to Hyderabad/Mumbai/Delhi NCR.
# Outside India: case-by-case, no visa sponsorship.
# Hard rule: must be in India OR willing to relocate.
REQUIRE_INDIA_OR_RELOCATE = True

INDIA_PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi",
    "bengaluru", "bangalore", "chennai", "gurgaon", "gurugram"
}

# ──────────────────────────────────────────────────────────────
# NOTICE PERIOD
# ──────────────────────────────────────────────────────────────
# JD: "We'd love sub-30-day notice. We can buy out up to 30 days.
# 30+ day notice candidates are still in scope but the bar gets higher."
# → NOT a hard filter. Used for scoring only.
NOTICE_PREFERRED_DAYS = 30
NOTICE_ACCEPTABLE_DAYS = 90   # beyond this, penalize heavily

# ──────────────────────────────────────────────────────────────
# HARD DISQUALIFYING CONDITIONS (from JD "disqualifiers we actually apply")
# ──────────────────────────────────────────────────────────────
# These are described in the JD as absolute knock-outs.
# We enforce them via title + career history inspection, NOT keyword matching.

# Titles that indicate the candidate is in a completely wrong domain.
# Must match current_title or the majority of career_history titles.
HARD_DISQUALIFYING_TITLE_FRAGMENTS = [
    "marketing manager",
    "hr generalist",
    "hr manager",
    "human resources",
    "recruiter",
    "talent acquisition",
    "sales manager",
    "account executive",
    "business development",
    "operations manager",
    "technical support",
    "project manager",       # usually distinct from eng lead
    "scrum master",
    "computer vision engineer",
    "cv engineer",
    "computer vision scientist"
]

# ──────────────────────────────────────────────────────────────
# HONEYPOT DETECTION (from submission_spec.txt section 7)
# ──────────────────────────────────────────────────────────────
# ~80 honeypot profiles with "subtly impossible" data are in the dataset.
# Ranked in top 10 → disqualification.

# Pattern 1: Claims expert-level in many skills with 0 duration
HONEYPOT_EXPERT_SKILLS_WITH_ZERO_DURATION = 5   # ≥ this many = honeypot
# Pattern 2: Large unrecorded career gap.
# If the fraction of claimed YoE that is NOT backed by career_history is >= this
# threshold, the profile is flagged as impossible/fabricated.
# e.g. 10 yr claimed but only 3 yr recorded → 70% gap → honeypot
HONEYPOT_UNRECORDED_GAP_PCT = 0.10   # 10% unrecorded = honeypot


# ──────────────────────────────────────────────────────────────
# AVAILABILITY / ENGAGEMENT SIGNALS
# ──────────────────────────────────────────────────────────────
# JD explicitly says: "Active on Redrob platform or has clear signal of being
# in the job market so we can actually talk to them."
# Hard rule: NOT open_to_work AND inactive > 365 days → remove.
MAX_INACTIVE_DAYS_IF_NOT_OPEN = 365

# ──────────────────────────────────────────────────────────────
# REQUIRED SKILLS — HARD (from "Things you absolutely need")
# ──────────────────────────────────────────────────────────────
# These are described as REQUIRED in the JD. A candidate with NONE of these
# in their skills/career text is almost certainly wrong.
# We check across: skill names + career history descriptions + headline/summary.
MUST_HAVE_SKILL_KEYWORDS = [
    # Embeddings / retrieval
    "embedding", "embeddings",
    "retrieval", "vector search", "semantic search",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch",
    # Ranking / search
    "ranking", "reranking", "re-ranking",
    "recommendation", "recommender",
    # Evaluation
    "ndcg", "mrr", "map", "precision@",
    # Core NLP / ML (python removed — too broad)
    "nlp", "natural language processing",
    "sentence-transformers", "sentence transformers",
    "bert", "transformers",
]

# ──────────────────────────────────────────────────────────────
# JD-RELEVANT ROLE KEYWORDS (for title / summary relevance check)
# ──────────────────────────────────────────────────────────────
# A candidate PASSES if ANY of these appear in:
#   current_title, any career_history title, or profile.summary.
# This replaces the old narrow disqualifying-title list with a positive
# relevance gate — if nothing here appears anywhere, the person has never
# worked in a related domain at all.
JD_RELEVANT_ROLE_KEYWORDS = [
    # Engineering roles
    "machine learning", "ml engineer", "ai engineer",
    "data scientist", "data science",
    "research engineer", "research scientist",
    "nlp engineer", "nlp scientist",
    "applied scientist", "applied ml",
    "software engineer",  # broad but keeps generalist engineers in
    # Search / IR / RecSys domain titles
    "search engineer", "search scientist",
    "information retrieval", "ranking engineer",
    "recommendation", "recommender",
    "retrieval",
    # Data roles that often overlap
    "data engineer", "analytics engineer",
    "deep learning", "computer vision",  # adjacent, not disqualified
    # Foundational tech terms in titles/summaries
    "nlp", "natural language",
    "embedding", "vector",
    "transformer", "bert", "llm", "large language",
    "rag", "retrieval augmented",
    "neural", "model",
]

# ──────────────────────────────────────────────────────────────
# PREFERRED SKILLS — SOFT (from "Things we'd like you to have")
# ──────────────────────────────────────────────────────────────
PREFERRED_SKILL_KEYWORDS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "xgboost", "lightgbm",
    "a/b testing", "ab testing",
    "distributed systems", "inference optimization",
    "bm25", "hybrid search",
    "production", "deployed", "shipped",
    "mlops", "model serving",
]

# ──────────────────────────────────────────────────────────────
# DISQUALIFYING TEXT SIGNALS — SOFT (from JD "Things we explicitly do NOT want")
# ──────────────────────────────────────────────────────────────
# These are NOT hard knock-outs on their own (JD is nuanced), but penalize heavily.
NOT_WANTED_SIGNALS = [
    "computer vision only",   # CV background with no NLP/IR
    "speech recognition only",
    "robotics",
    "langchain",              # "LangChain tutorial" type profiles
    "openai api",
    "consulting firm",        # pure consulting career (handled by career_history inspection)
]

# Pure-services companies (from JD: "only worked at consulting firms")
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree"
}
