"""
jd.py — Structured JD constants derived from job_description.docx.
This is the single source of truth for all JD-based filtering decisions.

Role: Senior AI Engineer — Founding Team @ Redrob AI
"""

MIN_YEARS_EXPERIENCE = 4

# Target band (used for scoring proximity, not hard filtering)
TARGET_YOE_MIN = 5
TARGET_YOE_MAX = 9
TARGET_YOE_IDEAL = 7   # midpoint

REQUIRE_INDIA_OR_RELOCATE = True

INDIA_PREFERRED_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi",
    "bengaluru", "bangalore", "chennai", "gurgaon", "gurugram"
}

NOTICE_PREFERRED_DAYS = 30
NOTICE_ACCEPTABLE_DAYS = 90   # beyond this, penalize heavily

# Disqualifying titles
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

# Honeypot detection thresholds
HONEYPOT_EXPERT_SKILLS_WITH_ZERO_DURATION = 5
HONEYPOT_UNRECORDED_GAP_PCT = 0.10   # 10% unrecorded = honeypot


MAX_INACTIVE_DAYS_IF_NOT_OPEN = 365

# Required skills
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

# JD relevant role keywords
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

# Preferred skills
PREFERRED_SKILL_KEYWORDS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "learning to rank", "xgboost", "lightgbm",
    "a/b testing", "ab testing",
    "distributed systems", "inference optimization",
    "bm25", "hybrid search",
    "production", "deployed", "shipped",
    "mlops", "model serving",
]

# Unwanted signals
NOT_WANTED_SIGNALS = [
    "computer vision only",   # CV background with no NLP/IR
    "speech recognition only",
    "robotics",
    "langchain",              # "LangChain tutorial" type profiles
    "openai api",
    "consulting firm",        # pure consulting career (handled by career_history inspection)
]

# Pure-services companies
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree"
}
