"""
config.py — Centralised configuration for the Discovery Engine.
All settings are loaded from a .env file via python-dotenv.
Never hardcode credentials here.

LLM Backend: Groq  (fast inference, free tier available)
Embeddings : OpenAI text-embedding-3-small
  NOTE — Groq does not offer an embedding API; OpenAI is used
         only for vector embeddings (Chroma storage, Phase 2).
         All text classification goes through Groq (Phase 3).
"""
from dotenv import load_dotenv
import os

load_dotenv()

def _get_secret(key: str, default: str = "") -> str:
    """Read configuration from environment or Streamlit Cloud secrets."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as _st
        if hasattr(_st, "secrets") and key in _st.secrets:
            return str(_st.secrets[key])
    except Exception:
        pass
    return default

# ── API Credentials ──────────────────────────────────────────────────────────
GROQ_API_KEY         = _get_secret("GROQ_API_KEY", "")          # LLM classification
OPENAI_API_KEY       = _get_secret("OPENAI_API_KEY", "")        # Embeddings only
REDDIT_CLIENT_ID     = _get_secret("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = _get_secret("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = _get_secret("REDDIT_USER_AGENT", "discovery-engine/1.0")
YOUTUBE_API_KEY      = _get_secret("YOUTUBE_API_KEY", "")
SERP_API_KEY         = _get_secret("SERP_API_KEY", "")          # optional web search

# ── LLM Model (Groq) ─────────────────────────────────────────────────────────
# Primary: llama-3.1-8b-instant  — fastest, cheap, good JSON adherence
# Fallback: llama-3.1-70b-versatile — slower but more accurate for edge cases
CLASSIFIER_MODEL         = "llama-3.1-8b-instant"
CLASSIFIER_MODEL_FALLBACK = "llama-3.1-70b-versatile"
CLASSIFIER_TEMPERATURE   = 0.0      # deterministic JSON output

# ── Embedding Model (OpenAI) ─────────────────────────────────────────────────
EMBEDDING_MODEL      = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536         # text-embedding-3-small output dim
EMBEDDING_BATCH_SIZE = 100          # items per OpenAI embedding API call
CHROMA_COLLECTION    = "feedback_items"
CHROMA_DISTANCE      = "cosine"     # metric for similarity search

# ── Pipeline Settings ────────────────────────────────────────────────────────
BATCH_SIZE        = 20       # items per Groq classification batch
MAX_RETRIES       = 3        # max API retry attempts per item
TARGET_RAW_COUNT  = 1000     # minimum raw items before proceeding to tagging

# Per-source collection limits
SOURCE_LIMITS = {
    "play_store":  300,
    "app_store":   200,
    "reddit":      300,
    "youtube":     150,
    "help_forum":  100,
    "web":         100,
}

# ── Storage Paths ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
LOGS_DIR     = os.path.join(BASE_DIR, "logs")
REPORTS_DIR  = os.path.join(BASE_DIR, "reports")

DB_PATH              = os.path.join(DATA_DIR, "discovery_engine.db")
CHROMA_PATH          = os.path.join(DATA_DIR, "chroma")
RAW_CSV_PATH         = os.path.join(DATA_DIR, "raw_items.csv")
TAGGED_JSON_PATH     = os.path.join(DATA_DIR, "tagged_items.json")
TAGGED_CSV_PATH      = os.path.join(DATA_DIR, "tagged_items.csv")
OPPORTUNITY_CSV_PATH = os.path.join(DATA_DIR, "opportunity_areas.csv")
SYNTHESIS_REPORT_PATH = os.path.join(REPORTS_DIR, "synthesis_report.md")

# ── Text Quality Filters ──────────────────────────────────────────────────────
MIN_TEXT_LENGTH   = 35    # characters — items shorter than this are discarded
MAX_TEXT_LENGTH   = 2000  # characters — items longer are truncated intelligently

# ── Scraping / Rate Limiting ──────────────────────────────────────────────────
REQUEST_DELAY_MIN = 1.5   # seconds — minimum delay between requests
REQUEST_DELAY_MAX = 3.0   # seconds — maximum delay (random within range)

# ── Ensure directories exist on import ───────────────────────────────────────
for _dir in [DATA_DIR, LOGS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
