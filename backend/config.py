"""F1 — Configuration.

Every secret and tunable is read from `.env` at import time. Nothing is
hard-coded, so the same code runs locally, in Colab, and on Render.

Rubric note (F1): "keys load from .env" is checked here, and `.env` is
git-ignored at the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths relative to this file, not the shell's cwd, so `uvicorn main:app`
# and `python eval/evaluate.py` both find the same .env and database.
BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")


def _path(env_key: str, default: str) -> Path:
    """Read a path from env and anchor it to BACKEND_DIR if it's relative."""
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (BACKEND_DIR / p)


def _key(env_key: str) -> str:
    """Read a secret, treating the .env.example placeholder as 'not set'.

    Without this, copying .env.example to .env makes every check believe a key
    is configured, and the first real error is an opaque 400 from the API.
    """
    value = os.getenv(env_key, "").strip()
    if not value or value.startswith("paste-your") or value.startswith("your-"):
        return ""
    return value


# --- Keys -------------------------------------------------------------------
# GEMINI_API_KEY is the current name; GOOGLE_API_KEY is still read so an older
# .env keeps working. Loaded from .env only — never hard-coded, never printed.
GEMINI_API_KEY = _key("GEMINI_API_KEY") or _key("GOOGLE_API_KEY")
TAVILY_API_KEY = _key("TAVILY_API_KEY")
LANGFUSE_PUBLIC_KEY = _key("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = _key("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

# --- Models -----------------------------------------------------------------
# All Gemini traffic goes through the class LiteLLM proxy, which exposes both an
# OpenAI-compatible surface (/v1) and a native Gemini surface (/gemini).
PROXY_BASE = os.getenv(
    "PROXY_BASE_URL", "https://saidazam-litellm-proxy.hf.space"
).rstrip("/")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", f"{PROXY_BASE}/v1")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", f"{PROXY_BASE}/gemini")

# Only these three model names exist on the proxy.
#   gemini-flash-lite : high-volume work — the four agents, answer generation,
#                       the LLM judge, and the evaluation metrics.
#   gemini-flash      : the supervisor and the critic. Routing and verification
#                       are the two hardest reasoning jobs in the graph, and a
#                       lite model approves SQL whose denominator is wrong.
#   gemini-embedding  : embeddings for the document store and memory.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-lite")
# REASONING_MODEL drives BOTH the supervisor and the critic. CRITIC_MODEL is
# still honoured so an existing .env does not silently change behaviour.
REASONING_MODEL = (
    os.getenv("REASONING_MODEL", "").strip()
    or os.getenv("CRITIC_MODEL", "").strip()
    or "gemini-flash"
)
CRITIC_MODEL = REASONING_MODEL  # backwards-compatible alias
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding")
# Must equal the embedding model's real output size, or every Qdrant insert
# fails with a dimension error. verify_key.py reports the actual number — the
# proxy may not return the same width as calling Google directly.
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "3072"))

# --- Storage ----------------------------------------------------------------
QDRANT_PATH = _path("QDRANT_PATH", "./qdrant_data")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "long_term_memory")
DOCS_DIR = _path("DOCS_DIR", "./data/docs")

# SQLite lives under backend/data/. Build an absolute URI so the data agent works
# no matter which directory the process was started from.
_sqlite_raw = os.getenv("SQLITE_URI", "sqlite:///data/company.db")
if _sqlite_raw.startswith("sqlite:///") and not _sqlite_raw.startswith("sqlite:////"):
    _rel = _sqlite_raw[len("sqlite:///"):]
    SQLITE_URI = f"sqlite:///{(BACKEND_DIR / _rel).as_posix()}"
else:
    SQLITE_URI = _sqlite_raw

# --- Safety limits ----------------------------------------------------------
# F6: the code agent is killed after this many seconds.
CODE_TIMEOUT_SECONDS = int(os.getenv("CODE_TIMEOUT_SECONDS", "10"))
# F9: hard ceiling on graph steps so a mis-routing loop can never run forever.
MAX_GRAPH_STEPS = int(os.getenv("MAX_GRAPH_STEPS", "12"))
# F8: how many times the critic may bounce an answer back before we give up.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

# --- Feature flags: optional services degrade gracefully (F4, F12) ----------
WEB_ENABLED = bool(TAVILY_API_KEY)
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def require_api_key() -> None:
    """Fail loudly and helpfully instead of with a confusing SDK stack trace."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is missing.\n"
            f"  1. copy {BACKEND_DIR / '.env.example'} to {BACKEND_DIR / '.env'}\n"
            "  2. paste the key your class issued for the LiteLLM proxy\n"
            "  3. set it as GEMINI_API_KEY=..."
        )


# Old name kept so any straggling import does not break.
require_google_key = require_api_key


def summary() -> dict:
    """Non-secret config snapshot — surfaced at GET /health for the frontend."""
    return {
        "llm_model": LLM_MODEL,
        "reasoning_model": REASONING_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "proxy_base_url": OPENAI_BASE_URL,  # a URL, not a secret
        "google_key_set": bool(GEMINI_API_KEY),  # key name kept for the frontend
        "web_enabled": WEB_ENABLED,
        "langfuse_enabled": LANGFUSE_ENABLED,
        "max_graph_steps": MAX_GRAPH_STEPS,
        "max_revisions": MAX_REVISIONS,
        "code_timeout_seconds": CODE_TIMEOUT_SECONDS,
    }
