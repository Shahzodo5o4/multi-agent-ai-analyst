"""Shared model factory — one place that knows how to build a Gemini client.

Everything else imports `get_llm()` / `get_embeddings()` so swapping models (or
providers) is a one-file change. Instances are cached: building the client on
every node call would waste time and connections.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import config

# Some Gemini models use fixed sampling and ignore `temperature`, warning once
# per call. The warning is correct but not actionable — temperature is still
# passed so the code behaves correctly on models that do honour it.
warnings.filterwarnings(
    "ignore",
    message=r".*uses fixed sampling defaults.*",
    category=UserWarning,
)


def _chat(model: str, temperature: float, name: str | None = None):
    """Build a chat client pointed at the class LiteLLM proxy.

    The proxy speaks the OpenAI protocol, so the OpenAI-compatible LangChain
    classes are used rather than the Google-native ones — same models, routed
    through the class endpoint instead of calling Google directly.

    `name` is the LangChain runnable name, which is what Langfuse (F12) shows as
    the observation title. Left unset, every LLM call in the trace is titled
    `ChatOpenAI`, so six identical rows give no clue which one routed the
    question and which one verified the answer. Passing a verb-first role name
    makes the trace readable.
    """
    config.require_api_key()
    from langchain_openai import ChatOpenAI

    kwargs = {}
    if name:
        kwargs["name"] = name
    return ChatOpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.GEMINI_API_KEY,
        model=model,
        temperature=temperature,
        max_retries=3,
        **kwargs,
    )


@lru_cache(maxsize=16)
def get_llm(temperature: float = 0.0, name: str | None = None):
    """High-volume chat model (`gemini-flash-lite`).

    Used by the four specialist agents, answer generation, the LLM judge and the
    evaluation metrics — everything where call count matters more than depth.
    temperature=0 by default: tool use should be deterministic, not creative.

    `max_retries` handles transient per-minute rate limits by backing off. It
    cannot help with a per-day cap, which is a hard stop.
    """
    return _chat(config.LLM_MODEL, temperature, name)


@lru_cache(maxsize=8)
def get_reasoning_llm(name: str | None = None):
    """Stronger model (`gemini-flash`) for the supervisor (F7) and critic (F8).

    Routing and verification are the two hardest reasoning jobs in the graph.
    A lite model mis-routes questions and will approve a SQL query whose
    denominator is wrong — both observed in this project, see the README's error
    analysis. Spending the better model only on these two nodes keeps the call
    volume low while fixing the failures that actually matter.
    """
    return _chat(config.REASONING_MODEL, 0.0, name)


# Old name kept: the critic was the first caller of the stronger model.
get_critic_llm = get_reasoning_llm


def is_quota_error(exc: Exception) -> bool:
    """True for a 429 — worth reporting differently from a real bug."""
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def quota_hint() -> str:
    return (f"Rate limit hit on '{config.LLM_MODEL}' via the class proxy. "
            "Wait a minute and retry; if it persists the shared proxy quota may "
            "be exhausted. `python pick_model.py` shows which models respond.")


@lru_cache(maxsize=1)
def get_embeddings():
    """Embedding model (`gemini-embedding`) for the document store (F2) and
    memory (F10), via the proxy's OpenAI-compatible endpoint.

    `check_embedding_ctx_length=False` matters here. By default OpenAIEmbeddings
    tokenises text with tiktoken and sends arrays of token IDs — an
    OpenAI-specific optimisation that a proxy fronting Gemini will reject or
    silently mis-embed. Disabling it sends plain strings, which is what the
    proxy expects.
    """
    config.require_api_key()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.GEMINI_API_KEY,
        model=config.EMBEDDING_MODEL,
        check_embedding_ctx_length=False,
    )


def text_of(response) -> str:
    """Gemini sometimes returns content as a list of parts rather than a string.
    Normalise both shapes to plain text."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and "text" in part:
                out.append(part["text"])
        return "".join(out)
    return str(content)
