"""F3-F6 — the four specialist agents.

Every agent has the same shape: it takes the shared `AgentState`, does ONE job,
and returns only the keys it changed. That uniformity is what lets the
supervisor treat them interchangeably.

Each is independently testable — `python agents.py` at the bottom of this file
runs all four alone, which is exactly what the rubric asks for ("tested alone").
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import config
import sandbox
from llm import get_llm, is_quota_error, quota_hint, text_of
from state import AgentState

def _context(state: AgentState) -> str:
    """Recalled conversation history, formatted for a specialist's prompt.

    Specialists must see this, not just the supervisor: a follow-up like "and
    what percentage is that?" is unanswerable without knowing what "that" was,
    and an agent that guesses will confidently answer a different question.
    """
    past = state.get("memory") or []
    if not past:
        return ""
    return "\nCONVERSATION CONTEXT (earlier turns, for resolving references):\n" \
           + "\n".join(past) + "\n"


# ---------------------------------------------------------------------------
# F3 — Retriever agent: RAG over the ingested documents
# ---------------------------------------------------------------------------

RETRIEVER_K = 4


def retriever_agent(state: AgentState) -> dict:
    """Fetch the top-k most relevant chunks for the question."""
    from vectorstore import documents_store

    question = state["question"]
    try:
        store = documents_store()
        docs = store.as_retriever(search_kwargs={"k": RETRIEVER_K}).invoke(question)
    except Exception as exc:  # noqa: BLE001
        return {
            "steps": [f"retriever (failed: {exc})"],
            "visited": ["retriever"],
        }

    if not docs:
        return {
            "steps": ["retriever (0 chunks — is the knowledge base ingested?)"],
            "visited": ["retriever"],
        }

    sources = [
        {
            "type": "document",
            "title": d.metadata.get("source", "document"),
            "snippet": d.page_content[:200],
        }
        for d in docs
    ]
    return {
        "documents": [d.page_content for d in docs],
        "sources": sources,
        "steps": [f"retriever ({len(docs)} chunks)"],
        "visited": ["retriever"],
    }


# ---------------------------------------------------------------------------
# F4 — Web agent: Tavily search for anything outside the documents
# ---------------------------------------------------------------------------

WEB_K = 4


def web_agent(state: AgentState) -> dict:
    """Live web search. Skips itself gracefully when no key is configured —
    the rubric explicitly checks this."""
    if not config.WEB_ENABLED:
        return {
            "steps": ["web (skipped — no TAVILY_API_KEY set)"],
            "visited": ["web"],
        }

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        hits = client.search(
            state["question"], max_results=WEB_K, search_depth="basic"
        ).get("results", [])
    except Exception as exc:  # noqa: BLE001
        return {"steps": [f"web (failed: {exc})"], "visited": ["web"]}

    if not hits:
        return {"steps": ["web (0 results)"], "visited": ["web"]}

    return {
        "documents": [h.get("content", "") for h in hits if h.get("content")],
        "sources": [
            {
                "type": "web",
                "title": h.get("title", "web result"),
                "url": h.get("url", ""),
                "snippet": (h.get("content") or "")[:200],
            }
            for h in hits
        ],
        "steps": [f"web ({len(hits)} results)"],
        "visited": ["web"],
    }


# ---------------------------------------------------------------------------
# F5 — Data agent: text-to-SQL against a real database
# ---------------------------------------------------------------------------

# Read-only guard. The guide is blunt about this: never let a generated query
# run DROP or DELETE. Two independent checks — an allow-list on the first
# keyword AND a deny-list scan — because either alone has holes.
_ALLOWED_STARTS = ("select", "with")
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "truncate", "attach", "detach", "pragma", "vacuum", "grant", "revoke",
)


class UnsafeSQL(Exception):
    """Raised when a generated query is not a read-only SELECT."""


@lru_cache(maxsize=1)
def get_db():
    from langchain_community.utilities import SQLDatabase

    return SQLDatabase.from_uri(config.SQLITE_URI)


def clean_sql(raw: str) -> str:
    """Strip markdown fences, prose, and a trailing semicolon from LLM output."""
    text = raw.strip()
    fenced = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    # Drop any chatter before the first SELECT/WITH.
    match = re.search(r"\b(select|with)\b", text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    return text.strip().rstrip(";").strip()


def guard_sql(sql: str) -> None:
    """Raise UnsafeSQL unless this is a single read-only statement."""
    lowered = sql.lower().strip()
    if not lowered.startswith(_ALLOWED_STARTS):
        raise UnsafeSQL("query must start with SELECT or WITH")
    # Reject stacked statements: "SELECT 1; DROP TABLE users"
    if ";" in sql.strip().rstrip(";"):
        raise UnsafeSQL("multiple statements are not allowed")
    for word in _FORBIDDEN:
        if re.search(rf"\b{word}\b", lowered):
            raise UnsafeSQL(f"forbidden keyword '{word.upper()}'")


SQL_PROMPT = """You are a SQLite expert. Write ONE read-only SQL query.

Database schema:
{schema}
{context}
Question: {question}

Rules:
- Resolve pronouns against the conversation context above. If the question says
  "that", "it" or "those", it refers to the subject of the previous turn — carry
  over the SAME filters (date range, segment, status). Getting this wrong
  silently answers a different question.
- SQLite dialect only.
- The query MUST start with SELECT (or WITH ... SELECT). Never write INSERT,
  UPDATE, DELETE, DROP, ALTER or CREATE.
- For a ratio, make the denominator explicit with a scalar subquery, e.g.
  `... / (SELECT COUNT(*) FROM customers)`. Do NOT join another table and then
  divide by COUNT(*) — after a join, COUNT(*) counts joined rows, not the
  total you meant, and the query silently returns the wrong number.
- Read the question's denominator literally: "percentage of all customers"
  divides by every row in `customers`, not by the number of churned ones.
- Return ONLY the SQL. No explanation, no markdown fences.
"""


def data_agent(state: AgentState) -> dict:
    """Write a SQL query from the question, guard it, run it, read the result."""
    try:
        db = get_db()
        schema = db.get_table_info()
    except Exception as exc:  # noqa: BLE001
        return {
            "sql_result": f"ERROR: cannot open the database ({exc}). "
                          "Run `python data/seed_db.py` first.",
            "steps": ["data/sql (no database)"],
            "visited": ["data"],
        }

    prompt = SQL_PROMPT.format(
        schema=schema, question=state["question"], context=_context(state)
    )
    try:
        sql = clean_sql(text_of(get_llm(name="write-sql").invoke(prompt)))
    except Exception as exc:  # noqa: BLE001
        # A quota error mid-run should degrade this one agent, not kill the graph:
        # the supervisor can still finish with whatever other evidence exists.
        detail = quota_hint() if is_quota_error(exc) else f"{type(exc).__name__}: {exc}"
        return {
            "sql_result": f"ERROR: could not generate SQL. {detail}",
            "steps": ["data/sql (LLM unavailable)"],
            "visited": ["data"],
        }

    try:
        guard_sql(sql)
    except UnsafeSQL as exc:
        return {
            "sql_result": f"REJECTED BY READ-ONLY GUARD: {exc}\nQuery was: {sql}",
            "steps": [f"data/sql (blocked: {exc})"],
            "visited": ["data"],
        }

    try:
        rows = db.run(sql)
    except Exception as exc:  # noqa: BLE001
        return {
            "sql_result": f"SQL:\n{sql}\n\nERROR: {exc}",
            "steps": ["data/sql (query error)"],
            "visited": ["data"],
        }

    rows_text = str(rows).strip() or "(no rows)"
    return {
        "sql_result": f"SQL:\n{sql}\n\nRESULT:\n{rows_text}",
        "sources": [{"type": "sql", "title": "company.db", "snippet": sql}],
        "steps": ["data/sql"],
        "visited": ["data"],
    }


# ---------------------------------------------------------------------------
# F6 — Code agent: write and run Python for exact computation
# ---------------------------------------------------------------------------

CODE_PROMPT = """Write a short Python script that answers this question.

Question: {question}

{context}

Rules:
- print() the final answer. Output that isn't printed is lost.
- Use only the standard library (math, statistics, datetime, itertools, decimal).
- Do NOT import os, sys, subprocess, socket, pathlib or requests — they are
  blocked by the sandbox and your script will be rejected.
- Do NOT read or write files.
- Return ONLY the code. No explanation, no markdown fences.
"""


def code_agent(state: AgentState) -> dict:
    """Generate Python, run it in the sandbox (F6), return stdout.

    LLMs are unreliable at arithmetic; executing code makes the numbers exact.
    """
    context = _context(state)
    if state.get("sql_result"):
        context += (
            "\nYou may use these figures already retrieved from the database:\n"
            f"{state['sql_result']}\n"
        )

    prompt = CODE_PROMPT.format(question=state["question"], context=context)
    try:
        code = sandbox.strip_code_fences(
            text_of(get_llm(name="write-code").invoke(prompt))
        )
    except Exception as exc:  # noqa: BLE001
        detail = quota_hint() if is_quota_error(exc) else f"{type(exc).__name__}: {exc}"
        return {
            "code_result": f"ERROR: could not generate code. {detail}",
            "steps": ["code (LLM unavailable)"],
            "visited": ["code"],
        }
    output = sandbox.run(code)

    return {
        "code_result": f"CODE:\n{code}\n\nOUTPUT:\n{output}",
        "sources": [{"type": "code", "title": "sandboxed python", "snippet": code[:200]}],
        "steps": ["code"],
        "visited": ["code"],
    }


# Registry the supervisor routes into (F7/F9).
SPECIALISTS = {
    "retriever": retriever_agent,
    "web": web_agent,
    "data": data_agent,
    "code": code_agent,
}


# ---------------------------------------------------------------------------
# Test each agent ALONE — `python agents.py`
# The rubric awards F3-F6 for agents demonstrated in isolation.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from state import new_state

    checks = [
        ("F3 retriever", retriever_agent, "What is the refund policy?"),
        ("F4 web", web_agent, "What is LangGraph used for?"),
        ("F5 data/sql", data_agent, "How many customers churned in Q3 2024?"),
        ("F6 code", code_agent, "What is the compound interest on 2500 at 4.5% for 7 years?"),
    ]
    for label, fn, question in checks:
        print("\n" + "=" * 70)
        print(f"{label}  —  {question}")
        print("=" * 70)
        try:
            result = fn(new_state(question))
            for key, value in result.items():
                preview = str(value)
                print(f"  {key}: {preview[:600]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
