"""F1 — Shared state.

One `AgentState` flows through every node in the graph. Each agent reads what it
needs and returns ONLY the keys it changed; LangGraph merges the rest.

`steps` and `documents` use `operator.add` reducers, so a node returns just its
new items (`{"steps": ["retriever"]}`) instead of rebuilding the whole list.
That removes a whole class of "agent accidentally wiped the history" bugs.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # --- input ---
    question: str

    # --- supervisor's decision (F7): which agent runs next, or "finish" ---
    plan: str

    # --- evidence gathered by the specialists (F3, F4, F5, F6) ---
    documents: Annotated[List[str], operator.add]   # retriever + web chunks
    sources: Annotated[List[dict], operator.add]    # {title, url|file} for the UI
    sql_result: Optional[str]                       # query + rows
    code_result: Optional[str]                      # stdout of executed Python

    # --- long-term memory (F10) ---
    memory: List[str]

    # --- output ---
    answer: str

    # --- control / audit ---
    steps: Annotated[List[str], operator.add]  # full agent path, streamed to UI
    revisions: int                             # F8: bounded by MAX_REVISIONS
    critique: str                              # why the critic rejected an answer
    visited: Annotated[List[str], operator.add]  # agents already run this turn


def new_state(question: str, memory: Optional[List[str]] = None) -> AgentState:
    """Build a fully-populated initial state. Every key is set so no node has to
    defend against a missing key."""
    return {
        "question": question,
        "plan": "",
        "documents": [],
        "sources": [],
        "sql_result": None,
        "code_result": None,
        "memory": memory or [],
        "answer": "",
        "steps": [],
        "revisions": 0,
        "critique": "",
        "visited": [],
    }


def evidence_block(state: AgentState, include_memory: bool = True) -> str:
    """Flatten everything the specialists found into one prompt-ready string.

    `include_memory=False` is used by the critic (F8). Past turns are *context*
    for resolving references, not *evidence* for verification — a previous wrong
    answer presented as evidence makes the critic more likely to approve the
    same mistake again, which is exactly how a bad answer becomes self-
    confirming across turns.
    """
    parts: List[str] = []
    if include_memory and state.get("memory"):
        parts.append("RELEVANT PAST TURNS:\n" + "\n".join(state["memory"]))
    if state.get("documents"):
        numbered = "\n\n".join(
            f"[{i}] {d}" for i, d in enumerate(state["documents"], start=1)
        )
        parts.append("RETRIEVED PASSAGES:\n" + numbered)
    if state.get("sql_result"):
        parts.append("DATABASE RESULT:\n" + state["sql_result"])
    if state.get("code_result"):
        parts.append("CODE EXECUTION RESULT:\n" + state["code_result"])
    return "\n\n".join(parts) if parts else "(no evidence gathered)"
