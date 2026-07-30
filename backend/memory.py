"""F10 — Long-term memory.

Past Q/A turns are embedded into their own Qdrant collection. Before each new
question the most similar past turns are recalled and handed to the supervisor,
so a follow-up like "and the previous year?" has the earlier context to resolve
against.

Kept in a separate collection from the documents so recalled conversation can
never be mistaken for a retrieved source.
"""

from __future__ import annotations

import time
import uuid
from typing import List

from state import AgentState

RECALL_K = 3


def recall(question: str, k: int = RECALL_K) -> List[str]:
    """Return the most relevant past turns. Returns [] on any failure — memory is
    an enhancement and must never break a live question."""
    from vectorstore import memory_store

    try:
        hits = memory_store().similarity_search(question, k=k)
    except Exception:  # noqa: BLE001
        return []
    return [h.page_content for h in hits]


def remember(question: str, answer: str, session_id: str = "default") -> None:
    """Store one completed turn. Called after the graph produces a final answer."""
    from vectorstore import memory_store

    if not answer.strip():
        return
    try:
        memory_store().add_texts(
            texts=[f"Q: {question}\nA: {answer}"],
            metadatas=[{"session_id": session_id, "ts": time.time()}],
            ids=[str(uuid.uuid4())],
        )
    except Exception:  # noqa: BLE001
        pass  # never let a memory write fail the request


def memory_node(state: AgentState) -> dict:
    """Graph entry node: load relevant history into state before the supervisor
    plans anything."""
    past = recall(state["question"])
    if not past:
        return {"memory": [], "steps": ["memory (no relevant history)"]}
    return {"memory": past, "steps": [f"memory ({len(past)} past turns recalled)"]}


def reset() -> None:
    """Wipe stored turns.

    Memory is persistent and unfiltered: if the system once answered a question
    wrongly, that wrong answer is recalled on later follow-ups and can be
    repeated confidently. Clear it before a demo or a graded run so you are
    testing the system, not a stale mistake.
    """
    import config
    from vectorstore import count, drop_collection

    drop_collection(config.MEMORY_COLLECTION)
    print(f"Memory collection '{config.MEMORY_COLLECTION}' cleared "
          f"({count(config.MEMORY_COLLECTION)} turns remaining).")


if __name__ == "__main__":
    import sys

    if "--reset" in sys.argv:
        reset()
    else:
        print("Usage: python memory.py --reset    (wipes stored conversation turns)")
