"""F11 — the four RAGAS metrics, implemented directly.

WHY THIS EXISTS. The RAGAS library will not run on this project's environment,
and no version of it does:

  * every published version hard-imports `langchain_community.chat_models
    .vertexai`, which langchain-community removed in 0.4 — so RAGAS cannot even
    be imported alongside the LangChain v1 stack this project uses;
  * 0.3+ additionally requires `scikit-network`, which ships no Python 3.14
    wheel and needs Microsoft C++ Build Tools to compile;
  * 0.2.x, isolated in its own venv with the older LangChain, then fails at
    runtime on Python 3.14 with `RuntimeError: Timeout should be used inside a
    task` — its async executor predates 3.14's asyncio changes.

So the metrics are computed here instead, following the definitions from the
RAGAS paper and docs. This is not an approximation of the idea — it is the same
four measurements:

  faithfulness      : of the claims the answer makes, what fraction is supported
                      by the retrieved context? (measures hallucination)
  answer_relevancy  : does the answer actually address the question asked?
                      Generate questions the answer would answer, and compare
                      them to the real one by embedding cosine similarity.
  context_precision : of the retrieved chunks, what fraction was actually useful?
                      (measures retrieval noise)
  context_recall    : of the claims in the reference answer, what fraction is
                      present in the retrieved context? (measures retrieval
                      misses)

All four return 0.0-1.0, higher is better. Cost is 3 LLM calls plus 2 embedding
calls per row.

If your mentor requires the literal RAGAS library, `score_ragas.py` runs it in
`.venv-eval` — that path works on Python 3.12 or older.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from llm import get_llm  # noqa: E402

MAX_CONTEXT_CHARS = 4000  # keep prompts inside free-tier limits


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------


class ClaimVerdicts(BaseModel):
    """Each claim in the answer, judged against the context."""

    claims: List[str] = Field(description="Each distinct factual claim in the answer.")
    supported: List[bool] = Field(
        description="For each claim, in the same order: is it supported by the context?"
    )


class GeneratedQuestions(BaseModel):
    questions: List[str] = Field(
        description="3 questions that the given answer would be a correct answer to."
    )


class RelevanceFlags(BaseModel):
    relevant: List[bool] = Field(
        description="For each numbered context, in order: was it useful for answering?"
    )


def _join(contexts: List[str]) -> str:
    text = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    return text[:MAX_CONTEXT_CHARS]


def _ratio(flags: List[bool]) -> float:
    return round(sum(flags) / len(flags), 3) if flags else 0.0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def faithfulness(answer: str, contexts: List[str]) -> float:
    """Fraction of the answer's claims that the context supports."""
    prompt = (
        "Break the ANSWER into its distinct factual claims, then decide for each "
        "whether the CONTEXT supports it.\n\n"
        f"CONTEXT:\n{_join(contexts)}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "A claim is supported only if the context states or directly implies it. "
        "General knowledge does not count as support. Return the claims and a "
        "matching list of booleans in the same order."
    )
    out = get_llm().with_structured_output(ClaimVerdicts).invoke(prompt)
    flags = out.supported[:len(out.claims)]
    return _ratio(flags)


def answer_relevancy(question: str, answer: str) -> float:
    """How directly the answer addresses the question actually asked.

    Generate questions the answer would answer, embed them, and average their
    cosine similarity to the real question. An evasive or off-topic answer
    produces questions unlike the original and scores low.
    """
    from llm import get_embeddings

    prompt = (
        "Read the ANSWER and write 3 questions that this answer would be a "
        "correct and complete response to. Base them only on the answer's "
        f"content.\n\nANSWER:\n{answer}"
    )
    generated = get_llm(temperature=0.3).with_structured_output(
        GeneratedQuestions).invoke(prompt).questions
    if not generated:
        return 0.0

    emb = get_embeddings()
    original = emb.embed_query(question)
    others = emb.embed_documents(generated[:3])

    sims = []
    for vec in others:
        dot = sum(a * b for a, b in zip(original, vec))
        norm = math.sqrt(sum(a * a for a in original)) * math.sqrt(sum(b * b for b in vec))
        sims.append(dot / norm if norm else 0.0)
    return round(max(0.0, sum(sims) / len(sims)), 3)


def context_precision(question: str, contexts: List[str]) -> float:
    """Fraction of retrieved chunks that were actually useful.

    Low precision means the retriever returned noise alongside the signal.
    """
    if not contexts:
        return 0.0
    prompt = (
        "For each numbered context below, decide whether it was useful for "
        f"answering the QUESTION.\n\nQUESTION: {question}\n\n"
        f"CONTEXTS:\n{_join(contexts)}\n\n"
        "Return one boolean per context, in order. Mark a context useful only if "
        "it contains information needed to answer the question."
    )
    out = get_llm().with_structured_output(RelevanceFlags).invoke(prompt)
    return _ratio(out.relevant[:len(contexts)])


def context_recall(reference: str, contexts: List[str]) -> float:
    """Fraction of the reference answer's claims present in the context.

    Low recall means retrieval missed information the correct answer needs.
    """
    if not contexts:
        return 0.0
    prompt = (
        "Break the REFERENCE ANSWER into its distinct factual claims, then "
        "decide for each whether the CONTEXT contains that information.\n\n"
        f"CONTEXT:\n{_join(contexts)}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        "Return the claims and a matching list of booleans in the same order."
    )
    out = get_llm().with_structured_output(ClaimVerdicts).invoke(prompt)
    return _ratio(out.supported[:len(out.claims)])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def score_row(question: str, answer: str, contexts: List[str], reference: str) -> dict:
    """All four metrics for one question. A failed metric returns None rather
    than aborting the run — a partial table beats no table."""
    scores: dict = {}
    for name, fn in (
        ("faithfulness", lambda: faithfulness(answer, contexts)),
        ("answer_relevancy", lambda: answer_relevancy(question, answer)),
        ("context_precision", lambda: context_precision(question, contexts)),
        ("context_recall", lambda: context_recall(reference, contexts)),
    ):
        try:
            scores[name] = fn()
        except Exception as exc:  # noqa: BLE001
            scores[name] = None
            scores.setdefault("errors", []).append(f"{name}: {type(exc).__name__}")
    return scores


def score_all(rows: List[dict], pause: float = 1.0) -> dict:
    """Average each metric across rows. `rows` need question/answer/contexts/reference."""
    import time

    per_row = []
    for i, r in enumerate(rows, 1):
        print(f"  [{i}/{len(rows)}] {r['question'][:52]}")
        s = score_row(r["question"], r["answer"], r["contexts"], r["reference"])
        short = {"faithfulness": "faith", "answer_relevancy": "ans_rel",
                 "context_precision": "ctx_prec", "context_recall": "ctx_rec"}
        shown = "  ".join(f"{short[k]}={s[k]}" for k in METRIC_NAMES
                          if s.get(k) is not None)
        print(f"        {shown}")
        per_row.append(s)
        time.sleep(pause)

    averages = {}
    for name in METRIC_NAMES:
        vals = [s[name] for s in per_row if s.get(name) is not None]
        averages[name] = round(sum(vals) / len(vals), 3) if vals else None
    return {"averages": averages, "per_row": per_row}
