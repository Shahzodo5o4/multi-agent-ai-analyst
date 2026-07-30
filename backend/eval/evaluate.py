"""F11 — Evaluation harness.

Scores the system on a fixed 14-question test set with two independent methods:

  1. LLM-AS-JUDGE  — a model scores each answer 1-5 against a reference answer,
     plus a strict correctness flag. Always runs.
  2. RAGAS         — faithfulness, answer relevancy, context precision/recall.
     Runs if `pip install -r requirements-eval.txt` succeeded; skipped with a
     clear message if not, so a heavy install can never block the rest.

It also runs a ROUTING check (did the supervisor pick the right agent?) and the
ABLATION the guide's "required visuals" asks for: the same test set with and
without the critic, side by side.

Run:
    python eval/evaluate.py                  # with critic only
    python eval/evaluate.py --ablation       # with AND without the critic
    python eval/evaluate.py --limit 4        # quick smoke run, saves free quota
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

# Make the backend package importable when run as `python eval/evaluate.py`.
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from pydantic import BaseModel, Field  # noqa: E402

import graph  # noqa: E402
import tracing  # noqa: E402
from llm import get_llm  # noqa: E402

TESTSET = EVAL_DIR / "testset.json"
RESULTS_DIR = EVAL_DIR / "results"


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


class Judgement(BaseModel):
    score: int = Field(description="1 = wrong or unsupported, 5 = fully correct and complete.")
    correct: bool = Field(description="True only if the answer is factually correct.")
    reason: str = Field(description="One sentence justifying the score.")


JUDGE_PROMPT = """You are grading an AI analyst's answer against a reference answer.

QUESTION: {question}

REFERENCE ANSWER (ground truth):
{reference}

SYSTEM'S ANSWER:
{answer}

Score 1-5:
  5 = matches the reference; all numbers and facts correct and complete
  4 = correct but missing a minor detail
  3 = partially correct, or one part of a multi-part question is missing
  2 = mostly wrong, or a key number disagrees with the reference
  1 = wrong, unsupported, or refuses to answer

Numbers matter most. A different number from the reference is at most a 2.
Wording may differ freely — judge the substance, not the phrasing.
"""


def judge(question: str, reference: str, answer: str) -> Judgement:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    try:
        return get_llm().with_structured_output(Judgement).invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        return Judgement(score=0, correct=False, reason=f"judge failed: {exc}")


# ---------------------------------------------------------------------------
# Run the system over the test set
# ---------------------------------------------------------------------------


def routed_correctly(case: dict, steps: List[str]) -> bool:
    """Did the supervisor call the agent(s) this question actually needs?

    `expected_agent` supports "+" for "all of these" and "|" for acceptable
    alternatives, e.g. "data+code|data" — because one SQL query that computes
    the percentage itself is a *better* route than data->code, not a failure.
    Scoring it wrong would penalise the supervisor for being smart.
    """
    path = " ".join(steps).lower()
    for option in case.get("expected_agent", "").split("|"):
        needed = [a.strip() for a in option.split("+") if a.strip()]
        if needed and all(agent in path for agent in needed):
            return True
    return False


def run_suite(cases: List[dict], use_critic: bool, pause: float) -> List[dict]:
    label = "WITH critic" if use_critic else "WITHOUT critic"
    print(f"\n{'=' * 74}\nRunning {len(cases)} questions — {label}\n{'=' * 74}")

    rows = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['id']}: {case['question'][:64]}")
        started = time.time()
        try:
            # store_memory=False: a graded answer must not leak into memory and
            # influence a later question in the same run.
            final = graph.run(case["question"], use_critic=use_critic,
                              store_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"    RUN FAILED: {exc}")
            rows.append({
                **case, "answer": f"ERROR: {exc}", "steps": [], "contexts": [],
                "score": 0, "correct": False, "reason": "run failed",
                "routed_ok": False, "revisions": 0, "latency": 0.0,
            })
            continue

        answer = final.get("answer", "")
        steps = final.get("steps", [])
        # RAGAS calls the evidence "contexts": everything the answer may rely on.
        contexts = list(final.get("documents", []))
        for key in ("sql_result", "code_result"):
            if final.get(key):
                contexts.append(final[key])

        verdict = judge(case["question"], case["reference"], answer)
        latency = time.time() - started

        print(f"    score {verdict.score}/5 · {len(steps)} steps · "
              f"{final.get('revisions', 0)} revision(s) · {latency:.1f}s")
        print(f"    {answer[:150].replace(chr(10), ' ')}")

        rows.append({
            **case,
            "answer": answer,
            "steps": steps,
            "contexts": contexts,
            "score": verdict.score,
            "correct": verdict.correct,
            "reason": verdict.reason,
            "routed_ok": routed_correctly(case, steps),
            "revisions": final.get("revisions", 0),
            "latency": round(latency, 2),
        })
        # Free-tier rate limits are per minute; a short pause avoids 429s.
        time.sleep(pause)

    return rows


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------


def run_metrics(rows: List[dict]) -> dict:
    """The four RAGAS metrics, computed by eval/ragas_metrics.py.

    See that module's docstring for why the RAGAS library itself cannot run on
    this stack. The measurements are the same four; only the implementation is
    local.
    """
    usable = [r for r in rows if r["contexts"] and r["answer"]
              and not r["answer"].startswith("ERROR")]
    if not usable:
        print("\n[metrics] skipped — no rows with both contexts and an answer.")
        return {}

    import ragas_metrics

    print(f"\n[metrics] scoring {len(usable)} rows "
          "(faithfulness, answer relevancy, context precision/recall)...")
    try:
        result = ragas_metrics.score_all(usable)
    except Exception as exc:  # noqa: BLE001
        print(f"[metrics] failed: {type(exc).__name__}: {exc}")
        return {}
    return result["averages"]


def write_ragas_input(rows: List[dict], label: str) -> Path | None:
    """Write the RAGAS input file for stage 2 of the evaluation.

    RAGAS cannot run in this environment. Every published version hard-imports
    `langchain_community.chat_models.vertexai`, which was removed in
    langchain-community 0.4 — so RAGAS is incompatible with the LangChain v1
    stack this project runs on, and no version bump fixes it (0.3+ additionally
    needs scikit-network, which has no Python 3.14 wheel).

    Rather than downgrade the whole project to satisfy a metrics library, the
    evaluation is split in two: this stage runs the agents and dumps what RAGAS
    needs, and `score_ragas.py` scores it inside `.venv-eval`, which pins the
    older stack. See the README for the two commands.
    """
    usable = [r for r in rows if r["contexts"] and r["answer"]
              and not r["answer"].startswith("ERROR")]
    if not usable:
        print("\n[RAGAS] nothing to write — no rows have both contexts and an answer.")
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"ragas-input-{label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps([{
        "question": r["question"],
        "answer": r["answer"],
        "contexts": r["contexts"],
        "ground_truth": r["reference"],
    } for r in usable], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[RAGAS] input for {len(usable)} rows written to {path.name}")
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarise(rows: List[dict]) -> dict:
    scored = [r for r in rows if r["score"] > 0]
    return {
        "questions": len(rows),
        "avg_judge_score": round(statistics.mean([r["score"] for r in scored]), 2) if scored else 0,
        "accuracy": round(100 * sum(r["correct"] for r in rows) / len(rows), 1) if rows else 0,
        "routing_accuracy": round(100 * sum(r["routed_ok"] for r in rows) / len(rows), 1) if rows else 0,
        "avg_revisions": round(statistics.mean([r["revisions"] for r in rows]), 2) if rows else 0,
        "avg_latency_s": round(statistics.mean([r["latency"] for r in rows]), 1) if rows else 0,
    }


def print_table(rows: List[dict]) -> None:
    print(f"\n{'id':<9} {'category':<11} {'score':>5} {'ok':>4} {'route':>6} {'rev':>4} {'sec':>6}")
    print("-" * 52)
    for r in rows:
        print(f"{r['id']:<9} {r['category']:<11} {r['score']:>4}/5 "
              f"{'Y' if r['correct'] else 'N':>4} "
              f"{'Y' if r['routed_ok'] else 'N':>6} {r['revisions']:>4} {r['latency']:>6.1f}")


def write_report(payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    (RESULTS_DIR / f"results-{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown table — paste straight into the README for the rubric.
    md = ["# Evaluation results", "", f"_Generated {stamp}_", ""]
    md += ["## Summary", "",
           "| Metric | With critic | Without critic |", "|---|---|---|"]
    with_c = payload["with_critic"]["summary"]
    without_c = payload.get("without_critic", {}).get("summary", {})
    for key in ("questions", "avg_judge_score", "accuracy", "routing_accuracy",
                "avg_revisions", "avg_latency_s"):
        md.append(f"| {key} | {with_c.get(key, '-')} | {without_c.get(key, '-')} |")

    ragas_with = payload["with_critic"].get("ragas", {})
    ragas_without = payload.get("without_critic", {}).get("ragas", {})
    if ragas_with or ragas_without:
        md += ["", "## RAGAS", "", "| Metric | With critic | Without critic |", "|---|---|---|"]
        for key in sorted(set(ragas_with) | set(ragas_without)):
            md.append(f"| {key} | {ragas_with.get(key, '-')} | {ragas_without.get(key, '-')} |")

    md += ["", "## Per-question (with critic)", "",
           "| id | category | score | correct | routed | revisions |", "|---|---|---|---|---|---|"]
    for r in payload["with_critic"]["rows"]:
        md.append(f"| {r['id']} | {r['category']} | {r['score']}/5 | "
                  f"{'yes' if r['correct'] else 'no'} | "
                  f"{'yes' if r['routed_ok'] else 'no'} | {r['revisions']} |")

    path = RESULTS_DIR / f"results-{stamp}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="F11 evaluation harness")
    parser.add_argument("--ablation", action="store_true",
                        help="also run without the critic, for the comparison table")
    parser.add_argument("--limit", type=int, default=0, help="only run the first N questions")
    parser.add_argument("--pause", type=float, default=1.5,
                        help="seconds between questions (free-tier rate limits)")
    parser.add_argument("--no-ragas", action="store_true", help="skip RAGAS")
    args = parser.parse_args()

    cases = json.loads(TESTSET.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[:args.limit]

    payload = {"testset_size": len(cases)}

    ragas_inputs = []

    rows = run_suite(cases, use_critic=True, pause=args.pause)
    print_table(rows)
    payload["with_critic"] = {
        "summary": summarise(rows),
        "ragas": {} if args.no_ragas else run_metrics(rows),
        "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows],
    }
    if not args.no_ragas:
        ragas_inputs.append(write_ragas_input(rows, "with-critic"))

    if args.ablation:
        rows_nc = run_suite(cases, use_critic=False, pause=args.pause)
        print_table(rows_nc)
        payload["without_critic"] = {
            "summary": summarise(rows_nc),
            "ragas": {} if args.no_ragas else run_metrics(rows_nc),
            "rows": [{k: v for k, v in r.items() if k != "contexts"} for r in rows_nc],
        }
        if not args.no_ragas:
            ragas_inputs.append(write_ragas_input(rows_nc, "without-critic"))

    tracing.flush()

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for label, key in (("with critic", "with_critic"), ("without critic", "without_critic")):
        if key in payload:
            print(f"\n{label}:")
            for k, v in payload[key]["summary"].items():
                print(f"  {k:<20} {v}")
            if payload[key].get("ragas"):
                for k, v in payload[key]["ragas"].items():
                    print(f"  ragas/{k:<14} {v}")

    report = write_report(payload)
    print(f"\nReport written to {report}")

    if any(p for p in ragas_inputs):
        print("\nMetrics above were computed by eval/ragas_metrics.py.")
        print("OPTIONAL — to cross-check with the literal RAGAS library "
              "(needs Python <=3.12):")
        print("  .\\.venv-eval\\Scripts\\python.exe eval\\score_ragas.py")


if __name__ == "__main__":
    main()
