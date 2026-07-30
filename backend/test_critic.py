"""F8 — prove the critic catches a deliberately wrong answer.

The rubric's "done when" for the critic is exactly this: a wrong answer is
caught and revised, a good one is approved. This runs both cases directly
against the critic node — 4 LLM calls instead of the ~24 a full graph run would
cost, so you can re-check the quality gate without burning the day's quota.

Run:  python test_critic.py
"""

from __future__ import annotations

import sys

import config
from graph import critic
from state import new_state

# Each case: (label, question, evidence, answer, should_be_rejected)
CASES = [
    (
        "wrong number vs SQL",
        "What percentage of all our customers churned in Q3 2024?",
        {"sql_result": "SQL:\nSELECT ... / (SELECT COUNT(*) FROM customers)\n\n"
                       "RESULT:\n[(10.416666666666666,)]"},
        "That is 58.1% of all our customers.",
        True,
    ),
    (
        "wrong denominator in the query itself",
        "What percentage of all our customers churned in Q3 2024?",
        {"sql_result": "SQL:\nSELECT CAST(SUM(CASE WHEN churn_date BETWEEN "
                       "'2024-07-01' AND '2024-09-30' THEN 1 ELSE 0 END) AS REAL) "
                       "* 100 / COUNT(*) FROM churn_events\n\n"
                       "RESULT:\n[(58.13953488372093,)]"},
        "That is 58.1% of all our customers.",
        True,
    ),
    (
        "hallucination not in the documents",
        "What is the refund policy?",
        {"documents": ["We offer a full refund within 30 days of the first "
                       "payment on any plan, no questions asked."]},
        "We offer a full refund within 90 days, and we also price-match any "
        "competitor.",
        True,
    ),
    (
        "non-answer to an answerable math question",
        "If I invest 2500 dollars at 4.5 percent compounded annually, what is "
        "it worth after 7 years?",
        {"documents": ["Refund policy: full refund within 30 days.",
                       "Support SLA: 8 business hours for billing."]},
        "The provided evidence does not contain information regarding "
        "investment returns or compound interest calculations.",
        True,  # the code agent could have answered this — refusing is a failure
    ),
    (
        "only half of a two-part question answered",
        "How many customers churned in Q3 2024, and what percentage of all "
        "customers is that?",
        {"sql_result": "SQL:\nSELECT ... / (SELECT COUNT(*) FROM customers)\n\n"
                       "RESULT:\n[(10.416666666666666,)]"},
        "The evidence does not state how many customers churned; it only gives "
        "the percentage, which is 10.4%.",
        True,  # the count was asked for and is obtainable — must not finish here
    ),
    (
        "correct and fully grounded",
        "How many customers churned in Q3 2024?",
        {"sql_result": "SQL:\nSELECT COUNT(*) FROM churn_events WHERE churn_date "
                       "BETWEEN '2024-07-01' AND '2024-09-30'\n\n"
                       "RESULT:\n[(25,)]"},
        "25 customers churned in Q3 2024.",
        False,
    ),
]


def main() -> int:
    print("=" * 72)
    print(f"F8 CRITIC CHECK — critic model: {config.CRITIC_MODEL}")
    print("=" * 72)

    failures = 0
    for label, question, evidence, answer, should_reject in CASES:
        state = new_state(question)
        state.update(evidence)
        state["answer"] = answer

        out = critic(state)
        rejected = bool(out.get("critique"))
        step = out.get("steps", [""])[0]

        if "verifier error" in step:
            print(f"[skip] {label}\n       critic could not run: {step[:120]}")
            failures += 1
            continue

        ok = rejected == should_reject
        want = "reject" if should_reject else "approve"
        got = "rejected" if rejected else "approved"
        print(f"{'[ok]  ' if ok else '[FAIL]'} {label}\n"
              f"       expected {want}, got {got}")
        if rejected:
            print(f"       reason: {out['critique'][:110]}")
        if not ok:
            failures += 1

    print("-" * 72)
    print(f"{len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        print("\nIf the critic approved a wrong answer, try a stronger")
        print("CRITIC_MODEL in .env — verifying SQL is harder than writing it.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
