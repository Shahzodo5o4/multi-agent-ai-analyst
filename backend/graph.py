"""F7, F8, F9 — the supervisor, the critic, and the graph that wires them.

The shape:

    memory → supervisor ─┬→ retriever ─┐
                         ├→ web ───────┤
                         ├→ data ──────┼→ back to supervisor
                         ├→ code ──────┘
                         └→ generate → critic ─┬→ END (approved)
                                               └→ supervisor (revise)

Termination is guaranteed three ways (the guide's "watch out" for F9):
  1. the supervisor is told which agents already ran and not to repeat them,
  2. a step budget forces "finish" after MAX_GRAPH_STEPS,
  3. the critic may only bounce an answer back MAX_REVISIONS times,
  4. LangGraph's own recursion_limit is a final backstop.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

import config
from agents import SPECIALISTS, code_agent, data_agent, retriever_agent, web_agent
from llm import get_llm, get_reasoning_llm, text_of
from memory import memory_node
from state import AgentState, evidence_block

# ---------------------------------------------------------------------------
# F7 — Supervisor / Router
# ---------------------------------------------------------------------------


class Route(BaseModel):
    """Structured output — forcing a schema is what makes routing reliable.
    Free-text routing ('I think we should use the data agent') is unparseable."""

    next: Literal["retriever", "web", "data", "code", "finish"] = Field(
        description="Which specialist runs next, or 'finish' if there is enough evidence."
    )
    reason: str = Field(description="One short sentence explaining the choice.")


SUPERVISOR_PROMPT = """You are the supervisor of a team of specialist agents.
Read the question and the evidence gathered so far, then choose the NEXT agent.

Your team:
- retriever : searches THIS COMPANY'S internal documents (handbook, postmortems).
              Use for policy, process, definitions — and for any question asking
              WHY something happened at this company, what CAUSED it, or for an
              EXPLANATION of it. The database stores one-word reason labels; only
              the documents contain the analysis, and label counts do not answer
              "why". Do NOT use it for general knowledge or for arithmetic — the
              internal documents contain neither.
- web       : live internet search. Use for current events, public facts, or
              anything the internal documents would not contain.
- data      : writes and runs SQL against the company database. Use for counts,
              sums, averages, rankings — any question about the actual data.
- code      : writes and runs Python. Use for arithmetic, growth rates,
              projections, or transforming numbers the data agent returned.
- finish    : enough evidence has been gathered; go write the answer.

QUESTION: {question}

AGENTS ALREADY RUN THIS TURN: {visited}
EVIDENCE SO FAR:
{evidence}
{critique}
Rules:
- Do NOT pick an agent that already ran unless the critic asked for it again.
- A pure arithmetic or finance question with no reference to this company's data
  (compound interest, CAGR, percentage of a stated number) goes straight to
  `code`. Searching documents for it wastes a step and finds nothing.
- A multi-part question needs multiple agents: get the number from `data`
  first, then use `code` to compute on it, then finish.
- Before choosing 'finish', check EVERY part of the question is covered. If the
  question asks for a count AND a percentage, the evidence must contain both —
  route again rather than finishing with half an answer.
- Steps used: {used}/{budget}. If you are near the budget, choose 'finish'.
"""


def supervisor(state: AgentState) -> dict:
    """Decide which specialist runs next, or that the answer is ready."""
    used = len(state.get("steps", []))
    visited = state.get("visited", [])

    # Hard budget: stop routing and write the answer with whatever we have.
    if used >= config.MAX_GRAPH_STEPS:
        return {
            "plan": "finish",
            "steps": [f"supervisor → finish (step budget {config.MAX_GRAPH_STEPS} reached)"],
        }

    critique = ""
    if state.get("critique"):
        critique = (
            f"\nTHE CRITIC REJECTED THE PREVIOUS ANSWER: {state['critique']}\n"
            "Pick the agent that can close that specific gap.\n"
        )

    prompt = SUPERVISOR_PROMPT.format(
        question=state["question"],
        visited=", ".join(visited) if visited else "(none yet)",
        evidence=evidence_block(state),
        critique=critique,
        used=used,
        budget=config.MAX_GRAPH_STEPS,
    )

    try:
        # Routing uses the stronger model (gemini-flash): a lite model sent a
        # pure-arithmetic question to the document retriever. See error #5.
        decision = (
            get_reasoning_llm(name="route-question")
            .with_structured_output(Route)
            .invoke(prompt)
        )
        nxt, reason = decision.next, decision.reason
    except Exception as exc:  # noqa: BLE001
        # If structured output fails, don't crash the graph — fall back to a
        # sensible default so the user still gets an answer.
        nxt = "finish" if visited else "retriever"
        reason = f"router error ({exc}); falling back to {nxt}"

    return {"plan": nxt, "steps": [f"supervisor → {nxt}"], "critique": ""}


def route_from_supervisor(state: AgentState) -> str:
    """Conditional edge: read the supervisor's decision off the state."""
    return state.get("plan", "finish")


# ---------------------------------------------------------------------------
# Answer generation — turns evidence into prose (part of F9)
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """Answer the question using ONLY the evidence below.

QUESTION: {question}

EVIDENCE:
{evidence}
{critique}
Rules:
- Ground every claim in the evidence. Do not add outside knowledge.
- If the evidence contains a number from the database or from code, use that
  exact number — never recompute it yourself.
- Round for readability: money to 2 decimals ($3,402.15), percentages to 1
  decimal (10.4%), counts as whole numbers. Never print a raw float like
  10.416666666666666 — but never change the underlying value.
- If the evidence does not answer the question, say so plainly.
- Be concise and direct. No preamble.
"""


def generate(state: AgentState) -> dict:
    """Draft the answer from the gathered evidence."""
    critique = ""
    if state.get("critique"):
        critique = (
            f"\nYour previous answer was rejected: {state['critique']}\n"
            "Write a corrected answer that fixes exactly that problem.\n"
        )

    prompt = GENERATE_PROMPT.format(
        question=state["question"],
        evidence=evidence_block(state),
        critique=critique,
    )
    answer = text_of(
        get_llm(temperature=0.2, name="compose-answer").invoke(prompt)
    ).strip()
    return {"answer": answer, "steps": ["generate"]}


# ---------------------------------------------------------------------------
# F8 — Critic / Verifier
# ---------------------------------------------------------------------------


class Verdict(BaseModel):
    """The quality gate's structured output."""

    ok: bool = Field(description="True only if the answer is correct AND fully supported by the evidence.")
    reason: str = Field(description="If not ok, state precisely what is wrong or unsupported.")


CRITIC_PROMPT = """You are a strict verifier. Judge the answer against the evidence.

QUESTION: {question}

EVIDENCE:
{evidence}

PROPOSED ANSWER:
{answer}

Check the EVIDENCE ITSELF, not just whether the answer matches it. An answer
that faithfully reports a query which asked the wrong question is still wrong.

If the evidence contains SQL, read the query and verify:
- it selects from the right table(s) for what was asked,
- its filters match the question (date range, status, segment),
- for a ratio, the DENOMINATOR is what the question asked for. "What percentage
  of all customers" means dividing by the total number of customers — not by
  the number of churned customers, or by any other subset.

Set ok = false if ANY of these is true:
- a claim is not supported by the evidence (hallucination),
- a number disagrees with the database or code output,
- the query answers a subtly different question than the one asked,
- the question has multiple parts and one is unanswered,
- the answer dodges the question.

Set ok = true if the answer is correct and every claim traces to the evidence.

"The evidence does not cover this" is ONLY acceptable when no agent on the team
could have obtained the information. It is NOT acceptable when:
- the question is arithmetic and the code agent was never run,
- the question is about company data and the data agent was never run,
- the question is about policy or process and the retriever was never run.
In those cases set ok = false and say which agent should have run. A confident
non-answer to an answerable question is a failure, not honesty.
"""


def critic(state: AgentState) -> dict:
    """Approve the answer, or send it back with a reason."""
    # Revision budget exhausted — approve to guarantee termination (F9).
    if state.get("revisions", 0) >= config.MAX_REVISIONS:
        return {
            "steps": [f"critic → accepted (revision limit {config.MAX_REVISIONS} reached)"],
            "critique": "",
        }

    prompt = CRITIC_PROMPT.format(
        question=state["question"],
        # Past turns are excluded: the critic must judge against evidence
        # gathered THIS turn, not against its own earlier answers.
        evidence=evidence_block(state, include_memory=False),
        answer=state.get("answer", ""),
    )
    try:
        verdict = (
            get_reasoning_llm(name="verify-answer")
            .with_structured_output(Verdict)
            .invoke(prompt)
        )
    except Exception as exc:  # noqa: BLE001
        return {"steps": [f"critic → accepted (verifier error: {exc})"], "critique": ""}

    if verdict.ok:
        return {"steps": ["critic → approved ✅"], "critique": ""}

    return {
        "revisions": state.get("revisions", 0) + 1,
        "critique": verdict.reason,
        "steps": [f"critic → REVISE: {verdict.reason}"],
    }


def route_after_critic(state: AgentState) -> str:
    """Conditional edge: finish, or loop back to the supervisor to gather more."""
    if state.get("critique"):
        return "revise"
    return "finish"


# ---------------------------------------------------------------------------
# F9 — Wire the graph
# ---------------------------------------------------------------------------


def build_graph(use_critic: bool = True):
    """Compile the supervisor graph.

    `use_critic=False` compiles the same graph with the verifier removed. That
    ablation is what produces the rubric's required "with vs without the critic"
    metrics table (F11) — it isolates how much quality the critic actually adds.
    """
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)

    g.add_node("memory", memory_node)
    g.add_node("supervisor", supervisor)
    g.add_node("retriever", retriever_agent)
    g.add_node("web", web_agent)
    g.add_node("data", data_agent)
    g.add_node("code", code_agent)
    g.add_node("generate", generate)
    if use_critic:
        g.add_node("critic", critic)

    g.set_entry_point("memory")
    g.add_edge("memory", "supervisor")

    # The supervisor fans out to whichever specialist it chose.
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retriever": "retriever",
            "web": "web",
            "data": "data",
            "code": "code",
            "finish": "generate",
        },
    )

    # Every specialist hands control straight back to the supervisor.
    for name in SPECIALISTS:
        g.add_edge(name, "supervisor")

    if use_critic:
        g.add_edge("generate", "critic")
        g.add_conditional_edges(
            "critic",
            route_after_critic,
            {"finish": END, "revise": "supervisor"},
        )
    else:
        g.add_edge("generate", END)

    return g.compile()


_APPS: dict[bool, object] = {}


def get_app(use_critic: bool = True):
    """Cached compiled graph, one per critic setting."""
    if use_critic not in _APPS:
        _APPS[use_critic] = build_graph(use_critic=use_critic)
    return _APPS[use_critic]


def recursion_limit() -> int:
    """LangGraph's backstop against a runaway loop.

    Worst case per turn: MAX_GRAPH_STEPS supervisor+agent pairs, plus a
    generate/critic round for every allowed revision, plus headroom.
    """
    return config.MAX_GRAPH_STEPS * 2 + config.MAX_REVISIONS * 3 + 6


def run(question: str, use_critic: bool = True, store_memory: bool = True) -> AgentState:
    """Synchronous single-shot run. Used by the eval harness (F11) and CLI tests.

    `store_memory` writes the completed turn back into long-term memory (F10) so
    a follow-up question can recall it. The eval harness passes False — letting
    graded answers leak into memory would contaminate later questions in the run.
    """
    import memory as memory_module
    import tracing
    from state import new_state

    # `traced_run` names the Langfuse trace and sets its root input/output to the
    # question and answer. Without it the trace root carries the whole AgentState
    # — every chunk, row and step — which is unreadable in the UI.
    with tracing.traced_run(question, use_critic=use_critic) as trace:
        final = get_app(use_critic).invoke(
            new_state(question), config=tracing.run_config(recursion_limit())
        )
        trace.answered(final.get("answer", ""))
    if store_memory and final.get("answer"):
        memory_module.remember(question, final["answer"])
    return final


# --- Manual end-to-end check: `python graph.py "your question"` -------------
if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or (
        "How many customers churned in Q3 2024, and what percentage of our "
        "total customers is that?"
    )
    print(f"\nQ: {q}\n" + "-" * 70)
    final = run(q)
    print("\nAGENT PATH:")
    for i, step in enumerate(final.get("steps", []), 1):
        print(f"  {i:2}. {step}")
    print(f"\nREVISIONS: {final.get('revisions', 0)}")
    print("\nANSWER:\n" + final.get("answer", "(none)"))
