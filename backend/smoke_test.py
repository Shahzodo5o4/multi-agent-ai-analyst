"""Offline self-check — runs WITHOUT any API key.

Verifies the parts that don't need Gemini: imports resolve against the installed
library versions, the SQL read-only guard holds, the code sandbox blocks what it
should, and the graph compiles with the right topology.

Run:  python smoke_test.py
"""

from __future__ import annotations

import sys
import traceback

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((PASS, name, str(detail)))
    except Exception as exc:  # noqa: BLE001
        results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
        if "-v" in sys.argv:
            traceback.print_exc()


# --- 1. imports resolve against the installed versions ----------------------
def t_imports():
    import langchain_core, langgraph  # noqa: F401
    from langchain_community.utilities import SQLDatabase  # noqa: F401
    from langchain_qdrant import QdrantVectorStore  # noqa: F401
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401
    from langgraph.graph import END, StateGraph  # noqa: F401
    # OpenAI-compatible classes: all Gemini traffic goes through the class proxy.
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: F401
    return f"langchain-core {langchain_core.__version__}, langgraph ok"


# --- 2. config + state (F1) -------------------------------------------------
def t_config():
    import config
    from state import evidence_block, new_state

    s = new_state("test question")
    assert set(s) >= {"question", "documents", "steps", "revisions", "visited"}
    assert evidence_block(s) == "(no evidence gathered)"
    s["sql_result"] = "SELECT 1 -> 1"
    assert "DATABASE RESULT" in evidence_block(s)
    # The proxy exposes exactly three model names; a typo here 404s at runtime.
    allowed = {"gemini-flash-lite", "gemini-flash", "gemini-embedding"}
    for label, name in (("LLM_MODEL", config.LLM_MODEL),
                        ("REASONING_MODEL", config.REASONING_MODEL),
                        ("EMBEDDING_MODEL", config.EMBEDDING_MODEL)):
        assert name in allowed, f"{label}={name!r} is not a proxy model name"
    return (f"llm={config.LLM_MODEL} reasoning={config.REASONING_MODEL} "
            f"steps<={config.MAX_GRAPH_STEPS}")


# --- 3. SQL read-only guard (F5) -------------------------------------------
def t_sql_guard():
    from agents import UnsafeSQL, clean_sql, guard_sql

    guard_sql("SELECT COUNT(*) FROM customers")
    guard_sql("WITH x AS (SELECT 1) SELECT * FROM x")

    blocked = [
        "DROP TABLE customers",
        "DELETE FROM customers WHERE 1=1",
        "SELECT 1; DROP TABLE customers",
        "UPDATE customers SET status='active'",
        "INSERT INTO customers VALUES (1)",
        "PRAGMA table_info(customers)",
        "ATTACH DATABASE 'evil.db' AS e",
    ]
    for sql in blocked:
        try:
            guard_sql(sql)
        except UnsafeSQL:
            continue
        raise AssertionError(f"guard let through: {sql}")

    assert clean_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert clean_sql("Here you go:\nSELECT a FROM b") == "SELECT a FROM b"
    return f"{len(blocked)} dangerous statements blocked, fences stripped"


# --- 4. code sandbox (F6) ---------------------------------------------------
def t_sandbox_blocks():
    import sandbox

    blocked = [
        "import os\nprint(os.listdir('.'))",
        "import subprocess",
        "open('x.txt','w').write('hi')",
        "print(eval('1+1'))",
        "print(().__class__.__bases__)",
        "import socket",
    ]
    for code in blocked:
        out = sandbox.run(code)
        assert out.startswith("BLOCKED BY SANDBOX"), f"not blocked: {code!r} -> {out[:60]}"
    return f"{len(blocked)} escape attempts blocked"


def t_sandbox_runs():
    import sandbox

    out = sandbox.run("import math\nprint(round(2500 * 1.045 ** 7, 2))")
    assert "3402.15" in out, out
    out2 = sandbox.run("print(sum(range(101)))")
    assert "5050" in out2, out2
    return "real computation works (compound interest + sum)"


def t_sandbox_timeout():
    import sandbox

    out = sandbox.run("while True: pass", timeout=3)
    assert out.startswith("TIMEOUT"), out
    return "infinite loop killed by the 3s cap"


def t_sandbox_allows_main_guard():
    import sandbox

    out = sandbox.run('if __name__ == "__main__":\n    print(6*7)')
    assert "42" in out, out
    return "__main__ guard allowed, other dunders still blocked"


# --- 5. database (F5) -------------------------------------------------------
def t_database():
    import sqlite3
    from pathlib import Path

    import config

    db_file = Path(config.SQLITE_URI.replace("sqlite:///", ""))
    if not db_file.exists():
        raise AssertionError(f"{db_file} missing — run: python data/seed_db.py")
    conn = sqlite3.connect(db_file)
    n = conn.execute(
        "SELECT COUNT(*) FROM churn_events "
        "WHERE churn_date BETWEEN '2024-07-01' AND '2024-09-30'"
    ).fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    conn.close()
    assert n == 25 and total == 240, f"unexpected data: {n}/{total}"
    return f"company.db ready — {total} customers, {n} churned in Q3 2024"


def t_sqldatabase_wrapper():
    from agents import get_db

    schema = get_db().get_table_info()
    for table in ("customers", "churn_events", "subscriptions", "support_tickets"):
        assert table in schema, f"{table} missing from schema"
    return "LangChain SQLDatabase reads all 4 tables"


# --- 6. graph topology (F9) -------------------------------------------------
def t_graph_compiles():
    from graph import build_graph

    app = build_graph(use_critic=True)
    nodes = set(app.get_graph().nodes)
    expected = {"memory", "supervisor", "retriever", "web", "data", "code",
                "generate", "critic"}
    missing = expected - nodes
    assert not missing, f"missing nodes: {missing}"
    return f"{len(expected)} nodes wired"


def t_graph_ablation():
    from graph import build_graph

    app = build_graph(use_critic=False)
    assert "critic" not in set(app.get_graph().nodes)
    return "critic-free graph compiles (F11 ablation)"


def t_termination_logic():
    """The critic and supervisor must both respect their budgets (F8, F9)."""
    import config
    from graph import critic, route_after_critic, supervisor

    over = {"question": "q", "steps": ["s"] * (config.MAX_GRAPH_STEPS + 1),
            "visited": ["data"], "documents": [], "revisions": 0}
    assert supervisor(over)["plan"] == "finish", "supervisor ignored the step budget"

    maxed = {"question": "q", "answer": "a", "revisions": config.MAX_REVISIONS,
             "documents": [], "steps": []}
    out = critic(maxed)
    assert not out.get("critique"), "critic ignored the revision limit"
    assert route_after_critic({"critique": ""}) == "finish"
    assert route_after_critic({"critique": "wrong number"}) == "revise"
    return "step budget + revision limit both enforced"


# --- 7. optional services degrade gracefully (F4, F12) ----------------------
def t_web_skips_without_key():
    import config
    from agents import web_agent
    from state import new_state

    if config.WEB_ENABLED:
        return "TAVILY key present — skip path not exercised"
    out = web_agent(new_state("anything"))
    assert "skipped" in out["steps"][0], out
    return "web agent skips cleanly with no key"


def t_tracing_optional():
    import tracing

    cfg = tracing.run_config(30)
    assert cfg["recursion_limit"] == 30
    return "no key -> no callbacks, config still valid"


CHECKS = [
    ("imports resolve", t_imports),
    ("F1  config + shared state", t_config),
    ("F5  SQL read-only guard", t_sql_guard),
    ("F6  sandbox blocks escapes", t_sandbox_blocks),
    ("F6  sandbox runs real code", t_sandbox_runs),
    ("F6  sandbox timeout cap", t_sandbox_timeout),
    ("F6  sandbox allows __main__", t_sandbox_allows_main_guard),
    ("F5  demo database seeded", t_database),
    ("F5  SQLDatabase wrapper", t_sqldatabase_wrapper),
    ("F9  graph compiles", t_graph_compiles),
    ("F11 ablation graph compiles", t_graph_ablation),
    ("F8/F9 termination guarantees", t_termination_logic),
    ("F4  web degrades without key", t_web_skips_without_key),
    ("F12 tracing optional", t_tracing_optional),
]

if __name__ == "__main__":
    print("=" * 74)
    print("OFFLINE SMOKE TEST — no API key required")
    print("=" * 74)
    for name, fn in CHECKS:
        check(name, fn)

    for status, name, detail in results:
        mark = "[ok]  " if status == PASS else "[FAIL]"
        print(f"{mark} {name:<32} {detail}")

    failed = sum(1 for s, _, _ in results if s == FAIL)
    print("-" * 74)
    print(f"{len(results) - failed}/{len(results)} passed")
    if failed:
        print("\nRe-run with -v for full tracebacks.")
    sys.exit(1 if failed else 0)
