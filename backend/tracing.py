"""F12 — Observability with Langfuse.

Every LLM call, tool call, token count and cost is traced, so you can open one
question in the Langfuse UI and see the full path:
supervisor → data → code → critic.

Entirely optional: with no Langfuse keys in `.env` this returns an empty
callback list and the system runs exactly as before.

Written against the Langfuse **v4** Python SDK (`langfuse>=3`), whose API differs
from v2 in three ways that each cause silent, invisible failures:

* `CallbackHandler` no longer exposes `flush()` or `.client`. Flushing goes
  through the singleton: `get_client().flush()`. Miss this and a short-lived
  script exits before its batch is sent — the run looks fine and no trace ever
  appears.
* The client reads credentials from the environment when it is first touched,
  and caches the result for the whole process. If anything builds a client
  before the keys are in `os.environ`, you get a permanently disabled client
  that logs one auth warning and then silently drops everything.
* Trace name, session and tags are set with `propagate_attributes(...)`, not by
  passing them to the handler.

v2 import paths are still handled so an older pinned install keeps working.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator, List

import config

_WARNED = False

# --- Credentials must land in the environment before ANY client is built ------
# `config` has already loaded `.env` by the time this module is imported, so
# this runs early enough — but only if `tracing` is imported before something
# else touches Langfuse. Keep this at module level for that reason.
if config.LANGFUSE_ENABLED:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", config.LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", config.LANGFUSE_SECRET_KEY)
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)


@lru_cache(maxsize=1)
def _handler():
    """Build the Langfuse callback handler, or None if unavailable."""
    global _WARNED
    if not config.LANGFUSE_ENABLED:
        return None

    try:
        # Langfuse v3/v4 — credentials come from the environment, set above.
        from langfuse.langchain import CallbackHandler  # type: ignore

        return CallbackHandler()
    except ImportError:
        pass

    try:
        # Langfuse v2
        from langfuse.callback import CallbackHandler  # type: ignore

        return CallbackHandler(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    except Exception as exc:  # noqa: BLE001
        if not _WARNED:
            print(f"[tracing] Langfuse disabled ({exc})")
            _WARNED = True
        return None


def callbacks() -> List:
    """Pass into `graph.invoke(..., config={'callbacks': callbacks()})`."""
    handler = _handler()
    return [handler] if handler else []


def run_config(recursion_limit: int, session_id: str = "default", **metadata) -> dict:
    """The config dict every graph invocation uses: recursion cap + tracing.

    The `langfuse_*` metadata keys are the documented fallback for setting trace
    attributes from a LangChain config. `traced_run()` below is the preferred
    route — it also names the trace and sets its input/output — but these keys
    stay so a bare `run_config()` call is still attributed correctly.
    """
    cfg: dict = {"recursion_limit": recursion_limit}
    handlers = callbacks()
    if handlers:
        cfg["callbacks"] = handlers
        cfg["metadata"] = {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["multi-agent-analyst"],
            **metadata,
        }
    return cfg


class _Trace:
    """Handle yielded by `traced_run()`. `.answered()` records the final answer."""

    def __init__(self, span=None) -> None:
        self._span = span
        self.trace_id: str | None = getattr(span, "trace_id", None)

    def answered(self, answer: str) -> None:
        """Set the trace's root output to the answer text."""
        self._output = answer

    _output: str = ""

    @property
    def url(self) -> str | None:
        """Direct link to this trace, for pasting into a writeup."""
        if not self.trace_id:
            return None
        return f"{config.LANGFUSE_HOST.rstrip('/')}/trace/{self.trace_id}"


@contextmanager
def traced_run(question: str, session_id: str = "default", **attributes) -> Iterator[_Trace]:
    """Wrap one graph invocation as a single well-formed trace.

    Without this the trace is named after whatever LangGraph calls its root
    chain, and its root input/output is the entire `AgentState` dict — every
    retrieved chunk, every SQL row, the full step list. That is unreadable in
    the UI and useless for trace-level evaluation. Here the trace gets:

    * a stable, low-cardinality name (`answer-question`) so traces are filterable
    * `session_id`, so a follow-up question groups with the turn it refers back
      to — which is the whole point of the F10 memory feature
    * root input/output of just the question and the final answer

    A no-op when Langfuse is disabled, so callers need no conditionals.
    """
    if not config.LANGFUSE_ENABLED:
        yield _Trace()
        return

    try:
        from langfuse import get_client, propagate_attributes  # type: ignore
    except ImportError:
        # v2, or an install without propagate_attributes — tracing still works
        # through run_config(), just without the root naming/IO.
        yield _Trace()
        return

    client = get_client()
    tags = ["multi-agent-analyst"]
    if "use_critic" in attributes:
        # Lets you compare the F11 ablation arms side by side in the UI.
        tags.append("critic-on" if attributes["use_critic"] else "critic-off")

    with propagate_attributes(
        trace_name="answer-question",
        session_id=session_id,
        tags=tags,
        metadata=attributes or None,
    ):
        with client.start_as_current_observation(
            as_type="span", name="answer-question"
        ) as span:
            handle = _Trace(span)
            try:
                yield handle
            finally:
                span.set_trace_io(
                    input={"question": question},
                    output={"answer": handle._output} if handle._output else None,
                )


def flush() -> None:
    """Langfuse batches events; flush before the process exits or a short script
    can finish before its trace is sent.

    v4's handler has no `flush()` and no `.client` — the only working route is
    the singleton client. Getting this wrong is invisible: the run succeeds and
    the trace simply never arrives.
    """
    if not config.LANGFUSE_ENABLED:
        return

    try:
        from langfuse import get_client  # type: ignore

        get_client().flush()
        return
    except ImportError:
        pass

    handler = _handler()  # v2 fallback
    if handler is None:
        return
    try:
        if hasattr(handler, "flush"):
            handler.flush()
        elif hasattr(handler, "client") and hasattr(handler.client, "flush"):
            handler.client.flush()
    except Exception:  # noqa: BLE001
        pass
