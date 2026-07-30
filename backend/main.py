"""FastAPI server — the bridge between the graph and the HTML frontend (F13).

Endpoints
    GET  /health        config + knowledge-base status (the UI polls this)
    POST /ask           non-streaming, returns the final state as JSON
    GET  /ask/stream    Server-Sent Events — emits each agent step AS IT HAPPENS

/ask/stream is what makes F13 convincing: the browser watches
supervisor → data → code → critic arrive one at a time, live.

Run:  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import memory
import tracing
from graph import get_app, recursion_limit
from state import new_state

app = FastAPI(title="Multi-Agent AI Analyst", version="1.0")

# The frontend is a static index.html opened from the filesystem or served from
# Vercel, so it is always a different origin from this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Everything the UI needs to render an honest status bar."""
    import vectorstore

    chunks = vectorstore.count(config.QDRANT_COLLECTION)
    return {
        "status": "ok",
        "knowledge_base_chunks": chunks,
        "knowledge_base_ready": chunks > 0,
        **config.summary(),
    }


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    """Run the graph to completion and return the final state."""
    graph = get_app()
    cfg = tracing.run_config(recursion_limit(), session_id=req.session_id)

    def run_traced() -> dict:
        # The Langfuse span must be opened on the SAME thread that runs the graph:
        # trace context is thread-local, so a span started on the event loop would
        # not become the parent of spans created in this worker thread.
        with tracing.traced_run(req.question, session_id=req.session_id) as trace:
            result = graph.invoke(new_state(req.question), cfg)
            trace.answered(result.get("answer", ""))
            return result

    final = await asyncio.to_thread(run_traced)

    answer = final.get("answer", "")
    await asyncio.to_thread(memory.remember, req.question, answer, req.session_id)
    tracing.flush()

    return {
        "question": req.question,
        "answer": answer,
        "steps": final.get("steps", []),
        "sources": final.get("sources", []),
        "revisions": final.get("revisions", 0),
        "sql_result": final.get("sql_result"),
        "code_result": final.get("code_result"),
    }


# ---------------------------------------------------------------------------
# Streaming (SSE) — F13
# ---------------------------------------------------------------------------


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _event_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    """Yield one SSE message per node completion.

    LangGraph's sync `.stream()` blocks, so it runs on a worker thread and pushes
    into an asyncio queue that this coroutine drains. That keeps the event loop
    free and the stream genuinely live.
    """
    graph = get_app()
    cfg = tracing.run_config(recursion_limit(), session_id=session_id)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def produce() -> None:
        # Open the Langfuse span here, not in the caller: trace context is
        # thread-local, so a span started on the event loop would not parent the
        # spans this thread creates — the trace would come out named after
        # LangGraph's root chain with the raw state as its input.
        final_answer = ""
        try:
            with tracing.traced_run(question, session_id=session_id) as trace:
                for update in graph.stream(new_state(question), cfg,
                                           stream_mode="updates"):
                    for node, delta in update.items():
                        if delta.get("answer"):
                            final_answer = delta["answer"]
                        loop.call_soon_threadsafe(
                            queue.put_nowait, ("node", node, delta))
                trace.answered(final_answer)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc), {}))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", "", {}))

    task = asyncio.get_running_loop().run_in_executor(None, produce)

    # Accumulate the pieces the UI needs at the end.
    answer, sources, revisions = "", [], 0
    seen_steps: list[str] = []

    yield _sse("start", {"question": question})

    while True:
        kind, name, delta = await queue.get()

        if kind == "done":
            break

        if kind == "error":
            yield _sse("error", {"message": name})
            break

        # Emit every new step this node produced.
        for step in delta.get("steps", []) or []:
            seen_steps.append(step)
            yield _sse("step", {"node": name, "step": step, "index": len(seen_steps)})

        if delta.get("sources"):
            sources.extend(delta["sources"])
            yield _sse("sources", {"sources": delta["sources"]})

        if delta.get("sql_result"):
            yield _sse("tool", {"tool": "sql", "output": delta["sql_result"]})

        if delta.get("code_result"):
            yield _sse("tool", {"tool": "code", "output": delta["code_result"]})

        if delta.get("answer"):
            answer = delta["answer"]
            yield _sse("draft", {"answer": answer})

        if delta.get("revisions") is not None and name == "critic":
            revisions = delta.get("revisions", revisions)

    await task

    if answer:
        await asyncio.to_thread(memory.remember, question, answer, session_id)
    tracing.flush()

    yield _sse("final", {
        "answer": answer,
        "steps": seen_steps,
        "sources": sources,
        "revisions": revisions,
    })


@app.get("/ask/stream")
async def ask_stream(question: str, session_id: str = "default") -> StreamingResponse:
    """SSE endpoint consumed by the frontend's EventSource."""
    return StreamingResponse(
        _event_stream(question, session_id or str(uuid.uuid4())),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx/Render from buffering the stream
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
