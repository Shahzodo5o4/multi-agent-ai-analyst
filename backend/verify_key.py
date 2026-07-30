"""Check that your Gemini key actually works — before you burn time debugging.

Makes two tiny API calls (a chat completion and one embedding). Costs a
negligible amount of your free quota and tells you exactly what is wrong if
something fails.

Run:  python verify_key.py
"""

from __future__ import annotations

import sys

import config


def mask(key: str) -> str:
    """Never print a full key — not to the terminal, not into a screenshot."""
    return f"{key[:6]}…{key[-4:]} ({len(key)} chars)" if len(key) > 12 else "(too short)"


def main() -> int:
    print("=" * 66)
    print("GEMINI KEY CHECK")
    print("=" * 66)

    # --- 1. is a key loaded at all? ---
    if not config.GEMINI_API_KEY:
        print("\n[FAIL] No key loaded.")
        print(f"       Edit {config.BACKEND_DIR / '.env'} and set:")
        print("         GEMINI_API_KEY=<the key your class issued>")
        print("\n       If you already pasted it, check you did not leave the")
        print("       placeholder text or wrap the key in quotes.")
        return 1

    print(f"\n[ok]   key loaded from .env : {mask(config.GEMINI_API_KEY)}")
    print(f"[ok]   proxy                : {config.OPENAI_BASE_URL}")

    # --- 2. chat models: the high-volume one and the reasoning one ---
    for label, getter, name in (
        ("chat model", "get_llm", config.LLM_MODEL),
        ("reasoning model", "get_reasoning_llm", config.REASONING_MODEL),
    ):
        print(f"\nTesting {label:<16}: {name}")
        try:
            import llm as llm_module

            model = getattr(llm_module, getter)()
            reply = llm_module.text_of(model.invoke("Reply with exactly the word: OK"))
            print(f"[ok]   model replied      : {reply.strip()[:40]}")
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {type(exc).__name__}: {exc}")
            print(explain(exc))
            return 1

    # --- 3. embeddings ---
    print(f"\nTesting embeddings  : {config.EMBEDDING_MODEL}")
    try:
        from llm import get_embeddings

        vector = get_embeddings().embed_query("hello")
        print(f"[ok]   got a {len(vector)}-dimension vector")
        if len(vector) != config.EMBEDDING_DIM:
            print(f"[FAIL] but EMBEDDING_DIM in .env says {config.EMBEDDING_DIM}.")
            print(f"       Set EMBEDDING_DIM={len(vector)} and delete the")
            print(f"       {config.QDRANT_PATH.name}/ folder, then re-run ingest.py.")
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        print(explain(exc))
        return 1

    print("\n" + "=" * 66)
    print("ALL GOOD — your key works. Next:")
    print("  python ingest.py --test")
    print("  uvicorn main:app --reload --port 8000")
    print("=" * 66)
    return 0


def explain(exc: Exception) -> str:
    """Translate the common proxy errors into what to actually do."""
    text = str(exc).lower()
    if "401" in text or "unauthorized" in text or "invalid api key" in text:
        return ("       -> The proxy rejected the key. Re-copy the key your class\n"
                "          issued into GEMINI_API_KEY, with no quotes and no\n"
                "          trailing space.")
    if "quota" in text or "429" in text or "rate limit" in text:
        return ("       -> Rate limited by the proxy. Its quota is shared across\n"
                "          the whole class, so this may not be your usage. Wait a\n"
                "          minute and retry.")
    if "permission" in text or "403" in text:
        return ("       -> Key accepted but not authorised for this model. Check\n"
                "          with whoever issued it which models you may call.")
    if "not found" in text or "404" in text:
        return ("       -> The proxy does not recognise that model name. This\n"
                "          project must use only: gemini-flash-lite, gemini-flash,\n"
                "          gemini-embedding. Check LLM_MODEL / REASONING_MODEL /\n"
                "          EMBEDDING_MODEL in .env.")
    if "ssl" in text or "connection" in text or "timeout" in text or "502" in text:
        return (f"       -> Could not reach {config.PROXY_BASE}. The Space may be\n"
                "          asleep or down — open the URL in a browser to wake it,\n"
                "          then retry.")
    return "       -> Unexpected error. Re-run with the full traceback if unclear."


if __name__ == "__main__":
    sys.exit(main())

