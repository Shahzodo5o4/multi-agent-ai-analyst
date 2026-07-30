"""Find out which chat models this key is actually allowed to call.

Traffic goes through the class LiteLLM proxy, and a proxy key is scoped to a
LIST OF MODEL NAMES. Asking for one outside that list fails with a 403
`key_model_access_denied` — not a quota error — and the message names the models
you may use. Keys differ, so never trust a hard-coded name.

This matters because REASONING_MODEL (supervisor + critic) is meant to run on
`gemini-flash`: a lite model approves SQL whose denominator is wrong. If this
script reports `gemini-flash` as denied, that critic case will fail and the
README's critic caveat explains why.

Note: quota on the proxy is a SHARED class resource, not Google's old
per-model-per-day free tier — switching model names does not buy a fresh
allowance, and there is no midnight reset to wait for.

Run:  python pick_model.py
"""

from __future__ import annotations

import sys

import config

# The only three names the class proxy exposes.
CANDIDATES = [
    "gemini-flash-lite",
    "gemini-flash",
]


def probe(model: str) -> tuple[str, str]:
    """One minimal call. Returns (status, detail)."""
    import os

    from google import genai

    os.environ.setdefault("GEMINI_API_KEY", config.GEMINI_API_KEY)
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"base_url": config.GEMINI_BASE_URL},
    )
    try:
        client.models.generate_content(model=model, contents="Say OK")
        return "OK", "usable now"
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text:
            limit = ""
            for part in text.split():
                if part.startswith("limit:"):
                    limit = part
            # Pull the daily quota value out of the error when present.
            if "quotaValue" in text:
                try:
                    limit = "daily limit " + text.split("'quotaValue': '")[1].split("'")[0]
                except Exception:  # noqa: BLE001
                    pass
            return "QUOTA", f"exhausted today ({limit or 'see error'})"
        if "404" in text or "NOT_FOUND" in text:
            return "N/A", "not available to this key"
        if "403" in text or "PERMISSION" in text:
            return "N/A", "no permission"
        return "ERR", text[:70]


def main() -> int:
    if not config.GEMINI_API_KEY:
        print("No GEMINI_API_KEY in .env.")
        return 1

    print("=" * 70)
    print("PROBING PROXY MODELS (one tiny request each)")
    print(f"  {config.GEMINI_BASE_URL}")
    print("=" * 70)
    print(f"configured: LLM_MODEL={config.LLM_MODEL}  "
          f"REASONING_MODEL={config.REASONING_MODEL}\n")

    usable = []
    for model in CANDIDATES:
        status, detail = probe(model)
        mark = {"OK": "[ok]  ", "QUOTA": "[full]", "N/A": "[--]  ", "ERR": "[err] "}[status]
        print(f"{mark} {model:<28} {detail}")
        if status == "OK":
            usable.append(model)

    print("\n" + "=" * 70)
    if len(usable) == len(CANDIDATES):
        print("Both models respond. Nothing to change.")
    elif usable:
        print(f"Only {', '.join(usable)} responded.")
        print("The proxy's quota is shared across the class, so a limit here is")
        print("not yours alone — wait and retry rather than switching models.")
    else:
        print("Neither model responded. Check that the proxy is up and that")
        print("GEMINI_API_KEY in .env is the key your class issued.")
    print("=" * 70)
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
