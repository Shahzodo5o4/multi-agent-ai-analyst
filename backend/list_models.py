"""Ask your key which models it can actually use.

Model availability changes over time and differs per key — guessing a name
wastes quota on 404s. This asks the API directly and prints the names you can
paste into .env.

Run:  python list_models.py
"""

from __future__ import annotations

import sys

import config


def main() -> int:
    if not config.GEMINI_API_KEY:
        print("No GEMINI_API_KEY in .env — nothing to ask.")
        return 1

    import os

    from google import genai

    # Native Gemini surface of the class proxy. Key comes from the environment
    # via .env — never hard-coded, never printed.
    os.environ.setdefault("GEMINI_API_KEY", config.GEMINI_API_KEY)
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"base_url": config.GEMINI_BASE_URL},
    )

    chat, embed = [], []
    for m in client.models.list():
        actions = list(getattr(m, "supported_actions", None) or [])
        name = m.name.replace("models/", "")
        if "generateContent" in actions:
            chat.append(name)
        if "embedContent" in actions:
            embed.append(name)

    print("=" * 66)
    print("MODELS AVAILABLE VIA THE CLASS PROXY")
    print(f"  {config.GEMINI_BASE_URL}")
    print("=" * 66)

    print(f"\nCHAT ({len(chat)}) — pick one for LLM_MODEL:")
    for name in sorted(chat):
        print(f"  {name}")

    print(f"\nEMBEDDING ({len(embed)}) — pick one for EMBEDDING_MODEL:")
    for name in sorted(embed):
        print(f"  {name}")

    print("\n" + "=" * 66)
    print("This project uses only the three names the proxy exposes:")
    print("  LLM_MODEL=gemini-flash-lite      (agents, generation, judging)")
    print("  REASONING_MODEL=gemini-flash     (supervisor + critic)")
    print("  EMBEDDING_MODEL=gemini-embedding (document store + memory)")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())
