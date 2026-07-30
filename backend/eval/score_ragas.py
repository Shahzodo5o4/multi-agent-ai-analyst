"""F11, stage 2 — RAGAS metrics.

Runs in `.venv-eval`, NOT the main environment:

    .\\.venv-eval\\Scripts\\python.exe eval\\score_ragas.py

Why a second venv: every published RAGAS hard-imports
`langchain_community.chat_models.vertexai`, which langchain-community removed in
0.4. The main project runs the LangChain v1 stack, so RAGAS cannot be imported
there at all — and RAGAS 0.3+ additionally needs scikit-network, which has no
Python 3.14 wheel. Pinning the whole project back to LangChain 0.3 to satisfy a
metrics library would be the tail wagging the dog, so the older stack is
isolated in `.venv-eval` and reads the JSON that `evaluate.py` produced.

Scores every `ragas-input-*.json` in eval/results/ that has no scores yet, and
writes `ragas-scores-*.json` next to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import config  # noqa: E402

RESULTS_DIR = EVAL_DIR / "results"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_models():
    """Chat + embedding clients pointed at the class proxy, wrapped for RAGAS."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.GEMINI_API_KEY,
        model=config.LLM_MODEL,
        temperature=0.0,
    )
    emb = OpenAIEmbeddings(
        base_url=config.OPENAI_BASE_URL,
        api_key=config.GEMINI_API_KEY,
        model=config.EMBEDDING_MODEL,
        check_embedding_ctx_length=False,  # proxy expects strings, not token ids
    )
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


def score(path: Path) -> dict:
    """Run the four RAGAS metrics over one input file."""
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    rows = json.loads(path.read_text(encoding="utf-8"))
    print(f"\n{path.name}: scoring {len(rows)} rows "
          f"with {config.LLM_MODEL}...")

    llm, emb = build_models()

    # RAGAS 0.2 uses the SingleTurnSample schema; fall back to the legacy
    # HuggingFace Dataset column names if that import is unavailable.
    try:
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=r["question"],
                response=r["answer"],
                retrieved_contexts=r["contexts"],
                reference=r["ground_truth"],
            ) for r in rows
        ])
    except ImportError:
        from datasets import Dataset

        dataset = Dataset.from_list([{
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        } for r in rows])

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        raise_exceptions=False,
    )

    scores = {}
    for key, value in dict(result).items():
        try:
            scores[key] = round(float(value), 3)
        except (TypeError, ValueError):
            continue
    return scores


def main() -> int:
    if not config.GEMINI_API_KEY:
        print("No GEMINI_API_KEY in .env.")
        return 1

    inputs = sorted(RESULTS_DIR.glob("ragas-input-*.json")) if RESULTS_DIR.exists() else []
    if not inputs:
        print("No ragas-input-*.json found in eval/results/.")
        print("Run the generation stage first:  python eval\\evaluate.py")
        return 1

    # Only score inputs that don't already have a matching scores file.
    todo = [p for p in inputs
            if not (p.parent / p.name.replace("ragas-input-", "ragas-scores-")).exists()]
    if not todo:
        print(f"All {len(inputs)} input file(s) already scored. "
              "Delete a ragas-scores-*.json to re-score it.")
        return 0

    print("=" * 70)
    print("F11 STAGE 2 — RAGAS")
    print("=" * 70)

    for path in todo:
        try:
            scores = score(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue

        out = path.parent / path.name.replace("ragas-input-", "ragas-scores-")
        out.write_text(json.dumps(scores, indent=2), encoding="utf-8")

        print(f"\n  {'metric':<22} score")
        print("  " + "-" * 30)
        for name in METRIC_NAMES:
            if name in scores:
                print(f"  {name:<22} {scores[name]}")
        for name, value in scores.items():
            if name not in METRIC_NAMES:
                print(f"  {name:<22} {value}")
        print(f"\n  written to {out.name}")

    print("\n" + "=" * 70)
    print("Paste these into the Results table in README.md.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
