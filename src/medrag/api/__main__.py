"""CLI entry point: `python -m chatbot "arveles dozu" --intent medical_fact`.

There is no `AnsweringModel` yet -- this command does NOT call the answering
model; it only prints the context gathered by the REAL retrievers that `Router`
builds from `.env` (top_n: Qdrant, sql_topn: db_query). The point is to see the
retrieval layer actually working end to end once `.env` is filled in (same
principle as the read-only search example in vectorize/vectorize/query.py:
completely separate from the write side, queries only).

The intent is NOT classified here; it is supplied manually via `--intent` -- a
deliberate simplification so top_n/db_query can be tested on their own without
LLM_* configured (without intent_classification).
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from medrag.api.retrieval.core import IntentLabel


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m chatbot")
    parser.add_argument("query", help="question to test")
    parser.add_argument(
        "--intent",
        required=True,
        choices=[label.value for label in IntentLabel],
        help="which intent to route as if classified (see config/default.toml [routing])",
    )
    parser.add_argument("--k", type=int, default=None, help="override for result count (default from config)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    from medrag.api.factory import build_router_from_env

    router = build_router_from_env()
    flow = router.flow_for(IntentLabel(args.intent))
    context = asyncio.run(flow.run(args.query))

    print(f"intent: {args.intent}")
    print(f"strategy_key: {args.intent}")  # strategy_key is now derived 1:1 from the intent
    print(f"{len(context.results)} sonuc:\n")
    for i, r in enumerate(context.results, 1):
        print(f"#{i}  id={r.id}  score={r.score:.4f}")
        if r.text:
            print(f"    text: {r.text[:200]!r}")
        if r.metadata:
            extra = {k: v for k, v in r.metadata.items() if k not in ("text", "score")}
            if extra:
                print(f"    metadata: {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
