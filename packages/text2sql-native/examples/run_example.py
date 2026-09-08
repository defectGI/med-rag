"""Runnable end-to-end example.

Usage:
    1. Copy ``.env.example`` to ``.env``. Fill in a real provider and API key.
    2. From the repo root, run:
           python examples/run_example.py "top 5 best-selling products last month"

Makes real LLM calls (both stages), so it needs valid credentials. Prints the
SQL and metadata only. Never touches a database.
"""

from __future__ import annotations

import sys

from text2sql_native import Text2SQL
from text2sql_native.errors import Text2SQLError


def main(argv: list[str]) -> int:
    request = (
        " ".join(argv[1:])
        if len(argv) > 1
        else "get me the top 5 best-selling products last month"
    )

    try:
        engine = Text2SQL.from_config()
        result = engine.run(request)
    except Text2SQLError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("=" * 70)
    print("Request:", result.request)
    print("=" * 70)
    print("\n--- Linked schema (Stage 1) ---")
    print("Tables:", ", ".join(result.linked_schema.tables))
    print("Rationale:", result.linked_schema.rationale)
    print("\n--- Generated SQL (Stage 2) ---")
    print(result.sql)
    if result.explanation:
        print("\nExplanation:", result.explanation)
    print("\n--- Metadata ---")
    meta = result.metadata
    print(f"Linking:    {meta.linking.provider}/{meta.linking.model} "
          f"({meta.linking.usage.total_tokens} tokens, {meta.linking.latency_seconds:.2f}s)")
    print(f"Generation: {meta.generation.provider}/{meta.generation.model} "
          f"({meta.generation.usage.total_tokens} tokens, {meta.generation.latency_seconds:.2f}s)")
    print(f"Total:      {meta.total_usage.total_tokens} tokens, "
          f"{meta.total_latency_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
