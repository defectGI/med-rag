#!/usr/bin/env python
"""Evaluate the intent classifier against a golden set.

RUN THIS ON THE INFERENCE MACHINE — it calls the configured remote LLM endpoint
(.env: LLM_BASE_URL/MODEL/...). It must NOT be run on the GPU-less dev machine
(AGENTS.md GPU rule); that machine has no endpoint and this would fail anyway.

Usage:
    python scripts/eval_intent.py [GOLDEN_PATH] [--no-logprobs] [--show-pairs|--show-errors]

    --show-pairs   print every query / expected / predicted / confidence
    --show-errors  print only the mismatches (query / expected / predicted)

Defaults to data/eval/intent_golden.jsonl, falling back to the example set.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from medrag.api.retrieval.eval.metrics import intent_metrics
from medrag.api.retrieval.eval import load_intent_golden
from medrag.api.retrieval.modules.intent_classification import (
    LLMIntentClassifier,
    build_default_chat_model,
)


def _resolve_golden(argv: list[str]) -> Path:
    for arg in argv:
        if not arg.startswith("-"):
            return Path(arg)
    root = Path(__file__).resolve().parents[1]
    real = root / "data" / "eval" / "intent_golden.jsonl"
    if real.exists():
        return real
    return root / "data" / "eval" / "intent_golden.example.jsonl"


def _print_report(report) -> None:
    print(f"\nn={report.n}")
    print(f"accuracy : {report.accuracy:.3f}")
    print(f"macro F1 : {report.macro_f1:.3f}")
    print(f"micro F1 : {report.micro_f1:.3f}")
    print("\nper-class  (P / R / F1 / support):")
    for label, s in sorted(report.per_class.items()):
        print(f"  {label:<18} {s.precision:.2f} / {s.recall:.2f} / {s.f1:.2f} / {s.support}")


async def _classify_all(clf: LLMIntentClassifier, golden, concurrency: int = 8):
    sem = asyncio.Semaphore(concurrency)

    async def one(item):
        async with sem:
            return await clf.classify(item.query)

    return await asyncio.gather(*(one(item) for item in golden))


def main(argv: list[str]) -> int:
    golden_path = _resolve_golden(argv)
    request_logprobs = "--no-logprobs" not in argv
    show_pairs = "--show-pairs" in argv
    show_errors = "--show-errors" in argv

    golden = load_intent_golden(golden_path)
    print(f"golden: {golden_path}  ({len(golden)} items)")

    model = build_default_chat_model(request_logprobs=request_logprobs)
    clf = LLMIntentClassifier(model)

    results = asyncio.run(_classify_all(clf, golden))

    if show_pairs or show_errors:
        print("\nquery -> expected / predicted (confidence)")
        for item, res in zip(golden, results):
            correct = res.label.value == item.expected_intent.value
            if show_errors and correct:
                continue
            mark = "  " if correct else "X "
            print(
                f"{mark}{item.query!r} -> {item.expected_intent.value} / "
                f"{res.label.value}  (conf={res.confidence})"
            )

    report = intent_metrics(
        [it.expected_intent.value for it in golden],
        [r.label.value for r in results],
    )
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
