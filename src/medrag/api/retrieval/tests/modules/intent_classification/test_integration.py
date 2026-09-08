"""Integration test against the REAL inference endpoint.

Marked ``integration`` and skipped by default (pyproject: addopts
``-m "not integration"``). Run ONLY on the inference machine:

    pytest -m integration tests/modules/intent_classification/test_integration.py

Requires .env with LLM_BASE_URL/LLM_MODEL and the ``intent_classification``
extra installed (langchain-openai). Never runs on the GPU-less dev machine.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from medrag.api.retrieval.core import IntentLabel

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_MODEL"),
    reason="LLM endpoint not configured (.env)",
)
def test_real_endpoint_classifies():
    from medrag.api.retrieval.modules.intent_classification import (
        LLMIntentClassifier,
        build_default_chat_model,
    )

    clf = LLMIntentClassifier(build_default_chat_model())
    res = asyncio.run(clf.classify("de1000 fiyati ne kadar"))
    assert isinstance(res.label, IntentLabel)


@pytest.mark.skipif(
    not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_MODEL"),
    reason="LLM endpoint not configured (.env)",
)
def test_real_endpoint_eval_on_example_golden():
    """Full accuracy/F1 pass on the example golden set (sanity, not a gate)."""
    from pathlib import Path

    from medrag.api.retrieval.eval import evaluate_intent, load_intent_golden
    from medrag.api.retrieval.modules.intent_classification import (
        LLMIntentClassifier,
        build_default_chat_model,
    )

    # `data/` is data, not code: it was never packaged and lives under the
    # top-level `retrieval/` directory.
    root = Path(__file__).resolve().parents[7] / "retrieval"
    golden = load_intent_golden(root / "data" / "eval" / "intent_golden.example.jsonl")
    clf = LLMIntentClassifier(build_default_chat_model())
    report = asyncio.run(evaluate_intent(golden, clf.classify))
    print(f"\n[integration] accuracy={report.accuracy:.3f} macro_f1={report.macro_f1:.3f}")
    assert report.n == len(golden)
