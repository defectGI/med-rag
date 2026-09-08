"""Golden-set loading + end-to-end runner tests (with fakes, no inference)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from medrag.api.retrieval.core import IntentLabel, IntentResult, RetrievalResult
from medrag.api.retrieval.eval import (
    evaluate_intent,
    evaluate_retrieval,
    load_intent_golden,
    load_retrieval_golden,
)
from medrag.api.retrieval.eval.golden import (
    _iter_jsonl,  # internal helper under test
)

# `data/` is data, not code: it was never packaged with the module and lives
# under the top-level `retrieval/` directory. parents[6] = repo root.
_RETRIEVAL_DATA_ROOT = Path(__file__).resolve().parents[6] / "retrieval"
EVAL_DIR = _RETRIEVAL_DATA_ROOT / "data" / "eval"


def test_example_intent_golden_loads_and_has_52_items():
    items = load_intent_golden(EVAL_DIR / "intent_golden.example.jsonl")
    assert len(items) == 52
    # every label is a valid v1 intent, all 9 classes represented
    labels = {it.expected_intent for it in items}
    assert labels == set(IntentLabel)


def test_example_retrieval_golden_loads():
    items = load_retrieval_golden(EVAL_DIR / "retrieval_golden.example.jsonl")
    assert len(items) >= 1
    assert all(it.relevant_ids for it in items)


def test_comments_and_blanks_are_skipped(tmp_path):
    p = tmp_path / "g.jsonl"
    p.write_text(
        "# header\n\n"
        '{"query": "q1", "expected_intent": "product_fact"}\n'
        '{"query": "q2", "expected_intent": "out_of_scope"}\n',
        encoding="utf-8",
    )
    assert len(list(_iter_jsonl(p))) == 2
    assert len(load_intent_golden(p)) == 2


def test_bad_label_raises_with_location(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"query": "q", "expected_intent": "not_a_real_label"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:1"):
        load_intent_golden(p)


def test_evaluate_intent_with_fake_classifier():
    golden = load_intent_golden(EVAL_DIR / "intent_golden.example.jsonl")

    # Oracle classifier: returns the true label. Exercises both async plumbing
    # and the IntentResult return path.
    truth = {it.query: it.expected_intent for it in golden}

    async def oracle(query: str) -> IntentResult:
        return IntentResult(label=truth[query])

    rep = asyncio.run(evaluate_intent(golden, oracle))
    assert rep.accuracy == 1.0
    assert rep.n == 52


def test_evaluate_intent_accepts_sync_classifier_returning_str():
    golden = load_intent_golden(EVAL_DIR / "intent_golden.example.jsonl")

    def always_oos(query: str) -> str:
        return "out_of_scope"

    rep = asyncio.run(evaluate_intent(golden, always_oos))
    # only the 5 genuinely out_of_scope items are correct
    assert rep.n == 52
    assert round(rep.accuracy, 2) == 0.10


def test_evaluate_retrieval_with_fake_retriever():
    golden = load_retrieval_golden(EVAL_DIR / "retrieval_golden.example.jsonl")

    class PerfectRetriever:
        """Returns each query's known relevant ids first."""

        def __init__(self, gold):
            self._by_query = {g.query: g.relevant_ids for g in gold}

        async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
            ids = self._by_query.get(query, [])
            return [RetrievalResult(id=i, score=1.0) for i in ids][:k]

    rep = asyncio.run(evaluate_retrieval(golden, PerfectRetriever(golden), ks=(1, 3)))
    assert rep.recall_at_k[3] == 1.0
    assert rep.mrr == 1.0
