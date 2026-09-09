"""Core interface tests: result shape, async retriever protocol, run_sync."""

from __future__ import annotations

import asyncio

import pytest

from medrag.api.retrieval.core import (
    IntentLabel,
    IntentResult,
    RetrievalResult,
    Retriever,
    run_sync,
)


class FakeRetriever:
    """Minimal async retriever — no network, structurally a Retriever."""

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        return [
            RetrievalResult(id=f"hit-{i}", score=1.0 - i * 0.1, text=f"{query}#{i}")
            for i in range(k)
        ]


def test_fake_retriever_satisfies_protocol():
    r = FakeRetriever()
    assert isinstance(r, Retriever)  # runtime_checkable Protocol


def test_run_sync_from_sync_context():
    results = run_sync(FakeRetriever(), "de1000", k=3)
    assert [x.id for x in results] == ["hit-0", "hit-1", "hit-2"]
    assert results[0].score == 1.0


def test_run_sync_rejects_running_loop():
    async def inner():
        with pytest.raises(RuntimeError):
            run_sync(FakeRetriever(), "q", k=1)

    asyncio.run(inner())


def test_intent_result_accepts_label():
    res = IntentResult(label=IntentLabel.MEDICAL_FACT, confidence=0.9)
    assert res.label is IntentLabel.MEDICAL_FACT
    assert res.label.value == "medical_fact"


def test_intent_label_has_six_classes():
    assert len(list(IntentLabel)) == 6
