"""Eval runners — glue golden sets to metrics.

Deliberately decoupled from every module (ARCHITECTURE.md #1): a runner takes a
*callable* classifier or a :class:`Retriever`, never imports
``retrieval.modules.*``. That keeps eval usable with fakes/mocks (no GPU, no
real inference on this machine — ARCHITECTURE.md #10) and with any real backend
the consuming project plugs in.

Async-first, mirroring the retrieval contract. Classifiers may be sync or async
callables; both are supported.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence

from medrag.api.retrieval.core.interfaces import IntentLabel, IntentResult, Retriever
from medrag.api.retrieval.eval.golden import IntentGoldenItem, RetrievalGoldenItem
from medrag.api.retrieval.eval.metrics import (
    IntentReport,
    RetrievalReport,
    intent_metrics,
    retrieval_metrics,
)

# A classifier maps a query string to a label (either an IntentResult or a bare
# IntentLabel/str). It may be sync or async.
ClassifierOutput = IntentResult | IntentLabel | str
Classifier = Callable[[str], ClassifierOutput | Awaitable[ClassifierOutput]]


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


def _to_label(out: ClassifierOutput) -> str:
    if isinstance(out, IntentResult):
        return out.label.value
    if isinstance(out, IntentLabel):
        return out.value
    return str(out)


async def evaluate_intent(
    golden: Sequence[IntentGoldenItem],
    classifier: Classifier,
    *,
    concurrency: int = 8,
) -> IntentReport:
    """Run ``classifier`` over every golden query and score it.

    ``concurrency`` bounds how many classifier calls run at once (they are
    typically remote LLM calls).
    """
    sem = asyncio.Semaphore(concurrency)

    async def classify_one(item: IntentGoldenItem) -> str:
        async with sem:
            return _to_label(await _maybe_await(classifier(item.query)))

    y_pred = await asyncio.gather(*(classify_one(it) for it in golden))
    y_true = [it.expected_intent.value for it in golden]
    return intent_metrics(y_true, y_pred)


async def evaluate_retrieval(
    golden: Sequence[RetrievalGoldenItem],
    retriever: Retriever,
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    concurrency: int = 8,
) -> RetrievalReport:
    """Run ``retriever`` over every golden query and score recall@k / MRR.

    Each query is retrieved with ``max(ks)`` (or the item's own ``k`` if larger)
    results, then scored against the golden ``relevant_ids``.
    """
    default_k = max(ks)
    sem = asyncio.Semaphore(concurrency)

    async def rank_one(item: RetrievalGoldenItem) -> tuple[list[str], list[str]]:
        k = max(default_k, item.k or 0)
        async with sem:
            results = await retriever.retrieve(item.query, k)
        return list(item.relevant_ids), [r.id for r in results]

    cases = await asyncio.gather(*(rank_one(it) for it in golden))
    return retrieval_metrics(cases, ks=ks)


__all__ = [
    "Classifier",
    "evaluate_intent",
    "evaluate_retrieval",
]
