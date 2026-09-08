"""No-op flow for intents that need no retrieval at all, like
`out_of_scope`. It makes no retrieval calls; the answering model produces the
answer from the strategy prompt alone (see strategies/out_of_scope.md).

Deliberately separate: routing this intent to `default_topn` would make even
an irrelevant question depend on top_n retrieval -- an intent that needs no
retrieval must not depend on retrieval.
"""

from __future__ import annotations

from medrag.api.flows.base import FlowContext
from medrag.api.trace import TraceFn


class NoRetrievalFlow:
    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        if on_trace:
            on_trace("no_retrieval", "retrieval atlandı")
        return FlowContext(results=[])


__all__ = ["NoRetrievalFlow"]
