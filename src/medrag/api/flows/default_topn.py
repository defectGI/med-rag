"""Default flow: plain top_n chunk retrieval. Every intent with no explicit
flow in `config/default.toml` [routing] (today, in effect, `doc_question`)
falls here.

`retrieval.modules.top_n` is usable now -- `top_n_retriever=None` (default)
exists only as a backward/test convenience; in production
`factory.py` passes a real retriever.
"""

from __future__ import annotations

import time

from medrag.api.flows.base import FlowContext
from medrag.api.retrieval.core import Retriever
from medrag.api.trace import TraceFn, emit_timing, results_to_trace


class DefaultTopNFlow:
    """A single top_n call on the raw query."""

    def __init__(self, top_n_retriever: Retriever | None = None, *, k: int = 10) -> None:
        self._retriever = top_n_retriever
        self._k = k

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        if self._retriever is None:
            raise NotImplementedError(
                "DefaultTopNFlow bir retrieval.modules.top_n retriever'ı bekliyor; "
                "chatbot/factory.py::build_flows_from_env gerçek bir tane kurmalı."
            )
        if on_trace:
            on_trace("top_n_query", query)
        t0 = time.perf_counter()
        results = await self._retriever.retrieve(query, k=self._k)
        if on_trace:
            on_trace("top_n_results", results_to_trace(results))
            emit_timing(on_trace, "top_n", t0)  # timing card after the results
        return FlowContext(results=results)


__all__ = ["DefaultTopNFlow"]
