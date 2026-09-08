"""aggregation flow: runs `SqlTopNFlow`'s single-sided retrieval core
(`run_pass`) MULTIPLE times -- once per category/filter dimension -- and
merges the results.

The `aggregation` intent covers multi-dimensional questions that can't be
comfortably expressed as ONE SQL query (e.g. "which is the priciest product
in each category", "how many products in 3 different power ranges" -- each
category/range may need its own top_n+SQL pair).
Text2SqlGenerator/SqlEngine (the retrieval package) stay SINGLE-query (their
own scope); multi-SQL orchestration is deliberately done at THIS flow layer --
instead of adding a "multi query" mode to retrieval, we repeatedly call its
existing single-query core at flow level (consistent with the standing
principle that chatbot uses retrieval as a dependency only, without touching
retrieval's own query engine).

This flow does NOT write its own retrieval logic -- it calls
`SqlTopNFlow.run_pass` N times with N different sub-queries in PARALLEL (the
N-sided generalization of `ComparisonFlow`'s two-sided pattern, see
flows/comparison.py). A change in `run_pass` (e.g. rewrite behavior, the
keyword-derivation logic of the second top_n) lands HERE without any code
change -- nothing is copied.

`splitter` is optional: splitting one aggregate query into N sub-queries can
stay `None` until retrieval's planned query-rewriting module ships; without
it, the raw query runs as a SINGLE pass (behavior identical to
`SqlTopNFlow.run`) -- the flow never blows up, it just can't split into
dimensions (same degrade-without-splitter principle as ComparisonFlow).

Merging top_n + SQL and preferring SQL on conflict is NOT implemented at CODE
level (same principle as `ComparisonFlow`) -- it's an instruction given to the
answering model via `strategies/aggregation.md` (advisory-only, see
strategies/README.md and the provenance rule in strategies/product_fact.md:
on conflict, the SQL row prevails). This flow only gathers each sub-query's
results tagged with `aggregation_part` so they stay distinguishable (for
trace/diagnostics + so the model can see which dimension came from which
evidence) -- it does NOT CHANGE conflict-resolution decisions, it only tags
sources.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

from medrag.api.flows.base import FlowContext
from medrag.api.flows.sql_topn import SqlTopNFlow
from medrag.api.retrieval.core import RetrievalResult
from medrag.api.trace import TraceFn, emit_timing

_METADATA_KEY = "aggregation_part"


@runtime_checkable
class AggregationSplitter(Protocol):
    """Splits one aggregate query into N sub-queries (one per
    category/filter dimension), e.g. "which is the priciest product in each
    category" -> ["category: cooler", "category: pump", "category:
    compressor"]."""

    async def split(self, query: str) -> list[str]: ...


class AggregationFlow:
    def __init__(
        self, aggregation_pass: SqlTopNFlow, *, splitter: AggregationSplitter | None = None
    ) -> None:
        self._pass = aggregation_pass
        self._splitter = splitter

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        if self._splitter is None:
            # No splitter (query rewriting not shipped): run the raw query as
            # a single pass -- behavior identical to SqlTopNFlow.run, the flow
            # doesn't blow up.
            results = await self._pass.run_pass(query, on_trace)
            return FlowContext(results=results)

        t0 = time.perf_counter()
        sub_queries = await self._splitter.split(query)
        if on_trace:
            on_trace("aggregation_split", sub_queries)
            emit_timing(on_trace, "split", t0)

        if not sub_queries:
            # If the splitter returns an empty list (e.g. it judged the query
            # single-dimensional), fall back to the raw query -- the flow must
            # NEVER return an empty context (same safety net as
            # ComparisonFlow's no-splitter fallthrough).
            results = await self._pass.run_pass(query, on_trace)
            return FlowContext(results=results)

        # The N parts run in PARALLEL (the GPU-gate `SerializedRetriever`
        # already serializes the SQL chain process-wide, see gpu_gate.py -- no
        # EXTRA serialization here). Each part's in-flow sub-durations
        # (rewrite/top_n/sql) are published separately by `run_pass` with a
        # `[<index>]` suffix (the N-sided generalization of ComparisonFlow's
        # side-stamping pattern); results are tagged with `aggregation_part`
        # (index, string) -- NOT mixed with `comparison_side` (a separate
        # metadata key, see the `metadata_key` of flows/sql_topn.py::run_pass).
        parts = await asyncio.gather(
            *(
                self._pass.run_pass(sub_query, on_trace, tag=str(i), metadata_key=_METADATA_KEY)
                for i, sub_query in enumerate(sub_queries)
            )
        )

        merged: list[RetrievalResult] = [result for part in parts for result in part]
        return FlowContext(results=merged)


__all__ = ["AggregationFlow", "AggregationSplitter"]
