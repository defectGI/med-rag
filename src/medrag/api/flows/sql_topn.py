"""sql_topn flow: a combined path with two top_n calls for the SQL-backed
intents (`product_fact`, `aggregation`, `visual_request`). (`doc_download` no
longer uses this flow -- it moved to its own deterministic `doc_download`
flow, see flows/doc_download.py.)

1. The raw query goes through an optional rewrite step into `db_query` (a
   rewrite suited to SQL generation may differ from the query suited to chunk
   search) -- the rewrite and the FIRST top_n call with the raw query run in
   PARALLEL.
2. SQL rows come back; a keyword step derives a short query from those rows.
3. That keyword query triggers a second top_n call.
4. The answering model receives [first top_n] + [second top_n] + [SQL rows]
   as the merged context. NONE of steps 1-3 (rewritten query, SQL string,
   extracted keywords) advances on its own -- the only thing reaching the
   Orchestrator is `FlowContext.results`.

If `on_trace` is given, these intermediate steps are shown to the USER (live
panel) -- this does NOT PUNCH A HOLE in the "doesn't leak to the model" rule,
because traces never enter the model's message history; they go to the
browser only.

`rewriter`/`keyword_extractor` are optional: they can stay `None` until
query-rewriting ships in retrieval; the day that module reaches production it
is injected without touching this class.

`run_pass` is this flow's SINGLE-SIDED retrieval core -- `run()` calls it once,
while `ComparisonFlow` (see flows/comparison.py) calls the same core
repeatedly for two different sub-queries. A change made inside this flow
(e.g. calling the rewrite step multiple times) automatically lands in both,
because no code is copied.

`tag`/`metadata_key` -- `run_pass` was initially intended only for
`ComparisonFlow`'s two sides (`metadata_key` fixed at `"comparison_side"`).
Since `AggregationFlow` (see flows/aggregation.py) calls the SAME core N times
(once per category/filter dimension) and merges the results, the tag key had
to diverge with its own meaning (which part number, which dimension) -- the `metadata_key`
parameter provides that, and its default (`"comparison_side"`) is unchanged,
so `ComparisonFlow` keeps working with zero code change.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from medrag.api.flows.base import FlowContext, compute_result_shape
from medrag.api.retrieval.core import RetrievalResult, Retriever
from medrag.api.trace import TraceFn, emit_timing, results_to_trace, timed


@runtime_checkable
class Rewriter(Protocol):
    """Turns the raw query into a query suited to SQL generation."""

    async def rewrite(self, query: str) -> str: ...


# Derives a short keyword query from the SQL results for the second top_n
# call. Pure/synchronous -- requires no LLM call (a simple field pick or
# template works); can move to an async seam later if ever needed.
KeywordExtractor = Callable[[list[RetrievalResult]], str]


async def _empty() -> list[RetrievalResult]:
    return []


def _tag(
    results: list[RetrievalResult], tag: str | None, metadata_key: str = "comparison_side"
) -> list[RetrievalResult]:
    """If `tag` is given, adds it under `metadata_key` to every result's
    metadata -- so `ComparisonFlow`'s two sides (`comparison_side`, the
    default) or `AggregationFlow`'s N parts (`aggregation_part`) can be
    presented to the answering model in a distinguishable way (see
    flows/comparison.py, flows/aggregation.py). For `None`, behavior is
    unchanged and no copying/mutation happens."""
    if tag is None:
        return results
    return [r.model_copy(update={"metadata": {**r.metadata, metadata_key: tag}}) for r in results]


class SqlTopNFlow:
    def __init__(
        self,
        sql_retriever: Retriever,
        *,
        top_n_retriever: Retriever | None = None,
        rewriter: Rewriter | None = None,
        keyword_extractor: KeywordExtractor | None = None,
        sql_k: int = 10,
        topn_k: int = 10,
        few_max: int = 5,
    ) -> None:
        """`sql_k`/`topn_k` are SEPARATE parameters -- previously a single `k`
        was passed both to `_run_sql` and to this flow's own top_n calls (first
        top_n + keyword top_n), so raising "sql_k" (10 rows was too few for a
        50-product aggregation) unintentionally grew top_n's context size (and
        thus token cost) too. The two evidence sources of `SqlTopNFlow` (SQL
        rows, top_n chunks) can have different size needs, hence `config/
        default.toml [flow]` also has two separate fields."""
        self._sql = sql_retriever
        self._top_n = top_n_retriever
        self._rewriter = rewriter
        self._keyword_extractor = keyword_extractor
        self._sql_k = sql_k
        self._topn_k = topn_k
        # Threshold for `FlowContext.result_shape` -- injected from `config/
        # default.toml [result_shape] few_max` (see
        # factory.py::build_flows_from_env). Only effective on paths that use
        # `run()` directly (see below) -- `run_pass` (the core shared by
        # comparison/aggregation/recommendation) doesn't build its own
        # `FlowContext` and never FILLS this field.
        self._few_max = few_max

    async def _run_sql(self, sql_query: str, on_trace: TraceFn | None) -> list[RetrievalResult]:
        # `tables` is a byproduct the linking stage has ALREADY computed (same
        # `emit_stage` data, no new LLM call) -- added deterministically to the
        # results' metadata so the evidence panel can show which table a SQL
        # row came from (see answering_model.py `extract_evidence_chunks`,
        # webapp.py `addEvidence`). Collected INDEPENDENT of `on_trace` -- the
        # evidence panel must see the table name even when tracing is off.
        tables: list[str] = []

        def emit_sql(sql: str) -> None:
            if on_trace is not None:
                on_trace("sql_generated", sql)

        def emit_stage(stage: str, detail: dict) -> None:
            # `linking_done`/`generation_done`/`sql_error` (retrieval 0.5.0,
            # db_query.generator.OnStage) -- carries the linking/generation
            # detail (tables/columns/rationale/duration/tokens), or stage +
            # raw text at error time, into the trace under the SAME names
            # directly (so a SQL explosion shows what the intermediate model
            # returned).
            if stage == "linking_done":
                tables.extend(detail.get("tables") or [])
            if on_trace is not None:
                on_trace(stage, detail)

        # `on_sql`/`on_stage` are optional hooks specific to
        # retrieval.modules.db_query.DbQueryRetriever (not part of the core
        # Retriever protocol) -- if a fake/other Retriever doesn't support
        # them, the three-level fallback below silently reverts to the old
        # behavior. Because `on_stage` is called INSIDE `generate()` itself
        # (including on errors), a raise still guarantees emit_stage ran --
        # the try/except does NOT PREVENT that; it only probes which kwargs
        # are supported.
        try:
            results = await self._sql.retrieve(
                sql_query, k=self._sql_k, on_sql=emit_sql, on_stage=emit_stage
            )
        except TypeError:
            try:
                results = await self._sql.retrieve(sql_query, k=self._sql_k, on_sql=emit_sql)
            except TypeError:
                results = await self._sql.retrieve(sql_query, k=self._sql_k)
        return _tag(results, ", ".join(tables) or None, metadata_key="sql_tables")

    async def run_pass(
        self,
        query: str,
        on_trace: TraceFn | None = None,
        *,
        tag: str | None = None,
        metadata_key: str = "comparison_side",
    ) -> list[RetrievalResult]:
        """Single-sided retrieval core: rewrite -> [top_n, sql] parallel ->
        keyword -> second top_n. If `tag` is given, results are stamped under
        `metadata_key` (default `"comparison_side"` for ComparisonFlow's two
        sides; `"aggregation_part"` for AggregationFlow's N parts -- see
        flows/aggregation.py).

        Every sub-stage publishes its own duration as a `timing` trace event
        -- it concretizes what took how long inside the single "retrieval"
        total published by the orchestrator (especially that the first top_n
        and the SQL chain are PARALLEL). When `tag` is given (ComparisonFlow
        runs N sides, AggregationFlow runs N parts in PARALLEL), stage names
        are disambiguated with `[0]`/`[1]`/... (index) so sides/parts don't
        mix -- ComparisonFlow used to stamp a fixed two sides with `[a]`/`[b]`;
        now that it generalizes to N products it uses the same index-based
        scheme AggregationFlow already used."""
        suffix = f" [{tag}]" if tag else ""

        if self._rewriter:
            t0 = time.perf_counter()
            sql_query = await self._rewriter.rewrite(query)
            if on_trace:
                on_trace("sql_rewrite", sql_query)
                emit_timing(on_trace, f"rewrite{suffix}", t0)
        else:
            sql_query = query

        # The first top_n (if any) and the SQL chain run in PARALLEL -- each
        # branch is wrapped in its own `timed(...)` to measure its individual
        # wall-clock time (not the gather total; it shows which branch is the
        # bottleneck).
        first_topn_call: Awaitable[list[RetrievalResult]]
        if self._top_n is not None:
            if on_trace:
                on_trace("top_n_query", query)
            first_topn_call = timed(
                on_trace, f"top_n{suffix}", self._top_n.retrieve(query, k=self._topn_k)
            )
        else:
            first_topn_call = _empty()

        first_topn, sql_results = await asyncio.gather(
            first_topn_call,
            timed(on_trace, f"sql{suffix}", self._run_sql(sql_query, on_trace)),
        )

        if on_trace:
            if self._top_n is not None:
                on_trace("top_n_results", results_to_trace(first_topn))
            on_trace("sql_rows", results_to_trace(sql_results))

        last_topn: list[RetrievalResult] = []
        if self._top_n is not None and self._keyword_extractor is not None:
            keywords = self._keyword_extractor(sql_results)
            if keywords:
                if on_trace:
                    on_trace("top_n_query", keywords)
                t0 = time.perf_counter()
                last_topn = await self._top_n.retrieve(keywords, k=self._topn_k)
                if on_trace:
                    on_trace("top_n_results", results_to_trace(last_topn))
                    emit_timing(on_trace, f"top_n·kw{suffix}", t0)

        return _tag([*first_topn, *sql_results, *last_topn], tag, metadata_key)

    async def top_n_pass(
        self, query: str, on_trace: TraceFn | None = None
    ) -> list[RetrievalResult]:
        """SQL-free, top_n-only core -- for `RecommendationFlow` to run a
        supporting search WHILE clarification is in progress (before
        `min_answered` is reached); unlike `run_pass` it never goes to SQL, and
        returns an empty list if `top_n_retriever` is absent (`None`) (zero
        behavior change, same pattern as `run_pass`)."""
        if self._top_n is None:
            return []
        if on_trace:
            on_trace("top_n_query", query)
        t0 = time.perf_counter()
        results = await self._top_n.retrieve(query, k=self._topn_k)
        if on_trace:
            on_trace("top_n_results", results_to_trace(results))
            emit_timing(on_trace, "top_n", t0)
        return results

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        """`result_shape` summarizes only the SUCCESSFUL (non-raising) result
        -- distinguishing SQL error / "unparseable" is DELIBERATELY not done
        here: if `run_pass` raises (see tests,
        `test_sql_topn_emits_sql_error_before_raising`), that error keeps
        propagating UNLESS swallowed (the Orchestrator's error behavior is
        unchanged); a "soft" error signal would require widening `run_pass`'s
        return type (`list[RetrievalResult]`), which is out of scope here
        (only the zero/one/few/many distinction was wanted)."""
        results = await self.run_pass(query, on_trace)
        return FlowContext(
            results=results,
            result_shape=compute_result_shape(len(results), few_max=self._few_max),
        )


__all__ = ["KeywordExtractor", "Rewriter", "SqlTopNFlow"]
