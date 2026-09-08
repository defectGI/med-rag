import asyncio

import pytest

from medrag.api.flows.aggregation import AggregationFlow
from medrag.api.flows.base import FlowContext, compute_result_shape
from medrag.api.flows.comparison import (
    ComparisonFlow,
    DeterministicComparisonSplitter,
    parse_comparison_query,
)
from medrag.api.flows.default_topn import DefaultTopNFlow
from medrag.api.flows.no_retrieval import NoRetrievalFlow
from medrag.api.flows.sql_topn import SqlTopNFlow
from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import PinnedFlow, SessionState


def _result(id_: str) -> RetrievalResult:
    return RetrievalResult(id=id_, score=1.0, text=id_)


class _FakeRetriever:
    """Does NOT support `on_sql` -- exercises SqlTopNFlow's TypeError
    fallback path (a Retriever that is not the db_query-specific one)."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results
        self.calls: list[str] = []

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append(query)
        return self._results


class _FakeSqlRetrieverWithOnSql:
    """Mimics the `DbQueryRetriever.retrieve(..., on_sql=...)` signature."""

    def __init__(self, results: list[RetrievalResult], *, sql: str):
        self._results = results
        self._sql = sql
        self.calls: list[str] = []

    async def retrieve(self, query: str, k: int = 10, *, on_sql=None) -> list[RetrievalResult]:
        self.calls.append(query)
        if on_sql is not None:
            on_sql(self._sql)
        return self._results


class _FakeSqlRetrieverWithOnStage:
    """Mimics the `DbQueryRetriever.retrieve(..., on_sql=..., on_stage=...)`
    signature -- emits the linking/generation detail."""

    def __init__(self, results: list[RetrievalResult], *, sql: str):
        self._results = results
        self._sql = sql

    async def retrieve(self, query: str, k: int = 10, *, on_sql=None, on_stage=None):
        if on_sql is not None:
            on_sql(self._sql)
        if on_stage is not None:
            on_stage("linking_done", {"tables": ["product"], "rationale": "matched model"})
            on_stage("generation_done", {"explanation": "ok"})
        return self._results


class _FakeSqlRetrieverThatFails:
    """Tests that even when generation blows up, `on_stage` is called with
    `sql_error` first and the error is still raised afterwards."""

    def __init__(self, exc: Exception):
        self._exc = exc

    async def retrieve(self, query: str, k: int = 10, *, on_sql=None, on_stage=None):
        if on_stage is not None:
            on_stage("sql_error", {"error_type": "StructuredOutputError",
                                   "message": "parse failed", "raw_text": "garbage"})
        raise self._exc


class _TraceCollector:
    def __init__(self):
        self.events: list[tuple] = []

    def __call__(self, step: str, data) -> None:
        self.events.append((step, data))


# --- compute_result_shape ----------------------------------------------------


def test_compute_result_shape_zero():
    assert compute_result_shape(0, few_max=5) == "zero"


def test_compute_result_shape_one():
    assert compute_result_shape(1, few_max=5) == "one"


def test_compute_result_shape_few_at_boundary():
    assert compute_result_shape(5, few_max=5) == "few"


def test_compute_result_shape_many_above_boundary():
    assert compute_result_shape(6, few_max=5) == "many"


# --- parse_comparison_query (pure parsing core) ------------------------------


def test_parse_comparison_query_finds_codes_and_strips_residual():
    sides, residual, near_miss = parse_comparison_query("de1000 mi de1100 mu daha iyi")
    assert sides == ["PN1015", "PN1036"]
    assert residual == "mi mu daha iyi"
    assert near_miss is None


def test_parse_comparison_query_no_codes_no_near_miss():
    sides, residual, near_miss = parse_comparison_query("hangisi daha iyi")
    assert sides == []
    assert residual == "hangisi daha iyi"
    assert near_miss is None


def test_parse_comparison_query_near_miss_too_few_digits():
    sides, _residual, near_miss = parse_comparison_query("DE12 hakkında bilgi ver")
    assert sides == []
    assert near_miss == "near_miss_product_code"


def test_parse_comparison_query_near_miss_too_many_digits():
    sides, _residual, near_miss = parse_comparison_query("PN1064 nedir")
    assert sides == []
    assert near_miss == "near_miss_product_code"


def test_parse_comparison_query_no_near_miss_when_valid_code_present():
    # Once a valid code is found, the near-miss CHECK never runs -- `sides`
    # is already populated, not "zero".
    sides, _residual, near_miss = parse_comparison_query("de1000 fiyatı")
    assert sides == ["PN1015"]
    assert near_miss is None


# --- FlowContext.result_shape/residual/parse_note ----------------------------


def test_comparison_run_sets_zero_shape_and_near_miss_when_ambiguous():
    sql = _FakeRetriever([])
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("DE12 hakkında bilgi ver"))

    assert ctx.results == []
    assert ctx.result_shape == "zero"
    assert ctx.parse_note == "near_miss_product_code"


def test_comparison_run_sets_few_shape_and_residual_on_resolved_query():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter(), few_max=5)

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert ctx.result_shape == "few"
    assert ctx.residual == "mi mu daha iyi"


def test_comparison_run_result_shape_many_above_few_max():
    sql = _FakeRetriever([_result("r1"), _result("r2"), _result("r3")])
    flow = ComparisonFlow(
        SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter(), few_max=2,
    )

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert ctx.result_shape == "many"


def test_comparison_without_splitter_leaves_new_fields_unset():
    # No splitter -- the result_shape/residual/parse_note fields are never
    # FILLED (old behavior).
    sql = _FakeRetriever([_result("row1")])
    flow = ComparisonFlow(SqlTopNFlow(sql))

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert ctx.result_shape is None
    assert ctx.residual is None
    assert ctx.parse_note is None


def test_sql_topn_run_sets_result_shape():
    sql = _FakeRetriever([_result("row1")])
    flow = SqlTopNFlow(sql, few_max=5)

    ctx = asyncio.run(flow.run("de1000 fiyati"))

    assert ctx.result_shape == "one"


def test_sql_topn_run_sets_zero_shape_for_empty_results():
    sql = _FakeRetriever([])
    flow = SqlTopNFlow(sql, few_max=5)

    ctx = asyncio.run(flow.run("bulunamayacak bir sey"))

    assert ctx.result_shape == "zero"


# --- DefaultTopNFlow ---------------------------------------------------------


def test_default_topn_without_retriever_raises():
    flow = DefaultTopNFlow()
    with pytest.raises(NotImplementedError):
        asyncio.run(flow.run("soru"))


def test_default_topn_with_retriever_returns_results():
    retriever = _FakeRetriever([_result("c1")])
    flow = DefaultTopNFlow(retriever, k=5)
    ctx = asyncio.run(flow.run("soru"))
    assert [r.id for r in ctx.results] == ["c1"]
    assert retriever.calls == ["soru"]


# --- NoRetrievalFlow ----------------------------------------------------------


def test_no_retrieval_flow_returns_empty_context():
    flow = NoRetrievalFlow()
    ctx = asyncio.run(flow.run("alakasiz soru"))
    assert ctx.results == []


# --- SqlTopNFlow ---------------------------------------------------------------


def test_sql_topn_without_top_n_returns_sql_only():
    sql = _FakeRetriever([_result("row1")])
    flow = SqlTopNFlow(sql)
    ctx = asyncio.run(flow.run("de1000 fiyati"))
    assert [r.id for r in ctx.results] == ["row1"]
    assert sql.calls == ["de1000 fiyati"]  # rewriter yoksa ham sorgu gider


def test_sql_topn_combines_first_and_last_topn_around_sql():
    sql = _FakeRetriever([_result("row1")])
    top_n = _FakeRetriever([_result("chunk-first")])

    def keyword_extractor(results: list[RetrievalResult]) -> str:
        return "anahtar-kelime"

    flow = SqlTopNFlow(sql, top_n_retriever=top_n, keyword_extractor=keyword_extractor)
    ctx = asyncio.run(flow.run("de1000 fiyati"))

    # first top_n + sql + second top_n merge in order (the single fake
    # retriever returns the same result for both top_n calls, but the call
    # list verifies both calls really happened).
    assert [r.id for r in ctx.results] == ["chunk-first", "row1", "chunk-first"]
    assert top_n.calls == ["de1000 fiyati", "anahtar-kelime"]


def test_sql_topn_uses_separate_k_for_sql_and_topn():
    # sql_k and topn_k are separate parameters so that growing sql_k does
    # not silently grow the top_n context size; this locks in that they are
    # genuinely passed separately.
    class _KRecordingRetriever:
        def __init__(self, results: list[RetrievalResult]):
            self._results = results
            self.ks: list[int] = []

        async def retrieve(self, query: str, k: int = 10, **_kwargs) -> list[RetrievalResult]:
            self.ks.append(k)
            return self._results

    sql = _KRecordingRetriever([_result("row1")])
    top_n = _KRecordingRetriever([_result("chunk-first")])

    flow = SqlTopNFlow(sql, top_n_retriever=top_n, sql_k=300, topn_k=10)
    asyncio.run(flow.run("de1000 fiyati"))

    assert sql.ks == [300]
    assert top_n.ks == [10]


def test_sql_topn_uses_rewriter_for_sql_query_only():
    sql = _FakeRetriever([_result("row1")])
    top_n = _FakeRetriever([_result("chunk-first")])

    class _FakeRewriter:
        async def rewrite(self, query: str) -> str:
            return f"SELECT-ready: {query}"

    flow = SqlTopNFlow(sql, top_n_retriever=top_n, rewriter=_FakeRewriter())
    asyncio.run(flow.run("de1000 fiyati"))

    assert sql.calls == ["SELECT-ready: de1000 fiyati"]
    assert top_n.calls == ["de1000 fiyati"]  # rewrite only goes to the SQL side


# --- trace (live background panel) -------------------------------------------


def test_default_topn_emits_query_and_results_trace():
    retriever = _FakeRetriever([_result("c1"), _result("c2")])
    flow = DefaultTopNFlow(retriever)
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert trace.events[0] == ("top_n_query", "soru")
    assert trace.events[1][0] == "top_n_results"
    assert [c["id"] for c in trace.events[1][1]] == ["c1", "c2"]


def test_no_retrieval_emits_trace_event():
    flow = NoRetrievalFlow()
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert trace.events == [("no_retrieval", "retrieval atlandı")]


# --- per-stage timing --------------------------------------------------------


def _timing_stages(trace: "_TraceCollector") -> set:
    return {d["stage"] for s, d in trace.events if s == "timing"}


def test_default_topn_emits_top_n_timing_after_results():
    # Every intermediate stage's duration should be visible. The timing card
    # must come AFTER the results card (without breaking the existing
    # top_n_query -> top_n_results positional contract).
    retriever = _FakeRetriever([_result("c1")])
    flow = DefaultTopNFlow(retriever)
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert "top_n" in _timing_stages(trace)
    steps = [s for s, _ in trace.events]
    assert steps.index("timing") > steps.index("top_n_results")


def test_default_topn_without_trace_emits_no_timing():
    # Without on_trace no timing is published (behavior identical).
    retriever = _FakeRetriever([_result("c1")])
    flow = DefaultTopNFlow(retriever)
    asyncio.run(flow.run("soru"))  # must not raise, no side effects


def test_sql_topn_emits_timing_for_each_substage():
    sql = _FakeRetriever([_result("row1")])
    top_n = _FakeRetriever([_result("c1")])

    class _FakeRewriter:
        async def rewrite(self, query: str) -> str:
            return f"rw: {query}"

    def keyword_extractor(results: list[RetrievalResult]) -> str:
        return "anahtar"

    flow = SqlTopNFlow(sql, top_n_retriever=top_n, rewriter=_FakeRewriter(),
                       keyword_extractor=keyword_extractor)
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    # rewrite + (parallel) first top_n + sql + the keyword top_n all publish
    # their own duration.
    assert {"rewrite", "top_n", "sql", "top_n·kw"} <= _timing_stages(trace)
    assert all(
        isinstance(d["seconds"], float) and d["seconds"] >= 0.0
        for s, d in trace.events if s == "timing"
    )


def test_sql_topn_sql_only_still_emits_sql_timing():
    sql = _FakeRetriever([_result("row1")])
    flow = SqlTopNFlow(sql)  # no top_n/rewriter/keyword
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    stages = _timing_stages(trace)
    assert "sql" in stages
    assert "top_n" not in stages  # no top_n retriever


def test_sql_topn_timing_still_emitted_on_sql_error():
    # Even if SQL blows up, the sql stage duration is published (timed's finally).
    sql = _FakeSqlRetrieverThatFails(RuntimeError("boom"))
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    with pytest.raises(RuntimeError):
        asyncio.run(flow.run("soru", trace))
    assert "sql" in _timing_stages(trace)


def test_sql_topn_falls_back_gracefully_when_retriever_lacks_on_sql():
    # _FakeRetriever does not support on_sql (TypeError fallback path) --
    # the "sql_generated" event must never be emitted, but the flow must not break.
    sql = _FakeRetriever([_result("row1")])
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    ctx = asyncio.run(flow.run("soru", trace))
    assert [r.id for r in ctx.results] == ["row1"]
    steps = [name for name, _ in trace.events]
    assert "sql_generated" not in steps
    assert "sql_rows" in steps


def test_sql_topn_emits_sql_generated_when_retriever_supports_on_sql():
    sql = _FakeSqlRetrieverWithOnSql([_result("row1")], sql="SELECT * FROM products")
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert ("sql_generated", "SELECT * FROM products") in trace.events


def test_sql_topn_emits_rewrite_trace_event():
    sql = _FakeRetriever([_result("row1")])

    class _FakeRewriter:
        async def rewrite(self, query: str) -> str:
            return f"SELECT-ready: {query}"

    flow = SqlTopNFlow(sql, rewriter=_FakeRewriter())
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert ("sql_rewrite", "SELECT-ready: soru") in trace.events


def test_sql_topn_forwards_stage_detail_when_retriever_supports_it():
    sql = _FakeSqlRetrieverWithOnStage([_result("row1")], sql="SELECT 1")
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    asyncio.run(flow.run("soru", trace))
    assert ("linking_done", {"tables": ["product"], "rationale": "matched model"}) in trace.events
    assert ("generation_done", {"explanation": "ok"}) in trace.events


def test_sql_topn_tags_sql_results_with_linking_tables():
    # `_run_sql` tags linking's `tables` into the results' metadata as
    # `sql_tables` even without `on_trace` -- so the evidence panel
    # (webapp.py) can show which table a SQL row came from.
    sql = _FakeSqlRetrieverWithOnStage([_result("row1")], sql="SELECT 1")
    flow = SqlTopNFlow(sql)
    ctx = asyncio.run(flow.run("soru"))
    assert ctx.results[0].metadata["sql_tables"] == "product"


def test_sql_topn_no_sql_tables_tag_when_retriever_lacks_on_stage():
    sql = _FakeSqlRetrieverWithOnSql([_result("row1")], sql="SELECT 1")
    flow = SqlTopNFlow(sql)
    ctx = asyncio.run(flow.run("soru"))
    assert "sql_tables" not in ctx.results[0].metadata


def test_sql_topn_falls_back_gracefully_when_retriever_lacks_on_stage():
    # _FakeSqlRetrieverWithOnSql supports only on_sql, not on_stage -- the
    # first attempt (both) fails with TypeError and falls back to on_sql-only.
    sql = _FakeSqlRetrieverWithOnSql([_result("row1")], sql="SELECT 1")
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    ctx = asyncio.run(flow.run("soru", trace))
    assert [r.id for r in ctx.results] == ["row1"]
    assert ("sql_generated", "SELECT 1") in trace.events
    steps = [name for name, _ in trace.events]
    assert "linking_done" not in steps


def test_sql_topn_emits_sql_error_before_raising():
    exc = RuntimeError("StructuredOutputError: parse failed")
    sql = _FakeSqlRetrieverThatFails(exc)
    flow = SqlTopNFlow(sql)
    trace = _TraceCollector()
    with pytest.raises(RuntimeError):
        asyncio.run(flow.run("soru", trace))
    assert ("sql_error", {"error_type": "StructuredOutputError",
                          "message": "parse failed", "raw_text": "garbage"}) in trace.events


def test_sql_topn_works_without_trace_argument():
    # on_trace is optional -- omitting it must not affect behavior/return value.
    sql = _FakeRetriever([_result("row1")])
    flow = SqlTopNFlow(sql)
    ctx = asyncio.run(flow.run("soru"))
    assert [r.id for r in ctx.results] == ["row1"]


class _QueryEchoRetriever:
    """Returns a result that varies with the query -- to test that
    AggregationFlow calls the N parts in the right order with the right
    sub-queries."""

    def __init__(self):
        self.calls: list[str] = []

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append(query)
        return [_result(f"row-{query}")]


# --- AggregationFlow (multi-SQL orchestration at the flow layer) ------------


class _FakeAggregationSplitter:
    def __init__(self, sub_queries: list[str]):
        self._sub_queries = sub_queries
        self.calls: list[str] = []

    async def split(self, query: str) -> list[str]:
        self.calls.append(query)
        return self._sub_queries


def test_aggregation_without_splitter_falls_back_to_single_pass():
    # No splitter (the retrieval-side one isn't in place yet) -- the raw
    # query goes to SqlTopNFlow.run_pass as a single pass and results stay
    # untagged.
    sql = _FakeRetriever([_result("row1")])
    aggregation_pass = SqlTopNFlow(sql)
    flow = AggregationFlow(aggregation_pass)

    ctx = asyncio.run(flow.run("kaç farklı kategori var"))

    assert [r.id for r in ctx.results] == ["row1"]
    assert ctx.results[0].metadata.get("aggregation_part") is None
    assert sql.calls == ["kaç farklı kategori var"]


def test_aggregation_with_splitter_runs_n_tagged_passes_in_order():
    sql = _QueryEchoRetriever()
    aggregation_pass = SqlTopNFlow(sql)
    splitter = _FakeAggregationSplitter(["kategori: sogutucu", "kategori: pompa", "kategori: kompresor"])
    flow = AggregationFlow(aggregation_pass, splitter=splitter)

    ctx = asyncio.run(flow.run("her kategoride en pahali urun"))

    assert splitter.calls == ["her kategoride en pahali urun"]
    assert sorted(sql.calls) == ["kategori: kompresor", "kategori: pompa", "kategori: sogutucu"]
    # results merge in part order (0, 1, 2), each carrying the row matching
    # its own query.
    assert [r.id for r in ctx.results] == [
        "row-kategori: sogutucu",
        "row-kategori: pompa",
        "row-kategori: kompresor",
    ]
    assert [r.metadata.get("aggregation_part") for r in ctx.results] == ["0", "1", "2"]
    # Does NOT collide with comparison_side -- a separate metadata key is used.
    assert all(r.metadata.get("comparison_side") is None for r in ctx.results)


def test_aggregation_splitter_returning_empty_list_falls_back_to_raw_query():
    sql = _FakeRetriever([_result("row1")])
    aggregation_pass = SqlTopNFlow(sql)
    splitter = _FakeAggregationSplitter([])
    flow = AggregationFlow(aggregation_pass, splitter=splitter)

    ctx = asyncio.run(flow.run("tek boyutlu soru"))

    assert [r.id for r in ctx.results] == ["row1"]
    assert sql.calls == ["tek boyutlu soru"]


def test_aggregation_emits_split_trace_event():
    sql = _QueryEchoRetriever()
    aggregation_pass = SqlTopNFlow(sql)
    splitter = _FakeAggregationSplitter(["a", "b"])
    flow = AggregationFlow(aggregation_pass, splitter=splitter)
    trace = _TraceCollector()

    asyncio.run(flow.run("her kategoride en pahali urun", trace))

    assert ("aggregation_split", ["a", "b"]) in trace.events


def test_aggregation_emits_split_and_part_tagged_timings():
    sql = _QueryEchoRetriever()
    aggregation_pass = SqlTopNFlow(sql)
    splitter = _FakeAggregationSplitter(["a", "b"])
    flow = AggregationFlow(aggregation_pass, splitter=splitter)
    trace = _TraceCollector()

    asyncio.run(flow.run("her kategoride en pahali urun", trace))

    stages = {d["stage"] for s, d in trace.events if s == "timing"}
    assert "split" in stages
    assert {"sql [0]", "sql [1]"} <= stages


def test_aggregation_works_without_trace_argument():
    sql = _QueryEchoRetriever()
    aggregation_pass = SqlTopNFlow(sql)
    splitter = _FakeAggregationSplitter(["a", "b"])
    flow = AggregationFlow(aggregation_pass, splitter=splitter)

    ctx = asyncio.run(flow.run("her kategoride en pahali urun"))

    assert [r.id for r in ctx.results] == ["row-a", "row-b"]


def test_comparison_emits_split_and_side_tagged_timings():
    # N products run PARALLEL; the split duration + each product's in-flow
    # SQL duration must be visible separately with index-based `[0]`/`[1]`
    # suffixes (no confusion) -- the old `[a]`/`[b]` scheme was generalized
    # to N products.
    sql = _FakeRetriever([_result("row")])
    comparison_pass = SqlTopNFlow(sql)
    flow = ComparisonFlow(comparison_pass, splitter=DeterministicComparisonSplitter())
    trace = _TraceCollector()

    asyncio.run(flow.run("de1000 mi de1100 mu daha iyi", trace))

    stages = {d["stage"] for s, d in trace.events if s == "timing"}
    assert "split" in stages
    assert {"sql [0]", "sql [1]"} <= stages


# --- ComparisonFlow (N-product comparison, real deterministic splitter,
# clarification/pin, product x attribute fan-out, code-level table,
# follow-up integration) ----------------------------------------------------


def _spec_row(
    id_: str, *, key: str, model: str, raw: object = None, status: str = "present"
) -> RetrievalResult:
    """Mimics the shape of real `spec_value` rows -- the metadata carries the
    SAME column names produced by
    `retrieval.modules.db_query.mapping.row_to_result`
    (`dict(zip(columns, row))`, the SQL's OWN column names). Note: this
    helper used to emit `"model"`/`"raw"` -- the real `spec_value` columns
    are `product_code`/`raw_text`; the old shape made
    `comparison.py::_comparison_table` get tested against a metadata shape
    that never actually matched (i.e. it MASKED the bug)."""
    return RetrievalResult(
        id=id_, score=1.0, text=f"{key}={raw}",
        metadata={"key": key, "product_code": model, "status": status, "raw_text": raw},
    )


class _FakeSpecRowsRetriever:
    """Query (exact match) -> predefined row list. Tests that the sub-queries
    ComparisonFlow builds per product/attribute are constructed CORRECTLY
    and that the table alignment works over the real row shape (contrast
    with `_FakeRetriever` -- it returns the SAME list for every query, while
    different rows per query are needed here)."""

    def __init__(self, rows_by_query: dict[str, list[RetrievalResult]]):
        self._rows_by_query = rows_by_query
        self.calls: list[str] = []

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        self.calls.append(query)
        return list(self._rows_by_query.get(query, []))


class _ProductOnlySplitter:
    """Implements `ComparisonSplitter` but NOT `ComparisonAngleSplitter` --
    tests that ComparisonFlow enables angle fan-out only for splitters that
    support it (via isinstance); the optional-capability pattern, SAME as
    `PinnableFlow`/`SelfPinningFlow`."""

    def __init__(self, sides: list[str]):
        self._sides = sides
        self.calls: list[str] = []

    async def split(self, query: str) -> list[str]:
        self.calls.append(query)
        return list(self._sides)


def test_comparison_without_splitter_falls_back_to_single_pass():
    # If no splitter is injected at all (current/old behavior), the raw
    # query goes to SqlTopNFlow.run_pass as a single side and results stay
    # untagged -- clarification/pin do NOT activate on this path
    # (purity cannot be measured without a splitter).
    sql = _FakeRetriever([_result("row1")])
    flow = ComparisonFlow(SqlTopNFlow(sql))

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert [r.id for r in ctx.results] == ["row1"]
    assert ctx.results[0].metadata.get("comparison_side") is None
    assert sql.calls == ["de1000 mi de1100 mu daha iyi"]


def test_comparison_deterministic_splitter_finds_two_products_tagged_by_index():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert sorted(sql.calls) == ["PN1015", "PN1036"]
    tags = {r.metadata.get("comparison_side") for r in ctx.results if r.id in ("r1", "r2")}
    assert tags == {"0", "1"}


def test_comparison_deterministic_splitter_finds_three_products():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
        "PN1197": [_spec_row("r3", key="weight", model="PN1197", raw="4 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000, de1100 ve de2200'ü karşılaştır"))

    assert sorted(sql.calls) == ["PN1015", "PN1036", "PN1197"]
    tags = {r.metadata.get("comparison_side") for r in ctx.results if r.id in ("r1", "r2", "r3")}
    assert tags == {"0", "1", "2"}


def test_comparison_ambiguous_query_skips_retrieval_and_returns_empty():
    # If fewer than 2 product codes are found (here: none), retrieval is
    # never called -- no SQL call is made in vain.
    sql = _FakeRetriever([_result("should-not-be-called")])
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("hangisi daha iyi"))

    assert ctx.results == []
    assert sql.calls == []


def test_comparison_single_product_code_is_still_ambiguous():
    sql = _FakeRetriever([_result("should-not-be-called")])
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 nasıl bir ürün"))

    assert ctx.results == []
    assert sql.calls == []


def test_comparison_emits_clarification_trace_event_when_ambiguous():
    sql = _FakeRetriever([])
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())
    trace = _TraceCollector()

    asyncio.run(flow.run("hangisi daha iyi", trace))

    assert ("comparison_clarification_needed", []) in trace.events


def test_comparison_angle_splitter_fans_out_per_product_per_angle():
    # 2 products x 2 attributes = 4 sub-queries, all PARALLEL.
    sql = _FakeSpecRowsRetriever({
        "PN1015 ağırlık": [_spec_row("w1", key="weight", model="PN1015", raw="2 kg")],
        "PN1015 fiyat": [_spec_row("p1", key="price", model="PN1015", raw="100 usd")],
        "PN1036 ağırlık": [_spec_row("w2", key="weight", model="PN1036", raw="3 kg")],
        "PN1036 fiyat": [_spec_row("p2", key="price", model="PN1036", raw="150 usd")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 ve de1100'ü ağırlık ve fiyat açısından karşılaştır"))

    assert sorted(sql.calls) == [
        "PN1015 ağırlık", "PN1015 fiyat", "PN1036 ağırlık", "PN1036 fiyat",
    ]
    assert {"w1", "p1", "w2", "p2"} <= {r.id for r in ctx.results}


def test_comparison_without_angle_splitter_runs_single_pass_per_product():
    # `_ProductOnlySplitter` implements only `ComparisonSplitter` -- the
    # "ağırlık" word in the query is ignored, ONE pass per product.
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=_ProductOnlySplitter(["PN1015", "PN1036"]))

    asyncio.run(flow.run("de1000 ve de1100 ağırlık"))

    assert sorted(sql.calls) == ["PN1015", "PN1036"]


def test_comparison_table_aligns_shared_spec_key_across_products():
    # The same `key` (weight) is "present" for both products -- a single
    # synthetic row aligned at code level is added.
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    table = next(r for r in ctx.results if r.id == "comparison_table")
    assert "weight: PN1015=2 kg, PN1036=3 kg" in table.text
    assert table.metadata == {"comparison_table": True}


def test_comparison_table_supports_three_products():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
        "PN1197": [_spec_row("r3", key="weight", model="PN1197", raw="4 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000, de1100 ve de2200'ü karşılaştır"))

    table = next(r for r in ctx.results if r.id == "comparison_table")
    assert "PN1015=2 kg" in table.text
    assert "PN1036=3 kg" in table.text
    assert "PN1197=4 kg" in table.text


def test_comparison_table_ignores_absent_status_rows():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", status="absent", raw=None)],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert not any(r.id == "comparison_table" for r in ctx.results)


def test_comparison_table_omitted_when_no_shared_key_across_products():
    sql = _FakeSpecRowsRetriever({
        "PN1015": [_spec_row("r1", key="weight", model="PN1015", raw="2 kg")],
        "PN1036": [_spec_row("r2", key="price", model="PN1036", raw="150 usd")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    assert not any(r.id == "comparison_table" for r in ctx.results)


def test_comparison_results_feed_followups_but_table_is_skipped():
    # No separate code was written -- SessionState.available_followups
    # derives directly from ComparisonFlow's last_results, and the synthetic
    # `comparison_table` row (no key field) is silently skipped.
    sql = _FakeSpecRowsRetriever({
        "PN1015": [
            _spec_row("r1", key="weight", model="PN1015", raw="2 kg"),
            _spec_row("r1b", key="price", model="PN1015", raw="100 usd"),
        ],
        "PN1036": [_spec_row("r2", key="weight", model="PN1036", raw="3 kg")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))
    state = SessionState(last_results=ctx.results)

    followups = state.available_followups()

    assert "PN1015: weight" in followups
    assert "PN1015: price" in followups
    assert "PN1036: weight" in followups
    assert not any("comparison_table" in f for f in followups)


def test_comparison_check_pin_holds_for_normal_reply():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    assert asyncio.run(flow.check_pin("de1000 ve de1100, ağırlık", pin)) is True


def test_comparison_check_pin_breaks_on_cancel_word():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    assert asyncio.run(flow.check_pin("iptal", pin)) is False


def test_comparison_check_pin_breaks_on_empty_query():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    assert asyncio.run(flow.check_pin("   ", pin)) is False


class _FakeContinuationChecker:
    def __init__(self, result: bool):
        self._result = result
        self.calls: list[tuple[str, str]] = []

    async def check(self, query: str, context: str) -> bool:
        self.calls.append((query, context))
        return self._result


def test_comparison_check_pin_delegates_to_continuation_checker_when_injected():
    checker = _FakeContinuationChecker(False)
    flow = ComparisonFlow(
        SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter(),
        continuation_checker=checker,
    )
    pin = PinnedFlow(
        flow="comparison", intent="comparison",
        expected={"raw_query": "karşılaştır", "partial_sides": ["PN1015"]},
    )

    result = asyncio.run(flow.check_pin("bu arada başka bir şey soracaktım", pin))

    assert result is False  # the checker's return value -- a phrase NOT in the cancel list
    assert len(checker.calls) == 1
    query, context = checker.calls[0]
    assert query == "bu arada başka bir şey soracaktım"
    assert "PN1015" in context


def test_comparison_check_pin_empty_query_short_circuits_even_with_checker():
    checker = _FakeContinuationChecker(True)
    flow = ComparisonFlow(
        SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter(),
        continuation_checker=checker,
    )
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    assert asyncio.run(flow.check_pin("   ", pin)) is False
    assert checker.calls == []  # the empty message was never asked to the checker


# --- relevance gate when NO checker is injected (old behavior: everything True) ---


def test_comparison_check_pin_fallback_drops_pin_for_irrelevant_message():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    # Neither a product code nor a cancel word -- the old "not in
    # _CANCEL_WORDS" check returned True, now the pin DROPS.
    assert asyncio.run(flow.check_pin("bu arada başka bir şey soracaktım", pin)) is False


def test_comparison_check_pin_fallback_keeps_pin_for_near_miss_product_code():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    # "DE12" -- product-code SHAPED but invalid (2 digits) -- still counts
    # as an answer attempt, the pin stays alive.
    assert asyncio.run(flow.check_pin("DE12 sanırım", pin)) is True


def test_comparison_check_pin_fallback_unaffected_when_checker_injected():
    # With a checker injected the behavior must not change AT ALL -- the
    # relevance gate is never visited; the same "irrelevant" message may
    # return True depending on the checker's result.
    checker = _FakeContinuationChecker(True)
    flow = ComparisonFlow(
        SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter(),
        continuation_checker=checker,
    )
    pin = PinnedFlow(flow="comparison", intent="comparison", expected={})

    assert asyncio.run(flow.check_pin("bu arada başka bir şey soracaktım", pin)) is True


def test_comparison_pin_request_none_without_splitter():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])))

    assert asyncio.run(flow.pin_request("soru", FlowContext(results=[]))) is None


def test_comparison_pin_request_returns_pin_when_still_ambiguous():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())

    pin = asyncio.run(flow.pin_request("hangisi daha iyi", FlowContext(results=[])))

    assert pin is not None
    assert pin.flow == "comparison"
    assert pin.intent == "comparison"
    assert pin.expected["raw_query"] == "hangisi daha iyi"


def test_comparison_pin_request_clears_when_resolved():
    flow = ComparisonFlow(SqlTopNFlow(_FakeRetriever([])), splitter=DeterministicComparisonSplitter())

    pin = asyncio.run(
        flow.pin_request("de1000 ve de1100", FlowContext(results=[_result("row1")]))
    )

    assert pin is None


def test_row_value_reads_range_min_typ_max_and_list_slots():
    """`_comparison_table` used to read only `raw_text`/`text_value`/
    `num_value`; but a `kind='range'`/`'min_typ_max'` row stores its value
    in `val_min`/`val_typ`/`val_max` and a `kind='list'` row in `items`.
    When `raw_text` was NULL (a real case seen on the facts side) these keys
    silently DROPPED out of the table -- the risk grew once `switch_voltage`
    was reclassified in the glossary from `single` to `range`."""
    from medrag.api.flows.comparison import _row_value

    # raw_text is still the first choice (the document's own wording is more readable)
    assert _row_value({"raw_text": "0 °C to +55 °C", "val_min": 0, "val_max": 55}) == "0 °C to +55 °C"
    # range: both bounds
    assert _row_value({"val_min": 0, "val_max": 250}) == "0–250"
    # range: both bounds equal (a single nominal coerced into a range)
    assert _row_value({"val_min": 24, "val_max": 24}) == 24
    # range: single bound
    assert _row_value({"val_max": 220}) == 220
    # min_typ_max: only typ
    assert _row_value({"val_typ": 3.3}) == 3.3
    # list: JSON text column (spec_value.items is TEXT)
    assert _row_value({"items": '[{"item_code": "hdmi", "item_text": "HDMI"}, '
                                '{"item_code": "usb", "item_text": "USB"}]'}) == "HDMI, USB"
    # list: an already-resolved list
    assert _row_value({"items": [{"item_code": "rs485"}]}) == "rs485"
    # boolean
    assert _row_value({"bool_value": 1}) == 1
    # no slot filled -> row is skipped
    assert _row_value({"key": "weight"}) is None
    # broken JSON silently falls back to the raw text (so the row isn't lost)
    assert _row_value({"items": "{bozuk"}) == "{bozuk"


def test_comparison_table_aligns_range_rows_without_raw_text():
    """Two `range` rows (switch_voltage) without `raw_text` must align."""
    sql = _FakeSpecRowsRetriever({
        "PN1015": [RetrievalResult(id="r1", score=1.0, text="x", metadata={
            "key": "switch_voltage", "product_code": "PN1015", "status": "present",
            "val_min": 0, "val_max": 250})],
        "PN1036": [RetrievalResult(id="r2", score=1.0, text="x", metadata={
            "key": "switch_voltage", "product_code": "PN1036", "status": "present",
            "val_min": 0, "val_max": 440})],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de1000 mi de1100 mu daha iyi"))

    table = next(r for r in ctx.results if r.id == "comparison_table")
    assert "switch_voltage: PN1015=0–250, PN1036=0–440" in table.text


def test_comparison_table_skips_conflicting_on_purpose():
    """`conflicting` rows do NOT enter the comparison table -- when sources
    conflict, the value waits. Presenting a possibly-wrong spec is worse
    than presenting nothing; this table is read to make customer decisions.
    This test exists to lock the behavior: it was once mistaken for a loss."""
    sql = _FakeSpecRowsRetriever({
        "PN1267": [_spec_row("r1", key="stub_count", model="PN1267", raw="4", status="conflicting")],
        "PN1274": [_spec_row("r2", key="stub_count", model="PN1274", raw="8", status="conflicting")],
    })
    flow = ComparisonFlow(SqlTopNFlow(sql), splitter=DeterministicComparisonSplitter())

    ctx = asyncio.run(flow.run("de8202 mi de8236 mi daha iyi"))

    assert not any(r.id == "comparison_table" for r in ctx.results)
