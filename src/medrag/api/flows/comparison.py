"""comparison flow: runs `SqlTopNFlow`'s single-sided retrieval core
(`run_pass`) for N products and merges the results (later generalized from
two to N products).

The `comparison` intent compares N (>=2) products/options ("is de1000 or
de1100 better", "compare de1000, de1100, de2200"). This flow does NOT write
its own retrieval logic -- it calls `run_pass` of the SAME `SqlTopNFlow`
instance for every product (and for every product x feature/"angle" pair if
present). A change inside `SqlTopNFlow.run_pass` reflects HERE with no code
change -- they share the same core rather than copying it.

## The splitter is REAL (N-product)

retrieval's planned query-rewriting module is still absent and no LLM/GPU call
runs on the dev box (standing rule) -- a "cheap LLM call" option couldn't be
really tested/verified here. So `DeterministicComparisonSplitter` is a
DETERMINISTIC regex/keyword parser: ACME product codes have a fixed shape
("DE" + 3-7 digits, see the `product.product_code` `sample_values` in
`facts/db/schema.yaml`) and can be caught network-free, model-free, 100%
reproducibly. The `ComparisonSplitter` protocol is structural (`Protocol`), so
swapping in a real LLM/rule-based splitter later (e.g. when retrieval's query
rewriting ships) requires no change to `ComparisonFlow` -- only the injected
object changes.

## Clarification (which products, which features)

If the splitter finds FEWER THAN 2 products (0 or 1), the query is ambiguous:
`run()` returns an empty `FlowContext` WITHOUT touching retrieval (no SQL call
is wasted) -- `strategies/comparison.md` turns that emptiness into a "which
products shall we compare" question (advisory-only; the flow does NOT produce
its own text). At the same time `ComparisonFlow` implements the `PinnableFlow`
+ `SelfPinningFlow` capabilities and pins the next turn to itself cheaply
(reconcile+classify skipped) -- when the user answers "de1000 and de1100,
weight", that turn also passes this flow's `run()`, now 2+ products are found,
the real comparison runs AND the pin clears itself automatically (see
`pin_request`).

## Product x feature ("angle") fan-out, reusing the aggregation core

The "call the SAME `run_pass` core N times, tag, merge" pattern of
`AggregationFlow` is NOT REWRITTEN here -- if a product has (optionally)
multiple features listed ("weight", "price", ...), `run_pass` is called for
that product once per feature (just like `AggregationFlow` does); N products
also run in PARALLEL (the N-product generalization of the two-product
pattern). No extra "multi-SQL" primary code was written -- the one existing
mechanism (call `run_pass` N times, tag, merge) is applied over two
dimensions (product x feature).

## The comparison table is aligned at CODE level

SQL rows are already structured (`key`/`model`/`status`/`raw` columns, see
`spec_value` in `facts/db/schema.yaml`); `_comparison_table` aligns the values
of the same `key` across N products into "key: model_a=X, model_b=Y" and
appends it as ONE synthetic `RetrievalResult` at the END of the results. This
is pure FORMATTING (it doesn't break the advisory-only/no-tool-calling
principle) -- which side is "better" is still decided by the answering model;
this only places the same attribute side by side. The "SQL prevails on
conflict" rule (strategies/comparison.md) is UNCHANGED.

## "shall we compare something else"

No separate code was written -- `SessionState.available_followups`/
`followups_prompt` already scans every `key`/`model`/`status` field over
`last_results`; the rows this flow tags (`comparison_side`) CARRY those fields
UNCHANGED (only `comparison_side` is added), so the mechanism works
automatically. `_comparison_table`'s synthetic row carries no `key` field, so
it's silently skipped by the follow-up scan (see tests).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from medrag.api.continuation import ContinuationChecker
from medrag.api.flows.base import FlowContext, compute_result_shape
from medrag.api.flows.sql_topn import SqlTopNFlow
from medrag.api.pin_relevance import looks_relevant_to_pin
from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import PinnedFlow
from medrag.api.trace import TraceFn, emit_timing

_COMPARISON_SIDE_KEY = "comparison_side"

# Short answers that count as "this is now irrelevant/cancelled" in a turn --
# the FALLBACK `check_pin` drops to when `_continuation_checker` is not
# injected (dev/test). Matching data: keep verbatim.
_CANCEL_WORDS = {
    "iptal", "boşver", "bosver", "geç", "gec", "vazgeç", "vazgec",
    "unut", "hayır", "hayir", "yok", "boş ver", "bos ver",
}

# "DE" + 3-7 digits -- the shape of `product.product_code` in
# `facts/db/schema.yaml` (examples like PN1309, PN1015, PN1085, PN1190,
# PN1169). A user query may mix cases ("de1000") or insert a space/dash
# ("DE 1000", "DE-1000") -- all are normalized.
_PRODUCT_CODE_RE = re.compile(r"\bDE[\s-]?\d{3,7}\b", re.IGNORECASE)

# A token that has the SHAPE of "DE" + a product code but is invalid (e.g.
# "DE12" -- 2 digits, or 8+ digits) -- unlike `_PRODUCT_CODE_RE`, here the
# digit count is in the INVALID range. Deliberately an INDEPENDENT copy from
# the equivalent in `flows/doc_download.py` (same rationale as the splitter
# section of the module docstring: the two flows must stay INDEPENDENT, no
# dependency is created). Engaged only when NO VALID code is found (`sides`
# empty) -- the point is to distinguish "no product code at all" from "a token
# that LOOKS like a product code but is invalid" via
# `FlowContext.parse_note`.
_NEAR_MISS_CODE_RE = re.compile(r"\bDE[\s-]?(?:\d{1,2}|\d{8,})\b(?!\d)", re.IGNORECASE)

_NEAR_MISS_PRODUCT_CODE = "near_miss_product_code"


def parse_comparison_query(query: str) -> tuple[list[str], str, str | None]:
    """PURE function (for testability) -- a side-effect-free, reproducible
    core of the product-code regex also used by
    `DeterministicComparisonSplitter.split`. Returns: (found codes, unconsumed
    residual text, near-miss note or `None`).

    `residual`: the text left after ALL matching product-code spans are cut
    out of the query (runs of whitespace collapse to one space, ends are
    trimmed). `near_miss`: `"near_miss_product_code"` only when NO VALID code
    was found AND the query holds a code-shaped-but-invalid token, else
    `None`."""
    seen: dict[str, None] = {}
    spans: list[tuple[int, int]] = []
    for match in _PRODUCT_CODE_RE.finditer(query):
        code = re.sub(r"[\s-]", "", match.group(0)).upper()
        seen.setdefault(code, None)
        spans.append(match.span())

    residual = query
    for start, end in sorted(spans, reverse=True):
        residual = residual[:start] + residual[end:]
    residual = re.sub(r"\s+", " ", residual).strip()

    near_miss = None
    if not seen and _NEAR_MISS_CODE_RE.search(query):
        near_miss = _NEAR_MISS_PRODUCT_CODE

    return list(seen.keys()), residual, near_miss

# v1 deterministic "angle" (comparison feature) dictionary -- canonical name
# -> surface forms to look for in the query. Scope deliberately
# limited/extensible (see the splitter rationale in the module docstring); if
# an LLM-based parser ever arrives, this dictionary is fully retired and the
# `ComparisonAngleSplitter` protocol doesn't change. Matching data: keep the
# values verbatim.
_ANGLE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ağırlık": ("ağırlık", "agirlik", "weight"),
    "fiyat": ("fiyat", "price"),
    "güç": ("güç", "guc", "power"),
    "sıcaklık": ("sıcaklık", "sicaklik", "temperature"),
    "boyut": ("boyut", "boyutlar", "dimension", "size"),
    "hız": ("hız", "hiz", "speed"),
    "kapasite": ("kapasite", "capacity"),
    "voltaj": ("voltaj", "gerilim", "voltage"),
    "akım": ("akım", "akim", "current"),
}


@runtime_checkable
class ComparisonSplitter(Protocol):
    """Splits a comparison query into N products/sides, e.g. "which of de1000
    or de1100 is better" -> ["de1000", "de1100"] (this used to be a fixed
    two-sided `tuple[str, str]`; the `list[str]` return supports N sides).
    If fewer than 2 items (0 or 1) come back, `ComparisonFlow` treats it as
    "ambiguous" and drops to the clarification path WITHOUT touching
    retrieval (see `run`)."""

    async def split(self, query: str) -> list[str]: ...


@runtime_checkable
class ComparisonAngleSplitter(Protocol):
    """Optional EXTRA capability: splits out which feature(s) the comparison
    should be made on ("weight and price" etc.). The same object can implement
    both `ComparisonSplitter` and this (see
    `DeterministicComparisonSplitter`) -- `ComparisonFlow` detects it
    optionally via `isinstance`; otherwise it falls back to ONE `run_pass` per
    product (current/legacy behavior, no crash)."""

    async def split_angles(self, query: str) -> list[str]: ...


class DeterministicComparisonSplitter:
    """The default, REAL (LLM-free) implementation of
    `ComparisonSplitter` + `ComparisonAngleSplitter`. See the splitter section
    of the module docstring for WHY deterministic (rule-based) was chosen -- no
    LLM/GPU call can run on the dev box, so pure code is the only
    verifiable/testable option."""

    async def split(self, query: str) -> list[str]:
        sides, _residual, _near_miss = parse_comparison_query(query)
        return sides

    async def split_angles(self, query: str) -> list[str]:
        lowered = query.lower()
        seen: dict[str, None] = {}
        for canonical, surface_forms in _ANGLE_KEYWORDS.items():
            if any(form in lowered for form in surface_forms):
                seen.setdefault(canonical, None)
        return list(seen.keys())


def _row_value(metadata: Mapping[str, object]) -> object | None:
    """The displayable value of a `spec_value` row -- regardless of which
    slot `kind` uses.

    A `kind='range'`/`'min_typ_max'` row stores its value in
    `val_min`/`val_typ`/`val_max`, and a `kind='list'` row stores it in
    `items`; when `raw_text` is NULL (a real case seen on the facts side)
    those keys would SILENTLY DROP out of the comparison table if only
    `raw_text`/`text_value`/`num_value` were consulted.

    `raw_text` remains the FIRST preference -- the document's own wording
    ('0 °C to +55 °C') is more readable than the structured endpoints, and the
    comparison table's job is FORMATTING."""
    raw = metadata.get("raw_text")
    if raw is not None:
        return raw
    for field in ("text_value", "num_value", "bool_value"):
        value = metadata.get(field)
        if value is not None:
            return value
    low, typ, high = metadata.get("val_min"), metadata.get("val_typ"), metadata.get("val_max")
    if low is not None or high is not None:
        if low is not None and high is not None:
            return f"{low}–{high}" if low != high else low
        return low if low is not None else high
    if typ is not None:
        return typ
    items = metadata.get("items")
    if items:
        parsed = items
        if isinstance(items, str):
            try:
                parsed = json.loads(items)
            except json.JSONDecodeError:
                return items
        if isinstance(parsed, list) and parsed:
            labels = [
                str(i.get("item_text") or i.get("item_code")) if isinstance(i, dict) else str(i)
                for i in parsed
            ]
            return ", ".join(label for label in labels if label and label != "None")
    return None


def _comparison_table(results: list[RetrievalResult]) -> RetrievalResult | None:
    """A synthetic evidence row PRODUCED IN CODE that aligns the same `key`
    (spec_value) across multiple products with `status="present"` into a SINGLE
    line. SQL rows are already structured (see `spec_value.key`/`product_code`/
    `status`/`raw_text` in `facts/db/schema.yaml`) -- this is pure FORMATTING,
    it doesn't break the advisory-only/no-tool-calling principle: which product
    is "better" is still decided by the answering model.

    `metadata.get("model")`/`.get("raw")` were NOT real column names
    (`retrieval.modules.db_query.mapping.row_to_result` fills metadata with the
    SQL's OWN column names; it does NOT normalize to fixed "model"/"raw") --
    the real names are `product_code`/`raw_text` (+ the typed slots
    `text_value`/`num_value`). Because of that, the old `if not key or not
    model` check skipped EVERY row and `_comparison_table` silently never
    produced a line; fixed to the real column names.

    Deliberately does NOT carry a `key` field (the metadata holds nothing but
    `comparison_table: True`) -- `SessionState.available_followups` scans only
    rows with a real `key` field and skips this synthetic row silently (the
    precondition for the "no extra code was written" claim in the follow-ups
    section)."""
    by_key: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for r in results:
        metadata = r.metadata
        key = metadata.get("key")
        model = metadata.get("product_code")
        if not key or not model:
            continue
        status = metadata.get("status")
        if status is not None and status != "present":
            # EVERY non-present row is skipped, `conflicting` INCLUDED -- this
            # is INTENTIONAL ("if it conflicts, it waits"): when two documents
            # state different values for the same feature, that value is NEVER
            # shown to the user; it waits for human approval. Don't mistake
            # this for a bug and OPEN it up to show `conflicting` as "two
            # sources disagree" -- presenting a possibly-wrong spec with a
            # banner is worse than not presenting it (the comparison table is
            # read for customer decisions).
            continue
        value = _row_value(metadata)
        if value is None:
            continue
        if key not in by_key:
            by_key[key] = {}
            order.append(key)
        by_key[key].setdefault(str(model), str(value))

    lines = []
    for key in order:
        models = by_key[key]
        if len(models) < 2:
            continue
        lines.append(f"{key}: " + ", ".join(f"{m}={v}" for m, v in models.items()))

    if not lines:
        return None
    return RetrievalResult(
        id="comparison_table",
        score=1.0,
        text=(
            "Comparison table (same attribute aligned across products by "
            "code):\n" + "\n".join(lines)
        ),
        metadata={"comparison_table": True},
    )


class ComparisonFlow:
    """`flow_name`/`intent_key` must exactly equal the flow name registered in
    the `Router` and the `strategies/<intent_key>.md` key -- while this
    component produces `PinnedFlow.flow`/`PinnedFlow.intent` it must manually
    match the wiring (`config/default.toml [routing]` +
    `factory.py::build_flows_from_env`); `PinnedFlow` itself doesn't validate
    this (see the session_state.py docstring -- a known limit of the generic
    mechanism)."""

    def __init__(
        self,
        comparison_pass: SqlTopNFlow,
        *,
        splitter: ComparisonSplitter | None = None,
        flow_name: str = "comparison",
        intent_key: str = "comparison",
        continuation_checker: ContinuationChecker | None = None,
        few_max: int = 5,
    ) -> None:
        self._pass = comparison_pass
        self._splitter = splitter
        self._angle_splitter = (
            splitter if isinstance(splitter, ComparisonAngleSplitter) else None
        )
        self._flow_name = flow_name
        self._intent_key = intent_key
        self._continuation_checker = continuation_checker
        # Threshold for `FlowContext.result_shape` -- injected from `config/
        # default.toml [result_shape] few_max` (see
        # factory.py::build_flows_from_env), not hardcoded in code.
        self._few_max = few_max

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        if self._splitter is None:
            # No splitter (query rewriting hasn't shipped / none injected in
            # tests): run the raw query as a single side -- behavior identical
            # to SqlTopNFlow.run, the flow doesn't blow up, and
            # clarification/pinning is NOT ENGAGED (purity can't be measured
            # without a splitter).
            results = await self._pass.run_pass(query, on_trace)
            return FlowContext(results=results)

        t0 = time.perf_counter()
        sides = await self._splitter.split(query)
        if on_trace:
            on_trace("comparison_split", sides)
            emit_timing(on_trace, "split", t0)

        # residual/near-miss can only be computed on the REAL deterministic
        # splitter (it relies on the regex) -- the `ComparisonSplitter`
        # protocol doesn't promise this in general (e.g. if an LLM-based
        # splitter is injected later, these fields stay silently empty and the
        # flow doesn't blow up -- same isinstance pattern as
        # `ComparisonAngleSplitter`).
        residual: str | None = None
        near_miss: str | None = None
        if isinstance(self._splitter, DeterministicComparisonSplitter):
            _, residual, near_miss = parse_comparison_query(query)
            residual = residual or None

        if len(sides) < 2:
            # Ambiguous how many/which products -- drop to the clarification
            # path without wasting a SQL call. `strategies/comparison.md`
            # turns the empty evidence into a "which products shall we
            # compare" question (advisory-only -- this flow does NOT produce
            # the text shown to the user itself).
            if on_trace:
                on_trace("comparison_clarification_needed", sides)
            return FlowContext(
                results=[],
                result_shape=compute_result_shape(0, few_max=self._few_max),
                residual=residual,
                parse_note=near_miss,
            )

        angles: list[str] = []
        if self._angle_splitter is not None:
            angles = await self._angle_splitter.split_angles(query)
            if on_trace and angles:
                on_trace("comparison_angles", angles)

        async def _run_side(index: int, side: str) -> list[RetrievalResult]:
            # If feature(s) are specified, one `run_pass` per product per
            # feature (same pattern as AggregationFlow's call-N-times-and-
            # merge core, here over the product x feature grid); otherwise
            # (current/legacy behavior) a single pass per product.
            sub_queries = [f"{side} {angle}" for angle in angles] or [side]
            parts = await asyncio.gather(*(
                self._pass.run_pass(
                    sub_query, on_trace, tag=str(index), metadata_key=_COMPARISON_SIDE_KEY
                )
                for sub_query in sub_queries
            ))
            return [r for part in parts for r in part]

        # N products run in PARALLEL (the N-product generalization of the
        # two-product pattern) -- `SerializedRetriever` (gpu_gate.py) already
        # serializes the SQL chain process-wide, so NO EXTRA serialization is
        # done here.
        per_side = await asyncio.gather(*(_run_side(i, side) for i, side in enumerate(sides)))
        merged = [r for part in per_side for r in part]

        table = _comparison_table(merged)
        if table is not None:
            merged = [*merged, table]

        return FlowContext(
            results=merged,
            result_shape=compute_result_shape(len(merged), few_max=self._few_max),
            residual=residual,
            parse_note=near_miss,
        )

    async def check_pin(self, query: str, pinned: PinnedFlow) -> bool:
        """`PinnableFlow` (REVISED; fallback narrowed): at the moment we're
        pinned, a clarification question ("which products shall we compare")
        is expected. If `_continuation_checker` is injected (see factory.py),
        the decision delegates to it; otherwise (dev/test) it falls to
        `pin_relevance.looks_relevant_to_pin` -- the pin DROPS if the message
        is empty, is a cancel word, OR carries NO product-code-like signal at
        all (previously only the first two broke it)."""
        if not query.strip():
            return False  # empty message -- nothing to ask the checker about; invalid under the old behavior too
        if self._continuation_checker is None:
            sides, _residual, near_miss = parse_comparison_query(query)
            return looks_relevant_to_pin(
                query, cancel_words=_CANCEL_WORDS, has_signal=bool(sides) or near_miss is not None,
            )
        partial_sides = (pinned.expected or {}).get("partial_sides") or []
        context = (
            "The user was asked which products they want to compare"
            + (f" (so far: {', '.join(partial_sides)})" if partial_sides else "")
            + ". The user is expected to answer or give up."
        )
        return await self._continuation_checker.check(query, context)

    async def pin_request(self, query: str, context: FlowContext) -> PinnedFlow | None:
        """`SelfPinningFlow`: after `run()`, decides whether this turn's
        session should pin itself for the next turn. INSTEAD OF TRUSTING that
        `context.results` is empty (which could also mean "no splitter" or a
        genuinely result-less SQL call), it RECOMPUTES the split -- free on the
        default (deterministic) `splitter`; if an expensive (e.g. LLM-based)
        `splitter` is injected, this means two calls, a known limit (see the
        module docstring)."""
        if self._splitter is None:
            return None
        sides = await self._splitter.split(query)
        if len(sides) >= 2:
            return None  # clarification resolved -- the pin is cleared.
        return PinnedFlow(
            flow=self._flow_name, intent=self._intent_key,
            expected={"raw_query": query, "partial_sides": sides},
        )


__all__ = [
    "ComparisonAngleSplitter",
    "ComparisonFlow",
    "ComparisonSplitter",
    "DeterministicComparisonSplitter",
    "parse_comparison_query",
]
