"""doc_download flow: resolves ACME product documents (datasheet, brochure, CE
declaration, technical drawing, ...) DETERMINISTICALLY from the `document`
table and presents them as evidence to the answering model (only
`file_name`/`doc_type`, NEVER `source_path`).

## Why a new flow instead of the existing `sql_topn`

`doc_download` used to route to `sql_topn` (LLM-based SQL generation) in
`config/default.toml`. A file DOWNLOAD path is accuracy-critical -- a wrong
`doc_id` sends the user the wrong file. Since the `document` table can already
be resolved by a fully deterministic query (model + doc_type filters, see the
`document` description in facts/db/schema.yaml), this flow NEVER goes to LLM
SQL generation -- same philosophy as `ComparisonFlow`'s product-code regex
(see the splitter rationale in the flows/comparison.py module docstring): an
LLM/GPU call can't be tested/verified on the dev box, deterministic code is
both testable and LLM-free in production.

## `source_path` NEVER passes through this flow's hands

`DocumentLookup.find` returns only `DocumentRow` -- a type that does NOT carry
a `source_path` field (see the definition below). Unlike the image-serving
endpoint (where the guarantee was discipline/docstring only, see the
`_source_suffix` comment in answering_model.py), the guarantee HERE is at the
TYPE level: no code path in this flow can READ `source_path`, because the
class carrying it is never imported into this module (see document_store.py --
the ONLY place that reads `source_path`, used solely by webapp.py's download
endpoint).

## Ambiguity clarification via the pinned-flow pattern

Two kinds of ambiguity: (a) the query has no product code at all ("send a
datasheet") -- which product is unclear; (b) product(s) found but no
`doc_type` given AND the product has multiple DIFFERENT document types --
which type is unclear. In both cases the flow returns an empty/short
`FlowContext` WITHOUT touching retrieval + pins the session to itself (the
SAME `PinnableFlow`/`SelfPinningFlow` pattern as
`ComparisonFlow`/`RecommendationFlow`) -- `strategies/doc_download.md` turns
the gap into a clarification question.

## "want another file?" follow-ups, generalizing the spec follow-ups

On a resolved request (once `run`/`run_pinned` settle on a product + doc_type),
the flow adds not only the requested rows but ALL other active documents of
the same product(s) to `FlowContext.results`, tagged `requested=False`. These
are not pushed to the answering model as TEXT --
`SessionState.available_document_followups` (session_state.py) reads them from
last_results over a SEPARATE channel; it's the SIBLING of
`available_followups` looking at `doc_id`/`file_name` instead of the `key`
field. So the "is there another relevant file" suggestion is derived from REAL
document rows, never invented.

## `source_path` is NEVER shown to the model/user

`_to_result`'s metadata carries only `doc_id`/`model`/`doc_type`/`file_name`/
`requested` -- `source_path` is NEVER added to this dict. `webapp.py` (via
`answering_model.py::extract_visible_documents`) lifts `requested=True` rows
containing `doc_id`+`file_name` out into a SEPARATE `documents` channel (same
principle as the images channel) -- the actual download goes through
`GET /api/documents/<doc_id>` (webapp.py) using document_store.py's
SEPARATE/sole code that reads `source_path`.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, NamedTuple, Protocol, runtime_checkable

from medrag.api.continuation import ContinuationChecker
from medrag.api.flows.base import FlowContext, compute_result_shape
from medrag.api.pin_relevance import looks_relevant_to_pin
from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import PinnedFlow
from medrag.api.trace import TraceFn
from medrag.core.db.sqlite import connect as _sqlite_connect

# "DE" + 3-7 digits -- same shape as `flows/comparison.py::_PRODUCT_CODE_RE`
# (facts/db/schema.yaml `product.model` sample_values), DELIBERATELY REPEATED
# HERE (importing from comparison.py would make the two flows depend on each
# other -- same rationale as the `_CANCEL_WORDS` note in
# flows/recommendation.py: no dependency is created where the two flows must
# stay INDEPENDENT).
_PRODUCT_CODE_RE = re.compile(r"\bDE[\s-]?\d{3,7}\b", re.IGNORECASE)

# Same shape as `flows/comparison.py::_NEAR_MISS_CODE_RE`, DELIBERATELY an
# INDEPENDENT copy (same as the module docstring's rationale for repeating the
# product-code regex: the two flows don't import from each other).
_NEAR_MISS_CODE_RE = re.compile(r"\bDE[\s-]?(?:\d{1,2}|\d{8,})\b(?!\d)", re.IGNORECASE)

_NEAR_MISS_PRODUCT_CODE = "near_miss_product_code"

# The `document.doc_type` VALUES in facts/db/schema.yaml (9 fixed types) ->
# surface forms to look for in the query. Same pattern as
# `ComparisonFlow._ANGLE_KEYWORDS`/`RecommendationFlow._FAMILY_SURFACE_FORMS`:
# a small, hand-picked, extensible dictionary. Matching data: keep values
# verbatim.
_DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "DATASHEET": ("datasheet", "data sheet", "teknik föy", "teknik foy"),
    "BROCHURE": ("broşür", "brosur", "brochure"),
    "CATALOGUE": ("katalog", "catalogue", "catalog"),
    "CE_DECLARATION": (
        "ce beyan", "ce uygunluk", "ce declaration", "uygunluk beyanı", "uygunluk beyani",
    ),
    "TECHNICAL_DRAWING": (
        "teknik çizim", "teknik cizim", "technical drawing", "çizim", "cizim",
    ),
    "STP": ("stp", "cad", "3d model", "step dosyası", "step dosyasi"),
    "USER_MANUAL": (
        "kullanım kılavuzu", "kullanim kilavuzu", "user manual", "manual",
        "kılavuz", "kilavuz",
    ),
    "QUICK_START_GUIDE": ("hızlı başlangıç", "hizli baslangic", "quick start"),
    "PRODUCT_IMAGE": (
        "ürün görseli", "urun gorseli", "product image", "fotoğraf", "fotograf", "resim",
    ),
}

# Readable labels shown to the user (instead of printing the DB's raw
# `doc_type` value directly). User-facing: keep values verbatim.
_DOC_TYPE_LABELS: dict[str, str] = {
    "DATASHEET": "Datasheet",
    "BROCHURE": "Brochure",
    "CATALOGUE": "Catalogue",
    "CE_DECLARATION": "CE declaration of conformity",
    "TECHNICAL_DRAWING": "Technical drawing",
    "STP": "STP/CAD file",
    "USER_MANUAL": "User manual",
    "QUICK_START_GUIDE": "Quick start guide",
    "PRODUCT_IMAGE": "Product image",
}

# The SAME small/deterministic set as `ComparisonFlow`/`RecommendationFlow` --
# kept as a separate constant (INSTEAD OF importing, for the same reason: the
# flows must stay independent of each other). Now only the FALLBACK engaged
# when `_continuation_checker` is never injected (see the "REVISED" note in
# flows/base.py::PinnableFlow). Matching data: keep values verbatim.
_CANCEL_WORDS = {
    "iptal", "boşver", "bosver", "geç", "gec", "vazgeç", "vazgec",
    "unut", "hayır", "hayir", "yok", "boş ver", "bos ver",
}


class DocumentRow(NamedTuple):
    """One row of the `document` table -- DELIBERATELY does NOT carry
    `source_path` (see module docstring). Only fields safe to show the user."""

    doc_id: str
    model: str
    doc_type: str
    file_name: str


@runtime_checkable
class DocumentLookup(Protocol):
    """Deterministic (LLM-free) access to the `document` table -- the real
    implementation is `SqliteDocumentLookup` (see below, built via
    `factory.py::build_document_lookup_from_env`); tests use a fake dict-based
    implementation."""

    async def find(
        self, models: list[str], doc_type: str | None = None, *, include_inactive: bool = False
    ) -> list[DocumentRow]: ...


def extract_product_codes(query: str) -> list[str]:
    """SAME regex/normalization as
    `DeterministicComparisonSplitter.split` (DELIBERATELY an independent copy,
    see module docstring)."""
    seen: dict[str, None] = {}
    for match in _PRODUCT_CODE_RE.finditer(query):
        code = re.sub(r"[\s-]", "", match.group(0)).upper()
        seen.setdefault(code, None)
    return list(seen.keys())


def extract_doc_type(query: str) -> str | None:
    """Returns the canonical `doc_type` (DB value) for the document-type
    surface forms occurring in the query -- if several match, the FIRST match
    (dictionary order) wins; if none match, `None` (treated as unspecified)."""
    lowered = query.lower()
    for canonical, forms in _DOC_TYPE_KEYWORDS.items():
        if any(form in lowered for form in forms):
            return canonical
    return None


def parse_doc_download_query(query: str) -> tuple[list[str], str | None, str | None, str | None]:
    """PURE function (for testability) -- reuses `extract_product_codes`/
    `extract_doc_type`, NO side effects. Returns: (found product codes,
    doc_type, unconsumed residual text, near-miss note or `None`).

    `residual`: the text left after removing ALL recognized product-code spans
    + the first matching doc_type surface form from the query (runs of
    whitespace collapse to one space, ends trimmed) -- `None` if empty.
    `near_miss`: `"near_miss_product_code"` only when no product code was found
    AND the query holds a code-shaped-but-invalid token, else `None` (same
    principle as `flows/comparison.py::parse_comparison_query`)."""
    models = extract_product_codes(query)
    doc_type = extract_doc_type(query)

    residual = query
    for match in reversed(list(_PRODUCT_CODE_RE.finditer(query))):
        residual = residual[: match.start()] + residual[match.end() :]
    if doc_type is not None:
        lowered = residual.lower()
        for form in _DOC_TYPE_KEYWORDS[doc_type]:
            idx = lowered.find(form)
            if idx != -1:
                residual = residual[:idx] + residual[idx + len(form) :]
                break
    residual = re.sub(r"\s+", " ", residual).strip()

    near_miss = None
    if not models and _NEAR_MISS_CODE_RE.search(query):
        near_miss = _NEAR_MISS_PRODUCT_CODE

    return models, doc_type, (residual or None), near_miss


def _doc_type_label(doc_type: str) -> str:
    return _DOC_TYPE_LABELS.get(doc_type, doc_type)


def _to_result(row: DocumentRow, *, requested: bool) -> RetrievalResult:
    """DELIBERATELY does not carry `source_path` -- since `DocumentRow` never
    had the field, "forgetting" it here is impossible; only the four fields
    below + the `requested` flag enter the metadata."""
    label = _doc_type_label(row.doc_type)
    return RetrievalResult(
        id=f"doc:{row.doc_id}:{row.model}",
        score=1.0 if requested else 0.5,
        text=f"{row.file_name} ({label}) -- product: {row.model}",
        metadata={
            "doc_id": row.doc_id,
            "model": row.model,
            "doc_type": row.doc_type,
            "file_name": row.file_name,
            "requested": requested,
        },
    )


def _available_types_result(models: list[str], available_types: list[str]) -> RetrievalResult:
    """A synthetic evidence row that shows the REAL options to the
    model/user when `doc_type` is ambiguous -- same principle as
    `ComparisonFlow._comparison_table` (advisory-only formatting, no invented
    data; just the readable labels of the actual `doc_type` values)."""
    labels = ", ".join(_doc_type_label(t) for t in available_types)
    return RetrievalResult(
        id="doc_download_types",
        score=1.0,
        text=(
            f"Multiple file types are available for {', '.join(models)}: {labels}. "
            "Do not pick/suggest downloading a file until the requested type is clear."
        ),
        metadata={"doc_download_clarify": True},
    )


class DocDownloadFlow:
    """`flow_name`/`intent_key` must exactly equal the flow name registered in
    the `Router` and the `strategies/<intent_key>.md` key (the same known
    limitation as `ComparisonFlow`/`RecommendationFlow`, see the
    session_state.py::PinnedFlow docstring)."""

    def __init__(
        self,
        document_lookup: DocumentLookup,
        *,
        flow_name: str = "doc_download",
        intent_key: str = "doc_download",
        continuation_checker: ContinuationChecker | None = None,
        few_max: int = 5,
    ) -> None:
        self._lookup = document_lookup
        self._flow_name = flow_name
        self._intent_key = intent_key
        self._continuation_checker = continuation_checker
        # Threshold for `FlowContext.result_shape` -- injected from `config/
        # default.toml [result_shape] few_max` (see
        # factory.py::build_flows_from_env).
        self._few_max = few_max

    async def _diagnose_empty(self, models: list[str], doc_type: str | None) -> str:
        """doc_download builds its OWN deterministic query (unlike `sql_topn`),
        so it can diagnose WHICH single clause emptied the result by dropping
        `is_active`/`doc_type`/`model` one at a time. A `DocumentLookup` that
        doesn't support `include_inactive` (e.g. older fakes in tests) dies
        with `TypeError` -- then only the `doc_type` diagnosis is attempted and
        the safest assumption (`"model"`) is returned (zero behavior change,
        the flow doesn't blow up)."""
        if doc_type is not None:
            without_doc_type = await self._lookup.find(models, None)
            if without_doc_type:
                return "doc_type"
        try:
            with_inactive = await self._lookup.find(models, doc_type, include_inactive=True)
        except TypeError:
            with_inactive = []
        if with_inactive:
            return "is_active"
        if doc_type is not None:
            try:
                with_inactive_any_type = await self._lookup.find(
                    models, None, include_inactive=True
                )
            except TypeError:
                with_inactive_any_type = []
            if with_inactive_any_type:
                return "is_active"
        return "model"

    async def _advance(
        self,
        raw_query: str,
        models: list[str],
        doc_type: str | None,
        on_trace: TraceFn | None,
        *,
        residual: str | None = None,
        near_miss: str | None = None,
    ) -> FlowContext:
        if not models:
            # Ambiguity (a): no product code at all -- drop to the
            # clarification path WITHOUT touching retrieval + pin the session
            # (see pin_request).
            if on_trace:
                on_trace("doc_download_clarification_needed", "hangi ürün")
            return FlowContext(
                results=[],
                scratch={"stage": "need_product", "raw_query": raw_query, "doc_type": doc_type},
                result_shape=compute_result_shape(0, few_max=self._few_max),
                residual=residual,
                parse_note=near_miss,
            )

        rows = await self._lookup.find(models, doc_type)
        if not rows:
            # No active file for this product/doc_type combination -- this is
            # NOT an ambiguity (nothing to clarify); `strategies/
            # doc_download.md` turns the empty evidence into a "not found"
            # answer.
            if on_trace:
                on_trace("doc_download_no_match", {"models": models, "doc_type": doc_type})
            # WHICH clause emptied the result -- doc_download-specific (see the
            # `_diagnose_empty` docstring).
            clause = await self._diagnose_empty(models, doc_type)
            return FlowContext(
                results=[],
                result_shape=compute_result_shape(0, few_max=self._few_max),
                residual=residual,
                parse_note=f"empty_result:{clause}",
            )

        if doc_type is None:
            distinct_types = sorted({r.doc_type for r in rows})
            if len(distinct_types) > 1:
                # Ambiguity (b): no type specified AND multiple different
                # types exist -- drop to the clarification path + pin.
                if on_trace:
                    on_trace("doc_download_clarification_needed", distinct_types)
                return FlowContext(
                    results=[_available_types_result(models, distinct_types)],
                    scratch={
                        "stage": "need_doc_type", "raw_query": raw_query,
                        "models": models, "available_types": distinct_types,
                    },
                    result_shape=compute_result_shape(len(rows), few_max=self._few_max),
                    residual=residual,
                    parse_note=near_miss,
                )

        # Resolved: the requested rows (`requested=True`) + the other active
        # documents of the SAME product(s) (`requested=False`, the follow-up
        # ground) -- deduped by `(doc_id, model)` against fan-out.
        requested_keys = {(r.doc_id, r.model) for r in rows}
        primary = [_to_result(r, requested=True) for r in rows]
        other_rows = await self._lookup.find(models, None)
        others = [
            _to_result(r, requested=False)
            for r in other_rows if (r.doc_id, r.model) not in requested_keys
        ]
        if on_trace:
            on_trace(
                "doc_download_resolved",
                {"models": models, "doc_type": doc_type, "count": len(primary)},
            )
        return FlowContext(
            results=[*primary, *others], scratch={},
            result_shape=compute_result_shape(len(primary), few_max=self._few_max),
            residual=residual, parse_note=near_miss,
        )

    async def run(self, query: str, on_trace: TraceFn | None = None) -> FlowContext:
        models, doc_type, residual, near_miss = parse_doc_download_query(query)
        return await self._advance(
            query, models, doc_type, on_trace, residual=residual, near_miss=near_miss
        )

    async def run_pinned(
        self, query: str, pinned: PinnedFlow, on_trace: TraceFn | None = None
    ) -> FlowContext:
        """`PinAwareRunFlow`: this turn's raw query is the ANSWER to the
        clarification question asked last turn -- the missing piece (product
        code or doc_type) is extracted from this turn and merged with the state
        carried from the previous turn (same pattern as
        `RecommendationFlow.run_pinned`)."""
        expected: dict[str, Any] = pinned.expected or {}
        raw_query = expected.get("raw_query", query)
        models = list(expected.get("models", []))
        doc_type = expected.get("doc_type")

        new_models, this_turn_doc_type, residual, near_miss = parse_doc_download_query(query)
        if new_models:
            models = new_models
        if this_turn_doc_type:
            doc_type = this_turn_doc_type

        return await self._advance(
            raw_query, models, doc_type, on_trace, residual=residual, near_miss=near_miss
        )

    async def check_pin(self, query: str, pinned: PinnedFlow) -> bool:
        """`PinnableFlow` (REVISED; fallback narrowed): while pinned, a
        clarification ANSWER (a product code or a file type) is EXPECTED. If
        `_continuation_checker` is injected (see factory.py), the decision
        delegates to it; otherwise (dev/test) it falls to
        `pin_relevance.looks_relevant_to_pin` -- looking for a product code or
        a doc_type depending on the expected slot (`stage`); if neither is
        found the pin DROPS (previously only the cancel-word could break it)."""
        if not query.strip():
            return False
        expected = pinned.expected or {}
        stage = expected.get("stage")
        if self._continuation_checker is None:
            models, doc_type, _residual, near_miss = parse_doc_download_query(query)
            if stage == "need_doc_type":
                has_signal = doc_type is not None
            else:
                has_signal = bool(models) or near_miss is not None
            return looks_relevant_to_pin(query, cancel_words=_CANCEL_WORDS, has_signal=has_signal)
        if stage == "need_doc_type":
            available = expected.get("available_types") or []
            context = (
                "The user was asked which file type they want"
                + (f" (options: {', '.join(available)})" if available else "")
                + ". The user is expected to answer or give up."
            )
        else:
            context = (
                "The user was asked which product they want to download a "
                "file for. The user is expected to answer or give up."
            )
        return await self._continuation_checker.check(query, context)

    async def pin_request(self, query: str, context: FlowContext) -> PinnedFlow | None:
        """`SelfPinningFlow`: if `stage` in `context.scratch` is filled (still
        clarifying), the pin is ESTABLISHED/UPDATED; once clarification ends
        (`_advance` returned real rows, `scratch` empty) the pin is
        CLEARED."""
        stage = context.scratch.get("stage")
        if not stage:
            return None
        expected: dict[str, Any] = {
            "raw_query": context.scratch["raw_query"],
            "doc_type": context.scratch.get("doc_type"),
        }
        if stage == "need_doc_type":
            expected["models"] = context.scratch["models"]
            expected["available_types"] = context.scratch["available_types"]
        return PinnedFlow(
            flow=self._flow_name, intent=self._intent_key,
            expected={"stage": stage, **expected},
        )


class SqliteDocumentLookup:
    """The real `DocumentLookup` implementation -- stdlib `sqlite3` against
    chatbot's OWN `specs.db` (see `factory.py::
    build_document_lookup_from_env`; the SAME `CHATBOT_DB_QUERY_DB_PATH` +
    bundled-fallback pattern is SHARED with
    `build_db_query_retriever_from_env`) via a deterministic SELECT. Blocking
    sqlite I/O is wrapped in `asyncio.to_thread` (same principle as
    `OpenAICompatAnsweringModel._call`'s own `asyncio.to_thread` support).
    DELIBERATELY never SELECTs `source_path` (see module docstring)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def _query(
        self, models: list[str], doc_type: str | None, *, include_inactive: bool
    ) -> list[DocumentRow]:
        # `document` no longer carries a `model` column -- the fan-out lives in
        # `document_owner`. `v_document_product` (facts/db/schema_rag.sql)
        # pre-expands that fan-out down to the product code; this query reads
        # the view where it used to look at `document` directly -- the
        # (doc_id, product_code) pair meaning and the `is_active`/`doc_type`
        # filter semantics are PRESERVED.
        conn = _sqlite_connect(self._db_path, readonly=True)
        try:
            placeholders = ",".join("?" for _ in models)
            sql = (
                "SELECT doc_id, product_code, doc_type, file_name FROM v_document_product "
                f"WHERE UPPER(product_code) IN ({placeholders})"
            )
            params: list[str] = [m.upper() for m in models]
            if not include_inactive:
                # `include_inactive` exists ONLY for DocDownloadFlow's own
                # zero-result diagnosis (`_diagnose_empty`) -- the normal
                # resolution path ALWAYS runs with the `is_active = 1` filter;
                # this default behavior is UNCHANGED.
                sql += " AND is_active = 1"
            if doc_type is not None:
                sql += " AND doc_type = ?"
                params.append(doc_type)
            sql += " ORDER BY doc_type, file_name"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        return [DocumentRow(doc_id=r[0], model=r[1], doc_type=r[2], file_name=r[3]) for r in rows]

    async def find(
        self, models: list[str], doc_type: str | None = None, *, include_inactive: bool = False
    ) -> list[DocumentRow]:
        if not models:
            return []
        return await asyncio.to_thread(self._query, models, doc_type, include_inactive=include_inactive)


__all__ = [
    "DocDownloadFlow",
    "DocumentLookup",
    "DocumentRow",
    "SqliteDocumentLookup",
    "extract_doc_type",
    "extract_product_codes",
    "parse_doc_download_query",
]
