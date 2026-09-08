"""`DocDownloadFlow` tests -- deterministic product-code/doc_type extraction,
clarification/pin flow, follow-up groundwork, and that `source_path` NEVER
reaches this flow. A fake `DocumentLookup` is used -- the real
sqlite/GPU is never touched."""

from __future__ import annotations

import asyncio

from medrag.api.flows.base import FlowContext
from medrag.api.flows.doc_download import (
    DocDownloadFlow,
    DocumentRow,
    extract_doc_type,
    extract_product_codes,
    parse_doc_download_query,
)
from medrag.api.session_state import PinnedFlow


class _FakeLookup:
    """Mimics `DocumentLookup` -- filters a subset of the `document` table
    from a fixed list (the pure-Python equivalent of the
    `is_active=1 AND model IN (...) [AND doc_type = ?]` query the real
    implementation runs -- see `SqliteDocumentLookup`). DELIBERATELY keeps
    the older signature (no `include_inactive` support) -- this locks in that
    `DocDownloadFlow._diagnose_empty` works with legacy fakes with zero
    behavior change via the TypeError fallback path."""

    def __init__(self, rows: list[DocumentRow]) -> None:
        self._rows = rows
        self.calls: list[tuple[list[str], str | None]] = []

    async def find(self, models: list[str], doc_type: str | None = None) -> list[DocumentRow]:
        self.calls.append((list(models), doc_type))
        upper_models = {m.upper() for m in models}
        rows = [r for r in self._rows if r.model.upper() in upper_models]
        if doc_type is not None:
            rows = [r for r in rows if r.doc_type == doc_type]
        return rows


class _FakeLookupWithInactive:
    """Identical to `_FakeLookup`, but SUPPORTS `include_inactive` -- to test
    that `_diagnose_empty` can really drop the `is_active` condition. The
    `active` field is a fixed set of active (doc_id, model) pairs -- with
    `include_inactive=False` rows OUTSIDE this set are filtered out."""

    def __init__(self, rows: list[DocumentRow], *, active: set[tuple[str, str]]) -> None:
        self._rows = rows
        self._active = active
        self.calls: list[tuple[list[str], str | None, bool]] = []

    async def find(
        self, models: list[str], doc_type: str | None = None, *, include_inactive: bool = False
    ) -> list[DocumentRow]:
        self.calls.append((list(models), doc_type, include_inactive))
        upper_models = {m.upper() for m in models}
        rows = [r for r in self._rows if r.model.upper() in upper_models]
        if not include_inactive:
            rows = [r for r in rows if (r.doc_id, r.model) in self._active]
        if doc_type is not None:
            rows = [r for r in rows if r.doc_type == doc_type]
        return rows


def _run(flow: DocDownloadFlow, query: str, on_trace=None) -> FlowContext:
    return asyncio.run(flow.run(query, on_trace))


def _run_pinned(flow: DocDownloadFlow, query: str, pinned: PinnedFlow, on_trace=None) -> FlowContext:
    return asyncio.run(flow.run_pinned(query, pinned, on_trace))


def _pin_request(flow: DocDownloadFlow, query: str, context: FlowContext):
    return asyncio.run(flow.pin_request(query, context))


# --- product code / doc_type extraction (LLM-free, deterministic) --------------


def test_extract_product_codes_finds_single_code_case_insensitive():
    assert extract_product_codes("de9001 için datasheet lazım") == ["PN1309"]


def test_extract_product_codes_normalizes_spaces_and_dashes():
    assert extract_product_codes("DE 9001 ve DE-1000 için dosya") == ["PN1309", "PN1015"]


def test_extract_product_codes_empty_when_none_found():
    assert extract_product_codes("datasheet gönderir misin") == []


def test_extract_product_codes_dedups_preserving_order():
    assert extract_product_codes("PN1309 ve tekrar PN1309") == ["PN1309"]


def test_extract_doc_type_matches_turkish_and_english_surface_forms():
    assert extract_doc_type("PN1309 datasheet gönder") == "DATASHEET"
    assert extract_doc_type("broşür istiyorum") == "BROCHURE"
    assert extract_doc_type("CE uygunluk beyanı lazım") == "CE_DECLARATION"
    assert extract_doc_type("teknik çizim var mı") == "TECHNICAL_DRAWING"


def test_extract_doc_type_none_when_unspecified():
    assert extract_doc_type("PN1309 için dosya gönder") is None


# --- missing product code: clarification + pin ----------------------------------


def test_no_product_code_asks_for_clarification_without_querying_lookup():
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "datasheet gönderir misin")

    assert context.results == []
    assert lookup.calls == []  # retrieval was never called
    assert context.scratch["stage"] == "need_product"


def test_pin_request_after_need_product_pins_session():
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)
    context = _run(flow, "datasheet gönderir misin")

    pin = _pin_request(flow, "datasheet gönderir misin", context)

    assert pin is not None
    assert pin.flow == "doc_download"
    assert pin.intent == "doc_download"
    assert pin.expected["stage"] == "need_product"
    assert pin.expected["raw_query"] == "datasheet gönderir misin"
    assert pin.expected["doc_type"] == "DATASHEET"


def test_second_turn_supplies_product_and_resolves():
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "PN5106_Datasheet.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={"stage": "need_product", "raw_query": "datasheet gönderir misin", "doc_type": "DATASHEET"},
    )

    context = _run_pinned(flow, "PN1309", pin)

    assert context.scratch == {}
    assert len(context.results) == 1
    assert context.results[0].metadata["doc_id"] == "d1"
    assert context.results[0].metadata["requested"] is True


# --- ambiguous doc type: clarification + pin ------------------------------------


def test_ambiguous_doc_type_asks_for_clarification_with_real_options():
    rows = [
        DocumentRow("d1", "PN1309", "DATASHEET", "PN5106_Datasheet.pdf"),
        DocumentRow("d2", "PN1309", "BROCHURE", "PN5106_Brochure.pdf"),
    ]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 için dosya gönder")

    assert context.scratch["stage"] == "need_doc_type"
    assert context.scratch["models"] == ["PN1309"]
    assert sorted(context.scratch["available_types"]) == ["BROCHURE", "DATASHEET"]
    # The clarification evidence lists REAL types -- nothing invented.
    assert len(context.results) == 1
    assert context.results[0].id == "doc_download_types"
    assert "Datasheet" in context.results[0].text
    assert "Brochure" in context.results[0].text
    assert context.results[0].metadata == {"doc_download_clarify": True}


def test_pin_request_after_need_doc_type_carries_models_and_types():
    rows = [
        DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf"),
        DocumentRow("d2", "PN1309", "BROCHURE", "b.pdf"),
    ]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)
    context = _run(flow, "PN1309 için dosya gönder")

    pin = _pin_request(flow, "PN1309 için dosya gönder", context)

    assert pin.expected["stage"] == "need_doc_type"
    assert pin.expected["models"] == ["PN1309"]
    assert sorted(pin.expected["available_types"]) == ["BROCHURE", "DATASHEET"]


def test_third_turn_supplies_doc_type_and_resolves():
    rows = [
        DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf"),
        DocumentRow("d2", "PN1309", "BROCHURE", "b.pdf"),
    ]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={
            "stage": "need_doc_type", "raw_query": "PN1309 için dosya gönder",
            "doc_type": None, "models": ["PN1309"], "available_types": ["BROCHURE", "DATASHEET"],
        },
    )

    context = _run_pinned(flow, "datasheet olsun", pin)

    assert context.scratch == {}
    assert len(context.results) == 2  # requested + the product's other active file
    requested = [r for r in context.results if r.metadata["requested"]]
    others = [r for r in context.results if not r.metadata["requested"]]
    assert len(requested) == 1
    assert requested[0].metadata["doc_type"] == "DATASHEET"
    assert len(others) == 1
    assert others[0].metadata["doc_type"] == "BROCHURE"


def test_pin_request_clears_once_doc_type_resolved():
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={
            "stage": "need_doc_type", "raw_query": "q", "doc_type": None,
            "models": ["PN1309"], "available_types": ["DATASHEET"],
        },
    )
    context = _run_pinned(flow, "datasheet", pin)

    next_pin = _pin_request(flow, "datasheet", context)

    assert next_pin is None


# --- resolved turns: follow-up groundwork (requested=False rows) ----------------


def test_resolved_single_type_includes_other_active_docs_as_not_requested():
    rows = [
        DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf"),
        DocumentRow("d2", "PN1309", "CE_DECLARATION", "b.pdf"),
    ]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 datasheet gönder")

    requested = [r for r in context.results if r.metadata["requested"]]
    others = [r for r in context.results if not r.metadata["requested"]]
    assert [r.metadata["doc_id"] for r in requested] == ["d1"]
    assert [r.metadata["doc_id"] for r in others] == ["d2"]


def test_resolved_result_never_carries_source_path_key():
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 datasheet gönder")

    for r in context.results:
        assert "source_path" not in r.metadata
        assert set(r.metadata) == {"doc_id", "model", "doc_type", "file_name", "requested"}


def test_single_active_document_no_ambiguity_even_without_doc_type():
    # If there is only ONE doc_type, there is no ambiguity (even when the
    # type is unspecified).
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 için dosya gönder")

    assert context.scratch == {}
    assert len(context.results) == 1
    assert context.results[0].metadata["requested"] is True


# --- not found: empty evidence, NOT ambiguity -------------------------------------


def test_no_matching_documents_returns_empty_results_without_pinning():
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1344 için datasheet gönder")

    assert context.results == []
    assert context.scratch == {}


def test_pin_request_none_when_context_has_no_stage():
    lookup = _FakeLookup([DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")])
    flow = DocDownloadFlow(lookup)
    context = _run(flow, "PN1309 datasheet gönder")

    assert _pin_request(flow, "PN1309 datasheet gönder", context) is None


# --- check_pin: cancel word breaks, normal reply holds -----------------------


def test_check_pin_holds_for_normal_reply():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={})
    assert asyncio.run(flow.check_pin("PN1309", pin)) is True


def test_check_pin_breaks_on_cancel_word():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={})
    assert asyncio.run(flow.check_pin("vazgeç", pin)) is False


def test_check_pin_breaks_on_empty_reply():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={})
    assert asyncio.run(flow.check_pin("   ", pin)) is False


class _FakeContinuationChecker:
    def __init__(self, result: bool):
        self._result = result
        self.calls: list[tuple[str, str]] = []

    async def check(self, query: str, context: str) -> bool:
        self.calls.append((query, context))
        return self._result


def test_check_pin_delegates_to_continuation_checker_when_injected():
    checker = _FakeContinuationChecker(True)
    flow = DocDownloadFlow(_FakeLookup([]), continuation_checker=checker)
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={"stage": "need_doc_type", "available_types": ["DATASHEET", "STP"]},
    )

    result = asyncio.run(flow.check_pin("aslında farklı bir şeyi konuşmak istiyorum ama devam", pin))

    assert result is True
    assert len(checker.calls) == 1
    _query, context = checker.calls[0]
    assert "DATASHEET" in context and "STP" in context


def test_check_pin_empty_reply_short_circuits_even_with_checker():
    checker = _FakeContinuationChecker(True)
    flow = DocDownloadFlow(_FakeLookup([]), continuation_checker=checker)
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={})

    assert asyncio.run(flow.check_pin("   ", pin)) is False
    assert checker.calls == []


# --- relevance gate when NO checker is injected (old behavior: everything True)


def test_check_pin_fallback_drops_pin_for_irrelevant_message_need_product():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={"stage": "need_product"})

    assert asyncio.run(flow.check_pin("bugün hava çok güzel", pin)) is False


def test_check_pin_fallback_keeps_pin_for_product_code_need_product():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={"stage": "need_product"})

    assert asyncio.run(flow.check_pin("PN1309 galiba", pin)) is True


def test_check_pin_fallback_drops_pin_for_irrelevant_message_need_doc_type():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={"stage": "need_doc_type", "models": ["PN1309"], "available_types": ["DATASHEET", "STP"]},
    )

    # No product code (already resolved) and no known doc_type either.
    assert asyncio.run(flow.check_pin("bilmem ki", pin)) is False


def test_check_pin_fallback_keeps_pin_for_doc_type_keyword_need_doc_type():
    flow = DocDownloadFlow(_FakeLookup([]))
    pin = PinnedFlow(
        flow="doc_download", intent="doc_download",
        expected={"stage": "need_doc_type", "models": ["PN1309"], "available_types": ["DATASHEET", "STP"]},
    )

    assert asyncio.run(flow.check_pin("datasheet olsun", pin)) is True


def test_check_pin_fallback_unaffected_when_checker_injected():
    checker = _FakeContinuationChecker(True)
    flow = DocDownloadFlow(_FakeLookup([]), continuation_checker=checker)
    pin = PinnedFlow(flow="doc_download", intent="doc_download", expected={"stage": "need_product"})

    assert asyncio.run(flow.check_pin("bugün hava çok güzel", pin)) is True


# --- fan-out: one doc_id across multiple model rows -------------------------------


def test_fan_out_document_shared_across_models_not_duplicated_in_others():
    rows = [
        DocumentRow("shared", "PN1309", "CATALOGUE", "catalog.pdf"),
        DocumentRow("shared", "PN1316", "CATALOGUE", "catalog.pdf"),
        DocumentRow("only9001", "PN1309", "DATASHEET", "ds.pdf"),
    ]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 datasheet gönder")

    # requested: only the datasheet; the fan-out catalogue must SHOW as
    # "other", but the SAME (doc_id, model) pair must not be added twice.
    requested = [r for r in context.results if r.metadata["requested"]]
    others = [r for r in context.results if not r.metadata["requested"]]
    assert [r.metadata["doc_id"] for r in requested] == ["only9001"]
    assert [r.metadata["doc_id"] for r in others] == ["shared"]


# --- parse_doc_download_query (pure parsing core) --------------------------------


def test_parse_doc_download_query_finds_code_and_type_strips_residual():
    models, doc_type, residual, near_miss = parse_doc_download_query(
        "PN1309 için lütfen datasheet gönder"
    )
    assert models == ["PN1309"]
    assert doc_type == "DATASHEET"
    assert residual == "için lütfen gönder"
    assert near_miss is None


def test_parse_doc_download_query_no_product_no_near_miss():
    models, doc_type, _residual, near_miss = parse_doc_download_query("datasheet gönderir misin")
    assert models == []
    assert doc_type == "DATASHEET"
    assert near_miss is None


def test_parse_doc_download_query_near_miss_product_code():
    models, _doc_type, _residual, near_miss = parse_doc_download_query("DE12 için dosya lazım")
    assert models == []
    assert near_miss == "near_miss_product_code"


def test_parse_doc_download_query_no_near_miss_when_valid_code_present():
    models, _doc_type, _residual, near_miss = parse_doc_download_query("PN1309 için dosya")
    assert models == ["PN1309"]
    assert near_miss is None


# --- FlowContext.result_shape/residual/parse_note ----------------------------------


def test_no_product_code_sets_zero_shape_and_residual():
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "lütfen datasheet gönderir misin")

    assert context.result_shape == "zero"
    assert context.residual == "lütfen gönderir misin"
    assert context.parse_note is None


def test_near_miss_product_code_sets_parse_note():
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "DE12 için dosya lazım")

    assert context.result_shape == "zero"
    assert context.parse_note == "near_miss_product_code"


def test_resolved_query_sets_one_result_shape_for_single_match():
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup, few_max=5)

    context = _run(flow, "PN1309 datasheet gönder")

    assert context.result_shape == "one"


def test_resolved_query_sets_few_result_shape_for_multiple_matches():
    rows = [DocumentRow(f"d{i}", "PN1309", "DATASHEET", f"f{i}.pdf") for i in range(3)]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup, few_max=5)

    context = _run(flow, "PN1309 datasheet gönder")

    assert context.result_shape == "few"


def test_resolved_query_result_shape_many_above_few_max():
    rows = [DocumentRow(f"d{i}", "PN1309", "DATASHEET", f"f{i}.pdf") for i in range(4)]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup, few_max=2)

    context = _run(flow, "PN1309 datasheet gönder")

    assert context.result_shape == "many"


# --- _diagnose_empty: WHICH clause emptied the result (doc_download only) ---------


def test_empty_result_diagnoses_doc_type_as_blocking_clause():
    # PN1309 is active but only has a BROCHURE -- the requested DATASHEET is missing.
    rows = [DocumentRow("d1", "PN1309", "BROCHURE", "a.pdf")]
    lookup = _FakeLookup(rows)
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 datasheet gönder")

    assert context.results == []
    assert context.parse_note == "empty_result:doc_type"


def test_empty_result_diagnoses_is_active_as_blocking_clause():
    # A DATASHEET exists for PN1309 but it is INACTIVE -- when the lookup
    # finds it with include_inactive=True, "is_active" is blamed.
    rows = [DocumentRow("d1", "PN1309", "DATASHEET", "a.pdf")]
    lookup = _FakeLookupWithInactive(rows, active=set())  # none of them active
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1309 datasheet gönder")

    assert context.results == []
    assert context.parse_note == "empty_result:is_active"


def test_empty_result_diagnoses_model_as_default_when_nothing_else_explains_it():
    # PN1344 does not exist at all (not active, not inactive, no other type)
    # -- if the lookup still returns empty even with `include_inactive`,
    # "model" is the default blame.
    lookup = _FakeLookupWithInactive([], active=set())
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1344 datasheet gönder")

    assert context.results == []
    assert context.parse_note == "empty_result:model"


def test_empty_result_diagnosis_falls_back_gracefully_without_include_inactive_support():
    # The legacy fake lookup doesn't support `include_inactive` -- the
    # TypeError fallback drops to the "model" default without blowing up.
    lookup = _FakeLookup([])
    flow = DocDownloadFlow(lookup)

    context = _run(flow, "PN1344 datasheet gönder")

    assert context.results == []
    assert context.parse_note == "empty_result:model"
