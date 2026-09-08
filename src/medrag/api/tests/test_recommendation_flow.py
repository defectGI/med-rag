import asyncio
import sqlite3

import yaml

from medrag.api.flows.base import FlowContext
from medrag.api.flows.recommendation import (
    RecommendationFlow,
    _question_for,
    parse_family_query,
    parse_recommendation_answer,
)
from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import PinnedFlow


class _FakeSqlTopNPass:
    """Mimics `SqlTopNFlow.run_pass`/`top_n_pass` -- records only the call
    count/arguments (to verify the existing core is used VERBATIM)."""

    def __init__(
        self,
        results: list[RetrievalResult] | None = None,
        *,
        top_n_results: list[RetrievalResult] | None = None,
    ):
        self.calls: list[str] = []
        self.top_n_calls: list[str] = []
        self._results = results if results is not None else []
        self._top_n_results = top_n_results if top_n_results is not None else []

    async def run_pass(self, query: str, on_trace=None, **kwargs) -> list[RetrievalResult]:
        self.calls.append(query)
        return list(self._results)

    async def top_n_pass(self, query: str, on_trace=None) -> list[RetrievalResult]:
        self.top_n_calls.append(query)
        return list(self._top_n_results)


def _run(flow: RecommendationFlow, query: str, on_trace=None) -> FlowContext:
    return asyncio.run(flow.run(query, on_trace))


def _run_pinned(flow: RecommendationFlow, query: str, pinned: PinnedFlow, on_trace=None) -> FlowContext:
    return asyncio.run(flow.run_pinned(query, pinned, on_trace))


def _pin_request(flow: RecommendationFlow, query: str, context: FlowContext):
    return asyncio.run(flow.pin_request(query, context))


def _rows(*codes: str) -> list[RetrievalResult]:
    """Fake SQL rows each carrying a different `product_code` -- used to set
    the "remaining candidates" count per test (see
    `RecommendationFlow._remaining_codes`)."""
    return [
        RetrievalResult(id=f"r{i}", score=1.0, metadata={"product_code": code})
        for i, code in enumerate(codes)
    ]


# --- no answer-COUNT threshold -- with any constraint in hand the SQL filter
# runs IMMEDIATELY; clarification ends when the candidate set narrows ---------


def test_first_turn_asks_family_without_calling_run_pass_but_calls_top_n():
    sql_pass = _FakeSqlTopNPass()
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "bana bir öneri yapar mısın")

    assert context.results == []  # the fake top_n_pass returned empty
    assert sql_pass.calls == []  # SQL enrichment was never called
    assert sql_pass.top_n_calls == ["bana bir öneri yapar mısın"]  # but the top_n helper ran
    assert context.scratch["asking"] == "family"


def test_clarifying_turn_surfaces_supportive_top_n_results():
    # Even when the family matches no surface form (e.g. "savaş uçağı") the
    # user is not left empty-handed -- supportive results from qdrant appear
    # TOGETHER with the clarifying question.
    top_n_hit = RetrievalResult(id="d1", score=0.8, metadata={"product_code": "PN1344"})
    sql_pass = _FakeSqlTopNPass(top_n_results=[top_n_hit])
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "savaş uçağına malzeme lazım")

    assert context.results == [top_n_hit]
    assert sql_pass.calls == []  # SQL still did not run
    assert context.scratch["asking"] == "family"
    assert context.result_shape == "one"


def test_pin_request_after_first_question_pins_session():
    sql_pass = _FakeSqlTopNPass()
    flow = RecommendationFlow(sql_pass)
    context = _run(flow, "bana bir öneri yapar mısın")

    pin = _pin_request(flow, "bana bir öneri yapar mısın", context)

    assert pin is not None
    assert pin.flow == "recommendation"
    assert pin.intent == "recommendation"
    assert pin.expected["asking"] == "family"
    assert pin.expected["collected"] == {}


def test_filter_starts_as_soon_as_any_constraint_is_in_hand():
    # Previously `min_answered=2` kept SQL from running even with ONE
    # constraint in hand. Now the filter runs as soon as `collected` is
    # non-empty; if the candidate set is still wide (few_max=5), asking continues.
    sql_pass = _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6"))
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert sql_pass.calls == ["bana bir öneri yapar mısın | DAQ SYSTEMS"]  # the filter ran IMMEDIATELY
    assert sql_pass.top_n_calls == []  # no fall-through to the top_n-only branch while a constraint exists
    assert context.scratch["collected"] == {"family": "DAQ SYSTEMS"}
    assert context.scratch["asking"] != "family"  # the NEXT (different) slot was asked
    assert context.scratch["asked"] == ["family", context.scratch["asking"]]


def test_clarification_ends_once_candidate_set_is_small_enough():
    # When the candidate set drops to `few_max` (default 5) asking STOPS --
    # what matters is what REMAINS, not the answer count.
    sql_pass = _FakeSqlTopNPass(results=_rows("DE1", "DE2"))
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert len(sql_pass.calls) == 1
    assert context.scratch == {}  # clarification done -- even with one answer
    assert context.suggested_next_flow == "comparison"


def test_clarification_ends_when_filter_leaves_no_candidate():
    # If the constraints match no product at all, one more question could
    # only narrow the set further -- the flow ends and the model should
    # suggest relaxing constraints (see strategies/recommendation.md).
    sql_pass = _FakeSqlTopNPass(results=[])
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert len(sql_pass.calls) == 1
    assert context.scratch == {}
    assert context.result_shape == "zero"


def test_unmatched_family_answer_is_kept_as_free_text_and_drives_the_filter():
    # Real log scenario: the user answered the family question with concrete
    # constraints ("voltaj aralığı 20 falan, kanal sayısı 8"); because it
    # matched no family surface form it USED TO be discarded entirely ->
    # `collected` stayed empty and the filter never started.
    sql_pass = _FakeSqlTopNPass(results=_rows("DE1", "DE2"))
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "hangi ürünü seçmeliyim?"},
    )

    context = _run_pinned(flow, "voltaj aralığı 20 falan, kanal sayısı 8", pin)

    assert len(sql_pass.calls) == 1  # text kept -> filter ran
    assert "voltaj aralığı 20 falan, kanal sayısı 8" in sql_pass.calls[0]
    assert context.scratch == {}  # 2 candidates left -> clarification ended
    # A non-matching family is NOT invented -- only the free text is stored.
    assert "family" not in dict(pin.expected["collected"])


def test_third_turn_completes_clarification_and_calls_run_pass_exactly_once():
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1015"}),
    ])
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={
            "collected": {"family": "DAQ SYSTEMS"},
            "asked": ["family", "operating_temperature"],
            "asking": "operating_temperature",
            "raw_query": "bana bir öneri yapar mısın",
        },
    )

    context = _run_pinned(flow, "0-50 derece arası olmalı", pin)

    assert len(sql_pass.calls) == 1  # exactly ONCE
    synthesized = sql_pass.calls[0]
    assert "DAQ SYSTEMS" in synthesized
    assert "operating_temperature" in synthesized
    assert "0-50 derece arası olmalı" in synthesized
    assert context.results == [RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1015"})]
    assert context.scratch == {}  # clarification done, no work state carried over


def test_pin_request_clears_once_clarification_is_done():
    sql_pass = _FakeSqlTopNPass(results=[])
    flow = RecommendationFlow(sql_pass)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family", "raw_query": "öner"},
    )
    context = _run_pinned(flow, "DAQ istiyorum", pin)

    next_pin = _pin_request(flow, "DAQ istiyorum", context)

    assert next_pin is None


# --- already-given info is NEVER ASKED AGAIN --------------------------------------


def test_opportunistic_family_match_on_first_turn_skips_asking_it_again():
    # Keep the candidate set wide so clarification continues (otherwise it
    # would end on the first turn) -- what's tested is that family is not
    # asked AGAIN.
    sql_pass = _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6"))
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "PXI Express için bir öneri istiyorum")

    assert context.scratch["collected"] == {"family": "PXI EXPRESS SYSTEMS"}
    assert context.scratch["asking"] != "family"  # family was already given, not re-asked


# --- the question pool is limited to fields that exist in the schema --------------


def test_clarifying_pool_never_asks_about_unknown_attribute_like_budget():
    # Without dynamic discovery configured (as in this test) the emergency
    # pool is used -- the `number_of_channels` drift found during an audit
    # was fixed to `channel_count` (its real counterpart in the glossary).
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pool = flow._legacy_pool_keys()
    assert "budget" not in pool
    assert "bütçe" not in pool
    assert pool == ["family", "operating_temperature", "channel_count", "power_consumption"]


def test_pool_exhaustion_with_all_skip_answers_still_terminates():
    # If the user skips EVERY question, no constraint accumulates -- with
    # nothing to filter, SQL never runs (top_n only), but the flow still
    # terminates (pool exhausted -- family + 3 attributes = 4 slots).
    sql_pass = _FakeSqlTopNPass(results=[])
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "bir şey öner")
    assert sql_pass.calls == []
    asked_slots = [context.scratch["asking"]]

    max_turns = 10  # safety net against infinite loops -- the real limit is 4 questions
    for _ in range(max_turns):
        if not context.scratch:  # clarification ended
            break
        pin = PinnedFlow(
            flow="recommendation", intent="recommendation", expected=dict(context.scratch),
        )
        context = _run_pinned(flow, "farketmez", pin)
        if context.scratch:
            asked_slots.append(context.scratch["asking"])

    assert sql_pass.calls == []  # no constraint accumulated -> nothing to filter
    assert asked_slots == ["family", "operating_temperature", "channel_count", "power_consumption"]
    assert context.scratch == {}  # no deadlock; ended when the pool ran out
    assert context.results == []


# --- check_pin: cancel word breaks, normal reply holds -----------------------


def test_check_pin_holds_for_normal_reply():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(flow="recommendation", intent="recommendation", expected={})
    assert asyncio.run(flow.check_pin("DAQ istiyorum", pin)) is True


def test_check_pin_breaks_on_cancel_word():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(flow="recommendation", intent="recommendation", expected={})
    assert asyncio.run(flow.check_pin("iptal", pin)) is False


# --- check_pin: delegates to continuation_checker when injected ---------------------


class _FakeContinuationChecker:
    """Implements the `ContinuationChecker` protocol -- returns a fixed
    result and records which (query, context) it was called with."""

    def __init__(self, result: bool):
        self._result = result
        self.calls: list[tuple[str, str]] = []

    async def check(self, query: str, context: str) -> bool:
        self.calls.append((query, context))
        return self._result


def test_check_pin_delegates_to_continuation_checker_when_injected():
    checker = _FakeContinuationChecker(True)
    flow = RecommendationFlow(_FakeSqlTopNPass(), continuation_checker=checker)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"asking": "family"},
    )

    result = asyncio.run(flow.check_pin("aslında farklı bir şey soracaktım", pin))

    assert result is True  # the checker's return value -- the word list was never consulted
    assert len(checker.calls) == 1
    query, context = checker.calls[0]
    assert query == "aslında farklı bir şey soracaktım"
    # Not the raw slot key (no dict-shaped "asking": "family"); the natural-
    # language question text must be passed -- since the English translation
    # coincidentally also contains "family" (product family), we look for the
    # full natural-language question, not the bare key.
    assert _question_for("family", {}) in context


def test_check_pin_with_checker_can_break_on_non_cancel_word_text():
    # Once a checker is injected, _CANCEL_WORDS is never consulted -- a
    # phrase outside the list can also break the pin if the checker says
    # "no" (impossible under the old behavior).
    checker = _FakeContinuationChecker(False)
    flow = RecommendationFlow(_FakeSqlTopNPass(), continuation_checker=checker)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"asking": "operating_temperature"},
    )

    result = asyncio.run(flow.check_pin("bu arada başka bir ürün var mı", pin))

    assert result is False


# --- relevance gate when NO checker is injected (old behavior: everything True) ---


def test_check_pin_fallback_drops_pin_for_irrelevant_message_family_slot():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(flow="recommendation", intent="recommendation", expected={"asking": "family"})

    assert asyncio.run(flow.check_pin("bugün hava çok güzel", pin)) is False


def test_check_pin_fallback_keeps_pin_for_family_match():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(flow="recommendation", intent="recommendation", expected={"asking": "family"})

    assert asyncio.run(flow.check_pin("DAQ istiyorum", pin)) is True


def test_check_pin_fallback_keeps_pin_for_skip_word_family_slot():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(flow="recommendation", intent="recommendation", expected={"asking": "family"})

    assert asyncio.run(flow.check_pin("farketmez", pin)) is True


def test_check_pin_fallback_drops_pin_for_irrelevant_message_attribute_slot():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"asking": "operating_temperature"},
    )

    assert asyncio.run(flow.check_pin("bu arada başka bir ürün var mı", pin)) is False


def test_check_pin_fallback_keeps_pin_for_numeric_answer_attribute_slot():
    flow = RecommendationFlow(_FakeSqlTopNPass())
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"asking": "operating_temperature"},
    )

    assert asyncio.run(flow.check_pin("-40 ile 85 derece arası", pin)) is True


def test_check_pin_fallback_unaffected_when_checker_injected():
    checker = _FakeContinuationChecker(True)
    flow = RecommendationFlow(_FakeSqlTopNPass(), continuation_checker=checker)
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"asking": "operating_temperature"},
    )

    assert asyncio.run(flow.check_pin("bu arada başka bir ürün var mı", pin)) is True


# --- multi-product handoff offer ---------------------------------------------------


def test_handoff_suggested_when_two_or_more_distinct_models_returned():
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1015"}),
        RetrievalResult(id="r2", score=0.9, metadata={"product_code": "PN1036"}),
    ])
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "DAQ sistemi öner")

    assert context.suggested_next_flow == "comparison"
    assert context.suggested_next_candidates == ["PN1015", "PN1036"]


def test_no_handoff_when_single_distinct_model_returned():
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1015"}),
        RetrievalResult(id="r2", score=0.9, metadata={"product_code": "PN1015"}),
    ])
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "DAQ sistemi öner")

    assert context.suggested_next_flow is None
    assert context.suggested_next_candidates == []


def test_handoff_candidates_capped_at_max():
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id=f"r{i}", score=1.0, metadata={"product_code": f"DE{i}"})
        for i in range(6)
    ])
    # few_max=10: treat the 6 candidates as "narrow enough to present" so
    # clarification ends and the handoff branch runs (this test measures the
    # cap, not the narrowing).
    flow = RecommendationFlow(sql_pass, max_handoff_candidates=3, few_max=10)

    context = _run(flow, "DAQ sistemi öner")

    assert context.suggested_next_flow == "comparison"
    assert len(context.suggested_next_candidates) == 3


# --- parse_family_query / parse_recommendation_answer (pure) -------------------------


def test_parse_family_query_finds_family_and_strips_residual():
    family, residual, near_miss = parse_family_query("PXI Express için bir öneri istiyorum")
    assert family == "PXI EXPRESS SYSTEMS"
    assert residual == "için bir öneri istiyorum"
    assert near_miss is None


def test_parse_family_query_no_match_keeps_full_query_as_residual():
    family, residual, near_miss = parse_family_query("bana bir öneri yapar mısın")
    assert family is None
    assert residual == "bana bir öneri yapar mısın"
    assert near_miss is None


def test_parse_recommendation_answer_records_free_text_slot():
    assert parse_recommendation_answer("operating_temperature", "0-50 derece arası") == (
        "0-50 derece arası"
    )


def test_parse_recommendation_answer_skip_word_returns_none():
    assert parse_recommendation_answer("operating_temperature", "farketmez") is None


def test_parse_recommendation_answer_empty_reply_returns_none():
    assert parse_recommendation_answer("number_of_channels", "   ") is None


def test_parse_recommendation_answer_family_slot_returns_matched_family():
    assert parse_recommendation_answer("family", "DAQ sistemi lazım") == "DAQ SYSTEMS"


def test_parse_recommendation_answer_family_slot_unmatched_returns_none():
    # A non-matching family is NOT invented.
    assert parse_recommendation_answer("family", "bilmiyorum ne istediğimi") is None


# --- FlowContext.result_shape/residual ------------------------------------------------


def test_first_turn_clarifying_sets_zero_shape_and_residual():
    sql_pass = _FakeSqlTopNPass()
    flow = RecommendationFlow(sql_pass)

    context = _run(flow, "PXI Express için bir öneri istiyorum")

    assert context.result_shape == "zero"
    assert context.residual == "için bir öneri istiyorum"


def test_resolved_recommendation_sets_result_shape():
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1015"}),
    ])
    flow = RecommendationFlow(sql_pass, few_max=5)

    context = _run(flow, "DAQ sistemi öner")

    assert context.result_shape == "one"


def test_resolved_recommendation_result_shape_many_above_few_max():
    # One product, 4 evidence rows -> the candidate set narrowed
    # (clarification ends) but `result_shape` keeps summarizing the EVIDENCE count.
    sql_pass = _FakeSqlTopNPass(results=[
        RetrievalResult(id=f"r{i}", score=1.0, metadata={"product_code": "PN1015"})
        for i in range(4)
    ])
    flow = RecommendationFlow(sql_pass, few_max=2)

    context = _run(flow, "DAQ sistemi öner")

    assert context.result_shape == "many"


# --- dynamic question pool (schema + LLM choice) ------------------------------------


class _FakeSlotSelector:
    """Implements the `SlotSelector` protocol -- returns a fixed key (or
    `None`), or raises (to simulate a network error)."""

    def __init__(self, choice: str | None = None, *, error: Exception | None = None):
        self._choice = choice
        self._error = error
        self.calls: list[tuple[str, list[str]]] = []

    async def choose(self, context: str, candidates) -> str | None:
        self.calls.append((context, [c.key for c in candidates]))
        if self._error is not None:
            raise self._error
        return self._choice


def _write_schema_yaml(tmp_path, glossary_rows):
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump({"attribute_glossary": glossary_rows}), encoding="utf-8")
    return path


def _write_spec_value_db(tmp_path, rows):
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE spec_value (product_code TEXT, family TEXT, block TEXT, key TEXT, "
        "status TEXT, raw_text TEXT)"
    )
    con.executemany(
        "INSERT INTO spec_value (product_code, family, block, key, status, raw_text) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return str(db_path)


_GLOSSARY_ROWS = [
    {"block": "core", "key": "operating_temperature", "kind": "range", "unit": "C",
     "question": "What temperature range?"},
    {"block": "analog_io", "key": "channel_count", "kind": "single", "unit": None,
     "question": "How many channels?"},
]


def test_dynamic_mode_asks_the_llm_chosen_key_instead_of_legacy_pool(tmp_path):
    schema_path = _write_schema_yaml(tmp_path, _GLOSSARY_ROWS)
    db_path = _write_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    selector = _FakeSlotSelector(choice="channel_count")
    flow = RecommendationFlow(
        _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6")),
        slot_selector=selector, db_path=db_path, schema_yaml_path=schema_path,
    )
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert context.scratch["asking"] == "channel_count"  # the legacy pool would have asked "operating_temperature" first
    assert len(selector.calls) == 1
    _ctx, candidate_keys = selector.calls[0]
    assert candidate_keys == ["channel_count"]  # the only key that is GENUINELY informative for the family


def test_dynamic_mode_falls_back_to_legacy_pool_when_selector_errors(tmp_path):
    schema_path = _write_schema_yaml(tmp_path, _GLOSSARY_ROWS)
    db_path = _write_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    selector = _FakeSlotSelector(error=RuntimeError("network down"))
    flow = RecommendationFlow(
        _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6")),
        slot_selector=selector, db_path=db_path, schema_yaml_path=schema_path,
    )
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    # The selector errored -- the flow did NOT break; it fell back to the emergency pool.
    assert context.scratch["asking"] == "operating_temperature"


def test_dynamic_mode_falls_back_to_legacy_pool_when_no_informative_candidates(tmp_path):
    schema_path = _write_schema_yaml(tmp_path, _GLOSSARY_ROWS)
    # A single value -- no key discriminates for the family (see slot_candidates tests).
    db_path = _write_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
    ])
    selector = _FakeSlotSelector(choice="channel_count")
    flow = RecommendationFlow(
        _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6")),
        slot_selector=selector, db_path=db_path, schema_yaml_path=schema_path,
    )
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert context.scratch["asking"] == "operating_temperature"  # legacy pool
    assert selector.calls == []  # no candidates; the selector was never called


def test_dynamic_mode_llm_can_decline_all_candidates(tmp_path):
    schema_path = _write_schema_yaml(tmp_path, _GLOSSARY_ROWS)
    db_path = _write_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    selector = _FakeSlotSelector(choice=None)  # LLM: none worth asking about
    flow = RecommendationFlow(
        _FakeSqlTopNPass(results=_rows("DE1", "DE2", "DE3", "DE4", "DE5", "DE6")),
        slot_selector=selector, db_path=db_path, schema_yaml_path=schema_path,
    )
    pin = PinnedFlow(
        flow="recommendation", intent="recommendation",
        expected={"collected": {}, "asked": ["family"], "asking": "family",
                  "raw_query": "bana bir öneri yapar mısın"},
    )

    context = _run_pinned(flow, "DAQ sistemi lazım", pin)

    assert len(selector.calls) == 1
    assert context.scratch["asking"] == "operating_temperature"  # the legacy pool kicked in


def test_dynamic_mode_disabled_without_all_three_dependencies():
    # slot_selector given but no db_path/schema_yaml_path -- dynamic mode
    # must NOT activate (graceful degrade: one missing means all off).
    selector = _FakeSlotSelector(choice="channel_count")
    flow = RecommendationFlow(_FakeSqlTopNPass(), slot_selector=selector)
    assert flow._dynamic_enabled() is False
