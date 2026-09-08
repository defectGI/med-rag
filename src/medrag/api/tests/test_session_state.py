from medrag.api.retrieval.core import RetrievalResult
from medrag.api.session_state import (
    ConstraintDelta,
    PinnedFlow,
    SessionState,
    SessionStateStore,
    Subject,
    TaskSnapshot,
)


def test_as_goal_prompt_empty_when_no_task_or_focus():
    assert SessionState().as_goal_prompt() == ""


def test_as_goal_prompt_renders_all_parts():
    state = SessionState(
        task="de1000 vs de1100 karşılaştırması",
        subjects=[Subject(entity="de1000", last_turn=3), Subject(entity="de1100", last_turn=3)],
        constraints=["soğuk ortam"],
        focus="hangisi düşük sıcaklıkta çalışır",
    )
    out = state.as_goal_prompt()
    assert "de1000 vs de1100" in out
    assert "de1000, de1100" in out
    assert "soğuk ortam" in out
    assert "düşük sıcaklıkta" in out


# --- transition note -----------------------------------------------------------


def test_as_goal_prompt_switch_appends_note():
    state = SessionState(task="yeni görev", focus="fiyat")
    out = state.as_goal_prompt(transition="switch")
    assert "yeni görev" in out
    assert "new goal" in out.lower()  # switch note appended


def test_as_goal_prompt_digress_appends_note():
    state = SessionState(task="karşılaştırma", focus="güç")
    out = state.as_goal_prompt(transition="digress")
    assert "digression" in out.lower()  # digress note appended
    assert "karşılaştırma" in out


def test_as_goal_prompt_continue_no_note():
    state = SessionState(task="karşılaştırma", focus="güç")
    out = state.as_goal_prompt(transition="continue")
    # continue/evolve run smoothly -- NO extra note.
    assert "NOTE:" not in out


def test_as_goal_prompt_note_suppressed_when_no_goal():
    # No task+focus means no goal -> not even a switch note; returns "".
    assert SessionState().as_goal_prompt(transition="switch") == ""


def test_snapshot_drops_suspended_and_turn():
    state = SessionState(task="t", focus="f", turn=5,
                         subjects=[Subject(entity="x", last_turn=5)])
    snap = state.snapshot()
    assert snap.task == "t"
    assert snap.focus == "f"
    assert [s.entity for s in snap.subjects] == ["x"]
    assert not hasattr(snap, "turn")  # TaskSnapshot has no turn/suspended


# --- last_results ---------------------------------------------------------------


def test_last_results_defaults_to_empty():
    assert SessionState().last_results == []


def test_task_snapshot_has_no_last_results_field():
    assert "last_results" not in TaskSnapshot.model_fields


def test_snapshot_drops_last_results():
    state = SessionState(
        task="t", focus="f",
        last_results=[RetrievalResult(id="r1", score=1.0)],
    )
    snap = state.snapshot()
    assert not hasattr(snap, "last_results")


# --- available_followups / followups_prompt -------------------------------------


def test_available_followups_empty_when_no_last_results():
    assert SessionState().available_followups() == []


def test_available_followups_derives_from_spec_key_column():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
    ])
    assert state.available_followups() == ["PN1309: operating_temperature"]


def test_available_followups_skips_non_present_status():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "input_voltage", "status": "absent",
        }),
        RetrievalResult(id="r2", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "not_specified",
        }),
    ])
    assert state.available_followups() == []


def test_available_followups_no_status_field_still_counts():
    # Some rows (e.g. spec_chunk view) may have no status -- absence must not
    # block; only rows explicitly NOT present are filtered out.
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "input_voltage"}),
    ])
    assert state.available_followups() == ["PN1309: input_voltage"]


def test_available_followups_ignores_top_n_hits_without_key_column():
    # top_n chunk metadata has no spec_key (different schema) -- silently skipped.
    state = SessionState(last_results=[
        RetrievalResult(id="c1", score=0.9, metadata={"doc_id": "PN1022", "page_start": 3}),
    ])
    assert state.available_followups() == []


def test_available_followups_dedups_preserving_order():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={"product_code": "PN1309", "key": "a", "status": "present"}),
        RetrievalResult(id="r2", score=1.0, metadata={"product_code": "PN1309", "key": "b", "status": "present"}),
        RetrievalResult(id="r3", score=1.0, metadata={"product_code": "PN1309", "key": "a", "status": "present"}),
    ])
    assert state.available_followups() == ["PN1309: a", "PN1309: b"]


def test_available_followups_without_model_column_uses_bare_key():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={"key": "input_voltage", "status": "present"}),
    ])
    assert state.available_followups() == ["input_voltage"]


def test_available_followups_respects_limit():
    results = [
        RetrievalResult(id=f"r{i}", score=1.0,
                         metadata={"product_code": "PN1309", "key": f"k{i}", "status": "present"})
        for i in range(5)
    ]
    state = SessionState(last_results=results)
    assert len(state.available_followups(limit=2)) == 2


def test_followups_prompt_empty_when_no_candidates():
    assert SessionState().followups_prompt() == ""


def test_followups_prompt_contains_candidates_when_present():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
    ])
    prompt = state.followups_prompt()
    assert "PN1309: operating_temperature" in prompt
    assert "MAKE UP" not in prompt  # that note belongs to answering_model, not here


# --- available_document_followups / followups_prompt ----------------------------


def test_available_document_followups_empty_when_no_last_results():
    assert SessionState().available_document_followups() == []


def test_available_document_followups_derives_from_not_requested_document_rows():
    state = SessionState(last_results=[
        RetrievalResult(id="doc:d1:PN1309", score=1.0, metadata={
            "doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
            "file_name": "a.pdf", "requested": True,
        }),
        RetrievalResult(id="doc:d2:PN1309", score=0.5, metadata={
            "doc_id": "d2", "model": "PN1309", "doc_type": "BROCHURE",
            "file_name": "b.pdf", "requested": False,
        }),
    ])
    assert state.available_document_followups() == ["b.pdf (BROCHURE)"]


def test_available_document_followups_skips_requested_rows():
    state = SessionState(last_results=[
        RetrievalResult(id="doc:d1:PN1309", score=1.0, metadata={
            "doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
            "file_name": "a.pdf", "requested": True,
        }),
    ])
    assert state.available_document_followups() == []


def test_available_document_followups_ignores_spec_rows_without_doc_id():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
    ])
    assert state.available_document_followups() == []


def test_available_document_followups_dedups_by_doc_id_across_fan_out_rows():
    state = SessionState(last_results=[
        RetrievalResult(id="doc:shared:PN1309", score=0.5, metadata={
            "doc_id": "shared", "model": "PN1309", "doc_type": "CATALOGUE",
            "file_name": "catalog.pdf", "requested": False,
        }),
        RetrievalResult(id="doc:shared:PN1316", score=0.5, metadata={
            "doc_id": "shared", "model": "PN1316", "doc_type": "CATALOGUE",
            "file_name": "catalog.pdf", "requested": False,
        }),
    ])
    assert state.available_document_followups() == ["catalog.pdf (CATALOGUE)"]


def test_available_document_followups_respects_limit():
    results = [
        RetrievalResult(id=f"doc:d{i}:PN1309", score=0.5, metadata={
            "doc_id": f"d{i}", "model": "PN1309", "doc_type": "PRODUCT_IMAGE",
            "file_name": f"img{i}.png", "requested": False,
        })
        for i in range(5)
    ]
    state = SessionState(last_results=results)
    assert len(state.available_document_followups(limit=2)) == 2


# --- followups_prompt: single candidate, deterministic priority -----------------


def test_followups_prompt_picks_only_spec_candidate_when_both_available():
    # When both a spec and a document candidate exist, only one spec candidate
    # is picked (priority order) -- the document candidate NEVER enters the
    # sentence.
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
        RetrievalResult(id="doc:d2:PN1309", score=0.5, metadata={
            "doc_id": "d2", "model": "PN1309", "doc_type": "BROCHURE",
            "file_name": "b.pdf", "requested": False,
        }),
    ])
    prompt = state.followups_prompt()
    assert "PN1309: operating_temperature" in prompt
    assert "b.pdf" not in prompt


def test_followups_prompt_picks_document_candidate_when_only_that_available():
    state = SessionState(last_results=[
        RetrievalResult(id="doc:d2:PN1309", score=0.5, metadata={
            "doc_id": "d2", "model": "PN1309", "doc_type": "BROCHURE",
            "file_name": "b.pdf", "requested": False,
        }),
    ])
    prompt = state.followups_prompt()
    assert "b.pdf (BROCHURE)" in prompt


def test_followups_prompt_empty_when_neither_available():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={"doc_id": "d1", "page_start": 3}),
    ])
    assert state.followups_prompt() == ""


def test_followups_prompt_deterministic_across_repeated_calls():
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "a", "status": "present",
        }),
        RetrievalResult(id="r2", score=1.0, metadata={
            "product_code": "PN1309", "key": "b", "status": "present",
        }),
    ])
    assert state.followups_prompt() == state.followups_prompt() == state.followups_prompt()


def test_followups_prompt_unchanged_when_no_document_candidates():
    # The existing (spec-only) behavior must stay EXACTLY the same without any
    # document-shaped metadata -- backward compatibility.
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1309", "key": "operating_temperature", "status": "present",
        }),
    ])
    assert state.followups_prompt() == (
        "The data returned last turn also had this attribute the user "
        "hasn't asked about yet -- suggest it briefly only if it fits the "
        "flow of the conversation and the user wants it, DON'T FORCE it: "
        "PN1309: operating_temperature"
    )


# --- constraint_delta / correction note ------------------------------------------


def test_constraint_delta_defaults_to_none():
    assert SessionState().constraint_delta is None


def test_as_goal_prompt_appends_constraint_delta_note():
    state = SessionState(
        task="karşılaştırma", focus="basınç",
        constraint_delta=ConstraintDelta(added=["40 bar basınç"], removed=["50 bar basınç"]),
    )
    out = state.as_goal_prompt(transition="continue")
    assert "Correction this turn" in out
    assert "40 bar basınç" in out
    assert "50 bar basınç" in out


def test_as_goal_prompt_no_note_when_delta_none():
    state = SessionState(task="karşılaştırma", focus="basınç")
    out = state.as_goal_prompt(transition="continue")
    assert "Correction this turn" not in out


def test_task_snapshot_has_no_constraint_delta_field():
    assert "constraint_delta" not in TaskSnapshot.model_fields


# --- pinned flow ----------------------------------------------------------------


def test_pinned_defaults_to_none():
    assert SessionState().pinned is None


def test_pinned_flow_round_trips_through_session_state():
    state = SessionState(
        task="karşılaştırma",
        pinned=PinnedFlow(flow="comparison", intent="comparison",
                          expected={"sides": ["de1000", "de1100"]}),
    )
    assert state.pinned is not None
    assert state.pinned.flow == "comparison"
    assert state.pinned.intent == "comparison"
    assert state.pinned.expected == {"sides": ["de1000", "de1100"]}


def test_pinned_expected_defaults_to_empty_dict():
    pin = PinnedFlow(flow="comparison", intent="comparison")
    assert pin.expected == {}


def test_task_snapshot_has_no_pinned_field():
    # The pin is not carried into the switch/digress suspension stack -- only
    # the light snapshot of task/subjects/constraints/focus.
    assert "pinned" not in TaskSnapshot.model_fields


def test_snapshot_drops_pinned():
    state = SessionState(
        task="t", focus="f",
        pinned=PinnedFlow(flow="comparison", intent="comparison"),
    )
    snap = state.snapshot()
    assert not hasattr(snap, "pinned")


def test_store_get_returns_isolated_copy_of_pinned():
    # SessionStateStore.get() returns a deep copy -- this locks in that nested
    # models like `pinned` are covered by that guarantee too.
    store = SessionStateStore()
    store.set("s1", SessionState(pinned=PinnedFlow(flow="comparison", intent="comparison")))
    got = store.get("s1")
    got.pinned.expected["mutated"] = "yes"
    assert store.get("s1").pinned.expected == {}


# --- SessionStateStore -------------------------------------------------------


def test_store_empty_returns_fresh_state():
    store = SessionStateStore()
    state = store.get("s1")
    assert state.task is None
    assert state.turn == 0


def test_store_get_returns_isolated_copy():
    store = SessionStateStore()
    store.set("s1", SessionState(task="orijinal"))
    got = store.get("s1")
    got.task = "değiştirildi"  # mutate the copy
    # the copy in the store must remain untouched
    assert store.get("s1").task == "orijinal"


def test_store_clear():
    store = SessionStateStore()
    store.set("s1", SessionState(task="t"))
    store.clear("s1")
    assert store.get("s1").task is None


def test_available_followups_skips_conflicting_on_purpose():
    """`conflicting` is DELIBERATELY filtered out: when two documents state
    different values for the same attribute, that attribute is NOT suggested
    as a follow-up -- it waits for human confirmation. This test exists to
    lock the behavior: it was once mistaken for a bug, so it must not be
    "fixed" again."""
    state = SessionState(last_results=[
        RetrievalResult(id="r1", score=1.0, metadata={
            "product_code": "PN1267", "key": "stub_count", "status": "conflicting",
        }),
    ])
    assert state.available_followups() == []
