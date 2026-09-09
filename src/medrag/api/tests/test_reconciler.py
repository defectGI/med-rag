import asyncio

import pytest

import medrag.api.reconciler as reconciler_mod
from medrag.api.answering_model import ProviderError
from medrag.api.reconciler import (
    LLMReconciler,
    _apply,
    _apply_dropped_constraints,
    _build_system_prompt,
    _extract_json,
    _merge_constraints,
    _merge_subjects,
    _reconciler_endpoint,
    _reconciler_num_ctx,
    _reconciler_provider,
)
from medrag.api.session_state import SessionState, Subject, TaskSnapshot

_LABELS = ["medical_fact", "comparison", "out_of_scope"]


# --- pure helpers ------------------------------------------------------------


def test_merge_subjects_stamps_and_decays():
    prev = [Subject(entity="de1000", last_turn=1)]
    # turn 8, decay_window 6 -> de1000 (last_turn 1) drops out; de1100 is added.
    merged = _merge_subjects(prev, ["de1100"], turn=8, decay_window=6)
    entities = {s.entity for s in merged}
    assert entities == {"de1100"}


def test_merge_subjects_refresh_keeps_entity():
    prev = [Subject(entity="de1000", last_turn=1)]
    merged = _merge_subjects(prev, ["de1000"], turn=8, decay_window=6)
    # mentioned again -> last_turn refreshed, does not drop out.
    assert [(s.entity, s.last_turn) for s in merged] == [("de1000", 8)]


def test_merge_constraints_dedup_preserves_order():
    assert _merge_constraints(["a"], ["a", "b"]) == ["a", "b"]


# --- correction delta ----------------------------------------------------------


def test_apply_dropped_constraints_removes_matching_entries():
    merged = ["50 bar", "soğuk ortam"]
    remaining, added, removed = _apply_dropped_constraints(merged, ["50 bar", "soğuk ortam"], ["50 bar"])
    assert remaining == ["soğuk ortam"]
    assert removed == ["50 bar"]
    assert added == []


def test_apply_dropped_constraints_is_case_and_whitespace_insensitive():
    merged = [" 50 Bar "]
    remaining, _added, removed = _apply_dropped_constraints(merged, [" 50 bar "], ["50 BAR"])
    assert remaining == []
    assert removed == [" 50 Bar "]


def test_apply_dropped_constraints_added_excludes_prev_entries():
    # An item present in both merged and prev does not count as "newly added".
    merged = ["a", "b"]
    remaining, added, removed = _apply_dropped_constraints(merged, ["a"], [])
    assert remaining == ["a", "b"]
    assert added == ["b"]
    assert removed == []


def test_extract_json_from_fenced_reply():
    assert _extract_json('```json\n{"transition": "continue"}\n```') == {"transition": "continue"}


def test_extract_json_returns_none_on_garbage():
    assert _extract_json("bu bir json değil") is None
    assert _extract_json("") is None


def test_reconciler_endpoint_falls_back_to_llm():
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small"}
    assert _reconciler_endpoint(env) == ("http://llm/v1", "small", None)


def test_reconciler_endpoint_override_wins():
    env = {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small",
           "RECONCILER_BASE_URL": "http://rec/v1", "RECONCILER_MODEL": "rec"}
    base, model, _ = _reconciler_endpoint(env)
    assert (base, model) == ("http://rec/v1", "rec")


# --- provider/num_ctx resolution ------------------------------------------------


def test_reconciler_provider_falls_back_to_llm_provider():
    assert _reconciler_provider({"LLM_PROVIDER": "ollama"}) == "ollama"


def test_reconciler_provider_override_wins():
    assert _reconciler_provider({"LLM_PROVIDER": "ollama", "RECONCILER_PROVIDER": "openai"}) == "openai"


def test_reconciler_provider_defaults_to_openai_when_unset():
    assert _reconciler_provider({}) == "openai"


def test_reconciler_num_ctx_falls_back_to_llm_num_ctx():
    assert _reconciler_num_ctx({"LLM_NUM_CTX": "32768"}, "ollama") == 32768


def test_reconciler_num_ctx_override_wins():
    env = {"LLM_NUM_CTX": "32768", "RECONCILER_NUM_CTX": "8192"}
    assert _reconciler_num_ctx(env, "ollama") == 8192


def test_reconciler_num_ctx_defaults_to_16384_for_ollama_when_unset():
    assert _reconciler_num_ctx({}, "ollama") == 16384


def test_reconciler_num_ctx_stays_none_for_non_ollama_when_unset():
    assert _reconciler_num_ctx({}, "openai") is None


def test_reconciler_num_ctx_invalid_value_raises():
    with pytest.raises(ProviderError):
        _reconciler_num_ctx({"RECONCILER_NUM_CTX": "abc"}, "ollama")


def test_llm_reconciler_uses_native_api_chat_when_ollama_and_num_ctx_set(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        captured["payload"] = payload
        body = '{"resolved_query": "de1100 gücü", "transition": "continue", ' \
               '"confidence": 0.9}'
        return {"message": {"content": body}}

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=32768,
    )
    res = asyncio.run(rec.reconcile(SessionState(), [], "peki ya diğeri"))

    assert res.resolved_query == "de1100 gücü"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["options"]["num_ctx"] == 32768


def test_llm_reconciler_without_num_ctx_still_uses_v1_path(monkeypatch):
    # Regression: without num_ctx the behavior goes EXACTLY down the old /v1 path.
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        body = '{"resolved_query": "de1100 gücü", "transition": "continue", "confidence": 0.9}'
        return {"choices": [{"message": {"content": body}}]}

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(base_url="http://localhost:11434/v1", model="m", provider="ollama")
    res = asyncio.run(rec.reconcile(SessionState(), [], "peki ya diğeri"))
    assert res.resolved_query == "de1100 gücü"
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"


# --- _apply: transition mechanics --------------------------------------------


def test_apply_none_data_graceful_fallback():
    state = SessionState(task="t", turn=2)
    res = _apply(state, "ham sorgu", None, decay_window=6)
    assert res.resolved_query == "ham sorgu"
    assert res.transition == "continue"
    assert res.new_state.turn == 3
    assert res.new_state.task == "t"  # preserved
    assert res.raw_text is None  # raw_text was not given


def test_apply_none_data_carries_raw_text_when_given():
    # If the model returned something but it couldn't be parsed -- whatever it
    # returned must be carried into ReconcileResult, not vanish silently.
    state = SessionState(task="t", turn=2)
    res = _apply(state, "ham sorgu", None, decay_window=6, raw_text="garbled output")
    assert res.raw_text == "garbled output"
    assert res.transition == "continue"  # still graceful


def test_apply_continue_merges_and_resolves():
    state = SessionState(task="karşılaştırma", turn=1,
                        subjects=[Subject(entity="de1000", last_turn=1)])
    data = {"resolved_query": "de1100'in gücü", "transition": "continue",
            "task": None, "subjects": ["de1100"], "constraints": ["güç"],
            "focus": "de1100 gücü", "confidence": 0.9}
    res = _apply(state, "peki ya diğeri", data, decay_window=6)
    assert res.resolved_query == "de1100'in gücü"
    assert res.new_state.task == "karşılaştırma"  # null -> prev preserved
    assert {s.entity for s in res.new_state.subjects} == {"de1000", "de1100"}
    assert res.new_state.constraints == ["güç"]
    assert res.confidence == 0.9


def test_apply_dropped_constraint_removed_and_delta_recorded():
    # The user says "no, I said 50, make it 40" -- the old constraint must really go.
    state = SessionState(task="t", turn=1, constraints=["50 bar basınç"])
    data = {"resolved_query": "40 bar olsun", "transition": "continue",
            "constraints": ["40 bar basınç"], "dropped_constraints": ["50 bar basınç"],
            "confidence": 0.9}
    res = _apply(state, "40 bar olsun", data, decay_window=6)
    assert res.new_state.constraints == ["40 bar basınç"]
    assert res.new_state.constraint_delta is not None
    assert res.new_state.constraint_delta.added == ["40 bar basınç"]
    assert res.new_state.constraint_delta.removed == ["50 bar basınç"]


def test_apply_new_constraint_added_without_drop():
    state = SessionState(task="t", turn=1, constraints=["soğuk ortam"])
    data = {"resolved_query": "50 bar da olsun", "transition": "continue",
            "constraints": ["50 bar basınç"], "confidence": 0.9}
    res = _apply(state, "50 bar da olsun", data, decay_window=6)
    assert res.new_state.constraints == ["soğuk ortam", "50 bar basınç"]
    assert res.new_state.constraint_delta is not None
    assert res.new_state.constraint_delta.added == ["50 bar basınç"]
    assert res.new_state.constraint_delta.removed == []


def test_apply_no_dropped_constraints_means_delta_none():
    # Backward compatibility: if the model never returns dropped_constraints
    # (old behavior), constraint_delta stays None and constraints keep working
    # as pure accumulation.
    state = SessionState(task="t", turn=1, constraints=["a"])
    data = {"resolved_query": "x", "transition": "continue",
            "constraints": ["a"], "confidence": 0.9}
    res = _apply(state, "x", data, decay_window=6)
    assert res.new_state.constraints == ["a"]
    assert res.new_state.constraint_delta is None


def test_apply_switch_suspends_previous_task():
    state = SessionState(task="eski görev", turn=3,
                        subjects=[Subject(entity="de1000", last_turn=3)])
    data = {"resolved_query": "PN1183 fiyatı", "transition": "switch",
            "task": "yeni görev", "subjects": ["PN1183"], "constraints": [],
            "focus": "fiyat", "confidence": 0.8}
    res = _apply(state, "bunu boşver PN1183 fiyatı", data, decay_window=6)
    assert res.new_state.task == "yeni görev"
    assert {s.entity for s in res.new_state.subjects} == {"PN1183"}  # fresh
    # the old task was pushed onto the suspension stack (not deleted).
    assert len(res.new_state.suspended) == 1
    assert res.new_state.suspended[0].task == "eski görev"


# --- resume -- a real return to the suspension stack ---------------------------


def test_apply_switch_resumes_matching_suspended_task():
    # While "de1100 karşılaştırması" sits in suspended and the user returns to
    # it -> resume (not switch), subjects/constraints merge old+new, and the
    # snapshot pops off the stack.
    old_snap = TaskSnapshot(
        task="de1100 karşılaştırması",
        subjects=[Subject(entity="de1100", last_turn=2)],
        constraints=["soğuk ortam"],
        focus="karşılaştırma odağı",
    )
    state = SessionState(task="güncel görev", turn=5,
                        subjects=[Subject(entity="de9000", last_turn=5)],
                        suspended=[old_snap])
    data = {"resolved_query": "de1100 karşılaştırmasına dönelim", "transition": "switch",
            "task": "de1100 karşılaştırması", "subjects": ["de2000"],
            "constraints": ["50 bar"], "focus": "yeni bilgi", "confidence": 0.7}
    res = _apply(state, "de1100 karşılaştırmasına dönelim", data, decay_window=6)
    assert res.transition == "resume"
    assert res.new_state.task == "de1100 karşılaştırması"
    assert {s.entity for s in res.new_state.subjects} == {"de1100", "de2000"}
    assert res.new_state.constraints == ["soğuk ortam", "50 bar"]
    assert res.new_state.focus == "yeni bilgi"
    # The old snapshot left the stack and the current task (state.snapshot()) was pushed.
    suspended_tasks = [s.task for s in res.new_state.suspended]
    assert "de1100 karşılaştırması" not in suspended_tasks
    assert "güncel görev" in suspended_tasks


def test_apply_switch_no_match_keeps_old_behavior():
    # suspended empty/no match -> old behavior EXACTLY (regression lock).
    state = SessionState(task="eski görev", turn=3,
                        subjects=[Subject(entity="de1000", last_turn=3)])
    data = {"resolved_query": "PN1183 fiyatı", "transition": "switch",
            "task": "yeni görev", "subjects": ["PN1183"], "constraints": [],
            "focus": "fiyat", "confidence": 0.8}
    res = _apply(state, "bunu boşver PN1183 fiyatı", data, decay_window=6)
    assert res.transition == "switch"
    assert res.new_state.task == "yeni görev"
    assert {s.entity for s in res.new_state.subjects} == {"PN1183"}
    assert len(res.new_state.suspended) == 1
    assert res.new_state.suspended[0].task == "eski görev"


def test_resume_goal_prompt_includes_resume_note():
    state = SessionState(task="de1100 karşılaştırması", focus="odak")
    prompt = state.as_goal_prompt(transition="resume")
    assert "returning to an earlier topic" in prompt


def test_apply_digress_preserves_active_task():
    state = SessionState(task="karşılaştırma", turn=4, focus="odak",
                        subjects=[Subject(entity="de1000", last_turn=4)])
    data = {"resolved_query": "bugün hava nasıl", "transition": "digress",
            "task": None, "subjects": [], "constraints": [], "focus": "",
            "confidence": 0.3}
    res = _apply(state, "bugün hava nasıl", data, decay_window=6)
    # The goal is PRESERVED; only the turn advances.
    assert res.new_state.task == "karşılaştırma"
    assert res.new_state.focus == "odak"
    assert {s.entity for s in res.new_state.subjects} == {"de1000"}
    assert res.new_state.turn == 5


def test_apply_invalid_transition_defaults_continue():
    state = SessionState(turn=0)
    data = {"resolved_query": "x", "transition": "garip", "confidence": 0.5}
    res = _apply(state, "x", data, decay_window=6)
    assert res.transition == "continue"


# --- fusion: intent --------------------------------------------------------------


def test_apply_fused_intent_accepts_valid_label():
    state = SessionState(turn=0)
    data = {"resolved_query": "arveles dozu", "transition": "continue",
            "confidence": 0.9, "intent": "medical_fact"}
    res = _apply(state, "arveles dozu", data, decay_window=6, intent_labels=_LABELS)
    assert res.intent == "medical_fact"


def test_apply_fused_intent_rejects_unknown_label():
    # The model returned something outside the label set -> intent None (no fabrication flows through).
    state = SessionState(turn=0)
    data = {"resolved_query": "x", "transition": "continue", "intent": "uydurma_etiket"}
    res = _apply(state, "x", data, decay_window=6, intent_labels=_LABELS)
    assert res.intent is None


def test_apply_no_labels_means_no_intent():
    # Fusion off (no intent_labels given) -> None even if the model returns an intent.
    state = SessionState(turn=0)
    data = {"resolved_query": "x", "transition": "continue", "intent": "medical_fact"}
    res = _apply(state, "x", data, decay_window=6)
    assert res.intent is None


def test_apply_fused_intent_on_switch_and_digress():
    state = SessionState(task="eski", turn=2)
    for transition in ("switch", "digress"):
        data = {"resolved_query": "q", "transition": transition,
                "task": "yeni", "intent": "comparison"}
        res = _apply(state, "q", data, decay_window=6, intent_labels=_LABELS)
        assert res.intent == "comparison", transition


def test_build_system_prompt_includes_intent_only_with_labels():
    with_labels = _build_system_prompt(_LABELS)
    without = _build_system_prompt(None)
    assert "medical_fact" in with_labels and '"intent"' in with_labels
    assert "medical_fact" not in without and '"intent"' not in without
    # Labels must come with a definition + few-shot catalog, NOT a bare name
    # list -- catches the bare-list regression: intent must not silently
    # regress without the signal that separates medical_fact from
    # clinical_decision.
    assert "a factual question whose answer is in the uploaded medical documents" in with_labels
    assert " -> " in with_labels


def test_build_system_prompt_includes_dropped_constraints_field():
    # No extra LLM call -- a new field in the one existing prompt.
    assert '"dropped_constraints"' in _build_system_prompt(None)


def test_llm_reconciler_fused_intent(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        body = '{"resolved_query": "arveles dozu", "transition": "continue", ' \
               '"task": "doz bilgisi", "subjects": ["arveles"], ' \
               '"constraints": [], "focus": "doz", "confidence": 0.9, ' \
               '"intent": "medical_fact"}'
        return {"choices": [{"message": {"content": body}}]}

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(base_url="http://x/v1", model="m", intent_labels=_LABELS)
    res = asyncio.run(rec.reconcile(SessionState(), [], "arveles dozu nedir"))
    assert res.intent == "medical_fact"
    assert res.resolved_query == "arveles dozu"


# --- LLMReconciler (mocked HTTP) ---------------------------------------------


def test_llm_reconciler_parses_and_applies(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        body = '{"resolved_query": "de1100 gücü", "transition": "continue", ' \
               '"task": "karşılaştırma", "subjects": ["de1100"], ' \
               '"constraints": [], "focus": "güç", "confidence": 0.9}'
        return {"choices": [{"message": {"content": body}}]}

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(base_url="http://x/v1", model="m")
    res = asyncio.run(rec.reconcile(SessionState(), [], "peki ya diğeri"))
    assert res.resolved_query == "de1100 gücü"
    assert res.new_state.task == "karşılaştırma"


def test_llm_reconciler_unparseable_content_carries_raw_text(monkeypatch):
    # The model returned something (network/response shape intact) but it is
    # not JSON -- it must be carried to _apply as raw_text, not swallowed silently.
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": "bu bir json degil, duz metin"}}]}

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(base_url="http://x/v1", model="m")
    res = asyncio.run(rec.reconcile(SessionState(), [], "soru"))
    assert res.raw_text == "bu bir json degil, duz metin"
    assert res.resolved_query == "soru"  # graceful fallback


def test_llm_reconciler_graceful_on_network_error(monkeypatch):
    from medrag.api.answering_model import ProviderError

    def fake_post(url, payload, *, api_key, timeout):
        raise ProviderError("cannot reach http://x/v1")

    monkeypatch.setattr(reconciler_mod, "_post_json", fake_post)
    rec = LLMReconciler(base_url="http://x/v1", model="m")
    res = asyncio.run(rec.reconcile(SessionState(task="t"), [], "soru"))
    # network error -> fallback: raw query, continue, goal preserved.
    assert res.resolved_query == "soru"
    assert res.transition == "continue"
    assert res.new_state.task == "t"
    assert res.raw_text is None  # on a network error there is no content to show
