import asyncio

import pytest

import medrag.api.answering_model as answering_model_mod
from medrag.api.answering_model import ProviderError
from medrag.api.slot_candidates import SlotCandidate
from medrag.api.slot_selection import (
    LLMSlotSelector,
    _slot_selector_endpoint,
    _slot_selector_num_ctx,
    _slot_selector_provider,
    build_slot_selector_from_env,
)

_CANDIDATES = [
    SlotCandidate(key="operating_temperature", block="core", kind="range", unit="C",
                  question="What temperature range?", n_distinct=5),
    SlotCandidate(key="channel_count", block="analog_io", kind="single", unit=None,
                  question="How many channels?", n_distinct=3),
]


def test_selector_parses_chosen_key(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": '{"key": "channel_count"}'}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    assert asyncio.run(selector.choose("ctx", _CANDIDATES)) == "channel_count"


def test_selector_parses_null_key_as_none(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": '{"key": null}'}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    assert asyncio.run(selector.choose("ctx", _CANDIDATES)) is None


def test_selector_rejects_a_key_outside_the_candidate_list(monkeypatch):
    """If the model returns a fabricated key (not among the candidates) it is
    not accepted -- fail-loud: instead of silently swallowing the fabrication
    and carrying it into the DB, a ProviderError is raised (the caller --
    recommendation.py -- catches it and falls back to the emergency pool)."""
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": '{"key": "made_up_key"}'}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    with pytest.raises(ProviderError):
        asyncio.run(selector.choose("ctx", _CANDIDATES))


def test_selector_raises_on_unparseable_response(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": "not json at all"}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    with pytest.raises(ProviderError):
        asyncio.run(selector.choose("ctx", _CANDIDATES))


def test_selector_raises_on_network_error(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        raise ProviderError("cannot reach http://x/v1")

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    with pytest.raises(ProviderError):
        asyncio.run(selector.choose("ctx", _CANDIDATES))


def test_selector_sends_candidate_keys_questions_and_n_distinct(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"key": "channel_count"}'}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(base_url="http://x/v1", model="m")
    asyncio.run(selector.choose("ctx", _CANDIDATES))

    user_msg = captured["payload"]["messages"][1]["content"]
    assert "channel_count" in user_msg
    assert "How many channels?" in user_msg
    assert "3" in user_msg  # n_distinct


# --- env resolution (SAME pattern as CONTINUATION_*) -------------------------


def test_slot_selector_endpoint_falls_back_to_llm_star():
    base, model, api_key = _slot_selector_endpoint(
        {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small", "LLM_API_KEY": "k"}
    )
    assert (base, model, api_key) == ("http://llm/v1", "small", "k")


def test_slot_selector_endpoint_prefers_override():
    base, model, api_key = _slot_selector_endpoint({
        "LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small",
        "SLOT_SELECTOR_BASE_URL": "http://sel/v1", "SLOT_SELECTOR_MODEL": "sel-model",
    })
    assert (base, model, api_key) == ("http://sel/v1", "sel-model", None)


def test_slot_selector_provider_defaults_to_openai_when_unset():
    assert _slot_selector_provider({}) == "openai"


def test_slot_selector_num_ctx_defaults_to_16384_for_ollama_when_unset():
    assert _slot_selector_num_ctx({}, "ollama") == 16384


def test_slot_selector_num_ctx_stays_none_for_non_ollama_when_unset():
    assert _slot_selector_num_ctx({}, "openai") is None


def test_build_from_env_returns_none_when_unconfigured():
    assert build_slot_selector_from_env({}) is None


def test_build_from_env_returns_selector_when_configured():
    selector = build_slot_selector_from_env({"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small"})
    assert isinstance(selector, LLMSlotSelector)


def test_selector_uses_native_api_chat_when_ollama_and_num_ctx_set(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": '{"key": "channel_count"}'}}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    selector = LLMSlotSelector(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=32768,
    )
    assert asyncio.run(selector.choose("ctx", _CANDIDATES)) == "channel_count"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["options"]["num_ctx"] == 32768
