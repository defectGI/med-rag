import asyncio

import pytest

import medrag.api.answering_model as answering_model_mod
from medrag.api.answering_model import ProviderError
from medrag.api.continuation import (
    LLMContinuationChecker,
    _continuation_endpoint,
    _continuation_num_ctx,
    _continuation_provider,
    build_continuation_checker_from_env,
)


def test_checker_parses_yes_as_true(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": "yes"}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(base_url="http://x/v1", model="m")
    assert asyncio.run(checker.check("0-50 degrees", "a question is expected")) is True


def test_checker_parses_no_as_false(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": "No"}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(base_url="http://x/v1", model="m")
    assert asyncio.run(checker.check("actually I meant to ask something else", "a question is expected")) is False


def test_checker_raises_on_unparseable_response(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {"choices": [{"message": {"content": "belki?"}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(base_url="http://x/v1", model="m")
    try:
        asyncio.run(checker.check("q", "ctx"))
        assert False, "an unexpected response must not be swallowed silently"
    except ProviderError:
        pass


def test_checker_raises_on_network_error(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        raise ProviderError("cannot reach http://x/v1")

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(base_url="http://x/v1", model="m")
    try:
        asyncio.run(checker.check("q", "ctx"))
        assert False, "a network error must not be swallowed -- the caller should fall back to its own safe default"
    except ProviderError:
        pass


def test_continuation_endpoint_falls_back_to_llm_star():
    base, model, api_key = _continuation_endpoint(
        {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small", "LLM_API_KEY": "k"}
    )
    assert (base, model, api_key) == ("http://llm/v1", "small", "k")


def test_continuation_endpoint_prefers_continuation_override():
    base, model, api_key = _continuation_endpoint({
        "LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small",
        "CONTINUATION_BASE_URL": "http://cont/v1", "CONTINUATION_MODEL": "cont-model",
    })
    assert (base, model, api_key) == ("http://cont/v1", "cont-model", None)


def test_build_from_env_returns_none_when_unconfigured():
    assert build_continuation_checker_from_env({}) is None


def test_build_from_env_returns_checker_when_configured():
    checker = build_continuation_checker_from_env(
        {"LLM_BASE_URL": "http://llm/v1", "LLM_MODEL": "small"}
    )
    assert isinstance(checker, LLMContinuationChecker)


# --- provider/num_ctx resolution ---------------------------------------------


def test_continuation_provider_falls_back_to_llm_provider():
    assert _continuation_provider({"LLM_PROVIDER": "ollama"}) == "ollama"


def test_continuation_provider_override_wins():
    assert _continuation_provider({"LLM_PROVIDER": "ollama", "CONTINUATION_PROVIDER": "openai"}) == "openai"


def test_continuation_provider_defaults_to_openai_when_unset():
    assert _continuation_provider({}) == "openai"


def test_continuation_num_ctx_falls_back_to_llm_num_ctx():
    assert _continuation_num_ctx({"LLM_NUM_CTX": "32768"}, "ollama") == 32768


def test_continuation_num_ctx_override_wins():
    env = {"LLM_NUM_CTX": "32768", "CONTINUATION_NUM_CTX": "8192"}
    assert _continuation_num_ctx(env, "ollama") == 8192


def test_continuation_num_ctx_defaults_to_16384_for_ollama_when_unset():
    assert _continuation_num_ctx({}, "ollama") == 16384


def test_continuation_num_ctx_stays_none_for_non_ollama_when_unset():
    assert _continuation_num_ctx({}, "openai") is None


def test_continuation_num_ctx_invalid_value_raises():
    with pytest.raises(ProviderError):
        _continuation_num_ctx({"CONTINUATION_NUM_CTX": "abc"}, "ollama")


def test_build_from_env_wires_provider_and_num_ctx_when_ollama():
    checker = build_continuation_checker_from_env({
        "LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "small",
        "LLM_PROVIDER": "ollama", "LLM_NUM_CTX": "32768",
    })
    assert checker._provider == "ollama"
    assert checker._num_ctx == 32768


def test_checker_uses_native_api_chat_when_ollama_and_num_ctx_set(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"message": {"content": "yes"}}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=32768,
    )
    assert asyncio.run(checker.check("q", "ctx")) is True
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["options"]["num_ctx"] == 32768


def test_checker_native_path_maps_thinking_off_to_native_think_false(monkeypatch):
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["payload"] = payload
        return {"message": {"content": "yes"}}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(
        base_url="http://localhost:11434/v1", model="m", provider="ollama", num_ctx=16384,
        extra_body={"reasoning_effort": "none"},
    )
    asyncio.run(checker.check("q", "ctx"))
    assert captured["payload"]["think"] is False
    assert "reasoning_effort" not in captured["payload"]


def test_checker_without_num_ctx_still_uses_v1_path(monkeypatch):
    # Regression: without num_ctx the behavior goes EXACTLY down the old /v1 path.
    captured = {}

    def fake_post(url, payload, *, api_key, timeout):
        captured["url"] = url
        return {"choices": [{"message": {"content": "yes"}}]}

    monkeypatch.setattr(answering_model_mod, "_post_json", fake_post)
    checker = LLMContinuationChecker(base_url="http://localhost:11434/v1", model="m", provider="ollama")
    assert asyncio.run(checker.check("q", "ctx")) is True
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
