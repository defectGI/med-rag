"""Native Ollama provider: /api/chat dispatch + num_ctx forwarding."""

from __future__ import annotations

import json

import pytest

from text2sql_native.errors import ProviderError
from text2sql_native.providers.ollama_provider import OllamaProvider


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, seen: dict, reply: dict):
    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(reply)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_complete_hits_native_api_chat_with_num_ctx(monkeypatch):
    seen: dict = {}
    _patch_urlopen(monkeypatch, seen, {"message": {"content": "hi"},
                                        "prompt_eval_count": 10, "eval_count": 5})
    provider = OllamaProvider(model="gemma4:26b", base_url="http://x:11434", num_ctx=16384)

    response = provider.complete(system="s", user="u", temperature=0.0, top_p=1.0, max_tokens=100)

    assert seen["url"] == "http://x:11434/api/chat"
    assert seen["payload"]["options"]["num_ctx"] == 16384
    assert seen["payload"]["stream"] is False
    assert response.text == "hi"
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5


def test_complete_strips_v1_suffix_from_base_url(monkeypatch):
    seen: dict = {}
    _patch_urlopen(monkeypatch, seen, {"message": {"content": "hi"}})
    provider = OllamaProvider(model="m", base_url="http://x:11434/v1", num_ctx=8192)

    provider.complete(system="s", user="u", temperature=0.0, top_p=1.0, max_tokens=10)

    assert seen["url"] == "http://x:11434/api/chat"


def test_complete_without_num_ctx_omits_it_from_options(monkeypatch):
    seen: dict = {}
    _patch_urlopen(monkeypatch, seen, {"message": {"content": "hi"}})
    provider = OllamaProvider(model="m", base_url="http://x:11434")

    provider.complete(system="s", user="u", temperature=0.0, top_p=1.0, max_tokens=10)

    assert "num_ctx" not in seen["payload"]["options"]


def test_complete_json_sends_format_schema(monkeypatch):
    seen: dict = {}
    _patch_urlopen(monkeypatch, seen, {"message": {"content": "{}"}})
    provider = OllamaProvider(model="m", base_url="http://x:11434", num_ctx=4096)
    schema = {"type": "object", "properties": {}}

    provider.complete_json(system="s", user="u", json_schema=schema, schema_name="x",
                           temperature=0.0, top_p=1.0, max_tokens=10)

    assert seen["payload"]["format"] == schema


def test_complete_json_falls_back_to_text_parsing_when_format_rejected(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise __import__("urllib.error", fromlist=["HTTPError"]).HTTPError(
                req.full_url, 400, "bad format", None, None,
            )
        return _FakeResponse({"message": {"content": '{"ok": true}'}})

    def fake_read_for_http_error(self):
        return b"format not supported"

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("urllib.error.HTTPError.read", fake_read_for_http_error, raising=False)

    provider = OllamaProvider(model="m", base_url="http://x:11434")
    response = provider.complete_json(
        system="s", user="u", json_schema={"type": "object"}, schema_name="x",
        temperature=0.0, top_p=1.0, max_tokens=10,
    )
    assert "ok" in response.text


def test_unexpected_response_shape_raises_provider_error(monkeypatch):
    _patch_urlopen(monkeypatch, {}, {"unexpected": "shape"})
    provider = OllamaProvider(model="m", base_url="http://x:11434")
    with pytest.raises(ProviderError):
        provider.complete(system="s", user="u", temperature=0.0, top_p=1.0, max_tokens=10)


def test_registered_under_ollama_name():
    from text2sql_native.providers.base import get_provider_class

    assert get_provider_class("ollama") is OllamaProvider


def test_other_providers_accept_and_ignore_num_ctx():
    # base.LLMProvider.__init__ now takes num_ctx -- openai/anthropic must
    # still accept it harmlessly (build_provider() passes it unconditionally).
    from text2sql_native.providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider(model="gpt-4o-mini", num_ctx=16384)
    assert provider._num_ctx == 16384
