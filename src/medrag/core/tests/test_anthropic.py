"""Tests for `core/llm/anthropic.py::call_anthropic`.

Two failures actually hit in production: (1) `temperature=0` returned 400 on
some Claude models (we lock down that this module never sends temperature),
(2) a `base_url` ending in `.../v1` fell through to `.../v1/v1/messages`
(404)."""

from __future__ import annotations

import pytest

from medrag.core.llm.anthropic import call_anthropic
from medrag.core.llm.errors import ProviderError

MESSAGES = [
    {"role": "system", "content": "sen bir asistansın"},
    {"role": "user", "content": "merhaba"},
]


def _fake_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def test_missing_api_key_raises():
    with pytest.raises(ProviderError, match="API_KEY"):
        call_anthropic(MESSAGES, model="claude-sonnet-4-5", base_url="https://api.anthropic.com",
                        api_key=None, timeout=30)


@pytest.mark.parametrize("base_url", [
    "https://api.anthropic.com",
    "https://api.anthropic.com/v1",
    "https://api.anthropic.com/v1/",
])
def test_url_never_doubles_v1(monkeypatch, base_url):
    """A real failure: when base_url ended in `.../v1` it fell through to
    `.../v1/v1/messages` (404) -- entering base_url with `/v1` (the same habit
    as the other providers) AND entering the bare root must produce the SAME
    correct URL."""
    captured = {}

    def fake_post_json(url, payload, *, api_key, timeout, extra_headers=None):
        captured["url"] = url
        return _fake_response("ok")

    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", fake_post_json)
    call_anthropic(MESSAGES, model="m", base_url=base_url, api_key="sk-ant-x", timeout=30)

    assert captured["url"] == "https://api.anthropic.com/v1/messages"


def test_no_temperature_sent(monkeypatch):
    """A real failure: `temperature=0` returned 400 with "`temperature` is
    deprecated for this model" on some Claude models -- this module must never
    send temperature."""
    captured = {}

    def fake_post_json(url, payload, *, api_key, timeout, extra_headers=None):
        captured["payload"] = payload
        return _fake_response("ok")

    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", fake_post_json)
    call_anthropic(MESSAGES, model="m", base_url="https://api.anthropic.com",
                   api_key="sk-ant-x", timeout=30)

    assert "temperature" not in captured["payload"]


def test_system_message_moved_to_own_field(monkeypatch):
    captured = {}

    def fake_post_json(url, payload, *, api_key, timeout, extra_headers=None):
        captured["payload"] = payload
        captured["headers"] = extra_headers
        return _fake_response("merhaba!")

    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", fake_post_json)
    result = call_anthropic(MESSAGES, model="m", base_url="https://api.anthropic.com",
                             api_key="sk-ant-x", timeout=30)

    assert result == "merhaba!"
    assert captured["payload"]["system"] == "sen bir asistansın"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "merhaba"}]
    assert captured["payload"]["max_tokens"] > 0
    assert captured["headers"]["x-api-key"] == "sk-ant-x"
    assert "anthropic-version" in captured["headers"]


def test_unexpected_response_shape_raises(monkeypatch):
    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", lambda *a, **k: {"nope": True})
    with pytest.raises(ProviderError, match="unexpected Anthropic response shape"):
        call_anthropic(MESSAGES, model="m", base_url="https://api.anthropic.com",
                        api_key="sk-ant-x", timeout=30)


def test_skips_leading_thinking_block_to_find_text(monkeypatch):
    """A real failure: on a model with extended thinking enabled, `content[0]`
    is `type=="thinking"` (NOT the answer text, no `text` field) -- taking
    `content[0]["text"]` blindly hit a KeyError."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "...", "signature": "xyz"},
            {"type": "text", "text": "merhaba!"},
        ]
    }
    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", lambda *a, **k: response)
    result = call_anthropic(MESSAGES, model="m", base_url="https://api.anthropic.com",
                             api_key="sk-ant-x", timeout=30)
    assert result == "merhaba!"


def test_thinking_only_response_raises_clear_error(monkeypatch):
    response = {"content": [{"type": "thinking", "thinking": "...", "signature": "xyz"}]}
    monkeypatch.setattr("medrag.core.llm.anthropic.post_json", lambda *a, **k: response)
    with pytest.raises(ProviderError, match="text bloğu yok"):
        call_anthropic(MESSAGES, model="m", base_url="https://api.anthropic.com",
                        api_key="sk-ant-x", timeout=30)
