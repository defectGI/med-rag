"""OpenAICompatClient: the "thinking toggle rejected by a non-reasoning
model" fallback (llm/__init__.py sends {prefix}_THINKING_ON as extra_body,
but a model with no thinking mode has no such parameter to accept -- the
client should retry once without it instead of hard-failing the call).
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from typing import Self

from medrag.pipeline.parser.llm import ollama_recovery
from medrag.pipeline.parser.llm.openai_compat import OpenAICompatClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        return False


def test_retries_without_extra_body_when_server_rejects_it(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        if "reasoning_effort" in body:
            err_body = b'{"error": "unknown parameter: reasoning_effort"}'
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", None, io.BytesIO(err_body)
            )
        return _FakeResponse({"choices": [{"message": {"content": "hello"}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatClient(
        base_url="http://localhost:11434/v1",
        model="some-non-thinking-model",
        extra_body={"reasoning_effort": "none"},
    )
    result = client.complete(system="s", user="u")

    assert result == "hello"
    assert len(calls) == 2
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]


def test_unrelated_http_error_is_not_retried(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode("utf-8")))
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", None,
            io.BytesIO(b"boom"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = OpenAICompatClient(
        base_url="http://localhost:11434/v1",
        model="some-model",
        extra_body={"reasoning_effort": "none"},
    )
    try:
        client.complete(system="s", user="u")
        assert False, "expected LLMError"
    except Exception as exc:  # noqa: BLE001 -- test: tip hemen assert type(exc).__name__ == "LLMError" ile dogrulaniyor
        assert type(exc).__name__ == "LLMError"

    assert len(calls) == 1  # no retry for an unrelated failure


def _ollama_client(**kwargs) -> OpenAICompatClient:
    return OpenAICompatClient(
        base_url="http://localhost:11434/v1", model="some-model", provider="ollama", **kwargs)


def test_garbage_output_recovers_on_plain_retry(monkeypatch):
    replies = iter(["?" * 40, "a real answer"])
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return _FakeResponse({"choices": [{"message": {"content": next(replies)}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    restart_calls = []
    monkeypatch.setattr(ollama_recovery, "restart_local_ollama",
                        lambda base_url: restart_calls.append(base_url) or True)

    result = _ollama_client().complete(system="s", user="u")

    assert result == "a real answer"
    assert len(calls) == 2  # first garbage, one plain retry -- no restart needed
    assert restart_calls == []


def test_garbage_output_persists_then_restart_recovers(monkeypatch):
    replies = iter(["?" * 40, "?" * 40, "?" * 40, "a real answer"])
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return _FakeResponse({"choices": [{"message": {"content": next(replies)}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    restart_calls = []
    monkeypatch.setattr(ollama_recovery, "restart_local_ollama",
                        lambda base_url: restart_calls.append(base_url) or True)

    result = _ollama_client().complete(system="s", user="u")

    assert result == "a real answer"
    assert len(calls) == 4  # 3 plain attempts all garbage, then post-restart retry
    assert restart_calls == ["http://localhost:11434/v1"]


def test_garbage_output_persists_after_restart_raises(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": "?" * 40}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(ollama_recovery, "restart_local_ollama", lambda base_url: True)

    try:
        _ollama_client().complete(system="s", user="u")
        assert False, "expected LLMError"
    except Exception as exc:  # noqa: BLE001 -- test: tip ve mesaj ("restart") hemen assert ediliyor
        assert type(exc).__name__ == "LLMError"
        assert "restart" in str(exc)


def test_garbage_output_on_remote_ollama_raises_without_restart(monkeypatch):
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": "?" * 40}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    restart_calls = []
    monkeypatch.setattr(ollama_recovery, "restart_local_ollama",
                        lambda base_url: restart_calls.append(base_url) or False)

    client = OpenAICompatClient(
        base_url="https://gpu-box.example.com/v1", model="some-model", provider="ollama")
    try:
        client.complete(system="s", user="u")
        assert False, "expected LLMError"
    except Exception as exc:  # noqa: BLE001 -- test: tip ve mesaj ("remote") hemen assert ediliyor
        assert type(exc).__name__ == "LLMError"
        assert "remote" in str(exc)

    assert restart_calls == ["https://gpu-box.example.com/v1"]


def test_non_ollama_provider_is_not_garbage_checked(monkeypatch):
    # A non-ollama provider returning a repeated-character reply is passed
    # through as-is -- the guard only ever applies to provider == "ollama".
    def fake_urlopen(req, timeout=None):
        return _FakeResponse({"choices": [{"message": {"content": "?" * 40}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatClient(base_url="https://api.openai.com/v1", model="gpt-4o")

    assert client.complete(system="s", user="u") == "?" * 40
