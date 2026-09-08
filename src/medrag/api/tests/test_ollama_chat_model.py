import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from medrag.api.answering_model import ProviderError
from medrag.api.ollama_chat_model import NativeOllamaChatModel


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _model(**overrides) -> NativeOllamaChatModel:
    kwargs = {
        "ollama_base_url": "http://localhost:11434/v1",
        "ollama_model": "gemma4:31b",
        "num_ctx": 131072,
    }
    kwargs.update(overrides)
    return NativeOllamaChatModel(**kwargs)


def test_generate_forces_num_ctx_and_hits_native_api_chat(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"message": {"content": "PRODUCT_QUERY"}})

    monkeypatch.setattr("medrag.api.ollama_chat_model.urllib.request.urlopen", fake_urlopen)

    model = _model()
    result = model.invoke([SystemMessage(content="sys"), HumanMessage(content="soru")])

    assert result.content == "PRODUCT_QUERY"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "gemma4:31b"
    assert captured["payload"]["options"]["num_ctx"] == 131072
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "soru"},
    ]


def test_strips_v1_suffix_before_appending_api_chat(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        return _FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("medrag.api.ollama_chat_model.urllib.request.urlopen", fake_urlopen)
    _model(ollama_base_url="http://localhost:11434/v1").invoke([HumanMessage(content="x")])
    assert captured["url"] == "http://localhost:11434/api/chat"


def test_unexpected_response_shape_raises_provider_error(monkeypatch):
    def fake_urlopen(req, timeout):
        return _FakeResponse({"unexpected": True})

    monkeypatch.setattr("medrag.api.ollama_chat_model.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ProviderError):
        _model().invoke([HumanMessage(content="x")])
