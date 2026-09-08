"""OpenAICompatEmbedder tests. No network -- `_post_json` is monkeypatched."""

from __future__ import annotations

import pytest

from medrag.api.retrieval.modules.top_n.embedder import (
    Embedder,
    OpenAICompatEmbedder,
    ProviderError,
    embedder_from_env,
)

# OpenAICompatEmbedder itself lives in medrag.core.llm.client now (a shared,
# byte-identical twin of the vectorization component's embedder) -- its
# `embed()` calls that module's `_post_json`, so the fake has to be patched
# there, not on this re-exporting module.
from medrag.core.llm import client as embedder_mod


def test_embed_empty_list_short_circuits():
    emb = OpenAICompatEmbedder(base_url="http://x", model="m")
    assert emb.embed([]) == []


def test_embed_orders_by_index(monkeypatch):
    def fake_post(url, payload, *, api_key, timeout):
        return {
            "data": [
                {"index": 1, "embedding": [0.2]},
                {"index": 0, "embedding": [0.1]},
            ]
        }

    monkeypatch.setattr(embedder_mod, "_post_json", fake_post)
    emb = OpenAICompatEmbedder(base_url="http://x", model="m")
    assert emb.embed(["a", "b"]) == [[0.1], [0.2]]


def test_embed_incomplete_response_raises(monkeypatch):
    monkeypatch.setattr(
        embedder_mod, "_post_json", lambda *a, **k: {"data": [{"index": 0, "embedding": [0.1]}]}
    )
    emb = OpenAICompatEmbedder(base_url="http://x", model="m")
    with pytest.raises(ProviderError):
        emb.embed(["a", "b"])


def test_embed_malformed_response_raises(monkeypatch):
    monkeypatch.setattr(embedder_mod, "_post_json", lambda *a, **k: {"nope": True})
    emb = OpenAICompatEmbedder(base_url="http://x", model="m")
    with pytest.raises(ProviderError):
        emb.embed(["a"])


def test_embedder_satisfies_protocol():
    assert isinstance(OpenAICompatEmbedder(base_url="http://x", model="m"), Embedder)


def test_embedder_from_env_missing_provider():
    with pytest.raises(ProviderError):
        embedder_from_env({})


def test_embedder_from_env_anthropic_rejected():
    with pytest.raises(ProviderError):
        embedder_from_env({"EMBEDDING_PROVIDER": "anthropic", "EMBEDDING_MODEL": "m"})


def test_embedder_from_env_ollama_default_base_url():
    emb = embedder_from_env({"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "qwen3-embedding:0.6b"})
    assert emb.base_url == "http://localhost:11434/v1"
    assert emb.model == "qwen3-embedding:0.6b"
