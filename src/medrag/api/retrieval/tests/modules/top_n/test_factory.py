"""build_default_retriever wiring tests. `embedder_from_env`/`QdrantVectorStore`
are monkeypatched so this never needs a real `qdrant_client` install or a
network call -- only the env-driven wiring logic is under test."""

from __future__ import annotations

import pytest

from medrag.api.retrieval.modules.top_n import factory as factory_mod
from medrag.api.retrieval.modules.top_n.retriever import TopNRetriever


class _FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class _FakeStore:
    def __init__(self, *, url, collection_name, api_key=None):
        self.url = url
        self.collection_name = collection_name
        self.api_key = api_key

    def search(self, vector, *, limit):
        return []


def test_missing_qdrant_url_raises(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(ValueError, match="qdrant_url"):
        factory_mod.build_default_retriever(collection_name="chunks")


def test_missing_collection_name_raises(monkeypatch):
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    with pytest.raises(ValueError, match="collection_name"):
        factory_mod.build_default_retriever(qdrant_url="http://x:6333")


def test_wires_params_over_env(monkeypatch):
    monkeypatch.setattr(factory_mod, "embedder_from_env", lambda timeout: _FakeEmbedder())
    monkeypatch.setattr(
        factory_mod.QdrantVectorStore, "from_env",
        classmethod(lambda cls, *, url, collection_name, api_key=None, timeout=60.0:
                    _FakeStore(url=url, collection_name=collection_name, api_key=api_key)),
    )
    retriever = factory_mod.build_default_retriever(
        qdrant_url="http://explicit:6333", collection_name="explicit_chunks",
        qdrant_api_key="secret",
    )
    assert isinstance(retriever, TopNRetriever)
    assert retriever._store.url == "http://explicit:6333"
    assert retriever._store.collection_name == "explicit_chunks"
    assert retriever._store.api_key == "secret"


def test_env_fallback(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://env:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "env_chunks")
    monkeypatch.setattr(factory_mod, "embedder_from_env", lambda timeout: _FakeEmbedder())
    monkeypatch.setattr(
        factory_mod.QdrantVectorStore, "from_env",
        classmethod(lambda cls, *, url, collection_name, api_key=None, timeout=60.0:
                    _FakeStore(url=url, collection_name=collection_name, api_key=api_key)),
    )
    retriever = factory_mod.build_default_retriever()
    assert retriever._store.url == "http://env:6333"
    assert retriever._store.collection_name == "env_chunks"
