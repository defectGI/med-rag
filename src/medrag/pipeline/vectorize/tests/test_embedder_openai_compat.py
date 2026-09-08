"""HTTP'ye dokunan tek yer `_post_json`tir; testler onu monkeypatch ile
stub'lar — gerçek sunucu/network yok (kök AGENTS.md kalıcı kural).

D-31: `OpenAICompatEmbedder`'ın kendisi artık `medrag.core.llm.client`de
yaşıyor (vectorize/retrieval'ın byte-birebir aynı ikizi oradan tek kaynağa
indi) — `embed()` o modülün `_post_json`'ını çağırıyor, stub bu yüzden
`oc` (bu bileşenin re-export modülü) yerine oraya patch'lenir."""

import pytest

from medrag.core.llm import client as oc
from medrag.pipeline.vectorize.embedder.openai_compat import (
    OpenAICompatEmbedder,
    ProviderError,
    embedder_from_env,
)


def _stub(monkeypatch, yanit, kayit=None):
    def sahte(url, payload, *, api_key, timeout):
        if kayit is not None:
            kayit.append((url, payload, api_key))
        return yanit
    monkeypatch.setattr(oc, "_post_json", sahte)


def test_embedder_payload_ve_endpoint(monkeypatch):
    kayit = []
    _stub(monkeypatch, {"data": [
        {"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}]}, kayit)
    e = OpenAICompatEmbedder(base_url="http://x/v1/", model="m", api_key="k")
    e.embed(["a", "b"])
    url, payload, api_key = kayit[0]
    assert url == "http://x/v1/embeddings"
    assert payload == {"model": "m", "input": ["a", "b"]}
    assert api_key == "k"


def test_embedder_index_alanina_gore_siralar(monkeypatch):
    _stub(monkeypatch, {"data": [
        {"index": 1, "embedding": [2.0, 2.0]},
        {"index": 0, "embedding": [1.0, 1.0]}]})
    e = OpenAICompatEmbedder(base_url="http://x/v1", model="m")
    assert e.embed(["a", "b"]) == [[1.0, 1.0], [2.0, 2.0]]


def test_embedder_eksik_vektor_yuksek_sesle(monkeypatch):
    _stub(monkeypatch, {"data": [{"index": 0, "embedding": [1.0]}]})
    e = OpenAICompatEmbedder(base_url="http://x/v1", model="m")
    with pytest.raises(ProviderError, match="1/2"):
        e.embed(["a", "b"])


def test_embedder_bos_girdi_http_cagirmaz(monkeypatch):
    kayit = []
    _stub(monkeypatch, {}, kayit)
    e = OpenAICompatEmbedder(base_url="http://x/v1", model="m")
    assert e.embed([]) == []
    assert kayit == []


def test_embedder_from_env_anthropic_reddedilir():
    with pytest.raises(ProviderError, match="anthropic"):
        embedder_from_env({"EMBEDDING_PROVIDER": "anthropic", "EMBEDDING_MODEL": "m"})


def test_embedder_from_env_ollama_varsayilan_base_url():
    e = embedder_from_env({"EMBEDDING_PROVIDER": "ollama", "EMBEDDING_MODEL": "m"})
    assert e.base_url == "http://localhost:11434/v1"
    assert e.name == "ollama:m"


def test_embedder_from_env_model_eksik():
    with pytest.raises(ProviderError, match="EMBEDDING_MODEL"):
        embedder_from_env({"EMBEDDING_PROVIDER": "ollama"})
