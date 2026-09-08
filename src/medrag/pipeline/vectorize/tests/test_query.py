"""`vectorize.query`nin offline testleri — embedder/Qdrant stub'lanır (kök
AGENTS.md kalıcı kural: bu makinede gerçek servis çağrısı yok)."""

import json

import pytest

import medrag.pipeline.vectorize.query as query_mod


class _SahteEmbedder:
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _SahteStore:
    def __init__(self, sonuclar):
        self.sonuclar = sonuclar
        self.aranan_vektor = None
        self.limit = None

    def search(self, vector, *, limit):
        self.aranan_vektor = vector
        self.limit = limit
        return self.sonuclar


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def ortam(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "http://x:6333")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")


def _stub_store(monkeypatch, sonuclar):
    store = _SahteStore(sonuclar)
    monkeypatch.setattr(query_mod.QdrantVectorStore, "from_env",
                        classmethod(lambda cls, **kw: store))
    monkeypatch.setattr(query_mod, "embedder_from_env", lambda *a, **kw: _SahteEmbedder())
    return store


def test_search_dogru_vektor_ve_limit_ile_cagirir(monkeypatch, ortam):
    store = _stub_store(monkeypatch, [{"score": 0.9, "doc_id": "DOC1", "text": "x"}])
    sonuc = query_mod.search("bir soru", k=3)
    assert store.aranan_vektor == [1.0, 0.0]
    assert store.limit == 3
    assert sonuc == [{"score": 0.9, "doc_id": "DOC1", "text": "x"}]


def test_qdrant_url_eksikse_provider_error(monkeypatch, ortam):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(query_mod.ProviderError):
        query_mod.search("bir soru")


def test_main_json_ciktisi(monkeypatch, ortam, capsys):
    _stub_store(monkeypatch, [{"score": 0.9, "doc_id": "DOC1", "text": "x"}])
    assert query_mod.main(["bir soru", "--json"]) == 0
    cikti = json.loads(capsys.readouterr().out)
    assert cikti == [{"score": 0.9, "doc_id": "DOC1", "text": "x"}]


def test_main_insan_okunur_cikti(monkeypatch, ortam, capsys):
    _stub_store(monkeypatch, [{"score": 0.9, "doc_id": "DOC1", "node_id": "DOC1::c0",
                              "text": "merhaba", "heading_path": ["A", "B"]}])
    assert query_mod.main(["bir soru"]) == 0
    cikti = capsys.readouterr().out
    assert "DOC1::c0" in cikti
    assert "merhaba" in cikti
    assert "A > B" in cikti


def test_main_qdrant_url_eksikse_2_doner(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert query_mod.main(["bir soru"]) == 2


def test_main_sonuc_yoksa_bos_mesaj(monkeypatch, ortam, capsys):
    _stub_store(monkeypatch, [])
    assert query_mod.main(["bir soru"]) == 0
    assert "sonuç yok" in capsys.readouterr().out
