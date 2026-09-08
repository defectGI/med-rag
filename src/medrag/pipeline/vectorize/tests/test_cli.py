"""`main()`un uçtan uca akışı; embedder/Qdrant sahte nesnelerle stub'lanır
(kök AGENTS.md kalıcı kural: bu makinede gerçek LLM/GPU/servis çağrısı yok).
Gerçek `.env` dosyasının testlere sızmaması için `dotenv.load_dotenv` no-op'a
alınır — testler yalnızca `monkeypatch.setenv` ile verilen ortamı görür."""

import json

import pytest

import medrag.pipeline.vectorize.cli as cli_mod
from medrag.pipeline.vectorize.core import ChunkNode, ChunkSet


class _SahteEmbedder:
    name = "fake:test"

    def embed(self, texts):
        return [[float(len(t)), 0.0, 1.0] for t in texts]


class _SahteStore:
    def __init__(self):
        self.ensure_calls = []
        self.replace_calls = []

    def ensure_collection(self, vector_size):
        self.ensure_calls.append(vector_size)

    def replace_scope(self, scope_id, points):
        self.replace_calls.append((scope_id, list(points)))


@pytest.fixture(autouse=True)
def _no_real_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)


@pytest.fixture
def sahte_store(monkeypatch):
    store = _SahteStore()
    monkeypatch.setattr(cli_mod.QdrantVectorStore, "from_env",
                        classmethod(lambda cls, **kw: store))
    return store


@pytest.fixture
def sahte_embedder(monkeypatch):
    embedder = _SahteEmbedder()
    monkeypatch.setattr(cli_mod, "embedder_from_env", lambda **kw: embedder)
    return embedder


def _write_chunks(path, doc_id="DOC1", images=None):
    node = {"node_id": f"{doc_id}::c0", "doc_id": doc_id, "tree_level": 0,
            "text": "merhaba dünya", "heading_path": ["Giriş"]}
    if images is not None:
        node["images"] = images
    path.write_text(json.dumps({
        "doc_id": doc_id,
        "provenance": {"chunker_version": "1.0", "generated_at": "t",
                      "source": {"raw_sha256": "sha256:abc"}},
        "nodes": [node],
    }), encoding="utf-8")


def test_node_payload_dogru_alanlari_tasir():
    cs = ChunkSet.model_validate({
        "doc_id": "DOC1",
        "scope": "document",
        "nodes": [{"node_id": "DOC1::c0", "doc_id": "DOC1", "tree_level": 0,
                  "text": "merhaba", "heading_path": ["Giriş"],
                  "page_start": 1, "page_end": 2, "source_path": "a.pdf",
                  "fmt": "pdf"}],
    })
    payload = cli_mod._node_payload(cs.nodes[0], cs)
    assert payload == {
        "node_id": "DOC1::c0", "doc_id": "DOC1", "scope": "document",
        "tree_level": 0, "text": "merhaba", "keywords": [],
        "heading_path": ["Giriş"], "page_start": 1, "page_end": 2,
        "source_path": "a.pdf", "fmt": "pdf", "images": [],
        "chunk_schema_version": 1, "chunker_version": None,
        "chunk_generated_at": None,
    }


def test_node_payload_provenance_version_alanlari_tasinir():
    cs = ChunkSet.model_validate({
        "doc_id": "DOC1", "schema_version": 6,
        "provenance": {"chunker_version": "1.2.3", "generated_at": "2026-08-06T00:00:00+03:00"},
        "nodes": [{"node_id": "DOC1::c0", "doc_id": "DOC1", "text": "merhaba"}],
    })
    payload = cli_mod._node_payload(cs.nodes[0], cs)
    assert payload["chunk_schema_version"] == 6
    assert payload["chunker_version"] == "1.2.3"
    assert payload["chunk_generated_at"] == "2026-08-06T00:00:00+03:00"


def test_node_payload_images_yazilir():
    node = ChunkNode.model_validate({
        "node_id": "DOC1::c0", "doc_id": "DOC1", "text": "merhaba",
        "images": [{"image_id": "img-1"}, {"image_id": None}],
    })
    cs = ChunkSet.model_validate({"doc_id": "DOC1", "nodes": []})
    payload = cli_mod._node_payload(node, cs)
    assert payload["images"] == [{"image_id": "img-1"}, {"image_id": None}]


def _env(monkeypatch, tmp_path, **extra):
    monkeypatch.setenv("VECTORIZE_INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("VECTORIZE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("QDRANT_URL", "http://x:6333")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")
    monkeypatch.setenv("VECTORIZE_PROGRESS", "off")
    for k, v in extra.items():
        monkeypatch.setenv(k, v)


def test_girdi_klasoru_tanimsizsa_2_doner(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    monkeypatch.delenv("VECTORIZE_INPUT_DIR", raising=False)
    assert cli_mod.main([]) == 2


def test_qdrant_url_tanimsizsa_2_doner(monkeypatch, tmp_path, sahte_embedder):
    (tmp_path / "in").mkdir()
    monkeypatch.setenv("VECTORIZE_INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_MODEL", "m")
    assert cli_mod.main([]) == 2


def test_bos_girdi_klasoru_2_doner(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _env(monkeypatch, tmp_path)
    assert cli_mod.main([]) == 2


def test_basarili_kosu_0_doner_ve_store_a_yazar(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _write_chunks(tmp_path / "in" / "DOC1.chunks.json")
    _env(monkeypatch, tmp_path)
    assert cli_mod.main([]) == 0
    assert sahte_store.replace_calls[0][0] == "DOC1"
    assert len(sahte_store.replace_calls[0][1]) == 1
    assert sahte_store.ensure_calls == [3]


def test_basarili_kosu_images_payloada_yazilir(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _write_chunks(tmp_path / "in" / "DOC1.chunks.json",
                  images=[{"image_id": "img-1"}])
    _env(monkeypatch, tmp_path)
    assert cli_mod.main([]) == 0
    points = sahte_store.replace_calls[0][1]
    assert points[0].payload["images"] == [{"image_id": "img-1"}]


def test_ikinci_kosu_degismemisse_atlanir(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _write_chunks(tmp_path / "in" / "DOC1.chunks.json")
    _env(monkeypatch, tmp_path)
    assert cli_mod.main([]) == 0
    sahte_store.replace_calls.clear()
    assert cli_mod.main([]) == 0
    assert sahte_store.replace_calls == []


def test_force_bayragi_bayatlik_kapisini_bypass_eder(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _write_chunks(tmp_path / "in" / "DOC1.chunks.json")
    _env(monkeypatch, tmp_path)
    assert cli_mod.main([]) == 0
    sahte_store.replace_calls.clear()
    assert cli_mod.main(["--force"]) == 0
    assert len(sahte_store.replace_calls) == 1


def test_limit_bayragi_kapsam_sayisini_sinirlar(monkeypatch, tmp_path, sahte_store, sahte_embedder):
    (tmp_path / "in").mkdir()
    _write_chunks(tmp_path / "in" / "DOC1.chunks.json", doc_id="DOC1")
    _write_chunks(tmp_path / "in" / "DOC2.chunks.json", doc_id="DOC2")
    _env(monkeypatch, tmp_path)
    assert cli_mod.main(["--limit", "1"]) == 0
    assert len(sahte_store.replace_calls) == 1
