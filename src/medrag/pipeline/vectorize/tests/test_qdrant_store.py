"""Gerçek Qdrant sunucusuna karşı DOĞRULANMAMIŞTIR (kök AGENTS.md kalıcı
kural — bu makinede LLM/GPU/servis inference çalıştırılmaz gibi bir servise
bağımlı testler de burada koşulmaz). Testler `qdrant_client.QdrantClient`in
kullandığımız yüzeyini (collection_exists/create_collection/get_collection/
delete/upsert) taklit eden sahte bir istemciyle çalışır."""

import pytest
from qdrant_client.http import models as qm

from medrag.pipeline.vectorize.store.qdrant_store import (
    DimensionMismatchError,
    QdrantVectorStore,
    VectorPoint,
    node_point_id,
)


class _SahteCollectionInfo:
    def __init__(self, size):
        self.config = type("C", (), {"params": type("P", (), {
            "vectors": type("V", (), {"size": size})()})()})()


class SahteQdrantClient:
    def __init__(self, mevcut_boyut=None):
        self.mevcut_boyut = mevcut_boyut
        self.created = []
        self.deleted_filters = []
        self.upserted = []

    def collection_exists(self, name):
        return self.mevcut_boyut is not None

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config.size, vectors_config.distance))
        self.mevcut_boyut = vectors_config.size

    def get_collection(self, name):
        return _SahteCollectionInfo(self.mevcut_boyut)

    def delete_collection(self, name):
        self.mevcut_boyut = None

    def delete(self, collection_name, points_selector):
        self.deleted_filters.append(points_selector.filter.must[0].match.value)

    def upsert(self, collection_name, points):
        self.upserted.extend(points)

    def query_points(self, collection_name, query, limit, with_payload):
        Nokta = type("ScoredPoint", (), {})
        sonuclar = []
        for i, p in enumerate(self.upserted[:limit]):
            n = Nokta()
            n.score = 1.0 - i * 0.1
            n.payload = p.payload
            sonuclar.append(n)
        return type("QueryResponse", (), {"points": sonuclar})()


def _store(client, **kw):
    return QdrantVectorStore(client=client, collection_name="c", distance="cosine",
                             upsert_batch_size=kw.get("upsert_batch_size", 128),
                             on_dim_mismatch=kw.get("on_dim_mismatch", "error"))


def test_node_point_id_deterministik_ve_uuid_bicimli():
    a = node_point_id("DOC1::c0")
    b = node_point_id("DOC1::c0")
    c = node_point_id("DOC1::c1")
    assert a == b
    assert a != c
    import uuid
    uuid.UUID(a)  # patlamazsa geçerli UUID demektir


def test_ensure_collection_yoksa_kurar():
    client = SahteQdrantClient(mevcut_boyut=None)
    store = _store(client)
    store.ensure_collection(vector_size=4)
    assert client.created == [("c", 4, qm.Distance.COSINE)]


def test_ensure_collection_ayni_boyut_dokunmaz():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client)
    store.ensure_collection(vector_size=4)
    assert client.created == []


def test_ensure_collection_boyut_uyusmazliginda_error_ile_patlar():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client, on_dim_mismatch="error")
    with pytest.raises(DimensionMismatchError):
        store.ensure_collection(vector_size=8)


def test_ensure_collection_recreate_ile_yeniden_kurar():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client, on_dim_mismatch="recreate")
    store.ensure_collection(vector_size=8)
    assert client.created == [("c", 8, qm.Distance.COSINE)]


def test_replace_scope_once_siler_sonra_upsert_eder():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client)
    points = [VectorPoint(node_id="DOC1::c0", vector=[0.1, 0.2], payload={"doc_id": "DOC1"})]
    store.replace_scope("DOC1", points)
    assert client.deleted_filters == ["DOC1"]
    assert len(client.upserted) == 1
    assert client.upserted[0].id == node_point_id("DOC1::c0")


def test_replace_scope_bos_liste_yalniz_siler():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client)
    store.replace_scope("DOC1", [])
    assert client.deleted_filters == ["DOC1"]
    assert client.upserted == []


def test_search_skor_ve_payload_dondurur():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client)
    store.replace_scope("DOC1", [
        VectorPoint(node_id="DOC1::c0", vector=[0.1], payload={"doc_id": "DOC1", "text": "a"}),
        VectorPoint(node_id="DOC1::c1", vector=[0.2], payload={"doc_id": "DOC1", "text": "b"}),
    ])
    sonuclar = store.search([0.15], limit=2)
    assert sonuclar == [
        {"score": 1.0, "doc_id": "DOC1", "text": "a"},
        {"score": 0.9, "doc_id": "DOC1", "text": "b"},
    ]


def test_replace_scope_batch_boyutuna_gore_boler():
    client = SahteQdrantClient(mevcut_boyut=4)
    store = _store(client, upsert_batch_size=2)
    points = [VectorPoint(node_id=f"DOC1::c{i}", vector=[0.0], payload={})
              for i in range(5)]
    calls = []
    orig_upsert = client.upsert
    def sayan(collection_name, points):
        calls.append(len(points))
        orig_upsert(collection_name, points)
    client.upsert = sayan
    store.replace_scope("DOC1", points)
    assert calls == [2, 2, 1]
    assert len(client.upserted) == 5
