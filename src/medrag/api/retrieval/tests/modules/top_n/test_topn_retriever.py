"""TopNRetriever end-to-end with a fake embedder + fake vector store."""

from __future__ import annotations

import asyncio

from medrag.api.retrieval.core import Retriever
from medrag.api.retrieval.modules.top_n.retriever import TopNRetriever


class _FakeEmbedder:
    def __init__(self, vector: list[float]):
        self._vector = vector
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [self._vector for _ in texts]


class _FakeStore:
    def __init__(self, hits: list[dict]):
        self._hits = hits
        self.calls: list[tuple] = []

    def search(self, vector, *, limit):
        self.calls.append((vector, limit))
        return self._hits[:limit]


def test_retrieve_embeds_query_and_maps_hits():
    embedder = _FakeEmbedder([0.1, 0.2])
    store = _FakeStore([
        {"score": 0.9, "node_id": "c1", "text": "hello", "doc_id": "d1"},
        {"score": 0.5, "node_id": "c2", "text": "world"},
    ])
    r = TopNRetriever(embedder, store)
    results = asyncio.run(r.retrieve("soru", k=2))

    assert embedder.calls == [["soru"]]
    assert store.calls == [([0.1, 0.2], 2)]
    assert [x.id for x in results] == ["c1", "c2"]
    assert [x.score for x in results] == [0.9, 0.5]
    assert results[0].text == "hello"
    assert results[0].metadata["doc_id"] == "d1"


def test_retrieve_falls_back_to_index_when_node_id_missing():
    embedder = _FakeEmbedder([0.0])
    store = _FakeStore([{"score": 0.1, "text": "no id"}])
    r = TopNRetriever(embedder, store)
    results = asyncio.run(r.retrieve("q", k=1))
    assert results[0].id == "0"


def test_retrieve_empty_hits():
    embedder = _FakeEmbedder([0.0])
    store = _FakeStore([])
    r = TopNRetriever(embedder, store)
    assert asyncio.run(r.retrieve("q", k=5)) == []


def test_retrieve_passes_arbitrary_metadata_fields_through_untouched():
    # `_hit_to_result` must not special-case or drop any hit key -- it copies
    # the whole dict into `metadata` as-is, so new fields written to a vector
    # store's payload (e.g. an "images" list) reach the caller with no code
    # change on this side.
    hit = {
        "score": 0.7,
        "node_id": "c1",
        "text": "hello",
        "images": [{"image_id": "img-1"}, {"image_id": "img-2"}],
    }
    embedder = _FakeEmbedder([0.0])
    store = _FakeStore([hit])
    r = TopNRetriever(embedder, store)
    results = asyncio.run(r.retrieve("q", k=1))
    # pydantic validation copies the dict (RetrievalResult.metadata is not the
    # same object), but every key -- including a new "images" field -- must
    # survive untouched.
    assert results[0].metadata == hit
    assert results[0].metadata["images"] == [{"image_id": "img-1"}, {"image_id": "img-2"}]


def test_retriever_satisfies_protocol():
    r = TopNRetriever(_FakeEmbedder([0.0]), _FakeStore([]))
    assert isinstance(r, Retriever)
