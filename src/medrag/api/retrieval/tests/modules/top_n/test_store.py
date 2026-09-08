"""QdrantVectorStore.search mapping tests. No network -- a fake client with
the same `query_points` shape as `qdrant_client.QdrantClient` is injected."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medrag.api.retrieval.modules.top_n.store import QdrantVectorStore, VectorStore


@dataclass
class _Point:
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class _QueryResult:
    points: list[_Point]


class _FakeClient:
    def __init__(self, points: list[_Point]) -> None:
        self._points = points
        self.calls: list[dict[str, Any]] = []

    def query_points(self, *, collection_name: str, query: list[float], limit: int,
                      with_payload: bool) -> _QueryResult:
        self.calls.append(
            {"collection_name": collection_name, "query": query, "limit": limit,
             "with_payload": with_payload}
        )
        return _QueryResult(points=self._points[:limit])


def test_search_flattens_score_and_payload():
    client = _FakeClient([_Point(score=0.9, payload={"node_id": "n1", "text": "hello"})])
    store = QdrantVectorStore(client=client, collection_name="chunks")
    hits = store.search([0.1, 0.2], limit=5)
    assert hits == [{"score": 0.9, "node_id": "n1", "text": "hello"}]
    assert client.calls[0] == {
        "collection_name": "chunks", "query": [0.1, 0.2], "limit": 5, "with_payload": True,
    }


def test_search_handles_missing_payload():
    client = _FakeClient([_Point(score=0.5, payload={})])
    store = QdrantVectorStore(client=client, collection_name="chunks")
    assert store.search([0.0], limit=1) == [{"score": 0.5}]


def test_store_satisfies_protocol():
    client = _FakeClient([])
    assert isinstance(QdrantVectorStore(client=client, collection_name="c"), VectorStore)
