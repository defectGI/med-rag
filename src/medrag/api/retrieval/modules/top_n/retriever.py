"""top_n retriever: embed the query, search the nearest neighbors in a vector
store.

Two seams keep it testable and dependency-isolated, same shape as `db_query`:
- :class:`~retrieval.modules.top_n.embedder.Embedder` (query -> vector)
- :class:`~retrieval.modules.top_n.store.VectorStore` (vector -> scored hits)

Both seams are synchronous (embedding HTTP call, Qdrant client call); they are
run in a thread via ``asyncio.to_thread`` so the async `Retriever` contract
never blocks the event loop (same pattern as `db_query.DbQueryRetriever`).
"""

from __future__ import annotations

import asyncio
from typing import Any

from medrag.api.retrieval.core import RetrievalResult
from medrag.api.retrieval.modules.top_n.embedder import Embedder
from medrag.api.retrieval.modules.top_n.store import VectorStore


def _hit_to_result(hit: dict[str, Any], *, index: int) -> RetrievalResult:
    node_id = hit.get("node_id")
    return RetrievalResult(
        id=str(node_id) if node_id is not None else str(index),
        score=float(hit.get("score", 0.0)),
        text=str(hit.get("text", "")),
        metadata=hit,
    )


class TopNRetriever:
    """Implements the core `Retriever` protocol over an embedder + vector store."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    async def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        def _search() -> list[dict[str, Any]]:
            vector = self._embedder.embed([query])[0]
            return self._store.search(vector, limit=k)

        hits = await asyncio.to_thread(_search)
        return [_hit_to_result(hit, index=i) for i, hit in enumerate(hits)]


__all__ = ["TopNRetriever"]
