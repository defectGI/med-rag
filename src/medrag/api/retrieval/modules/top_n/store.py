"""Read-only Qdrant search client.

This module never builds or writes a Qdrant collection -- it QUERIES one
that already exists (same stance as `db_query`: "this module does not build
the DB it runs against"). The producer of that collection in this project's
reference pipeline is the `vectorize` component's Qdrant store, but this is a
format contract, not a code dependency (ARCHITECTURE.md #5/#9): any project
writing points with the same payload shape (`text`, `node_id`, `doc_id`,
`heading_path`, `page_start`/`page_end`, `keywords`, ...) can be queried
through this class.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """The seam this module depends on -- unit tests inject a fake."""

    def search(self, vector: list[float], *, limit: int) -> list[dict[str, Any]]: ...


class QdrantVectorStore:
    """Thin, read-only wrapper around `qdrant-client`."""

    def __init__(self, *, client: Any, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    def from_env(
        cls,
        *,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> QdrantVectorStore:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        return cls(client=client, collection_name=collection_name)

    def search(self, vector: list[float], *, limit: int) -> list[dict[str, Any]]:
        """Nearest `limit` points with score -- each result carries every
        payload field the producer wrote (`text`, `node_id`, `doc_id`, ...)
        plus `score`."""
        result = self._client.query_points(
            collection_name=self._collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [{"score": point.score, **(point.payload or {})} for point in result.points]


__all__ = ["QdrantVectorStore", "VectorStore"]
