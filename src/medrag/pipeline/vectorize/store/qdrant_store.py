"""Qdrant client: collection setup + the "replace scope" contract.

Contract (root `chunker/chunker/core/chunk.py` docstring, node_id section --
KARAR-004): chunk node_ids are SET-SCOPED, stability ACROSS sets is NOT
PROMISED. When a document is re-chunked, the `{doc_id}::c{n}` numbering is
reassigned from scratch; old numbers may correspond to different content in a
new run. So the side writing to a vector DB (keeping chunk.py's own promise)
does NOT play "upsert only the changed one" when updating a scope -- it deletes
ALL old points of that scope (`doc_id`/`_corpus`/`_profile.*`) in Qdrant, then
writes the whole new set. `replace_scope` encapsulates this in a single method
so the calling side (cli.py) cannot accidentally drift into incremental upsert.

Point id: Qdrant only accepts unsigned int or UUID; chunk `node_id`s are free
text (like `{doc_id}::c3`) so they are converted via `uuid5` to a deterministic
UUID (same node_id → same point id, two separate runs do not collide). The
original `node_id` is stored in the `node_id` payload field so the readable
form of the id is not lost.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("vectorize.store")

# Fixed namespace: this package's own UUID5 root. If changed, EVERY node_id →
# point_id mapping changes (every point in the existing collection counts as "new").
_NAMESPACE = uuid.UUID("d3c1b8b4-8b7e-4c1a-9c2e-2b6f7f1a9a01")


def node_point_id(node_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, node_id))


@dataclass
class VectorPoint:
    node_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


class DimensionMismatchError(RuntimeError):
    """The collection was already set up at a different size and `on_dim_mismatch=error`."""


class QdrantVectorStore:
    """A thin wrapper around `qdrant-client` -- the client is constructed here in
    one place; the calling side (cli.py) only calls `ensure_collection`/`replace_scope`."""

    def __init__(self, *, client: Any, collection_name: str,
                 distance: Literal["cosine", "dot", "euclid"],
                 on_dim_mismatch: Literal["error", "recreate"] = "error",
                 upsert_batch_size: int = 128):
        self.client = client
        self.collection_name = collection_name
        self.distance = distance
        self.on_dim_mismatch = on_dim_mismatch
        self.upsert_batch_size = upsert_batch_size

    @classmethod
    def from_env(cls, *, collection_name: str, distance: str,
                 on_dim_mismatch: str, upsert_batch_size: int,
                 url: str, api_key: str | None = None,
                 timeout: float = 60.0) -> QdrantVectorStore:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=url, api_key=api_key or None, timeout=timeout)
        return cls(client=client, collection_name=collection_name,
                   distance=distance, on_dim_mismatch=on_dim_mismatch,
                   upsert_batch_size=upsert_batch_size)

    def ensure_collection(self, vector_size: int) -> None:
        """Creates the collection if missing; verifies its size if present. On a
        size mismatch, per `on_dim_mismatch`, either fails loudly or deletes and
        recreates the collection (data loss -- deliberate opt-in)."""
        from qdrant_client.http import models as qm

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qm.VectorParams(
                    size=vector_size, distance=self._qdrant_distance(qm)))
            log.info("qdrant koleksiyonu kuruldu: %s (size=%d, distance=%s)",
                     self.collection_name, vector_size, self.distance)
            return

        bilgi = self.client.get_collection(self.collection_name)
        mevcut_boyut = bilgi.config.params.vectors.size
        if mevcut_boyut == vector_size:
            return
        if self.on_dim_mismatch == "error":
            raise DimensionMismatchError(
                f"koleksiyon {self.collection_name!r} boyutu {mevcut_boyut} "
                f"ama gelen embedding boyutu {vector_size} — modeli mi "
                f"değiştirdiniz? Bilinçli geçiş için QDRANT_ON_DIM_MISMATCH="
                f"recreate (config/default.toml) VERİ KAYBI riskiyle kullanılır")
        log.warning("qdrant koleksiyonu %s yeniden kuruluyor: %d -> %d boyut "
                    "(on_dim_mismatch=recreate, mevcut TÜM veri siliniyor)",
                    self.collection_name, mevcut_boyut, vector_size)
        self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qm.VectorParams(
                size=vector_size, distance=self._qdrant_distance(qm)))

    def _qdrant_distance(self, qm: Any) -> Any:
        return {"cosine": qm.Distance.COSINE, "dot": qm.Distance.DOT,
                "euclid": qm.Distance.EUCLID}[self.distance]

    def replace_scope(self, scope_id: str, points: list[VectorPoint]) -> None:
        """Deletes ALL old points belonging to `scope_id`, then writes `points`
        (the KARAR-004 outcome -- see the module docstring). If `points` is empty
        only deletion happens (when the scope has no nodes left, e.g. empty doc)."""
        from qdrant_client.http import models as qm

        # CRITICAL: in an empty Qdrant the collection does not yet exist on the
        # first processing (`ensure_collection` has not been called yet); here
        # `delete` dies with a 404 and the first loaded document lands on "error"
        # (med-rag boot bug: first setup, first document guaranteed error). If the
        # collection is missing there are no points to delete either -- deletion
        # is a no-op, and the write path already sets the collection up in
        # `cli.py` via `ensure_collection`. If `points` is non-empty the
        # collection definitely exists (the caller ensures first); even so, if it
        # is missing the no-op deletion above protects the incomplete setup, while
        # the write still 404s -- which, per the "fail loudly" principle, exposes
        # the wrong alignment.
        if self.client.collection_exists(self.collection_name):
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=scope_id))
                ])))

        for i in range(0, len(points), self.upsert_batch_size):
            parca = points[i:i + self.upsert_batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[qm.PointStruct(id=node_point_id(p.node_id), vector=p.vector,
                                       payload=p.payload) for p in parca])
        log.info("qdrant: kapsam %s → %d nokta yazıldı", scope_id, len(points))

    def search(self, vector: list[float], *, limit: int) -> list[dict]:
        """Returns the `limit` nearest points together with their score -- each
        result carries ALL the fields `_node_payload` writes (text, doc_id,
        node_id, heading_path, ...) plus `score` (see `vectorize/query.py`)."""
        sonuc = self.client.query_points(
            collection_name=self.collection_name, query=vector, limit=limit,
            with_payload=True)
        return [{"score": nokta.score, **(nokta.payload or {})} for nokta in sonuc.points]
