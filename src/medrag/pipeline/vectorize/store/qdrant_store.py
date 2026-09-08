"""Qdrant istemcisi: koleksiyon kurulumu + "kapsamı değiştir" sözleşmesi.

Kontrat (kök `chunker/chunker/core/chunk.py` docstring'i, node_id bölümü —
KARAR-004): chunk node_id'leri SET-KAPSAMLIDIR, setler arası kararlılık VAAT
EDİLMEZ. Bir doküman yeniden chunk'landığında `{doc_id}::c{n}` numaralandırması
sıfırdan atanır; eski numaralar yeni koşuda farklı içeriğe karşılık gelebilir.
Bu yüzden bir vektör DB'ye yazan taraf (chunk.py'nin kendi sözünü tuttuğu
üzere) bir kapsamı güncellerken "yalnız değişeni upsert et" OYNAMAZ — o
kapsamın (`doc_id`/`_corpus`/`_profile.*`) Qdrant'taki TÜM eski noktalarını
siler, sonra yeni setin tamamını yazar. `replace_scope` bunu tek metotta
kapsüller ki çağıran taraf (cli.py) yanlışlıkla incremental upsert'e kaymasın.

Nokta kimliği: Qdrant yalnız unsigned int veya UUID kabul eder; chunk
`node_id`leri serbest metin (`{doc_id}::c3` gibi) olduğundan `uuid5` ile
deterministik bir UUID'ye çevrilir (aynı node_id → aynı nokta id'si, iki ayrı
koşu çakışmaz). Orijinal `node_id` payload'da `node_id` alanında saklanır —
kimliğin okunabilir hali kaybolmaz.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

log = logging.getLogger("vectorize.store")

# Sabit namespace: bu paketin kendi UUID5 kökü. Değiştirilirse TÜM node_id →
# point_id eşlemesi değişir (mevcut koleksiyondaki her nokta "yeni" sayılır).
_NAMESPACE = uuid.UUID("d3c1b8b4-8b7e-4c1a-9c2e-2b6f7f1a9a01")


def node_point_id(node_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, node_id))


@dataclass
class VectorPoint:
    node_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


class DimensionMismatchError(RuntimeError):
    """Koleksiyon zaten farklı boyutta kurulu ve `on_dim_mismatch=error`."""


class QdrantVectorStore:
    """`qdrant-client`in ince bir sarmalayıcısı — istemci burada tek yerde
    kurulur, çağıran taraf (cli.py) yalnızca `ensure_collection`/`replace_scope`
    çağırır."""

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
        """Koleksiyon yoksa kurar; varsa boyutunu doğrular. Boyut uyuşmazsa
        `on_dim_mismatch` kararına göre ya yüksek sesle patlar ya da
        koleksiyonu silip yeniden kurar (veri kaybı — bilinçli opt-in)."""
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
        """`scope_id`ye ait TÜM eski noktaları siler, ardından `points`i yazar
        (KARAR-004 sonucu — bkz. modül docstring'i). `points` boşsa yalnız
        silme yapılır (kapsamın artık hiç düğümü yoksa, ör. boş doküman)."""
        from qdrant_client.http import models as qm

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
        """En yakın `limit` noktayı skoruyla birlikte döndürür — her sonuç
        `_node_payload`nin yazdığı TÜM alanları (text, doc_id, node_id,
        heading_path, ...) + `score` taşır (bkz. `vectorize/query.py`)."""
        sonuc = self.client.query_points(
            collection_name=self.collection_name, query=vector, limit=limit,
            with_payload=True)
        return [{"score": nokta.score, **(nokta.payload or {})} for nokta in sonuc.points]
