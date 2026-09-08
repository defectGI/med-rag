"""Girdi sözleşmesi: `chunker`in ürettiği ChunkSet/ChunkNode JSON'ının OKUYUCU
tarafı (kök AGENTS.md mimari kuralı — bileşenler birbirini import etmez, tek
köprü `pipeline`dır; `vectorize` bu yüzden `chunker.core.chunk`ı import ETMEZ,
şekli burada AYNALAR).

Kural (`retrieval/` paketindeki aynı desenle, ARCHITECTURE.md #9): şema kod
bağımlılığı değil PYDANTIC MODELİ olarak tanımlanır, `extra="ignore"` ile —
chunker şeması yeni bir alan eklerse (schema_version yükselmeden) vectorize
sessizce tolere eder; yalnız BU modülün okuduğu alanlardan biri kaldırılır/adı
değişirse patlar (pydantic zorunlu alan hatası).

Bilinçli olarak taşınmayanlar: `summary`, `parent_id`, `child_ids`,
`source_block_ids`, `cross_refs`, split/flex diagnostikleri — vektörleştirme
bunlara ihtiyaç duymaz, payload'ı şişirmemek için hiç okunmaz.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ImageRef(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_id: str | None = None


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_sha256: str | None = None
    parser_version: str | None = None
    ir_version: int | None = None


class RaptorProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str | None = None
    model: str | None = None
    prompt_version: str | None = None


class ChunkProvenance(BaseModel):
    model_config = ConfigDict(extra="ignore")

    chunker_version: str | None = None
    generated_at: str | None = None
    source: SourceProvenance | None = None
    raptor: RaptorProvenance | None = None


class ChunkNode(BaseModel):
    """Vectorize'ın gördüğü alan alt kümesi (chunker'ın tam şeması için bkz.
    `chunker/chunker/core/chunk.py`)."""

    model_config = ConfigDict(extra="ignore")

    node_id: str
    doc_id: str
    tree_level: int = 0
    text: str
    keywords: list[str] = Field(default_factory=list)
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    source_path: str | None = None
    fmt: str | None = None
    token_count: int | None = None
    images: list[ImageRef] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.tree_level == 0


class ChunkSet(BaseModel):
    """Bir kapsamın (doküman/`_corpus`/`_profile.*`) chunk çıktısı."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    doc_id: str
    scope: Literal["document", "corpus", "profile"] = "document"
    member_doc_ids: list[str] = Field(default_factory=list)
    provenance: ChunkProvenance | None = None
    nodes: list[ChunkNode] = Field(default_factory=list)

    def signature(self, *, embedding_model: str) -> dict[str, Any]:
        """Bayatlık imzası: aynı embedding modeliyle aynı imza → yeniden
        embed etmeye gerek yok (bkz. `state.py`). `generated_at` BİLİNÇLİ
        olarak dışarıda — yeniden üretilen ama içeriği aynı kalan bir set
        gereksiz yere yeniden embed edilmesin (chunker'ın kendi
        `ChunkProvenance.same_inputs` ilkesiyle aynı gerekçe).

        `text_sha256` provenance alanlarının yakalamadığı in-place text
        düzenlemelerini (örn. `strip_injected_ocr.py`nin OCR metnini
        node.text'ten sıyırması — chunker_version/raw_sha256/node_count
        değişmez) yakalamak için var; onlarsız böyle bir edit sessizce
        "değişmemiş" sayılıp yeniden embed atlanırdı."""
        prov = self.provenance
        source = prov.source if prov else None
        raptor = prov.raptor if prov else None
        text_hash = hashlib.sha256(
            "\x00".join(node.text for node in self.nodes).encode("utf-8")
        ).hexdigest()
        return {
            "embedding_model": embedding_model,
            "chunker_version": prov.chunker_version if prov else None,
            "raw_sha256": source.raw_sha256 if source else None,
            "raptor_model": raptor.model if raptor else None,
            "raptor_prompt_version": raptor.prompt_version if raptor else None,
            "node_count": len(self.nodes),
            "text_sha256": text_hash,
        }
