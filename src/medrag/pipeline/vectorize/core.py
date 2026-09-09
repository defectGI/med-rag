"""Input contract: the READER side of the ChunkSet/ChunkNode JSON that
`chunker` produces (root AGENTS.md architectural rule -- components do not
import each other, the single bridge is `pipeline`; `vectorize` therefore does
NOT import `chunker.core.chunk`, it MIRRORS the shape here).

Rule (same pattern as the `retrieval/` package, ARCHITECTURE.md #9): the schema
is defined as a PYDANTIC MODEL, not a code dependency, with `extra="ignore"` --
if chunker's schema adds a new field (without bumping schema_version) vectorize
silently tolerates it; it only breaks if one of the fields THIS module reads is
removed/renamed (pydantic required-field error).

Deliberately not carried over: `summary`, `parent_id`, `child_ids`,
`source_block_ids`, `cross_refs`, split/flex diagnostics -- vectorization does
not need them and never reads them, to avoid bloating the payload.
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
    """The subset of fields vectorize sees (for chunker's full schema see
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
    """The chunk output of a scope (document/`_corpus`/`_profile.*`)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: int = 1
    doc_id: str
    scope: Literal["document", "corpus", "profile"] = "document"
    member_doc_ids: list[str] = Field(default_factory=list)
    provenance: ChunkProvenance | None = None
    nodes: list[ChunkNode] = Field(default_factory=list)

    def signature(self, *, embedding_model: str) -> dict[str, Any]:
        """Staleness signature: same signature + same embedding model → no
        need to re-embed (see `state.py`). `generated_at` is DELIBERATELY
        excluded -- a set that was re-produced but whose content stayed the
        same should not be re-embedded needlessly (same reasoning as chunker's
        own `ChunkProvenance.same_inputs` principle).

        `text_sha256` exists to catch in-place text edits the provenance
        fields do not capture (e.g. `strip_injected_ocr.py` stripping OCR text
        from node.text -- chunker_version/raw_sha256/node_count do not
        change); without it such an edit would silently count as "unchanged"
        and be re-embed skipped."""
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
