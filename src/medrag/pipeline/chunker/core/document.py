"""Parser-agnostic inner document model.

The structure adapters target: each parser's adapter converts its IR into
this model, and the splitting engine only knows this model. No parser
type/schema is imported here.

Design decisions
----------------
* **A list is a container block.** Even when source IRs keep list items
  in a flat stream with metadata (first_parse's `list_id`/`list_level`
  does this), a list is an atomic unit for the engine; rebuilding it is
  the adapter's job. Nested lists are NOT a separate container; they're
  represented via `ListItem.level` — the natural model of source formats
  (docx `w:ilvl`, PDF).
* **A table is a flat text matrix.** The engine's only job with a table
  is to split it with k-row overlap and re-render to text; in-cell
  structure (nested tables, formatting) is flattened in the adapter
  (first_parse's `Cell.plain_text()`). `description` is the parser's
  natural-language table description; injected into each split part.
* **Provenance is the core's own vocabulary.** Parser labels
  ("text-layer-verified" etc.) are mapped to this enum at the adapter
  boundary; deterministic sources leave it None. A chunk's
  `provenance_summary` is the lowest among its blocks'.
* **Page range is normalized.** Source's single page / range distinction
  is flattened by the adapter; in the inner model each block carries
  only `page_start`/`page_end` (single page => equal, unknown => None).
* **Links are cross-ref raw material.** `LinkRef` is a candidate, not a
  resolved reference; resolving to a doc-internal chunk id is
  `enrichment/`'s job.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Provenance(str, Enum):
    """Confidence in a block's content (high to low).

    `str`-based: written to JSON as a plain string. Deterministic parsers
    (docx, html, ...) don't produce a provenance => the field stays None
    and is read as "trusted"; the enum is only for cases where the source
    itself reports confidence.
    """

    VERIFIED = "verified"      # confirmed verbatim against source's own text
    CONSENSUS = "consensus"    # independent readers agreed
    UNVERIFIED = "unverified"  # could not be verified

    @property
    def rank(self) -> int:
        """Sort key; larger = higher confidence."""
        return {"unverified": 0, "consensus": 1, "verified": 2}[self.value]

    @classmethod
    def lowest(cls, values: Iterable[Provenance | None]) -> Provenance | None:
        """Lowest confidence among the given values (for chunk summaries).

        None values (blocks that didn't report provenance) are ignored; if
        no value is given, returns None.
        """
        known = [v for v in values if v is not None]
        return min(known, key=lambda v: v.rank) if known else None


class LinkRef(BaseModel):
    """A hyperlink in a block: display text + target.

    Whether the target is in-document or out-of-document is NOT
    interpreted here; it's carried raw, resolution happens in enrichment.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    target: str


class _BlockBase(BaseModel):
    """Common fields for every block.

    id
        Source IR's block identifier (for traceability + cross-ref
        targets). For blocks the adapter synthesizes (e.g. the list
        container), the adapter derives a deterministic id.
    heading_path
        Ancestor heading texts, outer-to-inner. Empty list = before the
        first heading or the source didn't produce one.
    page_start / page_end
        1-based page range; both equal for a single page, None when
        unknown.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    heading_path: list[str] = Field(default_factory=list)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    provenance: Provenance | None = None
    links: list[LinkRef] = Field(default_factory=list)


class Heading(_BlockBase):
    """`anchor_id`: the source's own anchor/bookmark id (HTML `id`, docx
    bookmark, explicit markdown anchor) — the ONLY reliable way to bind a
    link.target `#fragment` to this heading. Same open-item category as
    `header_rows`: first_parse IR doesn't carry this today. Until the
    parser side adds it, the field stays None and in-document links are
    logged as unresolvable (`chunker/enrichment/cross_ref.py`) — matching
    by heading TEXT (slugify etc.) is DELIBERATELY not done (wrong-binding
    risk)."""

    kind: Literal["heading"] = "heading"
    text: str
    level: int = Field(default=1, ge=1)  # 1 = top-level heading
    anchor_id: str | None = None


class Paragraph(_BlockBase):
    kind: Literal["paragraph"] = "paragraph"
    text: str


class Code(_BlockBase):
    """Verbatim code block. `text` is preserved as-is (only ever modified
    by splitting); `language` is the source's hint, or None."""

    kind: Literal["code"] = "code"
    text: str
    language: str | None = None


class Table(_BlockBase):
    """Flattened table: row-major plain-text matrix.

    cells
        Each cell is plain text (the adapter flattens in-cell structure).
        Merge slots become empty strings in the adapter — row integrity is
        enough for the engine; merge geometry is not carried.
    header_rows
        How many leading rows are headers. When the table is split with
        k-row overlap, header rows are repeated in every part. 0 =
        headerless table.
    description
        Parser-produced natural-language description of the table (the
        shared `description` field in the first_parse IR — see
        `DescribableBlock`: a table is not a separate branch of this
        category, just the more deterministic member). Injected into
        each split part.
    facts
        Natural-language rendering of the table sentence by sentence
        (first_parse's `facts`); produced for retrieval, carried with
        structure intact. Injection is a separate config decision (the
        splitting engine). Only populated for tables — digit-check
        verification requires real cell data, absent from a pixel-based
        image.
    source_crop
        Audit crop for an "unverified" table (parser IR's `Block.
        source_crop`, carried verbatim) — None for most tables (only
        populated when text-containment verification failed). Exclusion
        placeholders use this as a reference, otherwise they would have
        no reference.
    excluded_at_parse
        True if the parser-time `[visual] exclude_types` had "table" added
        — no content (description/facts) was produced, only `cells` remain.
        Same field/meaning as `Image.excluded_at_parse` (see parser IR's
        `DescribableBlock`) — placeholder rendering uses the same helper
        for both (`core/markdown.py`).
    """

    kind: Literal["table"] = "table"
    cells: list[list[str]] = Field(default_factory=list)
    header_rows: int = Field(default=1, ge=0)
    description: str | None = None
    facts: list[str] = Field(default_factory=list)
    source_crop: str | None = None
    excluded_at_parse: bool = False

    @property
    def n_rows(self) -> int:
        return len(self.cells)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.cells), default=0)


class Image(_BlockBase):
    """An image in the document stream.

    No raw pixels are carried; `image_id` is a reference into the blob
    store. `ocr_text` is embedded into the chunk text AND additionally
    carried in the chunk's structured `images` list so the answer can
    re-show the relevant image.

    visual_type, source_crop, excluded_at_parse, description
        Visual-region classification + description fields (carried
        verbatim from parser IR via the adapter — see
        adapters/first_parse.py). `visual_type` is the taxonomy label
        (table/chart/block_diagram/technical_drawing/flowchart/
        product_photo/decorative/unknown) or None if not yet
        classified. `source_crop` — when `image_id` isn't yet resolved
        (e.g. a table candidate reclassified by the parser), this is
        the only crop id a placeholder can reference; `image_id` wins
        when present. `excluded_at_parse` True means parser config
        produced no content for this block (no OCR/description) — chunk
        rendering uses this to distinguish "processed but empty output"
        from "had real content". `description` is the natural-language
        description of this visual region — either deterministic from a
        native chart/SmartArt's own data (no model call) or produced by
        a VLM from crop + context (see parser IR's `DescribableBlock`;
        same field/meaning as `Table.description` — a table is not a
        separate branch of this category, just the more deterministic
        member).
    """

    kind: Literal["image"] = "image"
    image_id: str | None = None
    ocr_text: str | None = None
    alt_text: str | None = None
    visual_type: str | None = None
    source_crop: str | None = None
    excluded_at_parse: bool = False
    description: str | None = None


# Blocks that may live inside a list item: in source formats an <li> can
# contain paragraph/table/image/code; a list-in-list is not a separate
# container — it's represented via `ListItem.level`.
ListItemContent = Annotated[
    Paragraph | Table | Image | Code,
    Field(discriminator="kind"),
]


class ListItem(BaseModel):
    """One list item. `level` is 0-based nesting depth; `ordered` is the
    numbered/bullet distinction (can vary per item in the same list)."""

    model_config = ConfigDict(extra="forbid")

    level: int = Field(default=0, ge=0)
    ordered: bool = False
    blocks: list[ListItemContent] = Field(default_factory=list)


class ListBlock(_BlockBase):
    """List container — the engine's atomic unit (a list is never split).

    The adapter collects items scattered in the source IR (blocks sharing
    the same `list_id` in first_parse) into one container; `id` uses the
    source list id. `heading_path`/page range are derived from the items
    and written on the container.
    """

    kind: Literal["list"] = "list"
    items: list[ListItem] = Field(default_factory=list)


AnyBlock = Annotated[
    Heading | Paragraph | ListBlock | Table | Image | Code,
    Field(discriminator="kind"),
]


class Document(BaseModel):
    """The document the engine sees: identity + ordered block list.

    `source_path`/`fmt` are passed through to chunks for citation;
    `metadata` is a free-form field for extras the adapter has that
    don't concern the engine (the engine never reads it).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source_path: str
    fmt: str
    page_count: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[AnyBlock] = Field(default_factory=list)

    def block_by_id(self, block_id: str) -> AnyBlock | None:
        return next((b for b in self.blocks if b.id == block_id), None)