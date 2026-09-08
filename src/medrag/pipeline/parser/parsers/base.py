"""BaseParser interface and ParsedDocument IR definition.

This module is the parser's single "contract" file: every format parser produces
a `ParsedDocument` (IR), and all later stages (images/, tables/, webapp/) read and
enrich this IR. The IR is stored as JSON under `storage/output/*.json`.

Design decisions
----------------
* Raw image bytes are NOT embedded in the IR; an image is referenced only by its
  `image_id`. The blob itself lives under `storage/images/`.
* Hash values in IR fields are written `sha256:<hex>` (`raw_sha256`, `image_id`,
  `source_crop`) — one self-describing format everywhere, see `sha256_id`
  below. On disk the blob store names files with the BARE
  hex (`:` is not a legal Windows filename character), so a field value maps to
  its file through `hash_hex`.
* Block ids (`b0`, `h1`, the `list_id` groups) are RUN-SCOPED: they are only
  meaningful inside the IR file they were emitted with, and carry no promise of
  stability across re-parses of the same source (a PDF's
  block count can genuinely differ between runs, since page reading is
  VLM-assisted). Anything referencing a block id (a chunk's `source_block_ids`,
  a cross-ref's `source_block_id`) must have been produced from THAT IR; chunk
  provenance (`raw_sha256` + `parser_version`) is what proves the pairing. A
  stable, content-addressed identity exists only where it is genuinely needed:
  `image_id` (the crop's own sha256), which is why the classification/description
  caches key on it and survive re-parses.
* Every block carries a `Span`: the byte range in the original file is preserved
  (markitdown was dropped because it lost this). For binary formats (docx/pptx/xlsx
  are zips) the offset may point into an internal stream; which stream is indicated
  by `Span.part`.
* Images are kept as block-level `ImageBlock` entries rather than inline strings, so
  the "text referring to an image -> image" link can be resolved via `Block.id`
  (the webapp's navigation requirement). At parse time `image_id`/`ocr_text`/
  `ocr_meaningful` are None; the images/ stage fills them.
* Tables are fully structured JSON including merges. A table's `description` is kept
  in the IR (on TableBlock); the LLM-check status/retry state likewise lives in the
  IR — there is no separate database.
* A "describable block" is any structural/visual block a natural-language description
  can be produced for — a table, a chart, a block diagram, a technical drawing, a
  flowchart. A table is NOT a category of its own here: it is the same kind of thing,
  only more deterministic (it has real cell data, so its description can be produced
  from the cells instead of from pixels). That shared nature is modeled by
  `DescribableBlock` (below): TableBlock and ImageBlock stay distinct block types
  because a table's `cells` grid is a genuine structural difference, but they carry
  the SAME description fields with the SAME names, and `describe/` fills them through
  one shared pass whose only per-type difference is the strategy used
  (`description_source` records which one ran).
* Lists are NOT a container block. The IR stays a flat block stream; list membership
  is metadata on ordinary blocks (`list_id`/`list_level`/`list_ordered`). This mirrors
  how docx (w:numId/w:ilvl) and pdf natively store lists, and keeps block-level content
  inside a list item (a table/image in a `<li>`) as a real TableBlock/ImageBlock instead
  of flattening it away. A "list" is reconstructed by grouping blocks that share a
  `list_id`.
* Provenance is additive metadata on any block. Deterministic parsers leave it
  None; the pdf parser labels every text-bearing block with how its content was
  obtained/confirmed: "text-layer-verified" (matched the PDF's own text layer),
  "consensus-verified" (two independent readers agreed), or "unverified"
  (could not be confirmed — `source_crop` then holds the sha256 of a page/region
  render in the blob store so the content can be audited). The citation pipeline
  reads its trust level from this tag.
* Inline formatting is additive, not a new tree. A text block keeps its plain `text`
  (the canonical view every consumer already uses) and gains an OPTIONAL `runs` list
  of `InlineRun`s carrying semantic marks (bold/italic/underline/strike/super/sub);
  `"".join(r.text for r in runs) == text`. Unformatted blocks store no `runs` (no JSON
  bloat). Only semantic marks are modeled — super/subscript because flattening them
  corrupts meaning (x² vs x2). Populated by docx; other parsers pending.
* `heading_path` gives every block its section breadcrumb (ancestor heading texts,
  outermost first) without requiring consumers to walk the flat block stream looking
  for the nearest preceding heading at each level. A parser builds this with a
  `HeadingStack` (below) as it emits blocks in reading order. None where a parser
  hasn't been wired to populate it yet, or for blocks before the first heading.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# IR schema version. Bump when the contract changes; readers migrate accordingly.
# v2: a table cell is no longer a bare string but a `Cell` (list of blocks), so a
# cell can hold nested tables/paragraphs losslessly. Readers still accept v1 (a
# cell serialized as a plain string) — see `_cell_from_dict`.
# v3: `Block.heading_path` added (ancestor heading texts). Absent/None on a block
# reads the same as "not populated by this parser yet" — no migration needed.
# v4: `CodeBlock` (BlockType.CODE) added — verbatim code with an optional language
# hint, instead of code degrading to a ParagraphBlock. `InlineRun.link` added — an
# optional hyperlink target on a run; absent reads as "no link". Both are additive
# for existing data (no v1-v3 file contains them); a reader older than v4 will not
# recognize the "code" block type, so only forward migration is a concern.
# v5: `TableBlock.table_confidence` / `table_flags` added — how well the parser
# pinned a table's grid (0..1) and notes on what was uncertain ("sparse-col0",
# "cols-from-gaps", ...). Additive/optional; absent reads as "not scored".
# v6: `TableData.header_rows` added — how many leading rows are header rows.
# None = unknown (source carried no header semantics), 0 = known headerless.
# Deterministic when the source states it (html <th>/<thead>, markdown's pipe
# header, docx w:tblHeader, pdf band header recovery); an LLM-inferred value is
# marked with a "header-llm" entry in `TableBlock.table_flags`. Additive/optional.
# v7: `HeadingBlock.anchor_id` added — the heading's own anchor/bookmark id in
# the source format (html `id` attribute, markdown explicit `{#id}` or its
# implicit GFM slug). Additive/optional; absent reads as "not populated by this
# parser" (docx/pdf/pptx don't fill it yet).
# v8: `ImageBlock.visual_type`/`visual_type_source`/`excluded_at_parse`/
# `chart_description` added — every visual region (image, geometric table
# candidate, native chart/SmartArt) can now carry a taxonomy label (table,
# chart, block_diagram, technical_drawing, flowchart, product_photo,
# decorative, unknown) set by classification (VLM), structural extraction
# (chart XML), or a firm heuristic (SmartArt is unambiguously a diagram from
# its own XML element). `excluded_at_parse` marks a block whose content was
# deliberately never generated (config-driven exclusion), so a consumer can
# tell that apart from "included but genuinely empty". `chart_description` is
# a deterministic natural-language rendering of a native chart's own data
# (no model call). All additive/optional; absent reads as "not classified".
# v9: the type-specific description fields are unified into one shared set on
# `DescribableBlock` (see module docstring, "describable block"):
# `TableBlock.table_description`/`table_facts` and `ImageBlock.chart_description`
# are GONE, replaced by `description`/`facts` plus `description_source` (which
# strategy produced it: "structural" = read from the element's own XML data with
# no model call, "llm-cells" = written from the table's cell text, "vlm-vision" =
# written from the region's pixels plus surrounding text context).
# `describe_status`/`describe_attempts` move from TableBlock onto the shared base
# unchanged, and `excluded_at_parse` moves from ImageBlock onto it likewise —
# "content was deliberately not generated" is not an image-only state, and a
# table excluded via `[visual] exclude_types` needs to say so the same way.
# This is the FIRST non-additive change: a v8 file is rewritten by
# `_migrate_v8_to_v9` (table_description -> description + source "llm-cells";
# chart_description -> description + source "structural"), nested cell blocks
# included, so no reader outside this module ever sees the old names.
# v10: `ImageBlock.visual_type_confidence` added — how confidently the classifier
# picked `visual_type` (0..1), the number the classify call already returned but
# the block used to drop. Purely additive/optional (absent reads as "no
# confidence recorded"), so no rewriting migration is needed; the bump only makes
# a re-saved file declare v10 and its presence explicit. Downstream signal for a
# use/skip filter on low-confidence visual types (see the field's docstring).
IR_VERSION = 10

# The ONE parser version. Bumped on every
# behavior-changing parser fix; the parse pipeline re-parses any document whose
# registry record carries an older value, and chunk provenance records which
# value produced the IR a chunk set was built from. Lives here -- next to
# the IR contract it versions -- and is imported by the pipeline.
# 1.4.0: the `sha256:<hex>` IR hash format + this version unification itself --
# IR output changed, full re-parse required.
# 1.5.0: anti-hallucination wave -- IR v10 adds
# `ImageBlock.visual_type_confidence`; describe/visual + classify prompts
# changed (their `_PROMPT_VERSION`s bumped); confidence/size gates and an
# opt-in VLM verify+retry added. IR output changed, full re-parse required.
PARSER_VERSION = "1.5.0"


def sha256_id(data: bytes) -> str:
    """The one way a content hash is written into any JSON/IR field
    `sha256:<hex>`. Disk artifacts (blob store,
    label cache) keep the bare hex as their filename -- `:` is not a legal
    filename character on Windows -- so a field value maps to its file via
    `hash_hex`."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_hex(value: str) -> str:
    """Bare hex of a hash value (`sha256:<hex>` -> `<hex>`), for
    resolving disk filenames. A legacy bare-hex value passes through, so
    readers stay tolerant of pre-1.4.0 IR."""
    return value.split(":", 1)[1] if ":" in value else value


class IRParseError(ValueError):
    """A serialized IR dict is malformed at a known location.

    `path` points at the offending spot (e.g. ``blocks[3].type``) so a corrupt
    parser output fails loudly *at the boundary* — naming where — instead of
    letting a missing field leak inward as a silent None and surface far away in
    a later stage. Optional fields that a parser simply hasn't populated stay
    None and are NOT errors; only genuinely required fields raise this.
    """

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _walk_block_dicts(blocks: Any) -> Any:
    """Yield every serialized block dict in `blocks`, INCLUDING blocks nested
    inside table cells (a cell body is a block list like any other, so a
    migration that skipped them would leave old field names alive one level
    down). Tolerant of a v1 cell (a bare string) and of malformed entries —
    validation is `from_dict`'s job, not a migration's."""
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        yield block
        table = block.get("table")
        if not isinstance(table, dict):
            continue
        for row in table.get("cells") or []:
            if not isinstance(row, list):
                continue
            for cell in row:
                if isinstance(cell, dict):
                    yield from _walk_block_dicts(cell.get("blocks"))


def _migrate_v8_to_v9(data: dict[str, Any]) -> dict[str, Any]:
    """v8 -> v9: fold the type-specific description fields into the shared
    `description`/`facts`/`description_source` set (see the IR_VERSION v9 note).

    `description_source` is inferred from WHICH old field held the text, since
    each old name implied exactly one producer: `table_description` was always
    written from the table's cells by an LLM ("llm-cells"), `chart_description`
    always read straight from a native chart's own XML ("structural").
    """
    for block in _walk_block_dicts(data.get("blocks")):
        btype = block.get("type")
        if btype == "table":
            description = block.pop("table_description", None)
            facts = block.pop("table_facts", None)
            if facts is not None:
                block["facts"] = facts
        elif btype == "image":
            description = block.pop("chart_description", None)
        else:
            continue
        if description is not None:
            block["description"] = description
            block["description_source"] = ("llm-cells" if btype == "table"
                                           else "structural")
    return data


def migrate_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a serialized IR dict from its stored `ir_version` up to the current
    `IR_VERSION` — a pure dict->dict step run BEFORE validation/typing, so version
    handling lives in one place rather than smeared through each `from_dict`.

    Every v1..v8 difference is additive or read tolerantly at parse time (a v1
    table cell is a bare string; see `_cell_from_dict`); v9 is the first version
    that renames fields, so it is also the first real rewriting step here. It
    never accepts a file newer than this reader.
    """
    if not isinstance(data, dict):
        raise IRParseError("<root>", f"expected an object, got {type(data).__name__}")
    v = data.get("ir_version", IR_VERSION)
    if not isinstance(v, int):
        raise IRParseError("ir_version", f"expected an int, got {v!r}")
    if v > IR_VERSION:
        raise IRParseError(
            "ir_version",
            f"file is IR v{v}, this reader supports up to v{IR_VERSION}")
    # Breaking migrations plug in here, oldest first:
    if v < 9:
        data = _migrate_v8_to_v9(data)
    # Stamp the version the data now conforms to. Required from v9 on: a
    # migration that RENAMES fields makes the stored number a lie the moment it
    # runs, and `ParsedDocument.from_dict` carries `ir_version` through to
    # `to_dict` -- so a migrated-then-resaved file would otherwise claim to be
    # the old version while holding the new field names.
    data["ir_version"] = IR_VERSION
    return data


class BlockType(str, Enum):
    """IR block types. `str`-based so it serializes to a plain string in JSON."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


# ---------------------------------------------------------------------------
# Location / source tracking
# ---------------------------------------------------------------------------


@dataclass
class Span:
    """A block's location in the original source.

    All fields are optional; if a format cannot provide a value it stays None.

    byte_start/byte_end
        Half-open byte range [start, end) in the source. For text formats
        (md/html) this points directly into the raw file. For binary formats it,
        together with `part`, denotes the offset within that internal stream.
    part
        Which stream the offsets address. None => the raw input file itself.
        E.g. "word/document.xml" for docx.
    page
        1-based page/slide number (pdf page, pptx slide). `page` is used for
        single-page ranges; blocks spanning multiple pages fill page_start/page_end.
    """

    byte_start: int | None = None
    byte_end: int | None = None
    part: str | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Span:
        if not data:
            return cls()
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Table structure
# ---------------------------------------------------------------------------


@dataclass
class Merge:
    """A merged cell region. The top-left cell (row, col) carries the content;
    the other covered cells are None in `cells`."""

    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"row": self.row, "col": self.col,
                "rowspan": self.rowspan, "colspan": self.colspan}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Merge:
        return cls(row=data["row"], col=data["col"],
                   rowspan=data.get("rowspan", 1), colspan=data.get("colspan", 1))


@dataclass
class TableData:
    """Structured table content.

    cells
        Row-major 2D matrix; each entry is a `Cell` (its content is a list of
        blocks) or None for a slot covered by a merge. `n_rows`/`n_cols` are the
        logical dimensions. A cell is block-structured rather than plain text so a
        table nested inside a cell is preserved losslessly; use `Cell.plain_text()`
        for a flat text view.
    merges
        Merge regions; the top-left cell carries the content.
    header_rows
        How many leading rows are header rows. None = unknown (the source
        format carried no header semantics, or the parser hasn't determined
        it); 0 = the table is known to have no header row. Consumers that
        must pick a header treat None as "assume the first row" (the
        pre-header_rows behavior).
    """

    n_rows: int
    n_cols: int
    cells: list[list[Cell | None]] = field(default_factory=list)
    merges: list[Merge] = field(default_factory=list)
    header_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "cells": [[c.to_dict() if c is not None else None for c in row]
                      for row in self.cells],
            "merges": [m.to_dict() for m in self.merges],
        }
        if self.header_rows is not None:
            d["header_rows"] = self.header_rows
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableData:
        return cls(
            n_rows=data["n_rows"],
            n_cols=data["n_cols"],
            cells=[[_cell_from_dict(c) for c in row]
                   for row in data.get("cells", [])],
            merges=[Merge.from_dict(m) for m in data.get("merges", [])],
            header_rows=data.get("header_rows"),
        )


# ---------------------------------------------------------------------------
# Inline text formatting
# ---------------------------------------------------------------------------


class Mark(str, Enum):
    """An inline character style. `str`-based so it serializes to a plain string.

    Only *semantic* styles are modeled — ones that can change meaning, not pure
    cosmetics (color, font, highlight). SUPERSCRIPT/SUBSCRIPT matter because
    flattening them corrupts text (x2 vs x², H2O vs H₂O).
    """

    BOLD = "bold"
    ITALIC = "italic"
    UNDERLINE = "underline"
    STRIKE = "strike"
    SUPERSCRIPT = "superscript"
    SUBSCRIPT = "subscript"


@dataclass
class InlineRun:
    """A maximal text span sharing the same set of active marks (and link).

    A text block keeps its plain `text` as the canonical view (unchanged for every
    text-only consumer); `runs` is an *optional* parallel detail whose concatenated
    text equals `text`. A run with no marks and no link is just unstyled text.

    link
        Optional hyperlink target the run's `text` points at (e.g. an `<a href>`
        or a markdown `[text](url)`). None => the run is not a link. The visible
        text stays in `text`, so text-only consumers are unaffected; only the
        destination is carried here.
    """

    text: str
    marks: tuple[Mark, ...] = ()
    link: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text, "marks": [m.value for m in self.marks]}
        if self.link:
            d["link"] = self.link
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InlineRun:
        return cls(text=data.get("text", ""),
                   marks=tuple(Mark(m) for m in data.get("marks", [])),
                   link=data.get("link"))


def runs_have_marks(runs: list[InlineRun]) -> bool:
    """True if any run carries non-redundant detail — a mark or a link (else
    `runs` is fully reconstructable from `text` and not worth storing)."""
    return any(r.marks or r.link for r in runs)


def finalize_runs(runs: list[InlineRun]) -> list[InlineRun]:
    """Merge adjacent runs with identical marks *and* link, drop empties, and strip
    the leading/trailing whitespace of the whole sequence so that
    `"".join(r.text for r in result)` equals the block's stripped plain text."""
    merged: list[InlineRun] = []
    for r in runs:
        if not r.text:
            continue
        if merged and merged[-1].marks == r.marks and merged[-1].link == r.link:
            merged[-1] = InlineRun(merged[-1].text + r.text, r.marks, r.link)
        else:
            merged.append(InlineRun(r.text, r.marks, r.link))
    if merged:
        merged[0] = InlineRun(merged[0].text.lstrip(), merged[0].marks, merged[0].link)
        merged[-1] = InlineRun(merged[-1].text.rstrip(), merged[-1].marks, merged[-1].link)
        merged = [r for r in merged if r.text]
    return merged


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """Common base for all blocks.

    id
        Stable block identifier within the document (e.g. "b0", "b1"). Used to
        establish links (webapp, chunker). The parser assigns these in order.
    span
        Location in the original source.

    List membership (see module docstring, "Lists"): a list is not a container
    block. Any block may be a list item; blocks sharing a `list_id` form one list.
    All three fields are None for blocks that are not part of a list.

    list_id
        Groups blocks belonging to the same list (docx w:numId analogue).
    list_level
        0-based nesting depth of the item (docx w:ilvl analogue).
    list_ordered
        True => numbered item, False => bullet. May vary per item within a list.

    heading_path
        Ancestor heading texts, outermost first (e.g. ["MIL-STD-1553 Bus
        Couplers", "Product Overview"]) — see module docstring, "heading_path".
        None if this parser doesn't populate it, or the block precedes any
        heading.

    Provenance (see module docstring): both fields are None for deterministic
    formats; the pdf parser fills them.

    provenance
        "text-layer-verified" | "consensus-verified" | "unverified".
    source_crop
        sha256 of a page/region render in the blob store (storage/images/)
        backing an unverified block, for human audit.
    """

    id: str
    span: Span = field(default_factory=Span)

    list_id: str | None = None
    list_level: int | None = None
    list_ordered: bool | None = None

    heading_path: list[str] | None = None

    provenance: str | None = None
    source_crop: str | None = None

    # Subclasses override this.
    type: BlockType = field(init=False, default=BlockType.PARAGRAPH)

    def _base_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value, "id": self.id}
        span = self.span.to_dict()
        if span:
            d["span"] = span
        for key in ("list_id", "list_level", "list_ordered",
                    "heading_path", "provenance", "source_crop"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    @staticmethod
    def _base_kwargs(data: dict[str, Any]) -> dict[str, Any]:
        """Shared base fields for subclass deserialization."""
        return {
            "id": data["id"],
            "span": Span.from_dict(data.get("span")),
            "list_id": data.get("list_id"),
            "list_level": data.get("list_level"),
            "list_ordered": data.get("list_ordered"),
            "heading_path": data.get("heading_path"),
            "provenance": data.get("provenance"),
            "source_crop": data.get("source_crop"),
        }

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict[str, Any], path: str = "block") -> Block:
        if not isinstance(data, dict):
            raise IRParseError(path, f"expected an object, got {type(data).__name__}")
        if "type" not in data:
            raise IRParseError(path, "required field 'type' missing")
        try:
            btype = BlockType(data["type"])
        except ValueError as e:
            raise IRParseError(f"{path}.type",
                               f"unknown block type {data['type']!r}") from e
        cls = _BLOCK_CLASSES[btype]
        try:
            return cls._from_dict(data)
        except IRParseError:
            raise
        except (KeyError, ValueError, TypeError) as e:
            # a required field was missing/ill-typed inside this block
            reason = (f"required field {e.args[0]!r} missing"
                      if isinstance(e, KeyError) else str(e))
            raise IRParseError(path, reason) from e

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Block:  # pragma: no cover
        raise NotImplementedError


@dataclass
class HeadingBlock(Block):
    text: str = ""
    level: int = 1  # 1 = top-level heading
    # Optional inline formatting detail; empty => no formatting captured/present.
    runs: list[InlineRun] = field(default_factory=list)
    # The heading's own anchor/bookmark id in the source format (html `id`
    # attribute, markdown explicit `{#id}` or its implicit GFM slug) — what a
    # same-document hyperlink's fragment (`InlineRun.link`, e.g. "#setup")
    # would need to match to resolve to this heading. None if the source had
    # none, or this parser doesn't populate it yet (docx/pdf/pptx).
    anchor_id: str | None = None

    type: BlockType = field(init=False, default=BlockType.HEADING)

    def to_dict(self) -> dict[str, Any]:
        d = {**self._base_dict(), "text": self.text, "level": self.level}
        if self.runs:
            d["runs"] = [r.to_dict() for r in self.runs]
        if self.anchor_id:
            d["anchor_id"] = self.anchor_id
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> HeadingBlock:
        return cls(**cls._base_kwargs(data),
                   text=data.get("text", ""), level=data.get("level", 1),
                   runs=[InlineRun.from_dict(r) for r in data.get("runs", [])],
                   anchor_id=data.get("anchor_id"))


class HeadingStack:
    """Tracks the current ancestor-heading breadcrumb while a parser emits
    blocks in reading order, for populating `Block.heading_path`.

    Call `enter(level, text)` for every heading as it's emitted (this also
    returns that heading's own `heading_path`, i.e. its ancestors), and read
    `path()` for every other block. A level-N heading closes any open heading
    at level >= N, mirroring how heading nesting works in every format here
    (docx outline levels, pdf font-size ranks, pptx placeholder levels, ...).
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def enter(self, level: int, text: str) -> list[str] | None:
        # Pop same-or-deeper headings BEFORE snapshotting: a closed sibling
        # (e.g. "1.1" when "1.2" arrives) is not an ancestor of this heading.
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        path = self.path()
        self._stack.append((level, text))
        return path

    def path(self) -> list[str] | None:
        return [text for _, text in self._stack] if self._stack else None


@dataclass
class ParagraphBlock(Block):
    text: str = ""
    # Optional inline formatting detail; empty => no formatting captured/present.
    runs: list[InlineRun] = field(default_factory=list)

    type: BlockType = field(init=False, default=BlockType.PARAGRAPH)

    def to_dict(self) -> dict[str, Any]:
        d = {**self._base_dict(), "text": self.text}
        if self.runs:
            d["runs"] = [r.to_dict() for r in self.runs]
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ParagraphBlock:
        return cls(**cls._base_kwargs(data), text=data.get("text", ""),
                   runs=[InlineRun.from_dict(r) for r in data.get("runs", [])])


@dataclass
class CodeBlock(Block):
    """A verbatim code block (fenced code, `<pre>`/`<code>`, ...).

    `text` is the raw code exactly as written — it is NEVER run through inline
    formatting (`runs`), because inside code `*` / `_` / `#` are literal characters,
    not markup. `language` is an optional hint when the source names it (a markdown
    fence info string ```` ```python ````, an html `<code class="language-x">`); None
    when unknown. A consumer renders this as a fenced block rather than a paragraph,
    which is why it is a distinct block type instead of a ParagraphBlock.
    """

    text: str = ""
    language: str | None = None

    type: BlockType = field(init=False, default=BlockType.CODE)

    def to_dict(self) -> dict[str, Any]:
        d = {**self._base_dict(), "text": self.text}
        if self.language:
            d["language"] = self.language
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> CodeBlock:
        return cls(**cls._base_kwargs(data), text=data.get("text", ""),
                   language=data.get("language"))


@dataclass
class DescribableBlock(Block):
    """Common base for blocks a natural-language description can be produced for.

    See the module docstring ("describable block"): a table and a chart/diagram
    are the same kind of thing here — a structural or visual region that can be
    restated in words — differing only in HOW reliably that restatement can be
    produced. So the fields are shared and only the producing strategy differs;
    `describe/` runs them all through one pass.

    All fields are None at parse time and filled by the `describe/` stage, with
    ONE exception: a native chart's/SmartArt's `description` is written by the
    format parser itself straight from the element's own XML (a deterministic,
    model-free read — `parsers/chart_extract.py`, `parsers/smartart_extract.py`),
    so it arrives already set and the describe pass leaves it alone.

    description
        Short plain-text summary of what this block is/shows.
    description_source
        Which strategy produced `description`:
        "structural"  — read from the element's own data (chart/SmartArt XML);
                        no model call, so it cannot be a hallucination.
        "llm-cells"   — an LLM wrote it from the table's real cell text.
        "vlm-vision"  — a VLM wrote it from the region's pixels plus the
                        surrounding document text; the weakest of the three
                        (nothing deterministic to check it against).
        None => no description, or not produced yet.
    facts
        The block's content restated as self-contained declarative sentences,
        one per item the table describes, each weaving that item's whole row of
        attribute values into a single cartesian sentence ("Product X withstands
        temperatures from 20 to 50 C, weighs 10 kg and is made of aluminium.")
        for downstream retrieval/QA. JSON-only material — deliberately NOT rendered into
        markdown by `render/markdown.py`. Only produced where fidelity can
        actually be checked (today: tables, whose digits must reoccur in the
        cells — see `describe/table.py`); None everywhere else.
    describe_status / describe_attempts
        Outcome of the optional verify+retry loop (no separate DB):
        "ok" | "flagged" | "empty" (nothing describable, no model call was
        made) | None (loop not run).
    excluded_at_parse
        True when parse-time config (`[visual] exclude_types`) excluded this
        block's type, so its content was deliberately never generated -- no
        OCR, no description. Distinct from a block that was processed but
        legitimately produced nothing (a photo with nothing to transcribe), so
        a consumer can render an excluded block as a placeholder instead of
        letting it vanish. The block's SOURCE data is always kept either way
        (an image's blob, a table's cells); only generated content is skipped,
        which is what makes the exclusion reversible without a re-parse.
    """

    description: str | None = None
    description_source: str | None = None
    facts: list[str] | None = None
    describe_status: str | None = None
    describe_attempts: int | None = None
    excluded_at_parse: bool = False

    #: Serialized under these names on every describable block. Not a dataclass
    #: field (no annotation) — just the single list both halves below read.
    _DESCRIBE_FIELDS = ("description", "description_source", "facts",
                        "describe_status", "describe_attempts")

    def _describe_dict(self) -> dict[str, Any]:
        d = {key: val for key in self._DESCRIBE_FIELDS
             if (val := getattr(self, key)) is not None}
        if self.excluded_at_parse:  # False is the default -- don't serialize it
            d["excluded_at_parse"] = True
        return d

    @staticmethod
    def _describe_kwargs(data: dict[str, Any]) -> dict[str, Any]:
        kwargs = {key: data.get(key) for key in DescribableBlock._DESCRIBE_FIELDS}
        kwargs["excluded_at_parse"] = data.get("excluded_at_parse", False)
        return kwargs


@dataclass
class TableBlock(DescribableBlock):
    table: TableData = field(default_factory=lambda: TableData(0, 0))
    # How confidently the parser pinned this table's row/column grid (0..1), and
    # notes on what was uncertain. Set by the pdf parser (deterministic band
    # builder / model referee); None for parsers that don't score tables. A
    # low value or a flag like "sparse-col0" / "content-dropout" marks a grid a
    # consumer may want to treat cautiously or have a human review -- it never
    # affects cell TEXT, which is always the source's own characters.
    table_confidence: float | None = None
    table_flags: list[str] | None = None

    type: BlockType = field(init=False, default=BlockType.TABLE)

    def to_dict(self) -> dict[str, Any]:
        d = {**self._base_dict(), "table": self.table.to_dict(),
             **self._describe_dict()}
        for key in ("table_confidence", "table_flags"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> TableBlock:
        return cls(**cls._base_kwargs(data), **cls._describe_kwargs(data),
                   table=TableData.from_dict(data["table"]),
                   table_confidence=data.get("table_confidence"),
                   table_flags=data.get("table_flags"))


@dataclass
class ImageBlock(DescribableBlock):
    """Block-level placeholder for an image in the document flow.

    Describable (see `DescribableBlock`): a chart/diagram/technical drawing is
    the same kind of describable region a table is, so its `description` lives
    in the same field under the same name — set either by the format parser
    itself (native chart/SmartArt, straight from XML) or by the describe pass
    (VLM reading the crop plus surrounding text).

    marker
        Sequential `<imageN>` marker within the document. `image_index` = N.
    locator
        Where the raw image lives in the source (image_handler fetches bytes with
        this). Format-specific free-form dict: e.g. {"part": "word/media/image1.png"}
        or {"page": 3, "xref": 12} or {"slide": 2, "shape": 4}.
    image_id, ocr_text, ocr_meaningful, ocr_confidence, mime, width, height
        Filled by the images/ stage; None at parse time. `ocr_meaningful is None`
        => not yet processed. False => blob+db record is kept but the text is not
        indexed. `ocr_text` is always OCR-derived (a VLM reading pixels, with no
        text layer to verify against) and therefore inherently unreliable;
        `ocr_confidence` (0.0-1.0, or None when the reviewer reported none) rates
        how trustworthy it is. image_handler withholds the text when the
        confidence is below its configured threshold, so a present `ocr_text`
        cleared the bar -- but consumers should still surface it as unverified.
    visual_type, visual_type_source
        Taxonomy label for what this visual region actually is (see IR_VERSION
        v8 note above) -- "table", "chart", "block_diagram",
        "technical_drawing", "flowchart", "product_photo", "decorative", or
        "unknown". None => not classified (classification disabled, or this
        parser/format doesn't classify yet). `visual_type_source` records how
        the label was produced: "vlm" (vision classification call),
        "text-llm" (text-only classification, no crop available),
        "structural" (deterministic, e.g. read straight from a chart's own
        XML data -- no model call, most trustworthy), "heuristic" (label is
        certain from the source element type itself, e.g. SmartArt is always
        a diagram, but no content could be extracted for it), "cache" (a
        prior run's cached label was reused), or "unavailable" (no
        classifier configured/reachable; label defaults to "unknown").
    visual_type_confidence
        The classifier's self-reported confidence in `visual_type` (0..1), or
        None (no classifier ran / deterministic source / no number in the
        reply). A downstream use/skip filter signal only -- it never changes the
        label. See IR_VERSION v10.
    Also carries the shared describe fields (`description`, `facts`,
    `excluded_at_parse`, ...) -- see `DescribableBlock`.
    """

    image_index: int = 0
    locator: dict[str, Any] = field(default_factory=dict)
    alt_text: str | None = None

    # Resolution fields filled by the images/ stage:
    image_id: str | None = None
    ocr_text: str | None = None
    ocr_meaningful: bool | None = None
    ocr_confidence: float | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None

    # Visual-type classification (v8):
    visual_type: str | None = None
    visual_type_source: str | None = None
    # How confidently the classifier picked `visual_type` (0..1), as reported by
    # the model (v10). None when no classifier ran, when the source is
    # deterministic ("structural"/"heuristic" -- confidence is meaningless there),
    # or when the reply carried no usable number. Never affects the label itself;
    # it is a downstream-filter signal (a consumer may choose to drop or flag a
    # low-confidence type before trusting its description) -- the same role
    # `table_confidence` plays for a table's grid.
    visual_type_confidence: float | None = None

    type: BlockType = field(init=False, default=BlockType.IMAGE)

    @property
    def marker(self) -> str:
        return f"<image{self.image_index}>"

    def to_dict(self) -> dict[str, Any]:
        d = {**self._base_dict(), "image_index": self.image_index,
             "locator": self.locator, **self._describe_dict()}
        for key in ("alt_text", "image_id", "ocr_text", "ocr_meaningful",
                    "ocr_confidence", "mime", "width", "height",
                    "visual_type", "visual_type_source", "visual_type_confidence"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> ImageBlock:
        return cls(
            **cls._base_kwargs(data),
            **cls._describe_kwargs(data),
            image_index=data.get("image_index", 0),
            locator=data.get("locator", {}),
            alt_text=data.get("alt_text"),
            image_id=data.get("image_id"),
            ocr_text=data.get("ocr_text"),
            ocr_meaningful=data.get("ocr_meaningful"),
            ocr_confidence=data.get("ocr_confidence"),
            mime=data.get("mime"),
            width=data.get("width"),
            height=data.get("height"),
            visual_type=data.get("visual_type"),
            visual_type_source=data.get("visual_type_source"),
            visual_type_confidence=data.get("visual_type_confidence"),
        )


_BLOCK_CLASSES: dict[BlockType, type[Block]] = {
    BlockType.HEADING: HeadingBlock,
    BlockType.PARAGRAPH: ParagraphBlock,
    BlockType.CODE: CodeBlock,
    BlockType.TABLE: TableBlock,
    BlockType.IMAGE: ImageBlock,
}


# ---------------------------------------------------------------------------
# Table cell content
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    """Content of one table cell.

    A cell body is structurally the same as a document body: an ordered list of
    blocks. So a cell can legally hold a paragraph, a nested `TableBlock`, or a
    paragraph+table+paragraph sequence — nothing about a cell is "text only".
    Merge-covered slots are represented as None in `TableData.cells`, never as a
    `Cell`.

    Text-only consumers must not walk `blocks`; call `plain_text()` instead — it
    is the single place that decides how a nested table degrades to text.
    """

    blocks: list[Block] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"blocks": [b.to_dict() for b in self.blocks]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cell:
        return cls(blocks=[Block.from_dict(b) for b in data.get("blocks", [])])

    def plain_text(self) -> str:
        """Flat text view of the cell.

        Paragraph/heading blocks contribute their text; a nested table is rendered
        inline (GitHub-flavored markdown, or `Header: value` pairs when it itself
        contains a table); an image contributes its `<imageN>` marker. Blocks are
        joined with newlines. This is the ONLY answer to "what if the cell holds a
        table?" for text-only consumers.
        """
        parts: list[str] = []
        images: list[ImageBlock] = []
        for b in self.blocks:
            if isinstance(b, TableBlock):
                parts.append(_render_cell_table(b.table))
            elif isinstance(b, ImageBlock):
                images.append(b)  # deferred: its marker is usually already in text
            else:  # ParagraphBlock / HeadingBlock
                parts.append(getattr(b, "text", ""))
        text = "\n".join(p for p in parts if p)
        for img in images:  # only surface a marker the paragraph text didn't carry
            if img.marker not in text:
                text = f"{text}\n{img.marker}" if text else img.marker
        return text


def text_cell(text: str | None) -> Cell | None:
    """Wrap plain text as a `Cell`. None (merge-covered/absent) stays None; an
    empty string becomes an empty cell (no paragraph)."""
    if text is None:
        return None
    return Cell(blocks=[ParagraphBlock(id="", text=text)] if text else [])


def _cell_from_dict(data: Any) -> Cell | None:
    """Deserialize one cell slot, tolerant of the v1 schema (a bare string)."""
    if data is None:
        return None
    if isinstance(data, str):  # legacy v1: cell was a plain string
        return text_cell(data)
    return Cell.from_dict(data)


def _render_cell_table(table: TableData) -> str:
    """Inline text rendering of a table found inside a cell."""
    if _table_has_subtable(table):
        return _render_header_value(table)  # markdown can't nest a table
    return _render_gfm(table)


def _table_has_subtable(table: TableData) -> bool:
    return any(
        isinstance(b, TableBlock)
        for row in table.cells for c in row if c is not None
        for b in c.blocks
    )


def _cell_text(c: Cell | None) -> str:
    return "" if c is None else c.plain_text().replace("\n", " ").strip()


def _render_gfm(table: TableData) -> str:
    rows = table.cells
    if not rows:
        return ""
    if table.header_rows == 0:
        # Known headerless: don't promote the first data row into the GFM
        # header slot — emit an empty header line instead.
        out = ["| " + " | ".join("" for _ in rows[0]) + " |",
               "| " + " | ".join("---" for _ in rows[0]) + " |"]
        body = rows
    else:
        out = ["| " + " | ".join(_cell_text(c).replace("|", r"\|") for c in rows[0]) + " |",
               "| " + " | ".join("---" for _ in rows[0]) + " |"]
        body = rows[1:]
    for row in body:
        out.append("| " + " | ".join(_cell_text(c).replace("|", r"\|") for c in row) + " |")
    return "\n".join(out)


def _render_header_value(table: TableData) -> str:
    rows = table.cells
    if not rows:
        return ""
    if table.header_rows == 0:
        headers: list[str] = []  # known headerless -> every column is "colN"
        data_rows = rows
    else:
        headers = [_cell_text(c) for c in rows[0]]
        data_rows = rows[1:]
    groups: list[str] = []
    for row in data_rows:
        pairs = []
        for i, c in enumerate(row):
            head = headers[i] if i < len(headers) and headers[i] else f"col{i}"
            # a cell may itself contain a table -> plain_text recurses one level down
            val = "" if c is None else c.plain_text()
            pairs.append(f"{head}: {val}")
        groups.append("\n".join(pairs))
    return "\n\n".join(groups)


# ---------------------------------------------------------------------------
# Document (IR)
# ---------------------------------------------------------------------------


@dataclass
class ParsedDocument:
    """Parser output: the common intermediate representation (IR).

    doc_id
        Document identifier (assigned by the pipeline; opaque and permanent --
        the registry preserves it across re-scans).
        `storage/output/{doc_id}.json`.
    source_path
        Path/name of the raw input under `storage/raw/` (provenance).
    fmt
        Format tag: "docx" | "pptx" | "xlsx" | "html" | "pdf" | "markdown".
    raw_sha256
        Hash of the raw file, `sha256:<hex>` (document-level provenance/dedup).
    parser_version
        The PARSER_VERSION that produced this IR (see that constant).
    access_level
        RESERVED, always None: no producer populates it today and nothing
        downstream reads it (kept rather than removed
        because dropping a field would cost an IR_VERSION bump, i.e. a full
        corpus re-parse, for a purely cosmetic gain). When a real access-control
        requirement appears, its source will be the registry, decided
        separately. Reading None here means "not populated", never "public".
    blocks
        Ordered list of blocks (document reading order). Block ids are
        run-scoped -- see the module docstring's block-id note.
    """

    doc_id: str
    source_path: str
    fmt: str
    raw_sha256: str | None = None
    mimetype: str | None = None
    page_count: int | None = None
    access_level: str | None = None
    parser_version: str | None = None
    ir_version: int = IR_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: list[Block] = field(default_factory=list)

    # -- Convenience accessors ------------------------------------------------

    def images(self) -> list[ImageBlock]:
        return [b for b in self.blocks if isinstance(b, ImageBlock)]

    def tables(self) -> list[TableBlock]:
        return [b for b in self.blocks if isinstance(b, TableBlock)]

    def block_by_id(self, block_id: str) -> Block | None:
        return next((b for b in self.blocks if b.id == block_id), None)

    def list_items(self, list_id: str) -> list[Block]:
        """Blocks belonging to one list (reconstructs a list from `list_id`)."""
        return [b for b in self.blocks if b.list_id == list_id]

    # -- Serialization --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ir_version": self.ir_version,
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "fmt": self.fmt,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        for key in ("raw_sha256", "mimetype", "page_count", "access_level",
                    "parser_version"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParsedDocument:
        """Validate + type a serialized IR dict. Raises `IRParseError` (naming the
        offending path) if a required field is missing/ill-typed, rather than
        letting a None leak inward. Call `migrate_dict` first for a versioned file
        (`from_json`/`load` do this already)."""
        if not isinstance(data, dict):
            raise IRParseError("<root>",
                               f"expected an object, got {type(data).__name__}")
        for key in ("doc_id", "source_path", "fmt"):
            if key not in data:
                raise IRParseError("<root>", f"required field {key!r} missing")
        raw_blocks = data.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raise IRParseError("blocks", "expected a list")
        return cls(
            doc_id=data["doc_id"],
            source_path=data["source_path"],
            fmt=data["fmt"],
            raw_sha256=data.get("raw_sha256"),
            mimetype=data.get("mimetype"),
            page_count=data.get("page_count"),
            access_level=data.get("access_level"),
            parser_version=data.get("parser_version"),
            ir_version=data.get("ir_version", IR_VERSION),
            metadata=data.get("metadata", {}),
            blocks=[Block.from_dict(b, f"blocks[{i}]")
                    for i, b in enumerate(raw_blocks)],
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ParsedDocument:
        return cls.from_dict(migrate_dict(json.loads(text)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ParsedDocument:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Parser contract
# ---------------------------------------------------------------------------


class BaseParser(ABC):
    """Abstract interface every format parser conforms to.

    A parser has a single responsibility: convert a raw file into the
    `ParsedDocument` IR. Enrichments like OCR, table description, and LLM checks
    are the job of later stages; the parser leaves those fields None.

    Subclasses fill the `extensions`/`mimetypes` class variables and the `parse()`
    method. `registry.py` uses `supports()` for selection.
    """

    #: File extensions this parser handles (with dot, lowercase): (".docx",)
    extensions: tuple[str, ...] = ()
    #: Mimetypes handled.
    mimetypes: tuple[str, ...] = ()
    #: Format tag (ParsedDocument.fmt).
    fmt: str = ""

    @abstractmethod
    def parse(self, raw_path: str | Path, doc_id: str) -> ParsedDocument:
        """Convert the raw file at `raw_path` into IR. `doc_id` comes from the pipeline."""
        raise NotImplementedError

    @classmethod
    def supports(cls, *, extension: str | None = None,
                 mimetype: str | None = None) -> bool:
        """Is this extension or mimetype supported by this parser?"""
        if extension and extension.lower() in cls.extensions:
            return True
        return bool(mimetype and mimetype in cls.mimetypes)
