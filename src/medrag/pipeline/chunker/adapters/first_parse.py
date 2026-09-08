"""first_parse IR v9 → inner document model adapter.

The source contract is `medrag.pipeline.parser.parsers.base`'s
`ParsedDocument` IR, imported DIRECTLY (it's a real package now; the
earlier vendor-copied IR module and the `_parser_path.ensure_parser_on_path`
sys.path bridge are gone). The IR's read/validation code — version gate
(`migrate_dict`), boundary-raising deserialization (`IRParseError`), and
cell flattening (`Cell.plain_text()`, nested tables included) — is
intentionally imported from there rather than copied: it lives in one
place. This dependency only lives in this file; `core/` never sees a
parser type.

Conversion decisions
--------------------
* **List rebuilding: contiguous groups**. IR lists are flat-stream metadata
  (`list_id`/`list_level`); consecutive blocks sharing the same `list_id`
  become a single `ListBlock`. If a non-list block slips between them the
  list closes at that point; the same `list_id` later opens a NEW
  container (id: `"L1"`, `"L1#2"`, ...) — reading order never breaks.
* **Item boundary is heuristic** (IR doesn't mark item boundaries):
  paragraph and heading open a new `ListItem`; table/image/code attach to
  the previous item (in source formats, `<li>` content is a continuation
  of the text); if there's no previous item they open their own. A
  heading inside a list isn't a `ListItemContent` member, so it's carried
  as a paragraph.
* **`header_rows`: IR field > config default**. IR doesn't carry header-
  row info today; the parser side agreed to add it. The adapter is
  forward-compatible: uses the IR field when present, otherwise the
  `[table_split] default_header_rows` config value (clipped to row count).
* **`anchor_id`: same forward-compatibility pattern**. Cross-reference
  resolution (`enrichment/cross_ref.py`) needs the heading's source
  anchor/bookmark id; IR doesn't carry it yet. The adapter reads it via
  `getattr(block, "anchor_id", None)` and picks it up automatically when
  the field exists; until then in-document links are NOT heuristically
  matched and are logged as unresolvable.
* **Provenance vocabulary is closed**: the parser's five labels are mapped
  to the core enum (v7 table labels included; mapping rationale in
  `_PROVENANCE`); unknown labels are rejected at the boundary with
  `AdapterError` (silently treating them as "unverified" would mean
  inventing a confidence level).
* **Text stays canonical**: IR `text` is carried as-is; inline marks
  (bold/super/subscript...) are NOT rendered into chunk text. From runs
  only `link` is taken (cross-ref raw material): consecutive runs pointing
  at the same target merge into one `LinkRef`; a link with empty text is
  dropped.
* **OCR text goes through a confidence filter**: an image marked
  `ocr_meaningful is False` does NOT carry `ocr_text` (the parser's
  "not to be indexed" decision); None (not processed) or True passes
  through as-is.
* **Document metadata isn't lost**: `parsed.metadata` (e.g. PDF `toc`)
  passes through; `raw_sha256`/`mimetype`/`access_level`/
  `parser_version`/`ir_version` are added to `Document.metadata` unless
  there's a collision (the engine doesn't read it — pass-through for
  citation/tracing).
"""

from __future__ import annotations

from pathlib import Path

import medrag.pipeline.parser.parsers.base as ir
from medrag.pipeline.chunker.config import ChunkerConfig, load_config
from medrag.pipeline.chunker.core.document import (
    Code,
    Document,
    Heading,
    Image,
    LinkRef,
    ListBlock,
    ListItem,
    Paragraph,
    Provenance,
    Table,
)


class AdapterError(ValueError):
    """IR is valid but the adapter cannot convert a value in it (e.g. an
    unknown provenance label or an unknown block type). Says which block
    it's in."""


# Parser provenance labels → core enum. The dict is closed: when IR
# introduces a new label, an explicit map must be added here (no silent
# assumption).
_PROVENANCE = {
    "text-layer-verified": Provenance.VERIFIED,
    "consensus-verified": Provenance.CONSENSUS,
    "unverified": Provenance.UNVERIFIED,
    # PDF table paths (IR v7, captured against the real corpus):
    # "table-bands" is fully deterministic — grid from page vector
    # geometry, cell text from the PDF's own characters → VERIFIED.
    # "table-structure-model" also has deterministic cell text but the
    # row/column structure is visual-model inference → CONSENSUS (a
    # structure error is exactly what chunking cares about).
    "table-bands": Provenance.VERIFIED,
    "table-structure-model": Provenance.CONSENSUS,
}


# ---------------------------------------------------------------------------
# Field converters
# ---------------------------------------------------------------------------


def _provenance(block: ir.Block) -> Provenance | None:
    if block.provenance is None:
        return None
    try:
        return _PROVENANCE[block.provenance]
    except KeyError:
        raise AdapterError(
            f"{block.id}: unknown provenance label {block.provenance!r}; "
            f"known: {sorted(_PROVENANCE)}") from None


def _pages(span: ir.Span) -> tuple[int | None, int | None]:
    """Normalize IR's single-page (`page`) / range (`page_start`/`page_end`)
    split; a missing end equals the other end."""
    if span.page_start is not None or span.page_end is not None:
        start = span.page_start if span.page_start is not None else span.page_end
        end = span.page_end if span.page_end is not None else span.page_start
        return start, end
    if span.page is not None:
        return span.page, span.page
    return None, None


def _links(runs: list[ir.InlineRun]) -> list[LinkRef]:
    """Collect hyperlinks from runs. Consecutive runs pointing at the same
    target (mark change inside a link) merge into one LinkRef."""
    merged: list[LinkRef] = []
    prev_link: str | None = None
    for run in runs:
        if run.link:
            if run.link == prev_link and merged:
                merged[-1] = LinkRef(text=merged[-1].text + run.text,
                                     target=run.link)
            else:
                merged.append(LinkRef(text=run.text, target=run.link))
        prev_link = run.link
    return [LinkRef(text=text, target=l.target)
            for l in merged if (text := l.text.strip())]


def _base_fields(block: ir.Block) -> dict:
    page_start, page_end = _pages(block.span)
    return {
        "id": block.id,
        "heading_path": list(block.heading_path or []),
        "page_start": page_start,
        "page_end": page_end,
        "provenance": _provenance(block),
        "links": _links(getattr(block, "runs", None) or []),
    }


# ---------------------------------------------------------------------------
# Block converters
# ---------------------------------------------------------------------------


def _table(block: ir.TableBlock, default_header_rows: int) -> Table:
    cells = [["" if cell is None else cell.plain_text() for cell in row]
             for row in block.table.cells]
    # Forward compatibility: when the parser adds a header_rows field to
    # its IR, the adapter uses it unchanged; today the field is absent →
    # config default (clipped to row count).
    header_rows = getattr(block, "header_rows", None)
    if header_rows is None:
        header_rows = default_header_rows
    return Table(
        **_base_fields(block),
        cells=cells,
        header_rows=min(header_rows, len(cells)),
        description=block.description,
        facts=list(block.facts or []),
        source_crop=getattr(block, "source_crop", None),
        excluded_at_parse=getattr(block, "excluded_at_parse", False),
    )


def _image(block: ir.ImageBlock) -> Image:
    ocr_text = block.ocr_text if block.ocr_meaningful is not False else None
    return Image(
        **_base_fields(block),
        image_id=block.image_id,
        ocr_text=ocr_text,
        alt_text=block.alt_text,
        visual_type=getattr(block, "visual_type", None),
        source_crop=getattr(block, "source_crop", None),
        excluded_at_parse=getattr(block, "excluded_at_parse", False),
        description=getattr(block, "description", None),
    )


def _convert(block: ir.Block, default_header_rows: int):
    """Single IR block → inner-model block (list membership is ignored
    here; `_build_list` builds them)."""
    if isinstance(block, ir.HeadingBlock):
        # Forward compatibility (same pattern as header_rows): when IR
        # adds anchor_id the adapter uses it unchanged; today the field
        # is absent → None.
        return Heading(**_base_fields(block), text=block.text, level=block.level,
                       anchor_id=getattr(block, "anchor_id", None))
    if isinstance(block, ir.ParagraphBlock):
        return Paragraph(**_base_fields(block), text=block.text)
    if isinstance(block, ir.CodeBlock):
        return Code(**_base_fields(block), text=block.text,
                    language=block.language)
    if isinstance(block, ir.TableBlock):
        return _table(block, default_header_rows)
    if isinstance(block, ir.ImageBlock):
        return _image(block)
    raise AdapterError(
        f"{block.id}: block type {type(block).__name__} unknown to the adapter")


def _item_content(block: ir.Block, default_header_rows: int):
    """List item content: a heading that isn't a `ListItemContent` member
    falls back to paragraph (text isn't lost; the level meaning is
    anyway lost outside the container)."""
    if isinstance(block, ir.HeadingBlock):
        return Paragraph(**_base_fields(block), text=block.text)
    return _convert(block, default_header_rows)


def _build_list(group: list[ir.Block], container_id: str,
                default_header_rows: int) -> ListBlock:
    """Build one one ListBlock from the contiguous block group sharing a
    `list_id`.

    Item boundary heuristic (module docstring): paragraph/heading opens
    a new item; other blocks attach to the previous item. The container's
    heading_path comes from the first block, the page range is the
    covering range of the items, provenance is the lowest of the items;
    links live on the item blocks (NOT copied to the container).
    """
    items: list[ListItem] = []
    for block in group:
        content = _item_content(block, default_header_rows)
        starts_item = isinstance(block, (ir.ParagraphBlock, ir.HeadingBlock))
        if starts_item or not items:
            items.append(ListItem(
                level=block.list_level or 0,
                ordered=bool(block.list_ordered),
                blocks=[content],
            ))
        else:
            items[-1].blocks.append(content)

    page_starts = [b.page_start for i in items for b in i.blocks
                   if b.page_start is not None]
    page_ends = [b.page_end for i in items for b in i.blocks
                 if b.page_end is not None]
    return ListBlock(
        id=container_id,
        heading_path=list(group[0].heading_path or []),
        page_start=min(page_starts) if page_starts else None,
        page_end=max(page_ends) if page_ends else None,
        provenance=Provenance.lowest(
            b.provenance for i in items for b in i.blocks),
        items=items,
    )


# ---------------------------------------------------------------------------
# Document converter
# ---------------------------------------------------------------------------


def adapt(parsed: ir.ParsedDocument, *,
          config: ChunkerConfig | None = None) -> Document:
    """`ParsedDocument` (IR v5) → inner `Document`. If `config` is not
    given, the default is loaded (only `table_split.default_header_rows`
    is read)."""
    default_header_rows = (config or load_config()).table_split.default_header_rows

    blocks = []
    seen_lists: dict[str, int] = {}
    i = 0
    while i < len(parsed.blocks):
        block = parsed.blocks[i]
        if block.list_id is None:
            blocks.append(_convert(block, default_header_rows))
            i += 1
            continue
        j = i
        while j < len(parsed.blocks) and parsed.blocks[j].list_id == block.list_id:
            j += 1
        count = seen_lists.get(block.list_id, 0) + 1
        seen_lists[block.list_id] = count
        container_id = block.list_id if count == 1 else f"{block.list_id}#{count}"
        blocks.append(_build_list(parsed.blocks[i:j], container_id,
                                  default_header_rows))
        i = j

    metadata = dict(parsed.metadata)
    for key in ("raw_sha256", "mimetype", "access_level",
                "parser_version", "ir_version"):
        value = getattr(parsed, key)
        if value is not None:
            metadata.setdefault(key, value)

    return Document(
        doc_id=parsed.doc_id,
        source_path=parsed.source_path,
        fmt=parsed.fmt,
        page_count=parsed.page_count,
        metadata=metadata,
        blocks=blocks,
    )


def adapt_json(text: str, *, config: ChunkerConfig | None = None) -> Document:
    """IR JSON text → inner `Document` (version gate + boundary validation
    live in `ParsedDocument.from_json`: v>5 files and malformed fields
    raise there)."""
    return adapt(ir.ParsedDocument.from_json(text), config=config)


def adapt_dict(data: dict, *, config: ChunkerConfig | None = None) -> Document:
    """Already-parsed IR dict → inner `Document`. The CLI loader reads the
    JSON once during discovery; here we don't re-serialize/parse but the
    version gate (`migrate_dict`) and boundary validation (`from_dict`)
    still run fully."""
    return adapt(ir.ParsedDocument.from_dict(ir.migrate_dict(data)), config=config)


def adapt_file(path: str | Path, *,
               config: ChunkerConfig | None = None) -> Document:
    """An IR file like `storage/output/*.json` → inner `Document`."""
    return adapt(ir.ParsedDocument.load(path), config=config)