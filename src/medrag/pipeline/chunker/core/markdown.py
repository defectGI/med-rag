"""GFM Markdown rendering from the inner model — the splitting engine's
small renderer.

A leaf chunk's `text` is rendered from the inner model to GFM; the token
budget is measured from this rendered OUTPUT (measured == stored). The
parser's render module is NOT imported (core must not see parser code);
the inner model is already flattened so this job is much smaller.

Render rules
------------
* **Body text is NOT escaped.** Block text is canonical; escaping `*`/`#`
  in it would break both the token count and readability. Only structure-
  breaking cases are repaired: `|` is escaped in table cells, cell
  line breaks become `<br>`, the code fence is chosen longer than the
  longest backtick run in the text.
* **Tables**: pipe table; the full `header_rows` block is written above
  the separator line (GFM recognizes a single header row but the
  decision is "above the separator"; multi-row headers stack as-is).
  When `header_rows == 0`, GFM's mandatory header row is filled with
  empty cells — data rows are not displayed as if they were headers.
  Short rows are padded with empty cells to the widest row.
* **Lists**: 4-space indent per level (level-independent of marker width,
  valid GFM for nested levels); ordered items are numbered consecutively
  within a level; the counter resets when the level shallows or the
  marker type changes (GFM starts a new list on marker-type change). The
  item's first block sits next to the marker; remaining lines are
  indented by the marker's width.
* **Images**: no pixels; `alt_text` as a single emphasized line,
  (`[images] inject_ocr` on) `ocr_text` as plain text,
  (`[visual] inject_description` on) `description` flowing into the text
  (all three are also carried in the chunk's structured `images` list —
  that's the engine's job). If everything is empty the block contributes
  nothing to the text. **`inject_ocr` defaults to OFF**: OCR text lives
  only in the structured channel. Tagging unverified OCR alone was tried
  and wasn't enough — even tagged, OCR doesn't separate from real
  paragraph/table text in the body and pushes retrieval toward wrong
  answers. `label_unverified_ocr` is only meaningful when `inject_ocr`
  is on.
* **"Describable block" exclusion placeholder goes through ONE helper**
  (`_describable_placeholder`, below): images and tables share the
  same line — a table is not a separate branch of this category, just
  the more deterministic member (see parser IR's `DescribableBlock`).
  `excluded_at_parse` (parser produced no content) OR
  `visual_type in [visual] exclude_types` (chunk-time exclusion,
  config-only, no re-parse) → a placeholder carrying type + reference
  ("[There is an X here: ...]") is rendered instead of the content —
  excluded blocks never DISAPPEAR from the output, only their content is
  not produced/rendered. When `visual_type` is known but content is
  still empty (e.g. SmartArt: type certain, no renderable data) the
  same placeholder applies; unclassified ("unidentifiable visual
  element") gets a generic variant. The `ref` field carries ONLY a
  file-path-style reference (`source_crop`); `image_id` is
  DELIBERATELY NOT embedded into placeholder text — the id lives only
  in structured metadata (`ChunkNode.images`/`ImageRef`), not in text
  (avoids the same id living in two channels). The table's placeholder
  is produced in `core/table_split.py::compose_table_text`
  (`visual_type="table"` literal — Table has no field of its own, being
  a table IS the type) but calls the same helper.
"""

from __future__ import annotations

import re

from medrag.core.markdown import gfm_empty_header_row
from medrag.pipeline.chunker.config import ChunkerConfig
from medrag.pipeline.chunker.core.document import (
    AnyBlock,
    Code,
    Heading,
    Image,
    ListBlock,
    ListItem,
    Paragraph,
    Provenance,
    Table,
)

# Label put before unverified OCR text when `label_unverified_ocr` is on.
_OCR_UNVERIFIED_LABEL = "*(OCR, doğrulanmamış):*"

# Taxonomy label -> human-readable name (used in the placeholder text).
_TYPE_LABEL_TR = {
    "table": "tablo", "chart": "grafik", "block_diagram": "blok diyagram",
    "technical_drawing": "teknik çizim", "flowchart": "akış şeması",
    "product_photo": "ürün fotoğrafı", "decorative": "dekoratif görsel",
    "unknown": "sınıflandırılmamış görsel",
}

# GFM recognizes at most 6 heading levels; deeper is clipped to 6.
_MAX_HEADING_LEVEL = 6
# Indent per list level.
_LIST_INDENT = 4


def render_heading(block: Heading) -> str:
    level = min(block.level, _MAX_HEADING_LEVEL)
    return f"{'#' * level} {block.text.strip()}"


def render_paragraph(block: Paragraph) -> str:
    return block.text.strip()


def render_code(block: Code) -> str:
    text = block.text.rstrip("\n")
    # Pick a fence longer than the longest backtick run in the text (min 3).
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    lang = block.language or ""
    return f"{fence}{lang}\n{text}\n{fence}"


def _cell(text: str) -> str:
    """Fit a cell value into a pipe table: escape `|`, line breaks become `<br>`."""
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>").strip()


def render_table(block: Table) -> str:
    if not block.cells:
        return ""
    n_cols = block.n_cols
    rows = [[_cell(c) for c in row] + [""] * (n_cols - len(row))
            for row in block.cells]
    header_rows = min(block.header_rows, len(rows))

    def line(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    out: list[str] = []
    if header_rows == 0:
        out.append(gfm_empty_header_row(n_cols))  # GFM's mandatory header: empty cells
    else:
        out.extend(line(r) for r in rows[:header_rows])
    out.append(line(["---"] * n_cols))
    out.extend(line(r) for r in rows[header_rows:])
    return "\n".join(out)


def _describable_placeholder(*, entity: str, block_id: str,
                             visual_type: str | None, ref: str | None) -> str:
    """Single placeholder helper — images and tables produce the same line
    (see module docstring, "goes through ONE helper"). `entity` is just a
    visible label passed in by the caller (the runtime labels are
    "Görsel" for images and "Blok" for tables); the logic itself is
    type-agnostic."""
    ref = ref or "—"
    if visual_type is None:
        return f"[Burada tanımlanamayan bir görsel öğe var: {entity} {block_id}]"
    label = _TYPE_LABEL_TR.get(visual_type, visual_type)
    return f"[Burada bir {label} var: {entity} {block_id}, {ref}]"


def _image_placeholder(block: Image) -> str:
    # `image_id` is DELIBERATELY NOT passed as `ref`: the goal is "id in
    # structured channel only, not in text" — `image_id` already travels
    # via ChunkNode.images/ImageRef in structured metadata; if it leaks
    # into text as well the same id sits in two channels. `source_crop`
    # is a file-path-style reference, not an id — it may stay.
    return _describable_placeholder(
        entity="Görsel", block_id=block.id, visual_type=block.visual_type,
        ref=block.source_crop)


def render_image(block: Image, cfg: ChunkerConfig | None = None) -> str:
    excluded = block.excluded_at_parse or (
        cfg is not None and block.visual_type
        and block.visual_type in cfg.visual.exclude_types)
    if excluded:
        return _image_placeholder(block)

    parts: list[str] = []
    if block.alt_text and block.alt_text.strip():
        parts.append(f"*{block.alt_text.strip()}*")
    inject_ocr = cfg is None or cfg.images.inject_ocr
    if inject_ocr and block.ocr_text and block.ocr_text.strip():
        ocr = block.ocr_text.strip()
        if (cfg is not None and cfg.images.label_unverified_ocr
                and block.provenance == Provenance.UNVERIFIED):
            ocr = f"{_OCR_UNVERIFIED_LABEL}\n{ocr}"
        parts.append(ocr)
    # IMAGE description goes through its own gate: it shares the flag with
    # the table description but the risks are NOT the same — this is a VLM
    # interpretation of a photo and doesn't separate from real spec text in
    # the body, same rationale as `[images] inject_ocr = false`.
    if block.description and (
        cfg is None
        or (cfg.visual.inject_description and cfg.visual.inject_image_description)
    ):
        parts.append(block.description)

    if not parts:
        # Processed (not excluded at parse time) but no content was
        # produced (e.g. SmartArt: type certain but no renderable data,
        # or a normal image where OCR/description is empty) — still let
        # at least the type (or the fact that we don't even know it)
        # show through, don't silently disappear.
        return _image_placeholder(block)
    return "\n\n".join(parts)


def _render_item_content(item: ListItem, cfg: ChunkerConfig | None = None) -> str:
    """Render the item's blocks without markers (empty blocks are skipped)."""
    rendered = [render_block(b, cfg) for b in item.blocks]
    return "\n\n".join(r for r in rendered if r)


def render_list(block: ListBlock, cfg: ChunkerConfig | None = None) -> str:
    lines: list[str] = []
    # level -> the ordered counter at that level; shallowing drops deeper counters.
    counters: dict[int, int] = {}
    prev_ordered: dict[int, bool] = {}
    for item in block.items:
        for deeper in [lv for lv in counters if lv > item.level]:
            del counters[deeper]
            del prev_ordered[deeper]
        if prev_ordered.get(item.level) != item.ordered:
            counters[item.level] = 0  # marker type changed: GFM starts a new list
        prev_ordered[item.level] = item.ordered

        if item.ordered:
            counters[item.level] = counters.get(item.level, 0) + 1
            marker = f"{counters[item.level]}. "
        else:
            marker = "- "

        indent = " " * (_LIST_INDENT * item.level)
        content = _render_item_content(item, cfg)
        if not content:
            lines.append(f"{indent}{marker}".rstrip())
            continue
        first, *rest = content.split("\n")
        lines.append(f"{indent}{marker}{first}")
        cont = indent + " " * len(marker)
        lines.extend(f"{cont}{line}".rstrip() for line in rest)
    return "\n".join(lines)


# Renderers that don't take cfg (heading/paragraph/code/table): nothing
# outside image/list contains an image, so cfg is not needed.
_RENDERERS = {
    "heading": render_heading,
    "paragraph": render_paragraph,
    "code": render_code,
    "table": render_table,
}
# Renderers that take cfg (can carry nested images: image directly, list inside).
_CFG_RENDERERS = {
    "image": render_image,
    "list": render_list,
}


def render_block(block: AnyBlock, cfg: ChunkerConfig | None = None) -> str:
    """Render one block to GFM; an unknown block type fails loudly.

    `cfg` is used only for image/list rendering (nested images) — for the
    `[images] label_unverified_ocr` policy.
    """
    if block.kind in _CFG_RENDERERS:
        return _CFG_RENDERERS[block.kind](block, cfg)
    try:
        renderer = _RENDERERS[block.kind]
    except KeyError:
        raise ValueError(f"unrenderable block type: {block.kind!r}") from None
    return renderer(block)


def render_blocks(blocks: list[AnyBlock], cfg: ChunkerConfig | None = None) -> str:
    """Render a block sequence as a single text separated by blank lines;
    blocks that contribute nothing (e.g. an image with no alt/OCR) are
    skipped."""
    rendered = [render_block(b, cfg) for b in blocks]
    return "\n\n".join(r for r in rendered if r)