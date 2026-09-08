"""The single describe pass: give every describable block a natural-language
description, whatever kind of block it is.

This is the seam the whole `describe/` package exists for. Before it, a table
went through one pipeline (`tables/table_describe.py`: context injection,
verify+retry, facts) and a chart went through a different one (its XML), and a
block diagram went through none at all -- it got a type label and nothing else,
so a product's architecture diagram contributed zero text to retrieval. Yet all
of them are the same thing: a region of the document that can be restated in
words. Tables are not a separate category, only a more deterministic member of
this one.

So the pass is shared and only the STRATEGY varies per type:

    table                        -> describe/table.py   (LLM reads real cells;
                                    can verify + produce digit-checked facts)
    chart / SmartArt             -> already described by the format parser from
                                    the element's own XML (parsers/chart_extract
                                    .py, parsers/smartart_extract.py) -- no
                                    model call at all, so this pass leaves it be
    block_diagram / technical_   -> describe/visual.py  (VLM reads the crop +
    drawing / flowchart / raster    the same surrounding text a table gets)
    charts

Ordering matters and is why this is a pass over the FINISHED document rather
than something folded into parsing or classification: context (the heading
chain, the paragraphs before and after) only exists once the block's neighbors
have been parsed. A classification call looking at one isolated crop cannot see
any of it.

Two stages, because a single-GPU Ollama pays a large price to swap models:
`stage="vlm"` does the pixel work (pipeline phase 1, alongside image OCR) and
`stage="llm"` does the text work (phase 2, where the table describer already
lived). Splitting by stage keeps that model affinity intact.

Fail-open throughout, matching classification's rule: if a model isn't
configured or reachable, blocks keep their type and simply stay undescribed --
never a crash, never a half-written field.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from medrag.pipeline.parser import progress
from medrag.pipeline.parser.config import ParserConfig, get_config
from medrag.pipeline.parser.describe.context import build_context
from medrag.pipeline.parser.describe.table import describe_table_block
from medrag.pipeline.parser.describe.visual import describe_crop
from medrag.pipeline.parser.llm import (
    LLMClient,
    LLMError,
    VLMClient,
    get_client,
    get_vlm_client,
)
from medrag.pipeline.parser.parsers.base import (
    DescribableBlock,
    ImageBlock,
    ParsedDocument,
    TableBlock,
    hash_hex,
)
from medrag.pipeline.parser.storage_paths import images_dir

__all__ = ["describe_blocks"]

# A TableBlock has no `visual_type` field: it IS the table type, structurally,
# and re-labeling it would just be a second place for the same fact to disagree.
# The taxonomy name it answers to when `[describe] types` is consulted:
_TABLE_TYPE = "table"


def _block_type(block: DescribableBlock) -> str | None:
    if isinstance(block, TableBlock):
        return _TABLE_TYPE
    return block.visual_type


def _wants(block: DescribableBlock, cfg: ParserConfig) -> bool:
    """Is this block a describe candidate at all (type-wise and state-wise)?

    Also the point where parse-time exclusion is honored for TABLES. Images get
    excluded earlier, in images/image_handler.py, because their exclusion also
    has to stop OCR; a table has no OCR stage, so its only generated content is
    what this pass would produce -- which makes here the one place to stop.
    Before this, putting "table" in `[visual] exclude_types` did nothing at
    parse time at all and only chunk-time exclusion honored it.
    """
    if not cfg.describe.enabled:
        # Description generation is unconditionally off (`[describe] enabled = false`).
        # `excluded_at_parse` is NOT marked: that flag means "this type is
        # deliberately left out" and draws a placeholder for the consumer, while
        # a global shut-off is a temporary cost decision that does not change
        # what the block means.
        return False
    btype = _block_type(block)
    if btype is None:
        return False
    if cfg.visual.classify and btype in cfg.visual.exclude_types:
        # Excluded type: record that the emptiness is deliberate, so a consumer
        # renders a placeholder rather than letting the block vanish. The
        # source data (cells/blob) stays -- only generation is skipped.
        block.excluded_at_parse = True
        return False
    if block.excluded_at_parse:
        # Excluded on an earlier pass/run; describing it now would smuggle back
        # exactly what the exclusion removed.
        return False
    if block.description is not None:
        # Already described -- the structural sources (chart/SmartArt XML) run
        # at parse time and are strictly more trustworthy than anything a model
        # could add, so this pass never overwrites them. Also what makes a
        # re-run idempotent.
        return False
    return btype in cfg.describe.types


def _crop_bytes(block: ImageBlock) -> tuple[bytes, str] | None:
    """The stored pixels for a visual block: its resolved blob, else the audit
    crop the pdf parser rendered for an unverified region.

    None => nothing to show a VLM. That is a normal state, not an error: a
    SmartArt block has no pixels anywhere (rendering it needs infrastructure
    this parser doesn't have), and its description comes from its own XML at
    parse time instead.

    Blobs are sha256-named with the extension of their mime (`_store_blob` in
    images/image_handler.py), so the extension is matched rather than assumed.
    """
    for ref, mime in ((block.image_id, block.mime), (block.source_crop, "image/png")):
        if not ref:
            continue
        for path in sorted(images_dir().glob(f"{hash_hex(ref)}.*")):
            if path.is_file() and path.suffix != ".tmp":
                return path.read_bytes(), (mime or "image/png")
    return None


def _describe_visual_one(block: ImageBlock, doc: ParsedDocument, idx: int,
                         vlm: VLMClient | None, cfg: ParserConfig) -> None:
    # An ImageBlock classified "table" is a RASTER table: it is pixels, not a
    # `TableBlock`, so it has no cells to feed the deterministic cell-based
    # strategy (`describe/table.py`, with verify + digit-checked facts). It is
    # therefore described here from pixels like any other visual -- deliberately,
    # not by accident: there is nothing to route it to. Its topic-level summary
    # is a `vlm-vision` description (never verified facts), and its actual cell
    # text, when legible, comes from the image OCR stage (`ocr_text`), not from
    # this pass. The type-specific guidance (describe/visual.py) tells the model
    # not to invent cell values for exactly this reason.
    if _too_small(block, cfg):
        return  # unreadably small -- leave it as type + placeholder (see below)
    crop = _crop_bytes(block)
    if crop is None:
        return  # no pixels to read -- leave it as type + placeholder
    data, mime = crop
    context = _context_for(doc, idx, cfg)
    prompt_type = _describe_type(block, cfg)
    result = describe_crop(data, mime, prompt_type, context, vlm, cfg.describe)
    if result.description:
        block.description = result.description
        block.description_source = "vlm-vision"
    # Verify+retry bookkeeping (None when [describe] visual_check is off), so a
    # flagged visual is as visible as a flagged table -- see describe/visual.py.
    block.describe_status = result.status
    block.describe_attempts = result.attempts


def _too_small(block: ImageBlock, cfg: ParserConfig) -> bool:
    """Is this crop below the min describable edge (`[describe]
    min_visual_edge_px`)?

    Model-independent, objective gate: a crop whose shortest edge is under the
    threshold carries no content any VLM can read, so asking for a 1-4 sentence
    description only invites a plausible-sounding fabrication (e.g. a 47x40 logo
    turned into "a block labeled System with an arrow"). The block keeps its
    type + a placeholder, exactly as when no VLM is available. Unknown
    dimensions (width/height None) never gate -- we only skip what we can prove
    is too small.
    """
    threshold = cfg.describe.min_visual_edge_px
    if threshold <= 0 or block.width is None or block.height is None:
        return False
    return min(block.width, block.height) < threshold


def _describe_type(block: ImageBlock, cfg: ParserConfig) -> str:
    """The visual type the DESCRIBE prompt should assert -- not necessarily the
    block's classified `visual_type`.

    A classification below `[describe] min_visual_confidence` is too weak to
    hand the describer as fact: asserting a wrong type ("this is a technical
    drawing, state its views and dimensions") is exactly what makes a VLM
    confabulate views and dimensions that aren't there. So a low-confidence
    label falls back to the humble "unknown" strategy, whose guidance tells the
    model to state only what it can actually see and to admit when it cannot
    tell. The block keeps its real `visual_type`/`visual_type_confidence` in the
    IR untouched -- only the prompt is softened, and only when a confidence is
    present AND below the (opt-in, default 0.0) threshold.
    """
    vtype = block.visual_type or "unknown"
    threshold = cfg.describe.min_visual_confidence
    conf = block.visual_type_confidence
    if threshold > 0 and conf is not None and conf < threshold:
        return "unknown"
    return vtype


def _describe_table_one(block: TableBlock, doc: ParsedDocument, idx: int,
                        client: LLMClient, cfg: ParserConfig) -> None:
    if block.table.cells and len(block.table.cells) > cfg.table.max_rows > 0:
        progress.write(f"!! table at block {idx}: {len(block.table.cells)} rows, "
                       f"only the first {cfg.table.max_rows} sent to the LLM "
                       f"(TABLE_MAX_ROWS) -- description and facts cover that "
                       f"prefix only")
    describe_table_block(
        client, block, _context_for(doc, idx, cfg),
        use_check=cfg.table.llm_check,
        max_attempts=max(1, cfg.table.check_retries) if cfg.table.llm_check else 1,
        use_facts=cfg.table.facts,
        max_rows=max(0, cfg.table.max_rows))


def _context_for(doc: ParsedDocument, idx: int, cfg: ParserConfig) -> str:
    ctx = cfg.describe.context
    if not ctx.enabled:
        return ""
    return build_context(doc.blocks, idx, ctx.max_chars,
                         max(0, ctx.before), max(0, ctx.after))


def describe_blocks(doc: ParsedDocument, *, stage: str,
                    llm: LLMClient | None = None,
                    vlm: VLMClient | None = None) -> ParsedDocument:
    """Describe every candidate block of `doc` for one `stage`.

    stage="vlm"
        Visual regions with pixels (block diagrams, technical drawings,
        flowcharts, rasterized charts). Runs where the VLM already is.
    stage="llm"
        Tables. Runs where the text LLM already is.

    Mutates `doc` in place and returns it. No-op when nothing qualifies, so a
    document without candidates needs no client or network at all. Idempotent:
    a block that already has a `description` is skipped, so a re-run (or a run
    after a partial failure) neither re-asks nor overwrites a structural
    description the parser produced.
    """
    if stage not in ("vlm", "llm"):
        raise ValueError(f"unknown describe stage {stage!r}; expected 'vlm' or 'llm'")

    cfg = get_config()
    wanted_cls = TableBlock if stage == "llm" else ImageBlock
    candidates = [(i, b) for i, b in enumerate(doc.blocks)
                  if isinstance(b, wanted_cls) and _wants(b, cfg)]
    if not candidates:
        return doc

    if stage == "llm":
        if llm is None:
            llm = get_client()  # raises: a table pass with no LLM IS a failure
        work = lambda idx, block: _describe_table_one(block, doc, idx, llm, cfg)
        label, unit = "describe tables", "table"
    else:
        if vlm is None:
            try:
                vlm = get_vlm_client()
            except LLMError:
                vlm = None  # fail-open: describe_crop returns None throughout
        work = lambda idx, block: _describe_visual_one(block, doc, idx, vlm, cfg)
        label, unit = "describe visuals", "visual"

    max_workers = max(1, cfg.describe.concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool, \
            progress.bar(len(candidates),
                         f"{os.path.basename(doc.source_path)}: {label}",
                         unit=unit) as pbar:
        futures = [pool.submit(work, idx, block) for idx, block in candidates]
        for f in as_completed(futures):
            f.result()  # re-raise the first failure (matches describe_tables)
            pbar.update()

    if stage == "llm" and cfg.table.llm_check:
        statuses = [b.describe_status for _, b in candidates]
        ok = statuses.count("ok")
        flagged = statuses.count("flagged")
        empty = statuses.count("empty")
        progress.write(f"table descriptions: {ok} ok, {flagged} flagged, "
                       f"{empty} empty (of {len(candidates)})")
        if flagged and ok == 0:
            progress.write("!! every checked table was flagged and none passed -- "
                           "description generation is likely broken (e.g. a thinking "
                           "model swallowing the token budget on hidden reasoning); "
                           "check LLM_THINKING_ON and the model/server logs")

    return doc
