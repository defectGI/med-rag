"""Classifies what a visual region actually is -- table, chart, block
diagram, technical drawing, flowchart, product photo, decorative, or
unknown -- so downstream code stops forcing every geometrically-detected
grid into a table (the root cause of e.g. a block diagram's repeated "AAF"
labels becoming a fake table) and so parse-time/chunk-time exclusion has a
label to filter on.

Three entry points:
    classify_page_regions(...) a full page render is available and no
                              candidate crop exists yet -- asks a grounding-
                              capable model to locate every whole visual
                              region on the page directly (one call per
                              page), instead of classifying pdfplumber's
                              already-fragmented candidates one at a time.
    classify_crop(...)       a rendered/cropped image is already available
                              (every ImageBlock, and every geometric table
                              candidate in pdf_parser.py).
    classify_table_text(...) no crop exists, only the cell text a VLM
                              already read off the page (pdf_parser.py's
                              ungrounded VLM-table fallback) -- a weaker,
                              text-only signal, used only there.

Idempotency: results are cached on disk keyed by the sha256 of whatever was
classified (crop bytes, or the flattened cell text) PLUS `_PROMPT_VERSION`
(every on-disk LLM cache key carries a prompt/logic version, so editing the
classification prompt below deliberately invalidates the cache instead of every
crop silently keeping its old label -- bump `_PROMPT_VERSION` whenever the
prompts or the parsing of their replies changes behavior). Cache files live
under `storage/labels/`, OUTSIDE the IR and outside
STORAGE_OUTPUT_DIR/STORAGE_IMAGES_DIR on purpose: an IR_VERSION bump
regenerates every document's IR from scratch, and a PARSER_VERSION bump
forces a full corpus re-parse for unrelated fixes -- neither should force
the same crop through the VLM a second time. A cache miss with no VLM
configured (or classification disabled) returns "unknown"/"unavailable"
WITHOUT writing the cache, so a later run with a VLM available still gets
a real answer instead of being stuck on a placeholder verdict forever.

Unknown labels are appended to `storage/labels/unknown.jsonl` (one line per
occurrence) so the taxonomy's blind spots are visible and reviewable,
instead of silently accumulating as unlabeled content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from medrag.pipeline.parser.config import Visual
from medrag.pipeline.parser.llm import LLMClient, LLMError, VLMClient
from medrag.pipeline.parser.storage_paths import labels_dir

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Part of every cache key (see module docstring). Bump on any change to the
# system prompts below or to how their replies are parsed into labels.
# v2: sharper "decorative" guidance so logos/brand marks/icons stop being
# labeled block_diagram and then described as circuitry.
_PROMPT_VERSION = "2"

_CLASSIFY_SYSTEM_TEMPLATE = (
    "You classify a single visual region cropped from a document page into "
    "exactly one of these types: {types}.\n"
    "Guidance:\n"
    "- \"table\": a genuine data table -- a grid whose cells hold distinct, "
    "varied values (numbers, labels, specs). A grid-like region whose cells "
    "are mostly empty or repeat the same short label is NOT a table -- it is "
    "almost always a block diagram or technical drawing whose frame lines "
    "happened to form a grid. Never label something \"table\" out of "
    "uncertainty.\n"
    "- \"chart\": a bar/line/pie/scatter chart showing plotted data.\n"
    "- \"block_diagram\": boxes or shapes connected by lines representing "
    "components, connections, or process steps, with little or no tabular "
    "data. A company logo or brand mark is NOT a block_diagram even if its "
    "shapes and strokes happen to resemble boxes and connecting lines -- see "
    "\"decorative\".\n"
    "- \"technical_drawing\": an engineering/mechanical drawing, schematic, "
    "or dimensioned diagram.\n"
    "- \"flowchart\": a decision/process flow diagram.\n"
    "- \"product_photo\": a photograph of a physical object or product.\n"
    "- \"decorative\": a company/brand logo (including a stylised or "
    "hand-drawn one), an app or UI icon, a header/footer ornament, or any "
    "purely decorative graphic that carries no technical information. Prefer "
    "this whenever the region is a logo or icon, even if it contains letters, "
    "lines or small shapes -- a logo is decorative regardless of what its "
    "strokes look like.\n"
    "- \"unknown\": you cannot confidently tell which type this is -- prefer "
    "this over guessing.\n\n"
    "Return ONLY a JSON object, no other text: "
    '{{"type": "<one of the types above>", "confidence": <0.0-1.0>}}.'
)

_TABLE_TEXT_SYSTEM_TEMPLATE = (
    "You are given the cell contents of a candidate table extracted from a "
    "document page (text only, no image). Decide whether this is genuinely "
    "one of: {types}.\n"
    "A real \"table\" has distinct, varied values per cell. If most cells "
    "are empty or repeat the same short label, this is more likely a "
    "\"block_diagram\" or other non-tabular visual that was misread as a "
    "grid -- do not label it \"table\" out of uncertainty; prefer "
    "\"unknown\".\n\n"
    "Return ONLY a JSON object, no other text: "
    '{{"type": "<one of the types above>", "confidence": <0.0-1.0>}}.'
)

# v1: page-level region grounding, added to stop treating
# every pdfplumber raster fragment (tile, arrowhead, dimension mark) of one
# technical drawing as its own separate figure -- see classify_page_regions.
_PAGE_REGIONS_PROMPT_VERSION = "1"

_PAGE_REGIONS_SYSTEM_TEMPLATE = (
    "You are looking at one full page rendered from a document. Find every "
    "distinct VISUAL region on the page: a table, chart, block diagram, "
    "technical/engineering drawing, flowchart, product photo, or decorative "
    "graphic (logo, icon, header/footer ornament). Ignore body text, "
    "headings, running headers/footers, and page numbers entirely -- do not "
    "report them, and do not read or transcribe anything.\n\n"
    "The single most important rule: a technical drawing, schematic, or "
    "diagram is frequently exported as many small separate pieces -- "
    "individual line segments, arrowheads, dimension ticks, callout boxes, "
    "small raster tiles -- that are all fragments of ONE drawing. Report ONE "
    "bounding box for the whole drawing, never one box per fragment. Judge "
    "this the way a person looking at the printed page would: if pieces "
    "touch, overlap, sit inside a shared frame/border, or are clearly part "
    "of the same figure and its caption, merge them into a single region "
    "that covers all of them. Only report separate regions for visuals that "
    "are genuinely independent of each other -- e.g. two unrelated photos "
    "placed side by side, or a table next to an unrelated chart. When in "
    "doubt between one merged region and several small ones, prefer the "
    "single merged region.\n\n"
    "For each region, give its type (one of: {types}) and its bounding box "
    "as fractions of the full page width/height -- [0, 0] is the top-left "
    "corner of the page, [1, 1] is the bottom-right corner -- in the form "
    "[x0, y0, x1, y1] with x0<x1 and y0<y1. Do not transcribe or describe "
    "the region's contents; only its type, location, and extent.\n\n"
    "Return ONLY a JSON object, no other text: "
    '{{"regions": [{{"type": "<one of the types above>", '
    '"bbox": [x0, y0, x1, y1], "confidence": <0.0-1.0>}}, ...]}}. '
    'If the page has no visual regions at all, return {{"regions": []}}.'
)

_unknown_log_lock = threading.Lock()


@dataclass
class ClassificationResult:
    visual_type: str
    source: str  # "vlm" | "text-llm" | "cache" | "unavailable"
    confidence: float | None = None
    model: str | None = None


@dataclass
class PageRegion:
    visual_type: str
    bbox: tuple[float, float, float, float]  # fractions of page: x0, y0, x1, y1
    confidence: float | None
    source: str  # "vlm" | "cache"


def _parse_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))


def _parse_verdict(raw: str, types: list[str]) -> tuple[str, float | None]:
    """Parse a `{"type", "confidence"}` reply; any malformed/out-of-taxonomy
    reply is "unknown" rather than trusting a guess the taxonomy doesn't
    recognize -- see module docstring on never forcing uncertainty into a
    real type."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        return "unknown", None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unknown", None
    label = str(parsed.get("type") or "").strip()
    confidence = _parse_confidence(parsed.get("confidence"))
    if label not in types:
        return "unknown", confidence
    return label, confidence


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    # Clamp to the page instead of dropping -- a model that overshoots
    # slightly past an edge still gave a usable region.
    x0, y0, x1, y1 = (max(0.0, min(1.0, v)) for v in (x0, y0, x1, y1))
    if x1 - x0 < 1e-6 or y1 - y0 < 1e-6:
        return None
    return (x0, y0, x1, y1)


def _parse_page_regions(raw: str, types: list[str], source: str
                        ) -> list[PageRegion]:
    """Parse a `{"regions": [{"type", "bbox", "confidence"}, ...]}` reply.
    Any entry that doesn't parse cleanly (bad JSON, out-of-taxonomy type,
    degenerate/missing bbox) is dropped rather than guessed at -- a region
    the model didn't clearly ground is worse than no region at all, since a
    caller treats these bboxes as authoritative."""
    match = _JSON_OBJECT.search(raw)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    entries = parsed.get("regions")
    if not isinstance(entries, list):
        return []
    out: list[PageRegion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("type") or "").strip()
        if label not in types:
            continue
        bbox = _parse_bbox(entry.get("bbox"))
        if bbox is None:
            continue
        out.append(PageRegion(visual_type=label, bbox=bbox,
                              confidence=_parse_confidence(entry.get("confidence")),
                              source=source))
    return out


def _cache_key(data: bytes) -> str:
    """sha256 over the classified bytes + `_PROMPT_VERSION`."""
    h = hashlib.sha256(data)
    h.update(b"\x00prompt:" + _PROMPT_VERSION.encode("utf-8"))
    return h.hexdigest()


def _cache_path(sha: str) -> Path:
    return labels_dir() / f"{sha}.json"


def _cache_read(sha: str) -> ClassificationResult | None:
    path = _cache_path(sha)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    visual_type = data.get("visual_type")
    if not isinstance(visual_type, str):
        return None
    return ClassificationResult(
        visual_type=visual_type, source="cache",
        confidence=data.get("confidence"), model=data.get("model"))


def _cache_write(sha: str, result: ClassificationResult) -> None:
    root = labels_dir()
    root.mkdir(parents=True, exist_ok=True)
    payload = {"visual_type": result.visual_type, "source": result.source,
              "confidence": result.confidence, "model": result.model}
    tmp = root / f"{sha}.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, root / f"{sha}.json")


def _page_regions_cache_key(png: bytes) -> str:
    """sha256 over the page render + `_PAGE_REGIONS_PROMPT_VERSION`. A
    separate subdir from the per-crop cache (below) -- same hash space,
    different payload shape (a list, not a single verdict), so the two must
    never collide on one file."""
    h = hashlib.sha256(png)
    h.update(b"\x00prompt:" + _PAGE_REGIONS_PROMPT_VERSION.encode("utf-8"))
    return h.hexdigest()


def _page_regions_cache_path(sha: str) -> Path:
    return labels_dir() / "pages" / f"{sha}.json"


def _page_regions_cache_read(sha: str) -> list[PageRegion] | None:
    path = _page_regions_cache_path(sha)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, list):
        return None
    out = []
    for entry in data:
        bbox = _parse_bbox(entry.get("bbox"))
        visual_type = entry.get("visual_type")
        if bbox is None or not isinstance(visual_type, str):
            continue
        out.append(PageRegion(visual_type=visual_type, bbox=bbox,
                              confidence=entry.get("confidence"), source="cache"))
    return out


def _page_regions_cache_write(sha: str, regions: list[PageRegion]) -> None:
    root = labels_dir() / "pages"
    root.mkdir(parents=True, exist_ok=True)
    payload = [{"visual_type": r.visual_type, "bbox": list(r.bbox),
               "confidence": r.confidence} for r in regions]
    tmp = root / f"{sha}.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, root / f"{sha}.json")


def log_unknown(doc_id: str, block_id: str, ref: str | None,
                extra: dict[str, Any] | None = None) -> None:
    """Append one line to `storage/labels/unknown.jsonl` recording a region
    classified as "unknown" -- reviewable evidence for growing the taxonomy,
    per the requirement that unknowns must never be silently absorbed."""
    entry = {
        "doc_id": doc_id, "block_id": block_id, "ref": ref,
        # Offset-aware local ISO — the single repo-wide timestamp standard.
        "classified_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **(extra or {}),
    }
    root = labels_dir()
    root.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with _unknown_log_lock, (root / "unknown.jsonl").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def classify_page_regions(png: bytes, vlm: VLMClient | None, cfg: Visual
                          ) -> list[PageRegion]:
    """One VLM call for a whole page: ask a grounding-capable model to
    locate every whole visual region directly on the full page render,
    instead of classifying candidate crops one at a time.

    Exists because geometric detection (pdfplumber's `page.images`) reports
    one real technical drawing as many small raster fragments (tiles,
    arrowheads, dimension marks) with no merge step, so the parser was
    emitting a dozen tiny ImageBlocks for what a reader sees as one figure --
    and because classifying each fragment individually scales the VLM call
    count with fragment count, not page count. This call replaces both: at
    most one VLM call per page, returning bboxes the caller treats as
    authoritative for grouping/cropping figures, in place of the raw
    fragment list.

    Cached by sha256(page png bytes + `_PAGE_REGIONS_PROMPT_VERSION`), in a
    "pages/" subdir separate from `classify_crop`'s per-crop cache -- same
    hash space, different payload shape, so a collision would silently
    corrupt whichever cache read second.

    Fail-open: `cfg.classify=False` or no `vlm` configured returns `[]`,
    NOT written to the cache, so the caller falls back to its own geometric
    detection and a later run with a VLM available still gets a real
    answer."""
    if not cfg.classify or vlm is None:
        return []

    sha = _page_regions_cache_key(png)
    cached = _page_regions_cache_read(sha)
    if cached is not None:
        return cached

    system = _PAGE_REGIONS_SYSTEM_TEMPLATE.format(types=", ".join(cfg.types))
    try:
        raw = vlm.complete_vision(system=system, user="Find every visual region on this page.",
                                  images=[("image/png", png)], max_tokens=cfg.max_tokens)
    except LLMError:
        return []

    regions = _parse_page_regions(raw, cfg.types, source="vlm")
    _page_regions_cache_write(sha, regions)
    return regions


def classify_crop(data: bytes, mime: str, vlm: VLMClient | None,
                  cfg: Visual) -> ClassificationResult:
    """Classify a rendered/cropped image. Cached by sha256(data + prompt
    version) -- identical crops (including on a later, unrelated re-parse)
    never pay for a second VLM call, while a prompt change re-classifies.

    `cfg.classify=False` or no `vlm` configured -> "unknown"/"unavailable",
    NOT written to the cache (so a later run with a VLM available still
    gets a real classification instead of being stuck on this verdict)."""
    sha = _cache_key(data)
    cached = _cache_read(sha)
    if cached is not None:
        return cached

    if not cfg.classify or vlm is None:
        return ClassificationResult(visual_type="unknown", source="unavailable")

    system = _CLASSIFY_SYSTEM_TEMPLATE.format(types=", ".join(cfg.types))
    try:
        raw = vlm.complete_vision(system=system, user="Classify this region.",
                                  images=[(mime, data)], max_tokens=cfg.max_tokens)
    except LLMError:
        return ClassificationResult(visual_type="unknown", source="unavailable")

    label, confidence = _parse_verdict(raw, cfg.types)
    result = ClassificationResult(visual_type=label, source="vlm", confidence=confidence)
    _cache_write(sha, result)
    return result


def classify_table_text(rows: list[list[str]], llm: LLMClient | None,
                        cfg: Visual) -> ClassificationResult:
    """Classify a candidate table by its cell text alone (no crop available
    -- pdf_parser.py's ungrounded VLM-table fallback, where the VLM emitted
    a `{"type": "table"}` spec with no geometric region to crop). A weaker
    signal than `classify_crop` since there are no pixels to look at, but
    still enough to catch the "mostly-empty/repeated-label grid" pattern
    that marks a misread diagram. Cached by sha256 of the flattened text +
    prompt version."""
    flattened = "\n".join("\t".join(row) for row in rows)
    sha = _cache_key(flattened.encode("utf-8"))
    cached = _cache_read(sha)
    if cached is not None:
        return cached

    if not cfg.classify or llm is None:
        return ClassificationResult(visual_type="unknown", source="unavailable")

    system = _TABLE_TEXT_SYSTEM_TEMPLATE.format(types=", ".join(cfg.types))
    user = f"Cell contents (row per line, tab-separated):\n{flattened}"
    try:
        raw = llm.complete(system=system, user=user, max_tokens=cfg.max_tokens)
    except LLMError:
        return ClassificationResult(visual_type="unknown", source="unavailable")

    label, confidence = _parse_verdict(raw, cfg.types)
    result = ClassificationResult(visual_type=label, source="text-llm", confidence=confidence)
    _cache_write(sha, result)
    return result
