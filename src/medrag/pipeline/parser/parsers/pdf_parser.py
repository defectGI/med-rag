"""PDF -> ParsedDocument.

Unlike the other formats, a PDF has no single lossless path. Each page is
triaged and routed to one of three strategies (see parsers/README.md,
"PDF pipeline"):

* code path    born-digital, simple layout: pdfplumber only. Lossless,
               deterministic, no model involved.
* hybrid path  born-digital but complex (tables / multi-column): the page is
               rendered and read by a VLM with the page's own text layer given
               as grounding; every VLM block is then verified against that
               text layer.
* scanned path no usable text layer: render -> independent text detector
               (bboxes + text) -> VLM read -> every VLM line is cross-checked
               against the detector; suspicious lines are crop-and-reread by a
               second, independent VLM.

Provenance: every text-bearing block gets `Block.provenance`:
    "text-layer-verified"   content matched the PDF's own text layer
    "consensus-verified"    two independent readers agreed (VLM + detector /
                            second VLM)
    "unverified"            could not be independently confirmed; the block's
                            `source_crop` holds the sha256 of a page/region
                            render stored under `storage/images/` so a human
                            (or the citation pipeline) can audit the claim.
Verification never auto-resolves a disagreement: a block that two sources
dispute stays "unverified" with its crop, it is not silently "fixed".

Model access is provider-agnostic: everything goes through `llm.VLMClient`
(`llm.get_vlm_client()`); the parser never names a provider or model. With no
VLM configured the parser degrades gracefully: hybrid pages fall back to the
code path (their text layer is real, so the result is still honest), scanned
pages produce a full-page ImageBlock for the images/ OCR stage to pick up.

The scanned-path text detector is likewise pluggable (`TextDetector`
protocol); the default uses pytesseract when it is importable and silently
disappears when it is not (verification then leans on the second VLM alone).

Known v1 limitations: on hybrid and scanned pages alike, a VLM "table" spec
is paired with a pdfplumber-detected table positionally (i-th VLM table <->
i-th geometric table, both in reading order) since the VLM returns no
coordinates -- a page where the model over/under-counts tables falls back to
its own ungrounded grid for the unpaired ones; a VLM "figure" spec is paired
the same way with pdfplumber's embedded raster objects, giving the figure
both a true reading-order position and a geometric bbox when the counts line
up (an unpaired spec still becomes an ImageBlock, just unverified with an
audit crop tight to the VLM's own bbox guess when it gave one, else no crop
at all -- a full-page dump is not a useful crop for a single figure; an
unpaired raster object is still appended, just without a
position). On scanned pages the raster that IS the scan itself (spanning
most of the page) is excluded from that pairing so it can't masquerade as a
verified figure. Inline formatting `runs` (bold/italic) come from each word's
own font name (e.g. "Arial-BoldMT") -- the PDF equivalent of reading a docx
run's rPr, deterministic and not a VLM guess. On the code path this is direct;
on the hybrid path it's backfilled onto a "text-layer-verified" block by
aligning its (VLM-read) words back onto the page's own font-tagged word
stream in reading order (`_align_runs`) -- the page is born-digital, so the
font source is real, it's just read once for content and once for formatting.
Blocks the text layer couldn't confirm are left unmarked rather than aligned,
since a wording mismatch means the alignment itself can't be trusted. Scanned
pages (no text layer at all) still don't fill `runs`: there is no independent
source to verify a bold/italic claim against, unlike text content itself.

Environment (all optional):
    PDF_VLM             "0" disables all VLM use (deterministic fallbacks only)
    PDF_RENDER_DPI      page render resolution for VLM/crops (default 150)
    PDF_CROP_DIR        blob dir for audit crops (default: STORAGE_IMAGES_DIR,
                        see storage_paths.py -- storage/images if that's unset too)
    PDF_VLM_MAX_TOKENS  token budget for a page read (default 8192)
    PDF_CONTAINMENT     token containment threshold, 0..1 (default 0.9)
    PDF_LINE_MATCH      per-line consensus similarity threshold (default 0.8)
    PDF_TESSERACT_LANG  language(s) for the default detector (e.g. "tur+eng")
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Any, Protocol

import pdfplumber
from PIL import Image

from medrag.pipeline.parser import progress
from medrag.pipeline.parser.config import get_config
from medrag.pipeline.parser.images.visual_classify import (
    classify_crop,
    classify_page_regions,
    classify_table_text,
    log_unknown,
)
from medrag.pipeline.parser.llm import (
    LLMClient,
    LLMError,
    VLMClient,
    get_client,
    get_vlm_classify_client,
    get_vlm_client,
)
from medrag.pipeline.parser.storage_paths import images_dir, tesseract_lang_override
from medrag.pipeline.parser.tables.structure import (
    TableStructureClient,
    TableStructureError,
    TextCellHint,
    get_table_structure_client,
)

from .base import (
    PARSER_VERSION,
    BaseParser,
    Block,
    HeadingBlock,
    HeadingStack,
    ImageBlock,
    InlineRun,
    Mark,
    Merge,
    ParagraphBlock,
    ParsedDocument,
    Span,
    TableBlock,
    TableData,
    finalize_runs,
    hash_hex,
    runs_have_marks,
    sha256_id,
    text_cell,
)
from .heading_heuristics import CAPTION
from .heading_reconcile import num_path, reconcile_headings, strip_num
from .pdf_lock import PDFIUM_LOCK
from .running_lines import (
    detect_running_lines,
    in_margin_band,
    looks_like_page_number,
    normalize,
)
from .table_bands import build_band_table, join_words, word_sep

PROV_TEXT_LAYER = "text-layer-verified"
PROV_CONSENSUS = "consensus-verified"
PROV_UNVERIFIED = "unverified"
# Row/column grid came from a table-structure model's visual inference rather
# than deterministic line geometry, but every cell's text was still placed
# from the PDF's own digital characters by containment (see
# _fill_cells_by_containment) -- the model never supplies cell text -- as
# trustworthy as PROV_TEXT_LAYER content-wise, tagged separately for
# transparency about where the grid itself came from.
PROV_TABLE_STRUCTURE = "table-structure-model"
# Row/column grid rebuilt deterministically from the page's own vector geometry
# (fill bands, ruling segments, whitespace channels -- see parsers/table_bands.py)
# with no model involved; cell text is the PDF's own words placed by that
# geometry. The default and most trustworthy table path for born-digital pages.
PROV_TABLE_BANDS = "table-bands"

_UNSET = object()  # sentinel: "resolve from environment lazily"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------


# Tuning thresholds come from config/default.toml now (get_config(); an env
# override beats the TOML, see config.py + the root CONFIG.md). There are no
# _env_bool/_env_int/_env_float helpers -- every call site reads
# get_config().<section>.<field>.


# ---------------------------------------------------------------------------
# Text normalization + fuzzy matching (pure helpers)
# ---------------------------------------------------------------------------

_PUNCT_EDGES = ".,;:!?()[]{}\"'`«»"

# Fold semantically-identical glyph VARIANTS onto one canonical form before any
# comparison. Two sources for the same table cell routinely disagree on which
# dash, space, or digit-variant codepoint they use while meaning the exact same
# thing: a PDF drawing "not applicable" with a horizontal bar (U+2015) and a VLM
# writing an ASCII hyphen; a footnote marker printed as superscript "¹" (U+00B9)
# that a general VLM reads as "1"; a non-breaking space inside "5 V". Left
# unfolded these cost a content match and, in tables, could sink a whole grid's
# accept score even though every character was right (see the e2e analysis:
# dash/superscript drift was the dominant reason VLM-refined grids were
# rejected). Only VARIANTS fold -- an ASCII hyphen "-" (U+002D) is left as-is,
# so genuine values like part number "99-1234" are untouched; digits/letters
# still have to match, so this never masks a real hallucination.
_GLYPH_FOLD: dict[int, str | None] = {0x00AD: None}  # soft hyphen: drop
for _c in "‐‑‒–—―−":  # ‐ ‑ ‒ – — ― −
    _GLYPH_FOLD[ord(_c)] = "-"
for _c in "    ":  # nbsp / figure / thin / narrow-nbsp
    _GLYPH_FOLD[ord(_c)] = " "
for _i, _c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹"):
    _GLYPH_FOLD[ord(_c)] = str(_i)  # superscript digits -> ASCII
for _i in range(10):
    _GLYPH_FOLD[0x2080 + _i] = str(_i)  # subscript digits -> ASCII
del _c, _i


def _fold_glyphs(text: str) -> str:
    return text.translate(_GLYPH_FOLD)


def _has_alnum(s: str) -> bool:
    return any(ch.isalnum() for ch in s)


def _norm(text: str) -> str:
    """Whitespace-collapsed, casefolded, glyph-folded view used for all
    comparisons (see `_GLYPH_FOLD`)."""
    return " ".join(_fold_glyphs(text).split()).casefold()


def _tok_list(text: str) -> list[str]:
    """Normalized content tokens: edge punctuation stripped, pure-symbol
    tokens dropped (e.g. a bare "=", "/", "-", math arrows "→"/"↑").

    Two independent transcriptions of the same math/formula line routinely
    disagree on operator glyphs (→ vs ->, − vs -) while agreeing on every
    letter and digit; counting those glyphs as content tokens dragged short
    formula lines below the containment threshold even when fully correct.
    Digits/letters still need to match — this only drops tokens that carry
    no alphanumeric content, so exact-number verification (`_criticals`) is
    unaffected.
    """
    out = []
    for t in _norm(text).split():
        t = t.strip(_PUNCT_EDGES)
        if t and any(ch.isalnum() for ch in t):
            out.append(t)
    return out


def _counter(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _containment(needle: list[str], hay: Counter) -> float:
    """Fraction of `needle` tokens present in `hay` (multiset semantics).

    Order-insensitive on purpose: the VLM may reflow line breaks or column
    order while still reading the same characters.
    """
    if not needle:
        return 1.0
    need = Counter(needle)
    found = sum(min(n, hay.get(t, 0)) for t, n in need.items())
    return found / sum(need.values())


def _criticals(text: str) -> Counter:
    """Digit-bearing tokens (numbers, dates, part numbers, units with counts).

    These are the tokens a hallucination hurts most, so they must match their
    independent source EXACTLY — fuzzy similarity is not enough for "500" vs
    "600".
    """
    return Counter(t for t in _tok_list(text) if any(ch.isdigit() for ch in t))


def _crit_ok(text: str, source_crit: Counter) -> bool:
    """Every critical token of `text` appears in the source (multiset)."""
    return not (_criticals(text) - source_crit)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ---------------------------------------------------------------------------
# VLM page reading: prompts + output parsing
# ---------------------------------------------------------------------------

_PAGE_SYSTEM = (
    "You transcribe one page of a document into structured blocks.\n"
    "Return ONLY a JSON array — no prose, no markdown fences. Each element is "
    "one of:\n"
    '  {"type": "heading", "level": 1, "text": "..."}\n'
    '  {"type": "paragraph", "text": "..."}\n'
    '  {"type": "list_item", "ordered": false, "level": 0, "text": "..."}\n'
    '  {"type": "table", "rows": [["cell", "cell"], ["cell", "cell"]]}\n'
    '  {"type": "figure", "bbox": [x0, y0, x1, y1]}\n'
    "Rules:\n"
    "- Keep reading order (in multi-column layouts finish the left column "
    "first).\n"
    "- Transcribe text EXACTLY as printed. Never correct, translate, "
    "summarize or invent anything.\n"
    '- A heading\'s "level" is its depth in the document hierarchy. For a '
    'numbered heading it is the count of dot-separated number components '
    '("3." is level 1, "3.5." is level 2, "2.2.1." is level 3). Sibling '
    "sections share one level.\n"
    '- A numbered section title (like "3.5. Ethernet Connector") is always '
    '{"type": "heading"} -- never a paragraph or list_item -- and its '
    'number stays in "text".\n'
    "- Include every piece of text on the page exactly once.\n"
    "- Inside a paragraph, preserve the printed line breaks as \\n.\n"
    '- Every row of a table must have the same number of cells; use "" for '
    "empty cells.\n"
    "- For each photo, illustration, or diagram, emit one "
    '{"type": "figure"} entry at its position in reading order. Do not '
    "transcribe or describe its contents. If it occupies a clearly separate "
    "region of the page (not the whole page), also give its approximate "
    '"bbox" as fractions of the page width/height, [x0, y0, x1, y1] with '
    "0,0 at the top-left corner and 1,1 at the bottom-right; omit bbox "
    "otherwise."
)

_TRANSCRIBE_SYSTEM = (
    "You transcribe text from an image exactly as printed, one line per "
    "printed line. Output plain text only — no commentary, no formatting. "
    "If the image contains no text, output nothing."
)


def _hybrid_user(pageno: int, grounding: str) -> str:
    return (
        f"The attached image is page {pageno} of a PDF. The PDF's own "
        "extracted text layer for this page is given below as grounding: "
        "where the image and the text layer agree, copy the text layer's "
        "characters exactly.\n\n"
        f"TEXT LAYER:\n{grounding}\n\n"
        "Return the JSON array now."
    )


def _scanned_user(pageno: int) -> str:
    return (
        f"The attached image is page {pageno} of a scanned document. "
        "Transcribe it. Return the JSON array now."
    )


def _parse_vlm_blocks(raw: str | None) -> list[dict[str, Any]] | None:
    """Parse the VLM's JSON array into validated block specs, or None."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):  # tolerate a fenced reply despite instructions
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    i, j = text.find("["), text.rfind("]")
    if i < 0 or j <= i:
        return None
    try:
        data = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None

    out: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        btype = item.get("type")
        if btype == "heading":
            txt = str(item.get("text") or "").strip()
            if txt:
                level = item.get("level")
                level = level if isinstance(level, int) and 1 <= level <= 6 else 1
                out.append({"type": "heading", "text": txt, "level": level})
        elif btype == "paragraph":
            txt = str(item.get("text") or "").strip()
            if txt:
                out.append({"type": "paragraph", "text": txt})
        elif btype == "list_item":
            txt = str(item.get("text") or "").strip()
            if txt:
                level = item.get("level")
                level = level if isinstance(level, int) and level >= 0 else 0
                out.append({"type": "list_item", "text": txt, "level": level,
                            "ordered": bool(item.get("ordered"))})
        elif btype == "table":
            rows = item.get("rows")
            if isinstance(rows, list) and rows:
                clean = [[("" if c is None else str(c)) for c in r]
                         for r in rows if isinstance(r, list)]
                clean = [r for r in clean if any(c.strip() for c in r)]
                if clean:
                    out.append({"type": "table", "rows": clean})
        elif btype == "figure":
            spec: dict[str, Any] = {"type": "figure"}
            bbox = item.get("bbox")
            if (isinstance(bbox, list) and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)):
                x0, y0, x1, y1 = (float(v) for v in bbox)
                if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1:
                    spec["bbox"] = (x0, y0, x1, y1)
            out.append(spec)
    return out or None


# ---------------------------------------------------------------------------
# Scanned-path text detector (pluggable)
# ---------------------------------------------------------------------------


@dataclass
class DetectedLine:
    """One detected text line: recognized text + bbox in rendered-image px."""

    text: str
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom)


class TextDetector(Protocol):
    """Independent text detector/reader used to cross-check the VLM."""

    def detect(self, png: bytes) -> list[DetectedLine]:
        ...


class TesseractDetector:
    """Default detector: pytesseract line boxes + text (optional dependency)."""

    def __init__(self, lang: str | None = None) -> None:
        self.lang = lang or tesseract_lang_override()

    def detect(self, png: bytes) -> list[DetectedLine]:
        import pytesseract  # deferred: optional dependency

        data = pytesseract.image_to_data(
            Image.open(io.BytesIO(png)), lang=self.lang,
            output_type=pytesseract.Output.DICT)
        lines: dict[tuple, list[int]] = {}
        for k in range(len(data["text"])):
            if not str(data["text"][k]).strip():
                continue
            key = (data["block_num"][k], data["par_num"][k], data["line_num"][k])
            lines.setdefault(key, []).append(k)
        out: list[DetectedLine] = []
        for key in sorted(lines):
            idxs = lines[key]
            text = " ".join(str(data["text"][k]).strip() for k in idxs)
            x0 = min(data["left"][k] for k in idxs)
            top = min(data["top"][k] for k in idxs)
            x1 = max(data["left"][k] + data["width"][k] for k in idxs)
            bottom = max(data["top"][k] + data["height"][k] for k in idxs)
            out.append(DetectedLine(text=text, bbox=(x0, top, x1, bottom)))
        return out


# --- broken-cmap detection / ligature repair ---------------------------------
#
# A PDF with a broken ToUnicode cmap decodes its ligatures to junk ("So�ware",
# "SoŌware", "(cid:31)"). The parser must not invent text, but it CAN (a) flag
# the document so consumers know its words are search-dead and a VLM/OCR
# re-read is worth paying for, and (b) apply the handful of KNOWN safe
# ligature substitutions -- only on flagged documents, so a legitimate "Ōsaka"
# in a healthy document is never rewritten.

_LIGATURE_REPAIRS = {
    "Ō": "ft", "Ʃ": "tt",          # the PN1204 quick-start guide's cmap junk
    "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬆ": "st",
}
_CMAP_JUNK_RE = re.compile(
    "[�" + "".join(_LIGATURE_REPAIRS) + r"]|\(cid:\d+\)")


def _walk_text_blocks(blocks: list[Block]):
    """Every Paragraph/Heading block, including ones nested in table cells."""
    for b in blocks:
        if isinstance(b, (ParagraphBlock, HeadingBlock)):
            yield b
        elif isinstance(b, TableBlock):
            for row in b.table.cells:
                for cell in row:
                    if cell is None:
                        continue
                    yield from _walk_text_blocks(cell.blocks)


def _scan_broken_cmap(blocks: list[Block], min_hits: int = 5
                      ) -> dict[str, Any] | None:
    """Count cmap-junk characters across the document's text; at `min_hits`
    or more, repair the known ligature substitutions in place (text and runs
    together, so their concatenation invariant holds) and return the stats
    for metadata["broken_cmap"]. Unrepairable junk (U+FFFD, "(cid:N)") is
    only counted -- rewriting it would be invention, not repair."""
    hits = sum(len(_CMAP_JUNK_RE.findall(b.text))
               for b in _walk_text_blocks(blocks))
    if hits < min_hits:
        return None

    def repair(text: str) -> str:
        for bad, good in _LIGATURE_REPAIRS.items():
            text = text.replace(bad, good)
        return text

    repaired = 0
    for b in _walk_text_blocks(blocks):
        fixed = repair(b.text)
        if fixed != b.text:
            repaired += 1
            b.text = fixed
            b.runs = [InlineRun(repair(r.text), r.marks, r.link)
                      for r in b.runs]
    remaining = sum(len(_CMAP_JUNK_RE.findall(b.text))
                    for b in _walk_text_blocks(blocks))
    return {"junk_chars": hits, "blocks_repaired": repaired,
            "junk_remaining": remaining,
            "note": ("document's ToUnicode cmap looks broken; known ligature"
                     " substitutions applied, remaining junk left as-is --"
                     " consider a VLM/OCR re-read for this document")}


def _det_paragraphs(det_lines: list[DetectedLine]) -> list[str]:
    """Detector lines -> paragraph texts for the tesseract-fallback path:
    consecutive lines belong to one paragraph until the vertical gap between
    them clearly exceeds normal leading (1.6x the median line height).
    Whitespace-collapsed, matching the IR's canonical single-space text."""
    lines = [dl for dl in det_lines if dl.text.strip()]
    if not lines:
        return []
    lines.sort(key=lambda dl: (dl.bbox[1], dl.bbox[0]))
    heights = sorted(dl.bbox[3] - dl.bbox[1] for dl in lines)
    med_h = heights[len(heights) // 2] or 1.0
    groups: list[list[str]] = [[lines[0].text]]
    for prev, dl in pairwise(lines):
        if dl.bbox[1] - prev.bbox[3] > 1.6 * med_h:
            groups.append([dl.text])
        else:
            groups[-1].append(dl.text)
    return [" ".join(" ".join(g).split()) for g in groups]


def _default_detector() -> TextDetector | None:
    try:
        import pytesseract  # noqa: F401
    except ModuleNotFoundError:
        return None
    return TesseractDetector()


# ---------------------------------------------------------------------------
# Render / crop / blob store
# ---------------------------------------------------------------------------


def _store_crop(data: bytes) -> str:
    """Write an audit crop into the image blob store; return its hash id
    (`sha256:<hex>`; the filename keeps the bare hex).

    Same store as document images (immutable, dedup by content hash) so the
    webapp/citation pipeline resolves both the same way. `PDF_CROP_DIR`
    overrides just this call site; otherwise it's the shared image store
    (`storage_paths.images_dir`, itself overridable via `STORAGE_IMAGES_DIR`).
    """
    hash_id = sha256_id(data)
    root = Path(os.getenv("PDF_CROP_DIR") or images_dir())
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{hash_hex(hash_id)}.png"
    if not path.exists():
        path.write_bytes(data)
    return hash_id


def _crop_png(png: bytes, bbox: tuple[float, float, float, float],
              margin: int = 8) -> bytes:
    """Crop a rendered page PNG to `bbox` (px) with a small margin."""
    img = Image.open(io.BytesIO(png))
    x0 = max(0, int(bbox[0]) - margin)
    y0 = max(0, int(bbox[1]) - margin)
    x1 = min(img.width, int(bbox[2]) + margin)
    y1 = min(img.height, int(bbox[3]) + margin)
    if x1 <= x0 or y1 <= y0:
        return png
    buf = io.BytesIO()
    img.crop((x0, y0, x1, y1)).save(buf, format="PNG")
    return buf.getvalue()


def _union_bbox(boxes: list[tuple[float, float, float, float]]
                ) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _figure_crop(png: bytes, locator: dict[str, Any]) -> bytes | None:
    """Audit crop for an unpaired figure: tight to the VLM's own bbox guess
    (`vlm_bbox`, normalized fractions of the page) when it gave one, else
    None -- a full-page dump is not a useful crop for a single figure, so we
    skip storing one rather than hand back something misleadingly ugly."""
    vlm_bbox = locator.pop("vlm_bbox", None)
    if vlm_bbox is None:
        return None
    img = Image.open(io.BytesIO(png))
    x0, y0, x1, y1 = vlm_bbox
    bbox = (x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height)
    return _crop_png(png, bbox)


# ---------------------------------------------------------------------------
# Code path layout analysis (pure helpers over pdfplumber word dicts)
# ---------------------------------------------------------------------------


@dataclass
class _Line:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float
    runs: list[InlineRun] = field(default_factory=list)
    bullet_len: int = 0  # >0: leading symbol-font bullet glyph, chars to strip
    label: bool = False  # a callout label split off its body line (9.5) --
                         # kept as its own block, never re-merged into a para


def _wsize(w: dict) -> float:
    return float(w.get("size") or 10.0)


def _word_marks(w: dict) -> tuple[Mark, ...]:
    """Bold/italic from the word's own font name -- the PDF equivalent of
    reading a docx run's rPr: deterministic, straight from the source's own
    metadata, not a visual guess. Covers the near-universal PDF font-naming
    convention (e.g. "Arial-BoldMT", "TimesNewRomanPS-BoldItalicMT")."""
    name = (w.get("fontname") or "").lower()
    marks: list[Mark] = []
    if "bold" in name:
        marks.append(Mark.BOLD)
    if "italic" in name or "oblique" in name:
        marks.append(Mark.ITALIC)
    return tuple(marks)


# A font used only to draw list-bullet/icon glyphs, whose codepoints map to
# arbitrary characters (a "bullet" glyph often lands on 'A', 'B', 'l', ...),
# so the extracted text is a stray letter rather than a real bullet character
# `_BULLET_RE` would recognize. Matched on the font *name* (after the subset
# prefix), covering the common icon/dingbat families and generator tools.
_SYMBOL_FONT_RE = re.compile(
    r"wingding|webding|dingbat|zapf|symbol|glyphter|glyph|fontawesome|"
    r"awesome|icomoon|material.?icons?|marlett|pictos?|bullets?|icons?",
    re.IGNORECASE)


def _font_family(name: str) -> str:
    """Base family of a PDF font name for equality tests: drop the random
    'ABCDEF+' subset prefix and any style/weight suffix, so
    'XWMDCD+RedHatText-Bold' and 'AAAAAA+RedHatText-Regular' compare equal
    (same family, different style) while 'XWMDCD+Glyphter' stays distinct."""
    name = re.sub(r"^[A-Z]{6}\+", "", name or "")
    return re.split(r"[-,]", name, 1)[0].strip().lower()


def _symbol_bullet_len(group: list[dict]) -> int:
    """Chars to strip if `group` (one visual line's words, left-to-right)
    begins with a bullet glyph drawn from a symbol/icon font, else 0.

    Font-based, not glyph-based, so it survives whatever letter a given icon
    font's bullet happens to decode to (an 'A' from "Glyphter", a 'l' from
    "Wingdings", ...) instead of hard-coding one symbol. Two independent
    signals, either sufficient, both requiring a single-character leading
    token with real text after it (a lone glyph is not a list):
      (a) that token's font name matches a known symbol/icon family
          (`_SYMBOL_FONT_RE`);
      (b) its font *family* differs from the very next word's -- a real
          leading initial or article ("A device", "I/O") shares the body
          font; a bullet glyph is set in its own distinct font.
    Style-only differences (Bold/Italic vs Regular of the same family) are
    ignored by `_font_family`, so a bold lead-in letter is never mistaken
    for a bullet.
    """
    if len(group) < 2:
        return 0
    lead, nxt = group[0], group[1]
    if len(lead["text"].strip()) != 1:
        return 0
    lead_font = lead.get("fontname") or ""
    if (_SYMBOL_FONT_RE.search(lead_font)
            or _font_family(lead_font) != _font_family(nxt.get("fontname") or "")):
        # the glyph + whatever joins it to the next word (see `word_sep`)
        return len(lead["text"]) + len(word_sep(lead, nxt))
    return 0


def _line_runs(group: list[dict]) -> list[InlineRun]:
    """Per-word marks -> merged InlineRuns whose concatenation equals the
    line's `join_words(group)` text exactly (gap-aware separators, so a word
    split at a font change carries no injected space)."""
    runs = [InlineRun(w["text"] if i == 0 else word_sep(group[i - 1], w) + w["text"],
                      _word_marks(w))
            for i, w in enumerate(group)]
    return finalize_runs(runs)


def _concat_runs(a: list[InlineRun], b: list[InlineRun]) -> list[InlineRun]:
    """`a` + a joining space + `b`, as run lists (merging identical-mark runs
    across the join). Used to glue per-line runs into a multi-line block."""
    if not a:
        return list(b)
    if not b:
        return list(a)
    return finalize_runs(a + [InlineRun(" " + b[0].text, b[0].marks)] + b[1:])


def _slice_runs(runs: list[InlineRun], start: int) -> list[InlineRun]:
    """Drop the first `start` characters (e.g. a stripped bullet marker)."""
    out: list[InlineRun] = []
    pos = 0
    for r in runs:
        end = pos + len(r.text)
        if end > start:
            out.append(InlineRun(r.text[max(0, start - pos):], r.marks))
        pos = end
    return finalize_runs(out)


def _word_marks_stream(prep: dict, tboxes: list
                       ) -> list[tuple[str, tuple[Mark, ...]]]:
    """Page words (minus table regions), in the same column-major reading
    order the code path reconstructs (`_parse_code`), each tagged with its
    deterministic font marks (`_word_marks`). Alignment source for backfilling
    hybrid-path runs onto VLM text that this same page's text layer has
    already confirmed -- not a new signal, just the code path's own one,
    read for a page the triage routed to the VLM instead."""
    words = [w for w in (prep.get("words") or []) if not _in_boxes(w, tboxes)]
    gutter = prep.get("gutter")
    if gutter is None:
        columns = [words]
    else:
        columns = [
            [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter],
            [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter],
        ]
    return [(w["text"], _word_marks(w))
            for cwords in columns
            for w in sorted(cwords, key=lambda w: (w["top"], w["x0"]))]


def _align_runs(text: str, stream: list[tuple[str, tuple[Mark, ...]]],
                pos: list[int], lookahead: int = 40) -> list[InlineRun]:
    """Backfill `text`'s own words with marks from `stream`, matching forward
    only from `pos[0]` (a single counter shared and advanced across every
    block on the page, in the page's reading order -- it never rewinds).

    A word is matched to the next equal (normalized) word within `lookahead`
    slots of the current position; on a match the counter jumps past it, on a
    miss the word is left unmarked and the counter doesn't move. This keeps
    later blocks aligned even when one block's wording doesn't line up
    perfectly (VLM paraphrase, OCR noise) -- it costs that block's marks, not
    everyone after it. Always returns runs whose concatenation is `text`
    exactly, matched or not.
    """
    runs: list[InlineRun] = []
    pending = ""
    for tok in re.findall(r"\S+|\s+", text):
        if tok.isspace():
            pending += tok
            continue
        target = _norm(tok).strip(_PUNCT_EDGES)
        marks: tuple[Mark, ...] = ()
        if target:
            limit = min(len(stream), pos[0] + lookahead)
            for j in range(pos[0], limit):
                if _norm(stream[j][0]).strip(_PUNCT_EDGES) == target:
                    marks = stream[j][1]
                    pos[0] = j + 1
                    break
        runs.append(InlineRun(pending + tok, marks))
        pending = ""
    if pending:
        runs.append(InlineRun(pending, ()))
    return finalize_runs(runs)


def _group_lines(words: list[dict]) -> list[list[dict]]:
    """Word boxes grouped into visual lines by `top` proximity, each group
    sorted by `x0` -- the shared first step behind both `_cluster_lines`
    (visual-line text/metrics) and `_split_gutter_groups` (per-line,
    gap-aware column split)."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if cur and w["top"] - cur[0]["top"] > max(2.0, 0.5 * _wsize(w)):
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    for group in lines:
        group.sort(key=lambda w: w["x0"])
    return lines


# A callout/warning box's short label ("Caution") printed at
# the same vertical position as an adjacent body line used to be x-merged
# into that sentence ("Caution the event of damage..."), scrambling reading
# order inside a "text-layer-verified" block. Three signals, all required
# (each alone is common in ordinary text; together they are the label/body
# seam -- measured on ACME_PN5064_Datasheet.pdf p.4: 12.6pt gap, 7pt Black
# label against 9pt Regular body):
#   * a horizontal gap past both the gutter scale (`_COLUMN_GAP_MIN`) and
#     `_LABEL_GAP_EM` ems -- far beyond word spacing;
#   * a font-SIZE discontinuity at the seam -- a spaced same-size key/value
#     line, justified text, or an emphasized (bold) word after a wide gap
#     all stay whole; a face-only change is deliberately not enough, since
#     bold/regular mixing inside one real line is routine;
#   * a short side of at most `_LABEL_MAX_WORDS` words -- labels are words,
#     not sentences.
_LABEL_GAP_EM = 1.2
_LABEL_MAX_WORDS = 2


def _split_label_group(group: list[dict]) -> list[list[dict]]:
    """Split one visual line at its widest qualifying label/body seam (see
    above), or return it whole."""
    if len(group) < 2:
        return [group]
    best_i, best_gap = None, 0.0
    for i in range(len(group) - 1):
        cur, nxt = group[i], group[i + 1]
        gap = nxt["x0"] - cur["x1"]
        em = min(_wsize(cur), _wsize(nxt))
        if gap < max(_COLUMN_GAP_MIN, _LABEL_GAP_EM * em) or gap <= best_gap:
            continue
        if abs(_wsize(cur) - _wsize(nxt)) < 0.5:
            continue  # same size across the seam: one logical line
        best_i, best_gap = i + 1, gap
    if best_i is None:
        return [group]
    left, right = group[:best_i], group[best_i:]
    if min(len(left), len(right)) > _LABEL_MAX_WORDS:
        return [group]
    return [left, right]


def _strip_symbol_glyphs(group: list[dict]) -> list[dict]:
    """Drop single-character words drawn from a symbol/icon font anywhere
    past the leading-bullet slot (a Glyphter checkmark encodes as a stray
    trailing "A" -- 300+ lines across 44 brochures, and a
    live confusion risk with real ampere values like "16 A", which are set
    in the body font and therefore untouched). The leading slot is kept:
    a leading glyph is a list bullet, which `_symbol_bullet_len` already
    recognizes and strips while recording the line as a list item."""
    def is_glyph(w: dict) -> bool:
        return (len(w["text"].strip()) == 1
                and bool(_SYMBOL_FONT_RE.search(w.get("fontname") or "")))

    if len(group) == 1:
        return [] if is_glyph(group[0]) else group
    return [group[0]] + [w for w in group[1:] if not is_glyph(w)]


def _cluster_lines(words: list[dict]) -> list[_Line]:
    """Group word boxes into visual lines by their `top` coordinate."""
    out: list[_Line] = []
    for raw_group in _group_lines(words):
        fragments = _split_label_group(raw_group)
        for group in fragments:
            is_label = (len(fragments) > 1
                        and len(group) <= _LABEL_MAX_WORDS)
            group = _strip_symbol_glyphs(group)
            if not group:
                continue
            out.append(_Line(
                text=join_words(group),
                x0=min(w["x0"] for w in group),
                x1=max(w["x1"] for w in group),
                top=min(w["top"] for w in group),
                bottom=max(w["bottom"] for w in group),
                size=median(_wsize(w) for w in group),
                runs=_line_runs(group),
                bullet_len=_symbol_bullet_len(group),
                label=is_label,
            ))
    return out


# Minimum horizontal gap (pt) between two adjacent words, straddling the
# gutter x-coordinate, for that gap to count as a genuine column boundary
# rather than ordinary word spacing (see `_split_gutter_groups`). Comfortably
# above normal inter-word spacing at these font sizes (~1-5pt) and below
# `_find_gutter`'s own minimum qualifying band width (page_width * 0.025,
# ~15pt on a typical page), so it never confuses one for the other.
_COLUMN_GAP_MIN = 8.0


def _split_gutter_groups(groups: list[list[dict]], gutter: float
                         ) -> tuple[list[dict], list[dict]]:
    """Partition line-groups (see `_group_lines`) into left/right gutter
    columns, splitting a group *at the gutter* only when there's a real gap
    there. A full-width line that merely crosses the gutter x-coordinate with
    no such gap (one continuous run of words, e.g. a wrapped sentence or a
    table's own header row on an otherwise two-column page) stays whole in
    whichever column its center falls into, instead of being cut into two
    disconnected fragments (ACME_PN5015_Datasheet.pdf p.6, where this used to
    split one sentence into three separate reading-order positions)."""
    col0: list[dict] = []
    col1: list[dict] = []
    for group in groups:
        split_at = None
        for i in range(len(group) - 1):
            if group[i]["x1"] <= gutter <= group[i + 1]["x0"]:
                if group[i + 1]["x0"] - group[i]["x1"] >= _COLUMN_GAP_MIN:
                    split_at = i + 1
                break  # the gutter can only fall between one adjacent pair
        if split_at is not None:
            col0.extend(group[:split_at])
            col1.extend(group[split_at:])
        else:
            center = (group[0]["x0"] + group[-1]["x1"]) / 2
            (col0 if center < gutter else col1).extend(group)
    return col0, col1


# A pdfplumber-detected "table" this small in both dimensions is never real
# tabular content in this corpus -- it's a decorative vector mark (e.g. a
# footer logo) whose bounding box happens to close into a 1-2 cell grid.
# Real tables always span a meaningful fraction of the page in at least one
# dimension, so gating on *both* being tiny avoids dropping a legitimate
# single-row-tall or single-column-wide table.
_MIN_TABLE_WIDTH = 20.0
_MIN_TABLE_HEIGHT = 12.0

# A region covering almost the whole page is a full-page technical drawing /
# schematic whose frame closed into a coarse grid, not a data table (the
# TECHNICAL_DRAWING corpus does this; e.g. a 10x35 "grid" holding a handful of
# scattered dimension callouts). Rejecting it here (rather than emitting a
# vast, ~empty table) lets its few labels fall back into the ordinary text
# flow, since a region dropped from `tables` is no longer excluded from it.
_PAGE_COVER_MAX = 0.85


def _is_degenerate_table(t) -> bool:
    x0, top, x1, bottom = t.bbox
    return (x1 - x0) < _MIN_TABLE_WIDTH and (bottom - top) < _MIN_TABLE_HEIGHT


def _covers_most_of_page(t, page) -> bool:
    x0, top, x1, bottom = t.bbox
    pw, ph = float(page.width), float(page.height)
    if pw <= 0 or ph <= 0:
        return False
    return (x1 - x0) * (bottom - top) >= _PAGE_COVER_MAX * pw * ph


# How much of a geometric table candidate's own area must be covered by a
# single VLM page-region bbox before that region's type is trusted for the
# candidate, skipping the per-candidate classify_crop call. Deliberately
# generous (not a strict IoU): the VLM's bbox is a coarse fraction-of-page
# estimate, not grounded against pdfplumber's own coordinates, so a real
# table must never be misclassified/dropped just because the two boxes
# don't line up pixel-for-pixel. Below this, `_best_region_match` reports no
# match and the caller falls back to the old, more expensive but more
# precise classify_crop call for that candidate -- a real table is never
# trusted to page-region agreement alone if the geometry doesn't back it.
_TABLE_REGION_OVERLAP_MIN = 0.5


def _best_region_match(bbox_pt: tuple[float, float, float, float], page,
                       regions: list) -> tuple[object, float] | None:
    """Best-overlap `PageRegion` (see visual_classify.py) for a candidate
    bbox in page-point coordinates, or None if nothing clears
    `_TABLE_REGION_OVERLAP_MIN`. Overlap is intersection-over-candidate-area
    (not IoU): a VLM region that fully contains a small candidate still
    counts as a confident match even if the region itself is much larger
    (e.g. it merged several fragments -- see classify_page_regions)."""
    pw, ph = float(page.width), float(page.height)
    if pw <= 0 or ph <= 0 or not regions:
        return None
    x0, y0, x1, y1 = bbox_pt
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    best: tuple[object, float] | None = None
    for r in regions:
        rx0, ry0, rx1, ry1 = (r.bbox[0] * pw, r.bbox[1] * ph,
                              r.bbox[2] * pw, r.bbox[3] * ph)
        ix0, iy0 = max(x0, rx0), max(y0, ry0)
        ix1, iy1 = min(x1, rx1), min(y1, ry1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        frac = (ix1 - ix0) * (iy1 - iy0) / area
        if frac >= _TABLE_REGION_OVERLAP_MIN and (best is None or frac > best[1]):
            best = (r, frac)
    return best


def _find_gutter(words: list[dict], page_width: float) -> float | None:
    """X coordinate of a two-column gutter, or None for single-column pages.

    Looks for a near-empty vertical band in the middle of the page with a
    substantial share of the words on each side. A small tolerance lets a
    centered title span the gutter without hiding it.
    """
    if len(words) < 40 or page_width <= 0:
        return None
    n = 200
    cover = [0] * n
    for w in words:
        b0 = max(0, min(n - 1, int(w["x0"] / page_width * n)))
        b1 = max(0, min(n - 1, int(w["x1"] / page_width * n)))
        for b in range(b0, b1 + 1):
            cover[b] += 1
    lo, hi = int(n * 0.30), int(n * 0.70)
    allow = max(1, int(0.02 * len(words)))
    best_width, best_center = 0, None
    i = lo
    while i <= hi:
        if cover[i] <= allow:
            j = i
            while j <= hi and cover[j] <= allow:
                j += 1
            if j - i > best_width:
                best_width, best_center = j - i, (i + j) / 2
            i = j
        else:
            i += 1
    if best_center is None or best_width < n * 0.025:
        return None
    gx = best_center / n * page_width
    left = sum(1 for w in words if (w["x0"] + w["x1"]) / 2 < gx)
    if min(left, len(words) - left) < 0.25 * len(words):
        return None
    return gx


# A "Contents" / "Table of Contents" heading -- see `_extract_toc`. Covers
# the corpus languages plus the common European ones so a TOC in a Turkish
# or German manual is recognized the same way.
_TOC_HEADING_RE = re.compile(
    r"^((table of )?contents|i[çc]indekiler|inhalt(sverzeichnis)?|sommaire"
    r"|table des mati[eè]res|[ií]ndice( general| de contenidos?)?|contenidos?)$",
    re.IGNORECASE)


def _toc_entry(text: str) -> tuple[str, int] | None:
    """(title, page) parsed from one flattened TOC line, or None if `text`
    doesn't end in a bare page number at all (the shape every GLUED TOC line
    in this corpus shares, whether it arrived as a numbered list item --
    "2.3. Relay Type 2", its own "2.3." marker already stripped by
    `_BULLET_RE` -- or a dot-leader paragraph with no marker at all --
    "DESCRIPTION .......................... 2"). See `_read_toc_entry` for
    the other physical shape a TOC line can take."""
    m = re.search(r"(\d+)\s*$", text)
    if not m:
        return None
    title = text[:m.start()].rstrip()
    title = re.sub(r"[.\s]{2,}$", "", title).rstrip()
    if not title:
        return None
    return title, int(m.group(1))


def _bare_page_number(text: str) -> int | None:
    m = re.match(r"^\s*(\d{1,4})\s*$", text)
    return int(m.group(1)) if m else None


def _read_toc_entry(blocks: list[Block], j: int
                    ) -> tuple[dict[str, Any], int] | None:
    """One TOC entry starting at `blocks[j]`, as ({"title", "page", "level"},
    blocks consumed), or None if `blocks[j]` doesn't open either physical
    shape a TOC line takes in this corpus:

    (a) glued -- one ParagraphBlock whose own trailing digits are the page
        number (`_toc_entry`; the code route's shape, and most hybrid pages').
    (b) split -- a numbered heading/paragraph title immediately followed by
        a SEPARATE block holding nothing but its page number. Seen on hybrid
        pages whose printed TOC rules off each row with a horizontal line
        instead of a dot leader (title ... big gap ... right-aligned page
        number): visually indistinguishable from a two-column table row, so
        a VLM reading the page image (unlike the code route, which glues
        every word on one physical line into a single string regardless of
        the gap between them) reads title and page number as two entries.
        Consumes 2 blocks; the title is stored number-stripped (`strip_num`)
        to match shape (a)'s convention, and its level comes from the
        number's own depth rather than list nesting."""
    b = blocks[j]
    if isinstance(b, ParagraphBlock):
        entry = _toc_entry(b.text)
        if entry is not None:
            title, page = entry
            return {"title": title, "page": page,
                    "level": (b.list_level or 0) + 1}, 1
    if isinstance(b, (HeadingBlock, ParagraphBlock)) and j + 1 < len(blocks):
        path = num_path(b.text)
        if path and _bare_page_number(b.text) is None:
            nb = blocks[j + 1]
            if isinstance(nb, ParagraphBlock):
                page = _bare_page_number(nb.text)
                if page is not None:
                    return {"title": strip_num(b.text), "page": page,
                            "level": min(len(path), 6)}, 2
    return None


def _extract_toc(blocks: list[Block]) -> tuple[list[Block], list[dict[str, Any]]]:
    """Pulls a table-of-contents run out of `blocks` into structured
    `metadata["toc"]` entries, so a line like
    "2.3. Relay Type 2" doesn't sit in `blocks` indistinguishable from the
    real "Relay Type" section heading elsewhere in the document -- retrieval
    over `blocks` would otherwise have no way to tell the two apart, and the
    glued-on page number is pure noise either way.

    Needs no page-position or font heuristic: a genuine TOC run is the only
    place consecutive entries' page numbers hold non-decreasing across a
    whole run (1, 1, 1, 2, 2, 3, ...) in reading order -- true across both
    physical shapes an entry can take (see `_read_toc_entry`), and true even
    when the run itself crosses a page-route boundary (one contiguous printed
    TOC spanning a code-route page then hybrid-route pages, each shaped
    differently). A run under 2 entries is left alone (mirrors the
    "a real list has >=1 sibling" rule) since a single coincidental
    match is far more likely than a one-line TOC.

    The "Contents" heading itself is kept -- only the noisy entries under it
    are removed -- so the document's own section marker still reads naturally
    in place."""
    out: list[Block] = []
    toc: list[dict[str, Any]] = []
    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]
        out.append(b)
        if not (isinstance(b, HeadingBlock)
                and _TOC_HEADING_RE.match(b.text.strip())):
            i += 1
            continue
        run: list[dict[str, Any]] = []
        kept: list[Block] = []
        last_page = -1
        j = i + 1
        skips = 0
        while j < n:
            result = _read_toc_entry(blocks, j)
            if result is None or result[0]["page"] < last_page:
                # A stray block interrupting an established run (a revision
                # note pasted between TOC columns) doesn't end it: up to two
                # consecutive non-entry blocks are stepped over -- kept as
                # ordinary blocks -- and the run resumes if entries with
                # still-non-decreasing pages follow. Without this the run's
                # tail survives as flattened list noise in `blocks`.
                if run and skips < 2:
                    kept.append(blocks[j])
                    skips += 1
                    j += 1
                    continue
                break
            entry, consumed = result
            run.append(entry)
            last_page = entry["page"]
            skips = 0
            j += consumed
        if len(run) >= 2:
            toc.extend(run)
            out.extend(kept)
            i = j
        else:
            i += 1
    return out, toc


# A line starting like a bullet / numbered item. The marker is stripped from
# the text; membership goes to list_* metadata (see base.py, "Lists"). The
# dotted-numbering branch (`\d{1,3}(?:\.\d{1,3}){1,4}\.?`) matches multi-level
# numbering like "1.1 " or "2.3.4 " without requiring trailing punctuation --
# the internal dots between digit groups are themselves the signal, mirroring
# heading_heuristics._NUM_PREFIX (used for the same shape in docx headings).
# A bare single number still needs trailing "." or ")" (`\d{1,3}[.)]`) since
# nothing else marks it as a list item rather than the start of a sentence.
_BULLET_RE = re.compile(
    r"^\s*([•◦▪‣●○∙·*–—-]"
    r"|\d{1,3}(?:\.\d{1,3}){1,4}\.?"
    r"|\d{1,3}[.)]"
    r"|[A-Za-z][.)])\s+")


@dataclass
class _Spec:
    """One prospective text block before Block/id assignment."""

    kind: str            # "heading" | "paragraph" | "item"
    text: str
    top: float
    bottom: float = 0.0
    x0: float = 0.0
    level: int = 1       # heading level
    list_group: int = -1
    list_level: int = 0
    list_ordered: bool = False
    runs: list[InlineRun] = field(default_factory=list)


def _is_heading(line: _Line, body_size: float) -> bool:
    text = line.text.strip()
    if not text or len(line.text.split()) > 14 or CAPTION.match(text):
        return False
    if num_path(text) and not text.endswith((".", ";", ",", ":")):
        # A numbered section title ("5.4.2.1. Controlling Cells") is heading-
        # shaped by its numbering alone, independent of size -- a deep
        # sub-heading is often set barely larger than body text, well under
        # the 1.15x an unnumbered heading needs to stand out on looks alone
        # (see ACME_PN5057_User_Manual.pdf p.20: 10pt heading over 9pt body,
        # ratio 1.111, missed the 1.15x bar by a hair and fell through to
        # _BULLET_RE's list-item branch, stripping the very number that
        # would have identified it as a heading). The trailing-punctuation
        # exclusion keeps this from ever claiming an ordinary numbered list
        # sentence ("1. Do this.") that merely renders a touch larger --
        # section titles don't end in sentence punctuation, list items do.
        # A real, if modest, gap is still required -- exact body size stays
        # excluded -- so this never engulfs a plain sentence at the same size.
        return line.size > body_size and line.size >= body_size * 1.05
    return line.size >= body_size * 1.15 and line.size >= body_size + 0.4


def _heading_candidate_sizes(lines: list[_Line], body: float) -> set[float]:
    return {round(l.size, 1) for l in lines if _is_heading(l, body)}


def _specs_from_lines(lines: list[_Line],
                      heading_sizes: list[float] | None = None
                      ) -> list[_Spec]:
    """Classify visual lines into heading / paragraph / list-item specs.

    `heading_sizes` (sorted, descending, distinct sizes -> level 1..6) ranks
    across the whole document by default (see `_doc_heading_sizes`), not just
    this call's own lines, so the same physical font size gets the same level
    everywhere -- a chapter heading on page 50 isn't demoted just because
    page 50 alone has no bigger text on it. Recomputed from just `lines` when
    omitted (e.g. standalone use)."""
    if not lines:
        return []
    body = median(l.size for l in lines)
    if heading_sizes is None:
        heading_sizes = sorted(_heading_candidate_sizes(lines, body),
                               reverse=True)

    specs: list[_Spec] = []
    para: list[_Line] = []
    head_buf: list[_Line] = []
    prev: _Line | None = None
    group = -1

    def flush_para() -> None:
        nonlocal para
        if para:
            runs: list[InlineRun] = []
            for l in para:
                runs = _concat_runs(runs, l.runs)
            specs.append(_Spec(kind="paragraph",
                               text=" ".join(l.text for l in para),
                               top=para[0].top, bottom=para[-1].bottom,
                               x0=min(l.x0 for l in para), runs=runs))
            para = []

    def flush_head() -> None:
        """A heading-sized line rarely stands alone -- a wrapped title spans
        2-3 lines just like a paragraph does, and an oversized body sentence
        (wrong font size on ordinary prose) also spans lines the same way.
        Both look identical line-by-line, so lines are buffered here the same
        way `para` buffers paragraph continuations, and only classified once
        the whole run is known.

        A run ending in sentence punctuation is prose that happened to be set
        in a heading-sized font, not a title -- checked unconditionally, even
        for a single line, since a misfont sentence's own word-count cap
        (`_is_heading`, <=14 words/line) often fails its longer neighbor
        lines first, leaving only the sentence's last, short fragment
        heading-sized and alone (see ACME_PN5015_Datasheet.pdf p.6: two
        16-word lines of a sentence fall to `para` on the word-count cap,
        but the trailing 9-word fragment "...applications." doesn't, and
        would stay a lone false heading without this check). A merged run
        longer than a real title (>14 words total) is demoted the same way,
        but only once >1 line has actually merged -- a single already-short
        line can never trip that length check on its own."""
        nonlocal head_buf
        if not head_buf:
            return
        text = " ".join(l.text.strip() for l in head_buf)
        runs: list[InlineRun] = []
        for l in head_buf:
            runs = _concat_runs(runs, l.runs)
        demote = (text.rstrip().endswith((".", ";", ","))
                 or (len(head_buf) > 1 and len(text.split()) > 14))
        if demote:
            specs.append(_Spec(kind="paragraph", text=text,
                               top=head_buf[0].top, bottom=head_buf[-1].bottom,
                               x0=min(l.x0 for l in head_buf), runs=runs))
        else:
            level = min(6, heading_sizes.index(round(head_buf[0].size, 1)) + 1)
            specs.append(_Spec(kind="heading", text=text,
                               top=head_buf[0].top, bottom=head_buf[-1].bottom,
                               x0=head_buf[0].x0, level=level, runs=runs))
        head_buf = []

    for line in lines:
        gap = line.top - prev.bottom if prev is not None else 0.0
        if line.label:
            # A split-off callout label (see `_split_label_group`): its own
            # block, never a continuation of the surrounding prose -- the
            # whole point of the split was to stop "Caution" being glued
            # into the sentence beside it.
            flush_head()
            flush_para()
            specs.append(_Spec(kind="paragraph", text=line.text,
                               top=line.top, bottom=line.bottom,
                               x0=line.x0, runs=line.runs))
        elif _is_heading(line, body):
            if (head_buf and gap <= 0.6 * body
                    and abs(line.size - head_buf[-1].size) <= 0.6):
                head_buf.append(line)  # same-size wrapped continuation
            else:
                flush_para()
                flush_head()
                head_buf = [line]
        elif (m := _BULLET_RE.match(line.text)) or line.bullet_len:
            flush_head()
            flush_para()
            if not (specs and specs[-1].kind == "item") or gap > 1.6 * body:
                group += 1
            # A textual marker (`_BULLET_RE`, e.g. "1." / "-") pins ordered-ness
            # by its own shape; a symbol-font bullet glyph (`bullet_len`, its
            # decoded letter meaningless) is always unordered.
            if m:
                cut, ordered = m.end(), m.group(1)[0].isalnum()
            else:
                cut, ordered = line.bullet_len, False
            specs.append(_Spec(kind="item", text=line.text[cut:].strip(),
                               top=line.top, x0=line.x0, list_group=group,
                               list_ordered=ordered,
                               runs=_slice_runs(line.runs, cut)))
        elif (specs and specs[-1].kind == "item" and not para and not head_buf
                and line.x0 > specs[-1].x0 + 1 and gap <= 0.9 * body):
            specs[-1].text += " " + line.text  # wrapped continuation of an item
            specs[-1].runs = _concat_runs(specs[-1].runs, line.runs)
        else:
            flush_head()
            if para and gap > 0.6 * body:
                flush_para()
            para.append(line)
        prev = line
    flush_head()
    flush_para()

    # List levels: within one contiguous list run, indent buckets -> depth.
    for g in {s.list_group for s in specs if s.kind == "item"}:
        xs: list[float] = []
        for s in specs:
            if (s.kind == "item" and s.list_group == g
                    and not any(abs(s.x0 - x) <= 4 for x in xs)):
                xs.append(s.x0)
        xs.sort()
        for s in specs:
            if s.kind == "item" and s.list_group == g:
                s.list_level = next(i for i, x in enumerate(xs)
                                    if abs(s.x0 - x) <= 4)
    return specs


def _code_columns(prep: dict, extra_boxes: tuple[tuple[float, float, float, float], ...] = ()
                  ) -> list[list[dict]]:
    """Page words (minus table regions, plus any caller-supplied `extra_boxes`
    -- e.g. a recovered table-header row, see `_recover_table_headers`) split
    into gutter columns -- the same partition `_parse_code` builds blocks
    from, factored out so the document-wide heading-size pass
    (`_doc_heading_sizes`) sees exactly the lines the real pass will."""
    tboxes = _table_boxes(prep) + list(extra_boxes)
    words = [w for w in (prep.get("words") or []) if not _in_boxes(w, tboxes)]
    gutter = prep.get("gutter")
    if gutter is None:
        return [words]
    return list(_split_gutter_groups(_group_lines(words), gutter))


def _is_margin_noise(l: _Line, page_height: float,
                     running_lines: frozenset[str] | None) -> bool:
    """A line to drop before block classification: a running header/footer
    (exact text repeats document-wide, see `detect_running_lines`) or a page
    number (own shape marks it, see `looks_like_page_number` -- its text
    differs every page, so it could never satisfy the repeat check). Both
    only ever fire inside the page-margin band, so real content elsewhere on
    the page is never touched by either."""
    if not in_margin_band(l.top, l.bottom, page_height):
        return False
    return looks_like_page_number(l.text) or bool(
        running_lines and normalize(l.text) in running_lines)


def _page_heading_candidates(prep: dict, page_height: float = 0.0,
                             running_lines: frozenset[str] | None = None
                             ) -> set[float]:
    """Candidate heading font sizes for one page, across its gutter columns
    -- the same partition/threshold/exclusions `_parse_code` builds blocks
    from (recovered table-header words via `prep["header_boxes"]`, and
    margin noise via `_is_margin_noise`), reused so the document-wide ranking
    (see `PdfParser.parse`) sees exactly the lines the real pass will
    encounter. Any mismatch here -- a word or line excluded in one pass but
    not the other -- can shift the local median (`body`) enough that a line
    crosses the heading threshold in the real pass without its size ever
    having been ranked, crashing `_specs_from_lines`'s
    `heading_sizes.index(...)` lookup."""
    sizes: set[float] = set()
    header_boxes = tuple(b for b in (prep.get("header_boxes") or ()) if b is not None)
    for cwords in _code_columns(prep, header_boxes):
        lines = [l for l in _cluster_lines(cwords)
                if not _is_margin_noise(l, page_height, running_lines)]
        if not lines:
            continue
        sizes |= _heading_candidate_sizes(lines, median(l.size for l in lines))
    return sizes


def _page_margin_lines(prep: dict) -> list[tuple[float, float, str]]:
    """(top, bottom, text) for every visual line on one page, table-interior
    words already excluded. Pooled across both gutter columns (unlike
    `_code_columns`) since running header/footer text spans the full page
    width regardless of the body's two-column layout -- splitting it by
    gutter first would just cut a repeating line in two and hide the
    repetition `detect_running_lines` is looking for."""
    tboxes = _table_boxes(prep)
    words = [w for w in (prep.get("words") or []) if not _in_boxes(w, tboxes)]
    return [(l.top, l.bottom, l.text) for l in _cluster_lines(words)]


def _in_boxes(w: dict, boxes: list[tuple]) -> bool:
    cx = (w["x0"] + w["x1"]) / 2
    cy = (w["top"] + w["bottom"]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _table_boxes(prep: dict) -> list[tuple]:
    """Bboxes of every region already claimed elsewhere: a real table, OR a
    region diverted to a non-table visual by classification (see
    `PdfParser._build_region_tables`). Every text-reconstruction pass that
    excludes "table-interior" words from its own output (`_code_columns`,
    `_page_margin_lines`, `_recover_dropped_headings`, `_word_marks_stream`)
    must exclude a diverted region's words too, or a block diagram's own
    labels (e.g. the repeated "AAF" boxes) leak back in as a stray
    paragraph/heading once the region stops being a table."""
    return ([t.bbox for t in (prep.get("tables") or [])]
            + [d["bbox"] for d in (prep.get("diverted_images") or [])])


def _span(vals: list[float], i0: int, hi: float, tol: float = 1.0) -> int:
    """How many grid steps (starting at i0) a cell's far edge `hi` covers,
    given the sorted grid boundary positions `vals`."""
    n = 1
    while i0 + n < len(vals) and vals[i0 + n] < hi - tol:
        n += 1
    return n


def _table_to_data(table) -> TableData:
    """pdfplumber Table -> TableData, reconstructing merges from cell geometry.

    pdfplumber gives no rowspan/colspan directly, but a merged cell shows up
    as one bbox spanning multiple grid boundaries (`table.cells`), while the
    grid positions it covers have no bbox of their own (`table.rows[r].cells`
    is None there) -- exactly the signal needed to rebuild `Merge` entries.
    A grid position with no bbox that *no* other cell's span covers is a
    genuine borderless/blank cell, not a merge, and stays an empty `Cell`.
    """
    raw_cells = table.cells
    if not raw_cells:
        return TableData(n_rows=0, n_cols=0, cells=[])

    xs = sorted({c[0] for c in raw_cells})
    tops = sorted({c[1] for c in raw_cells})
    nrows, ncols = len(tops), len(xs)

    chars = table.page.chars
    grid: list[list] = [[None] * ncols for _ in range(nrows)]
    covered: set[tuple[int, int]] = set()
    merges: list[Merge] = []

    for r, row in enumerate(table.rows):
        row_chars = [ch for ch in chars
                     if row.bbox[1] <= (ch["top"] + ch["bottom"]) / 2 < row.bbox[3]]
        for c, bbox in enumerate(row.cells):
            if bbox is None:
                continue
            colspan = _span(xs, c, bbox[2])
            rowspan = _span(tops, r, bbox[3])
            cell_chars = [ch for ch in row_chars
                          if bbox[0] <= (ch["x0"] + ch["x1"]) / 2 < bbox[2]]
            text = pdfplumber.utils.extract_text(cell_chars) if cell_chars else ""
            grid[r][c] = text_cell(text)
            if rowspan > 1 or colspan > 1:
                merges.append(Merge(row=r, col=c, rowspan=rowspan, colspan=colspan))
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if dr or dc:
                            covered.add((r + dr, c + dc))

    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] is None and (r, c) not in covered:
                grid[r][c] = text_cell("")

    return TableData(n_rows=nrows, n_cols=ncols, cells=grid, merges=merges)


@dataclass
class _PreparedTable:
    """A table whose `TableData` is already fully built (by the band builder or
    a structure-model referee), bypassing `_table_to_data`'s pdfplumber-geometry
    reconstruction. `bbox` is in the same PDF-point coordinate space as
    pdfplumber's own `Table.bbox`, so the existing reading-order code in
    `_parse_code`/`_blocks_from_vlm` that reads `.bbox` works on either kind
    of table without change.

    `provenance` records how the grid was built (`PROV_TABLE_BANDS` /
    `PROV_TABLE_STRUCTURE`); `confidence` (0..1) and `flags` carry the builder's
    self-assessment through to the emitted `TableBlock` for transparency."""

    bbox: tuple[float, float, float, float]
    data: TableData
    provenance: str = PROV_TABLE_STRUCTURE
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)


def _as_table_data(t) -> TableData:
    """`t` is either a pdfplumber `Table` (needs reconstruction) or a
    `_PreparedTable` (already built) -- the single seam _parse_code and
    _blocks_from_vlm both go through, so neither cares which detector a
    table came from."""
    return t.data if isinstance(t, _PreparedTable) else _table_to_data(t)


def _char_center(ch) -> tuple[float, float]:
    return (ch["x0"] + ch["x1"]) / 2.0, (ch["top"] + ch["bottom"]) / 2.0


def _rect_contains(bbox: tuple[float, float, float, float], cx: float, cy: float) -> bool:
    return bbox[0] <= cx < bbox[2] and bbox[1] <= cy < bbox[3]


def _rect_overlap_area(a: tuple[float, float, float, float],
                       b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def _best_overlap_index(region_pt: tuple[float, float, float, float],
                        geom_tables: list) -> int | None:
    """Index of the geom region whose bbox overlaps `region_pt` most, or None.
    Maps a model's returned table (which echoes the region bbox it was given)
    back to its input region slot without relying on positional order -- the
    adapter may skip regions, so order alone is not a safe key."""
    best_i, best_area = None, 0.0
    for i, t in enumerate(geom_tables):
        area = _rect_overlap_area(region_pt, tuple(t.bbox))
        if area > best_area:
            best_i, best_area = i, area
    return best_i


# Minimum fraction of a finished table's own (non-merge-covered) cells that must
# carry alphanumeric text for it to ship AS A TABLE. Below this the region is a
# false-positive detection -- an empty ruled box, or a drawing whose frame
# closed into a coarse grid -- worth dropping.
_TABLE_MIN_FILL = 0.15


def _table_worth_emitting(data: TableData) -> bool:
    """Whether a finished grid carries enough tabular content to ship as a table.

    A region that fails this is NOT lost: dropped from the page's `tables`
    list, its words stop being excluded from the ordinary text flow (every
    exclusion box is derived from that same list -- see `_word_marks_stream`
    and `_parse_code`) and resurface as normal paragraphs. That makes dropping
    strictly text-preserving, so it's safe to reject the corpus's three
    recurring false positives here:

      * all-empty ruled box (no cell carries text at all);
      * single-column strip that is a wrapped run of PROSE -- a paragraph
        mis-boxed as a 1-wide "table" (a real ruled single-column table of
        short labels is legitimate and kept; only flowing multi-word text is
        reflowed);
      * large sparse grid -- a technical drawing whose frame closed into a grid
        holding only a few scattered callouts.
    """
    if data.n_rows <= 0 or data.n_cols <= 0:
        return False
    total = filled = 0
    word_counts: list[int] = []
    for row in data.cells:
        for c in row:
            if c is None:
                continue
            total += 1
            txt = c.plain_text()
            if _has_alnum(txt):
                filled += 1
                word_counts.append(len(txt.split()))
    if filled == 0:
        return False  # all-empty ruled box
    if data.n_cols == 1:
        # One column conveys no row/column relationship. Keep it only if it
        # reads as a compact list of short entries (a real single-column
        # table); drop a wrapped paragraph (few rows, many words per cell) so
        # it reflows as text.
        mean_words = sum(word_counts) / len(word_counts)
        return not (len(word_counts) >= 3 and mean_words >= 2.0)
    return filled / total >= _TABLE_MIN_FILL


def _apply_health(pt: _PreparedTable, words: list[dict]) -> None:
    """Source-agnostic sanity pass on a finished table, whatever built it.
    Never changes cell TEXT; only lowers `confidence` and adds `flags` so a
    lossy or oddly-shaped grid is visible downstream (and to a human review):

      * content-dropout -- content words inside the table's region that appear
        in NO cell. This is the metric that actually matters ("did we lose
        text?"), and being blind to which builder ran it catches a bad grid a
        per-builder gate might have passed.
      * sparse-col0 -- a large share of body rows have an empty first cell while
        the rest of the row has content: the exact signature of the old lattice
        column-0 dropout, worth flagging wherever it recurs, regardless of source.
    """
    data = pt.data
    x0, top, x1, bot = pt.bbox
    region_toks: Counter = Counter()
    for w in words:
        if not w["text"].strip():
            continue
        cx = (w["x0"] + w["x1"]) / 2.0
        cy = (w["top"] + w["bottom"]) / 2.0
        if x0 <= cx <= x1 and top <= cy <= bot:
            for tok in _norm(w["text"]).split():
                if _has_alnum(tok):
                    region_toks[tok] += 1
    cell_toks: Counter = Counter()
    for row in data.cells:
        for c in row:
            if c is None:
                continue
            for tok in _norm(c.plain_text()).split():
                if _has_alnum(tok):
                    cell_toks[tok] += 1
    total = sum(region_toks.values())
    dropout = sum((region_toks - cell_toks).values()) / total if total else 0.0

    flags = list(pt.flags)
    if dropout > 0.05:
        flags.append("content-dropout")
    if _sparse_first_column(data):
        flags.append("sparse-col0")
    seen: set[str] = set()  # dedupe, preserve order
    pt.flags = [f for f in flags if not (f in seen or seen.add(f))]
    if pt.confidence is not None:
        pt.confidence = round(min(pt.confidence, 1.0 - dropout), 3)


def _sparse_first_column(data: TableData, min_frac: float = 0.3) -> bool:
    """A large share of body rows (header excluded) have no content in column 0
    while the rest of the row does -- the lattice column-0 dropout signature."""
    rows = data.cells[1:] if data.n_rows > 1 else data.cells
    considered = empty = 0
    for row in rows:
        if not row:
            continue
        first_has = row[0] is not None and _has_alnum(row[0].plain_text())
        rest_has = any(c is not None and _has_alnum(c.plain_text()) for c in row[1:])
        if rest_has:
            considered += 1
            if not first_has:
                empty += 1
    return considered > 0 and empty / considered >= min_frac


def _fill_cells_by_containment(chars: list, dt, to_pt: float) -> tuple[TableData, float]:
    """Build a `TableData` for a structure model's detected grid `dt`, placing
    each cell's text from the PDF's own characters `chars` (pdfplumber dicts,
    PDF-point space) by CENTER-CONTAINMENT. The model decides only the grid
    SHAPE; the text is always real PDF text, never the model's own reading --
    the same guarantee as `_table_to_data`, applied to the model's grid.

    Why containment and not an exact text-clip at each cell's bbox: a general
    VLM gets the grid topology (row/col/merges) right but its per-cell boxes
    are not pixel-exact, so an exact clip lands beside the glyphs and returns
    nothing (the "structure correct, cells empty" failure). Center-containment
    is tolerant -- a character belongs to whichever cell its centre falls in.

    Returns `(data, coverage)`, where `coverage` is the fraction of the
    region's real (non-blank) characters that landed in some cell. The caller
    reads a low coverage as "this grid is too wrong to trust" and falls back
    to pdfplumber's geometry -- so a badly misplaced grid degrades to the
    line-geometry table instead of emitting a half-empty one.
    """
    n_rows = max(c.row + c.rowspan for c in dt.cells)
    n_cols = max(c.col + c.colspan for c in dt.cells)
    region_pt = tuple(v * to_pt for v in dt.bbox)
    region_chars = [ch for ch in chars
                    if ch.get("text", "").strip()
                    and _rect_contains(region_pt, *_char_center(ch))]

    cells: list[list[Any]] = [[None] * n_cols for _ in range(n_rows)]
    merges: list[Merge] = []
    placed: set[int] = set()
    for c in dt.cells:
        bbox_pt = tuple(v * to_pt for v in c.bbox)
        cell_chars = []
        for i, ch in enumerate(region_chars):
            if _rect_contains(bbox_pt, *_char_center(ch)):
                cell_chars.append(ch)
                placed.add(i)
        text = pdfplumber.utils.extract_text(cell_chars) if cell_chars else ""
        cells[c.row][c.col] = text_cell(text)
        if c.rowspan > 1 or c.colspan > 1:
            merges.append(Merge(row=c.row, col=c.col,
                                rowspan=c.rowspan, colspan=c.colspan))

    covered = {(c.row + dr, c.col + dc)
               for c in dt.cells
               for dr in range(c.rowspan) for dc in range(c.colspan)
               if dr or dc}
    for r in range(n_rows):
        for cc in range(n_cols):
            if cells[r][cc] is None and (r, cc) not in covered:
                cells[r][cc] = text_cell("")

    coverage = 1.0 if not region_chars else len(placed) / len(region_chars)
    return TableData(n_rows=n_rows, n_cols=n_cols, cells=cells, merges=merges), coverage


def _match_norm_tok(a: str, b: str) -> bool:
    """Normalized token equality, tolerating edge-punctuation differences
    between two transcriptions of the same word ("GND," vs "GND")."""
    if a == b:
        return True
    a2, b2 = a.strip(_PUNCT_EDGES), b.strip(_PUNCT_EDGES)
    return bool(a2) and a2 == b2


def _is_hard_number(tok: str) -> bool:
    """A claimed token substantial enough that inventing it is a hard reject:
    at least two characters AND containing a digit (e.g. "6V", "264VAC",
    "2026"). A lone digit or a single footnote marker ("1", or "¹" folded to
    "1") is deliberately excluded -- too weak to condemn a whole grid on, and
    after glyph-folding a superscript footnote marker would otherwise trip it.
    A real fabricated value (a wrong measurement, a wrong part number) is
    almost always multi-character, so this keeps the hallucination guard while
    dropping the false positives that were rejecting good grids."""
    return len(tok) >= 2 and any(ch.isdigit() for ch in tok)


def _fill_cells_by_content(words: list[dict], dt, to_pt: float,
                           lookahead: int = 30
                           ) -> tuple[TableData, float, int]:
    """Build a `TableData` for a structure model's detected grid `dt` whose
    cells carry the model's own CLAIMED text (`DetectedCell.text`), by
    matching each claim against the region's real text-layer `words`
    (pdfplumber dicts, PDF-point space) -- content-anchored placement for
    models whose pixel boxes can't be trusted (a general VLM).

    The model decides only WHICH words belong together in a cell; every
    emitted cell is rebuilt from the PDF's own words, so the model's string
    never enters the output -- a misread there costs the match, not fidelity.
    Matching is reading-order constrained (cells row-major, words in visual
    line order) with a bounded lookahead, so a token like "GND" repeating in
    ten cells resolves to its next occurrence, not an arbitrary one; a second
    pass retries leftovers anywhere unconsumed (wrapped multi-line cells trail
    their row in the word stream), including run-concatenation fallbacks for
    tokenization drift between the two sources ("5VDC,1A" vs "5 V DC, 1 A").

    Returns `(data, placed, invented)`:
      placed   -- fraction of the region's real words that landed in some
                  cell. Every unplaced word is REAL text the grid would drop,
                  so the caller reads a low value as "this grid loses data"
                  and falls back to line geometry. Unlike the old
                  center-containment coverage this can't be gamed by a
                  misaligned grid: a word only counts as placed by matching a
                  cell's claimed content, not by falling inside some box.
      invented -- count of digit-bearing claimed tokens that matched nothing
                  real even after fallbacks: the model asserted a number the
                  region doesn't contain. Any invention is a hard reject at
                  the call site -- hallucinated numbers must never be able to
                  shape a grid that then looks trustworthy.
    """
    n_rows = max(c.row + c.rowspan for c in dt.cells)
    n_cols = max(c.col + c.colspan for c in dt.cells)
    region_pt = tuple(v * to_pt for v in dt.bbox)
    region_words = [w for w in words
                    if w["text"].strip()
                    and _rect_contains(region_pt, (w["x0"] + w["x1"]) / 2,
                                       (w["top"] + w["bottom"]) / 2)]
    stream = [w for line in _group_lines(region_words) for w in line]
    norms = [_norm(w["text"]) for w in stream]
    consumed = [False] * len(stream)

    ordered = sorted(dt.cells, key=lambda c: (c.row, c.col))
    matched: dict[tuple[int, int], list[int]] = {(c.row, c.col): [] for c in ordered}
    pending: list[tuple[tuple[int, int], str]] = []  # (cell key, claim tok)

    # Pass 1: forward, order-constrained (duplicate-safe fast path).
    pos = 0
    for c in ordered:
        key = (c.row, c.col)
        for tok in _norm(c.text or "").split():
            hit = None
            for j in range(pos, min(len(stream), pos + lookahead)):
                if not consumed[j] and _match_norm_tok(tok, norms[j]):
                    hit = j
                    break
            if hit is None:
                pending.append((key, tok))
            else:
                consumed[hit] = True
                matched[key].append(hit)
                pos = hit + 1

    # Pass 2: leftovers may match anywhere unconsumed (wrapped-cell text sits
    # after its row in the stream), then a run-concatenation fallback for
    # claim tokens the text layer split into several words.
    still: list[tuple[tuple[int, int], str]] = []
    for key, tok in pending:
        hit = next((j for j in range(len(stream))
                    if not consumed[j] and _match_norm_tok(tok, norms[j])), None)
        if hit is not None:
            consumed[hit] = True
            matched[key].append(hit)
            continue
        run = _find_concat_run(tok, norms, consumed)
        if run is not None:
            for j in run:
                consumed[j] = True
            matched[key].extend(run)
        else:
            still.append((key, tok))

    # Final fallback per cell, other direction (one glued text-layer word
    # covering several claim tokens): the cell's whole claim, spaces removed,
    # against a run of unconsumed words with spaces removed.
    still_by_cell: dict[tuple[int, int], list[str]] = {}
    for key, tok in still:
        still_by_cell.setdefault(key, []).append(tok)
    invented = 0
    for key, toks in still_by_cell.items():
        run = _find_concat_run("".join(toks), norms, consumed)
        if run is not None:
            for j in run:
                consumed[j] = True
            matched[key].extend(run)
        else:
            invented += sum(1 for t in toks if _is_hard_number(t))

    cells: list[list[Any]] = [[None] * n_cols for _ in range(n_rows)]
    merges: list[Merge] = []
    for c in ordered:
        idxs = sorted(matched[(c.row, c.col)])
        cells[c.row][c.col] = text_cell(
            " ".join(stream[j]["text"] for j in idxs))
        if c.rowspan > 1 or c.colspan > 1:
            merges.append(Merge(row=c.row, col=c.col,
                                rowspan=c.rowspan, colspan=c.colspan))

    covered = {(c.row + dr, c.col + dc)
               for c in ordered
               for dr in range(c.rowspan) for dc in range(c.colspan)
               if dr or dc}
    for r in range(n_rows):
        for cc in range(n_cols):
            if cells[r][cc] is None and (r, cc) not in covered:
                cells[r][cc] = text_cell("")

    # `placed` is measured over CONTENT words only (those carrying a letter or
    # digit). A punctuation-only word -- a lone dash "-"/"―" meaning "n/a", a
    # stray bullet -- left unmatched must not drag the score down and reject an
    # otherwise-complete grid: a model that renders a dash cell as "" or as a
    # different dash glyph is a cosmetic miss, not lost data, and the dominant
    # ACME spec tables are full of such dashes. Dropped *content* is what a low
    # score should mean, so only content words count toward it.
    content = [i for i in range(len(stream)) if _has_alnum(norms[i])]
    placed = 1.0 if not content else sum(consumed[i] for i in content) / len(content)
    return (TableData(n_rows=n_rows, n_cols=n_cols, cells=cells, merges=merges),
            placed, invented)


def _find_concat_run(target: str, norms: list[str], consumed: list[bool],
                     max_run: int = 6) -> list[int] | None:
    """Indices of the first run of consecutive unconsumed words whose
    concatenated normalized text equals `target` (itself normalized,
    spaces removed), or None. Tolerates edge punctuation on the full run."""
    target = target.replace(" ", "")
    if not target:
        return None
    for i in range(len(norms)):
        if consumed[i]:
            continue
        acc = ""
        run: list[int] = []
        for j in range(i, min(len(norms), i + max_run)):
            if consumed[j]:
                break
            acc += norms[j]
            run.append(j)
            if len(acc) >= len(target):
                break
        if _match_norm_tok(target, acc) and len(run) > 1:
            return run
    return None


# A table's header row is sometimes set in a shaded band with no ruling line
# directly above it, so find_tables()'s line-geometry grid doesn't include it
# -- the row's words then sit just outside the table bbox as ordinary page
# text (ACME_PN5036_Datasheet.pdf p.4: "Symbol Min Typ Max Units" surfacing
# as a stray heading instead of the table's own column labels).
# `_HEADER_GAP_MAX` is generous (the shaded band + its own padding easily
# exceeds one line height) since the token-count match below
# carries the real precision; `_HEADER_X_TOL` only absorbs sub-pixel/rounding
# slack, not genuine misalignment.
_HEADER_GAP_MAX = 30.0
_HEADER_X_TOL = 3.0


def _recover_table_headers(words: list[dict], tables: list,
                           table_data: list[TableData]
                           ) -> list[tuple[float, float, float, float] | None]:
    """Recover a missing header row into each of `table_data` (same order as
    `tables`) in place, conservatively: within `_HEADER_GAP_MAX` above the
    table (roughly within its column span), scan visual lines nearest-first
    and take the first whose whitespace-separated token count exactly equals
    the table's column count -- an unrelated caption, paragraph, or (as in
    ACME_PN5036_Datasheet.pdf p.4) a row-group label sitting *between*
    the true header and the table's bbox never has the right token count by
    coincidence often enough to matter, so nearest-first with an exact-count
    gate finds the real header past such lines instead of stopping at them.
    Deliberately does not try to align individual tokens to columns by
    x-position -- an exact count match already pins each token to its column
    by position in the row (leftmost token -> leftmost column, ...).

    Returns one entry per table, in the same order: the matched line's
    bounding box when a row was recovered, else None -- so the caller can
    both exclude those words from the ordinary text flow (otherwise they'd
    *also* surface as a stray heading/paragraph) and use the recovered box's
    top for the table's reading-order position.
    """
    other_boxes = [t.bbox for t in tables]
    out: list[tuple[float, float, float, float] | None] = []
    for t, data in zip(tables, table_data):
        if isinstance(t, _PreparedTable):
            # A prepared table already has its final grid, header included (the
            # band builder pulls a shaded-band header in as row 0; the model
            # referee returns the whole grid). Re-running the text-flow header
            # recovery here would insert a duplicate row, so skip it -- its
            # reading-order position comes from the region bbox (header-inclusive).
            out.append(None)
            continue
        x0, top, x1, _ = t.bbox
        strip = [w for w in words
                if not _in_boxes(w, other_boxes)
                and top - _HEADER_GAP_MAX <= (w["top"] + w["bottom"]) / 2 < top
                and w["x0"] >= x0 - _HEADER_X_TOL and w["x1"] <= x1 + _HEADER_X_TOL]
        lines = sorted(_cluster_lines(strip), key=lambda l: -l.top)  # nearest table first
        match = next((l for l in lines if len(l.text.split()) == data.n_cols), None)
        if match is None:
            out.append(None)
            continue
        tokens = match.text.split()
        data.cells.insert(0, [text_cell(tok) for tok in tokens])
        data.n_rows += 1
        for m in data.merges:
            m.row += 1
        data.header_rows = 1  # the recovered line is by construction the header
        out.append((match.x0, match.top, match.x1, match.bottom))
    return out


def _page_images(page) -> list[dict]:
    """Embedded raster objects worth keeping (tiny decorations skipped)."""
    out = []
    for im in page.images:
        if im["x1"] - im["x0"] < 8 or im["bottom"] - im["top"] < 8:
            continue
        out.append(im)
    return out


def _vlm_figure_regions(page, png: bytes, vlm: VLMClient | None
                        ) -> list[dict] | None:
    """Whole-figure bboxes from `classify_page_regions` (one VLM_CLASSIFY
    call for the whole page), reshaped into the same dict shape
    `_page_images` returns (`x0`/`top`/`x1`/`bottom`, plus `visual_type*`)
    so every downstream consumer (`_img_locator`, `_mask_figures`, the
    `geom_images` positional pairing in `_blocks_from_vlm`) works unchanged.

    Exists in place of `_page_images`: pdfplumber's raw raster-object list
    reports one real technical drawing as many small fragments (tiles,
    arrowheads, dimension marks) with no merge step, so the parser was
    emitting a dozen tiny ImageBlocks for what a reader sees as one figure.
    `classify_page_regions`'s prompt explicitly merges touching/overlapping
    pieces into one region, so its bboxes replace the fragments instead of
    being reconciled with them -- reconciling two disagreeing region sources
    (geometric fragments vs. VLM-merged regions) would need its own
    geometric-overlap logic, the exact complexity this is meant to avoid.

    `table`-typed regions are dropped: table detection/extraction stays on
    the existing `find_tables()`/band-builder path, untouched.

    Returns None (not []) when nothing came back -- "no VLM configured",
    "classify disabled", and "page genuinely has no figures" are all
    indistinguishable from here, so the caller falls back to `_page_images`
    rather than risk silently losing a real figure to an unreachable
    classifier."""
    cfg = get_config().visual
    if not cfg.classify or vlm is None:
        return None
    regions = classify_page_regions(png, vlm, cfg)
    figures = [r for r in regions if r.visual_type != "table"]
    if not figures:
        return None
    w, h = float(page.width), float(page.height)
    return [{
        "x0": round(r.bbox[0] * w, 2), "top": round(r.bbox[1] * h, 2),
        "x1": round(r.bbox[2] * w, 2), "bottom": round(r.bbox[3] * h, 2),
        "visual_type": r.visual_type,
        "visual_type_source": r.source,
        "visual_type_confidence": r.confidence,
    } for r in figures]


def _mask_figures(page, boxes: Sequence[tuple[float, float, float, float]]
                  ) -> bytes | None:
    """A fresh render of `page` with every `boxes` bbox blanked to a flat,
    opaque gray rectangle, or None if `boxes` is empty (nothing to mask, and
    no render to pay for). Reduces the fine-detail load a complex figure
    (e.g. a block diagram with a dozen tiny embedded labels) puts on the
    primary hybrid VLM read without erasing the fact that something occupies
    that reading-order slot -- the model still needs to notice each region
    and emit a `{"type": "figure"}` placeholder there for
    `_blocks_from_vlm`'s positional pairing to keep working.

    Diagnosed on ACME_PN5029_Datasheet.pdf p.5: a heading directly above
    a dense diagram, and another right before a table further down the same
    page, were both dropped entirely from the VLM's read -- along with a
    paragraph duplicated verbatim elsewhere on that same page, a known
    small-model "lost my reading-order position" symptom. Scoped to figures
    only, not tables: a table's own placeholder still needs the model to
    describe *some* row/column content to be worth emitting, whereas a
    figure's contents are never transcribed anyway (see `_PAGE_SYSTEM`), so
    blanking one loses nothing the pipeline uses -- and an unpaired table is
    already recovered separately (`_merge_unpaired_tables`) regardless of
    why the VLM missed it.

    Callers use this ONLY for the primary read; the real, unmasked PNG
    still backs crops / the second reader / the scanned path, which need
    real pixels."""
    if not boxes:
        return None
    dpi = get_config().pdf.render_dpi
    pi = page.to_image(resolution=dpi)
    for bbox in boxes:
        pi.draw_rect(bbox, fill=(160, 160, 160, 255),
                    stroke=(160, 160, 160, 255), stroke_width=0)
    buf = io.BytesIO()
    pi.annotated.save(buf, format="PNG")
    return buf.getvalue()


def _img_locator(pageno: int, im: dict) -> dict[str, Any]:
    loc: dict[str, Any] = {
        "page": pageno,
        "bbox": [round(float(im["x0"]), 2), round(float(im["top"]), 2),
                 round(float(im["x1"]), 2), round(float(im["bottom"]), 2)],
    }
    if im.get("name"):
        loc["name"] = str(im["name"])
    return loc


def _line_top(lines: list[_Line], text: str) -> float | None:
    """`top` of the text-layer line that begins `text`, or None -- used to
    give a block (whose own wording came from that text layer) an approximate
    vertical position so a recovered heading can be slotted into reading
    order relative to it. Matches on normalized full-line equality or a
    prefix either way (a block's text spans several physical lines; a line is
    a prefix of the block, and vice versa when the block is a single line),
    preferring the longest overlap so the most specific line wins."""
    key = normalize(text)
    if len(key) < 3:
        return None
    best: tuple[int, float] | None = None
    for l in lines:
        ln = normalize(l.text)
        if not ln:
            continue
        if ln == key or key.startswith(ln) or ln.startswith(key):
            overlap = min(len(ln), len(key))
            if best is None or overlap > best[0]:
                best = (overlap, l.top)
    return best[1] if best else None


def _insert_at_top(blocks: list[Block], new_block: Block, new_top: float,
                   lines: list[_Line]) -> None:
    """Insert `new_block` into a page's reading-order `blocks` at the position
    its text-layer `top` implies: before the first existing block that sits
    lower on the page (larger `top`). Blocks whose own top can't be located
    (`_line_top` None) don't anchor the search, so a stray unlocatable block
    never forces the new one to the end; if none sits lower, it appends.
    Single-column assumption (the pages this recovers on -- see
    `_recover_dropped_headings`); a gutter split would need per-column tops."""
    for i, b in enumerate(blocks):
        t = _line_top(lines, _block_text(b))
        if t is not None and t > new_top + 0.5:
            blocks.insert(i, new_block)
            return
    blocks.append(new_block)


def _recover_dropped_headings(blocks: list[Block], page, prep: dict,
                              pageno: int, running_lines: frozenset[str] | None,
                              ct: _Counters,
                              declared: frozenset[str]) -> int:
    """Recover heading lines the primary VLM read omitted ENTIRELY -- present
    verbatim in the page's own text layer, but in no block the VLM produced.
    Returns how many were inserted (into `blocks`, in place).

    Distinct from what the document-wide reconcile pass does: reconcile can
    only re-level or promote a block that EXISTS; a heading the VLM never
    emitted has no block to promote. Seen on ACME_PN5029_Datasheet.pdf p.5
    -- '2.1. Circuitry' and '2.2.1. Electrical Specifications' are printed
    digital text (in `extract_text()`), yet the small VLM dropped both while
    transcribing the dense block diagram sitting between them (and duplicated
    a nearby paragraph -- the 'lost my reading-order place' symptom that
    masking the diagram, `_mask_figures`, reduces but doesn't guarantee away).

    Deterministic, so it costs no extra model call and the recovered text is
    the text layer's own characters -- hence PROV_TEXT_LAYER. `declared` is
    the set of headings the document's own structure (its native outline)
    promises exist, each key `normalize(strip_num(title))`: recovery fires
    ONLY for a text-layer line whose key is declared. That gate is what keeps
    it from promoting a big-font vector-diagram label ('Main', 'Controller')
    or a dimension callout ('98.40 mm', which even parses as a section number)
    into a heading -- those look heading-shaped in the text layer but the
    document never declares them. Beyond the gate, a line still has to read as
    a heading by the code path's own test (`_is_heading`, numbered-title size
    discount included) and be genuinely absent: its number-stripped text a
    substring of no existing block -- number-insensitive, so a heading the VLM
    kept but stripped the number from ('Circuitry' for '2.1. Circuitry') reads
    as already-present, never duplicated. Level is left at 1; reconcile sets
    it from the number's depth. Table-interior and margin lines are excluded
    as the code path and running-line detector exclude them."""
    if not declared:
        return 0
    tboxes = _table_boxes(prep)
    words = [w for w in (prep.get("words") or []) if not _in_boxes(w, tboxes)]
    lines = _cluster_lines(words)
    if not lines:
        return 0
    body = median(l.size for l in lines)
    seen = [normalize(_block_text(b)) for b in blocks]
    recovered = 0
    for l in lines:
        if _is_margin_noise(l, page.height, running_lines):
            continue
        if not _is_heading(l, body):
            continue
        key = normalize(strip_num(l.text))
        if len(key) < 4 or key not in declared or any(key in bt for bt in seen):
            continue
        hb = HeadingBlock(id=ct.bid(), span=Span(page=pageno),
                          text=l.text.strip(), level=1,
                          provenance=PROV_TEXT_LAYER)
        _insert_at_top(blocks, hb, l.top, lines)
        seen.append(key)  # a physically repeated line isn't recovered twice
        recovered += 1
    return recovered


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Counters:
    """Document-wide id counters (block ids, image markers, list ids)."""

    def __init__(self) -> None:
        self._block = 0
        self._image = 0
        self._list = 0

    def bid(self) -> str:
        s = f"b{self._block}"
        self._block += 1
        return s

    def img(self) -> int:
        self._image += 1
        return self._image

    def lid(self) -> str:
        self._list += 1
        return f"L{self._list}"


class PdfParser(BaseParser):
    extensions = (".pdf",)
    mimetypes = ("application/pdf",)
    fmt = "pdf"

    def __init__(self, *, vlm: VLMClient | None = _UNSET,
                 vlm2: VLMClient | None = _UNSET,
                 detector: TextDetector | None = _UNSET,
                 table_struct: TableStructureClient | None = _UNSET,
                 text_llm: LLMClient | None = _UNSET) -> None:
        # All five are injectable for tests; the _UNSET sentinel means
        # "resolve lazily from the environment on first use" while an explicit
        # None disables that component outright.
        self._vlm = vlm
        self._vlm2 = vlm2
        self._detector = detector
        self._table_struct = table_struct
        self._llm = text_llm
        self._vlm_failures = 0
        # Read-outcome forensics: per-kind counters ("ok" / "transport" /
        # "empty-reply" / "bad-json") surfaced as metadata["vlm_health"], the
        # last failure's evidence for the page note, and — when the transport
        # kill-switch fires — the reason, so later pages say WHY the VLM is
        # gone instead of the misleading "no VLM configured".
        self._read_stats: Counter = Counter()
        self._last_read_failure: tuple[str, str] | None = None
        self._vlm_disabled: str | None = None

    # -- component resolution --------------------------------------------

    def _primary(self) -> VLMClient | None:
        if self._vlm is _UNSET:
            if not get_config().pdf.vlm:
                self._vlm = None
            else:
                try:
                    self._vlm = get_vlm_client()
                except LLMError:
                    self._vlm = None
        return self._vlm

    def _secondary(self) -> VLMClient | None:
        if self._vlm2 is _UNSET:
            if not get_config().pdf.vlm:
                self._vlm2 = None
            else:
                try:
                    self._vlm2 = get_vlm_client("secondary")
                except LLMError:
                    self._vlm2 = None
        return self._vlm2

    def _text_llm(self) -> LLMClient | None:
        """Text-only LLM for `classify_table_text` (the ungrounded VLM-table
        fallback in `_blocks_from_vlm` has no crop to classify by vision --
        just the VLM's own transcribed cell text). Gated on `[visual].classify`
        independently of `pdf.vlm` (this is a different toggle for a
        different call), so classification can be disabled without touching
        the PDF page-read VLM setting."""
        if self._llm is _UNSET:
            if not get_config().visual.classify:
                self._llm = None
            else:
                try:
                    self._llm = get_client()
                except LLMError:
                    self._llm = None
        return self._llm

    def _table_structure(self) -> TableStructureClient | None:
        """None means "not configured" -- unset TABLE_STRUCT_PROVIDER (the
        default) or a bad config, either way callers fall back to
        find_tables(). A live but momentarily-failing client raises
        TableStructureError per-call instead; that's handled where it's
        actually invoked (_structure_model_tables), not here."""
        if self._table_struct is _UNSET:
            try:
                self._table_struct = get_table_structure_client()
            except TableStructureError:
                self._table_struct = None
        return self._table_struct

    def _get_detector(self) -> TextDetector | None:
        if self._detector is _UNSET:
            self._detector = _default_detector()
        return self._detector

    # -- top level ---------------------------------------------------------

    @staticmethod
    def _pdf_outline(pdf) -> list[tuple[int, str]]:
        """(depth, title) per native outline (bookmark) entry, in document
        order -- the author's own heading hierarchy, exact titles and exact
        depths, present in most Word/LaTeX/InDesign-produced PDFs. The
        strongest heading-level signal there is (see heading_reconcile).

        Caption bookmarks are dropped: an outline often lists every figure
        and table too ("Figure 28: ...", "Table 3: ...", the entries a
        list-of-figures / list-of-tables is built from -- 25 of ACME_PN5057_
        User_Manual.pdf's 142 outline entries are these). Those are not
        section headings; leaving them in would let reconcile try to match
        them (polluting the unmatched-heading audit list with captions that
        were never headings) and let recovery treat one as a declared
        heading. `CAPTION` needs a digit after "Figure"/"Table", so the real
        section headings "Figures" / "Tables" (the caption LISTS themselves)
        are kept.

        Broad except: a malformed outline tree (dangling refs, cycles) must
        never take down parsing when the signal is merely nice to have."""
        out: list[tuple[int, str]] = []
        try:
            for entry in pdf.doc.get_outlines():
                level, title = entry[0], entry[1]
                if isinstance(title, bytes):  # pdfminer normally decodes
                    title = title.decode("utf-8", "replace")
                title = str(title or "").strip()
                if title and isinstance(level, int) and level > 0 \
                        and not CAPTION.match(title):
                    out.append((min(level, 6), title))
        except Exception:  # noqa: BLE001 -- the outline is only an auxiliary signal (see docstring); a pdfminer error returns an empty list, parsing does not fall
            return []
        return out

    def classify_only(self, raw_path: str | Path, doc_id: str) -> None:
        """Pre-warms visual_classify.py's on-disk classification caches for
        every table-region candidate in this PDF, WITHOUT reading a single
        page (no primary-VLM call, ever). Return value is discarded -- the
        only effect is `classify_crop`'s per-crop cache AND
        `classify_page_regions`'s per-page cache (see visual_classify.py,
        keyed by crop sha256 / page-png sha256 respectively) getting
        populated as a side effect of `_triage`. `_triage` warms the
        page-region cache both for pages with geometric table candidates
        (via `_build_region_tables`) and for hybrid-routed pages with no
        table candidates but qualifying images (a dedicated warm-up branch
        in `_triage`, since `_build_region_tables` is never reached for
        those) -- so `_parse_hybrid`'s own `_vlm_figure_regions` call always
        hits a warm cache instead of calling VLM_CLASSIFY for the first time
        interleaved with the primary VLM read.

        This runs the exact same per-page `_triage` call `parse()` runs
        below, in the exact same order, before doing anything else -- so
        `parse()`'s own triage loop, run later, hits the cache for every
        region already seen here and never calls VLM_CLASSIFY again. This
        exists so a caller (run_parse_pipeline.py's phase 0) can warm the
        (possibly genuinely different) VLM_CLASSIFY model for the whole
        corpus in one pass, before the primary VLM/LLM phases start --
        otherwise a single document already alternates VLM_CLASSIFY (triage)
        then the primary VLM (page read), and DOC_CONCURRENCY runs many
        documents' triage and page-read concurrently, so the model keeps
        getting swapped throughout the run instead of once.

        Deliberately NOT exposed as a generic `BaseParser` method: only
        pdf_parser.py classifies BEFORE the rest of parsing (table-region
        triage, see `_build_region_tables`) -- every other format's images are
        only known once the block tree exists, so their classify pass has to
        go through `images.image_handler.handle_images(doc,
        classify_only=True)` on an already-`parse()`d document instead (see
        run_parse_pipeline.py's phase 0)."""
        raw_path = Path(raw_path)
        with (PDFIUM_LOCK, pdfplumber.open(raw_path) as pdf,
              progress.bar(len(pdf.pages), f"{raw_path.name}: classify",
                           unit="page") as pbar):
            for pageno, page in enumerate(pdf.pages, start=1):
                self._triage(page, pageno, raw_path, doc_id)
                pbar.update()

    def parse(self, raw_path: str | Path, doc_id: str) -> ParsedDocument:
        raw_path = Path(raw_path)
        raw_sha256 = sha256_id(raw_path.read_bytes())

        ct = _Counters()
        self._read_stats = Counter()
        self._last_read_failure = None
        blocks: list[Block] = []
        page_meta: list[dict[str, Any]] = []

        with PDFIUM_LOCK, pdfplumber.open(raw_path) as pdf:
            page_count = len(pdf.pages)
            triaged = []
            with progress.bar(page_count, f"{raw_path.name}: triage",
                              unit="page") as pbar:
                for pageno, page in enumerate(pdf.pages, start=1):
                    triaged.append(
                        (pageno, page, *self._triage(page, pageno, raw_path, doc_id)))
                    pbar.update()

            # Running headers/footers (repeating margin text: page numbers,
            # URLs, "Caution" boilerplate, ...) need a whole-document view,
            # same reasoning as heading_sizes below -- a single page can't
            # tell a genuine one-off line from one that recurs on every
            # other page. Computed first since heading_sizes must already
            # exclude what this finds (see _page_heading_candidates).
            running_lines = detect_running_lines(
                [(page.height, _page_margin_lines(prep))
                 for _, page, route, _, prep in triaged
                 if route in ("code", "hybrid")])

            # Heading levels rank font sizes once across the whole document,
            # not per page/column, so the same physical size always maps to
            # the same level (see _specs_from_lines). Both "code" pages and
            # "hybrid" pages are scanned for candidates: a hybrid page falls
            # back to the font-based code path whenever the VLM is
            # unavailable, and that fallback must never hit a size the
            # ranking doesn't know about.
            heading_sizes = sorted(
                {sz for _, page, route, _, prep in triaged
                 if route in ("code", "hybrid")
                 for sz in _page_heading_candidates(prep, page.height, running_lines)},
                reverse=True)

            # Inputs the post-parse heading reconciliation needs from the
            # open document: the native outline, and each page's visual
            # lines with their font sizes (plain data, so both survive the
            # `with` block; pdfplumber objects would not).
            outline = self._pdf_outline(pdf)
            line_sizes = {
                pageno: [(normalize(l.text), round(l.size, 1))
                         for l in _cluster_lines(prep["words"])]
                for pageno, _, route, _, prep in triaged
                if route in ("code", "hybrid")}

            # Headings the document's own outline promises exist, each keyed
            # number-stripped -- the gate the hybrid path recovers dropped
            # headings against (see _recover_dropped_headings). Empty when the
            # PDF carries no outline, disabling recovery entirely (no declared
            # structure to trust). The printed TOC isn't available yet (it's
            # extracted after the page loop), and isn't needed: every
            # structured doc in scope carries an outline whose titles cover
            # its TOC's.
            declared = frozenset(
                normalize(strip_num(title)) for _, title in outline)

            with progress.bar(page_count, f"{raw_path.name}: pages",
                              unit="page") as pbar:
                for pageno, page, route, info, prep in triaged:
                    meta = {"page": pageno, "route": route, "used": route, **info}
                    n_before = len(blocks)
                    if route == "empty":
                        meta["used"] = "empty"
                        meta["note"] = ("no text layer, raster, or vector ink;"
                                        " page skipped")
                    elif route == "code":
                        blocks.extend(
                            self._parse_code(page, pageno, prep, ct, heading_sizes,
                                             running_lines))
                    elif route == "hybrid":
                        blocks.extend(
                            self._parse_hybrid(page, pageno, prep, ct, meta,
                                               heading_sizes, running_lines,
                                               declared))
                    else:  # scanned
                        blocks.extend(self._parse_scanned(page, pageno, ct, meta))
                    # Vector-heavy text page: a technical
                    # drawing's real value is the drawing itself -- pure vector
                    # linework, invisible both to `_page_images` (raster objects
                    # only) and to the text path, so 0/50 drawings kept any
                    # full-page visual. Keep a render of the page as an
                    # ImageBlock: the images stage already knows how to fetch a
                    # bbox'd locator (render + crop), so it gets stored and
                    # OCR'd with no extra plumbing.
                    # Default threshold from the corpus' own distribution: every
                    # technical drawing sampled sits at 285-791 vector objects,
                    # ordinary pages at a median of ~39 (p90 ~466 -- a
                    # graphics-dense datasheet page above the threshold just gets
                    # a render too, which only costs one extra image block).
                    if (route in ("code", "hybrid")
                            and info["vector_objects"]
                            >= get_config().pdf.fullpage_render_min_vector):
                        meta["full_page_render"] = True
                        blocks.append(ImageBlock(
                            id=ct.bid(), span=Span(page=pageno),
                            image_index=ct.img(),
                            locator={"page": pageno,
                                     "region": "full_page_render",
                                     "bbox": [0, 0, round(float(page.width), 2),
                                              round(float(page.height), 2)]}))
                    # Last line of defense: whatever route
                    # handled -- or mishandled -- the page, a page with ink
                    # (chars, raster, or vector) must never leave the parser
                    # with zero blocks AND zero notes; that is exactly the
                    # silent total loss of the SLSC case. Route-level fixes
                    # make this rare; sitting at the exit, it holds even for
                    # failure modes nobody has met yet (e.g. every line of a
                    # sparse code page eaten by the margin-noise filter).
                    if (route != "empty" and len(blocks) == n_before
                            and (info["chars"] > 0
                                 or info["image_coverage"] >= 0.01
                                 or info["vector_objects"] >= self._VECTOR_INK_MIN)):
                        note = "inked page produced no blocks; page kept as image"
                        meta["note"] = (f"{meta['note']}; {note}"
                                        if meta.get("note") else note)
                        blocks.append(ImageBlock(
                            id=ct.bid(), span=Span(page=pageno),
                            image_index=ct.img(),
                            locator={"page": pageno, "region": "full_page",
                                     "bbox": [0, 0, round(float(page.width), 2),
                                              round(float(page.height), 2)]}))
                    page_meta.append(meta)
                    pbar.update()

        # TOC entries removed before heading_path assignment (below) both
        # skips wasted breadcrumb work on blocks about to be dropped and
        # keeps block ids gapless (renumbered right after, in reading order).
        pre_toc = blocks
        blocks, toc = _extract_toc(blocks)
        if toc:
            # Say where the entries went: a page whose whole
            # visible content was its TOC otherwise shows zero blocks in the
            # page meta and reads as a silent drop in any per-page audit,
            # when the content in fact moved to metadata["toc"].
            kept_ids = {id(b) for b in blocks}
            toc_pages = Counter(b.span.page for b in pre_toc
                                if id(b) not in kept_ids)
            for m in page_meta:
                if toc_pages.get(m["page"]):
                    m["toc_extracted"] = toc_pages[m["page"]]

        # Heading levels rebuilt document-wide from the strongest available
        # signals (outline > numbered titles > printed TOC > anchored font
        # size) -- the per-page sources they replace (font ladder on code
        # pages, the VLM's own guess on hybrid pages) never agreed with each
        # other across a route boundary. See parsers/heading_reconcile.py.
        hstats: dict[str, Any] = {}
        if get_config().pdf.heading_reconcile:
            blocks, hstats = reconcile_headings(
                blocks, outline=outline, toc=toc, line_sizes=line_sizes,
                heading_sizes=heading_sizes, running_lines=running_lines)

        # Broken-cmap flag + known-ligature repair.
        # After reconcile (whose line matching must see the same unrepaired
        # text the lines carry), before heading_path (so breadcrumbs carry
        # the repaired heading texts).
        cmap_stats = _scan_broken_cmap(
            blocks, min_hits=get_config().pdf.broken_cmap_min)

        for i, b in enumerate(blocks):
            b.id = f"b{i}"

        # heading_path: one pass over the whole document in final reading
        # order (pages/columns/routes are already flattened into `blocks`
        # correctly by this point), so a heading on page N still parents
        # blocks on page N+1 until closed by an equal-or-higher-level one.
        stack = HeadingStack()
        for b in blocks:
            if isinstance(b, HeadingBlock):
                b.heading_path = stack.enter(b.level, b.text)
            else:
                b.heading_path = stack.path()

        metadata: dict[str, Any] = {"pdf_pages": page_meta}
        if cmap_stats:
            metadata["broken_cmap"] = cmap_stats
        if self._read_stats:
            # Per-doc VLM read outcomes ("ok"/"transport"/"empty-reply"/
            # "bad-json") — the pipeline's health breaker reads this to catch
            # a systemically failing endpoint within a few documents.
            metadata["vlm_health"] = dict(self._read_stats)
        if toc:
            metadata["toc"] = toc
        if any(hstats.values()):
            metadata["heading_reconcile"] = {k: v for k, v in hstats.items() if v}

        return ParsedDocument(
            doc_id=doc_id,
            source_path=str(raw_path),
            fmt=self.fmt,
            raw_sha256=raw_sha256,
            mimetype="application/pdf",
            page_count=page_count,
            parser_version=PARSER_VERSION,
            metadata=metadata,
            blocks=blocks,
        )

    # Minimum count of vector path objects (curves + rects + lines) for a
    # page with no text layer to count as inked rather than blank. A blank
    # page with a decorative border or a couple of rules sits well under
    # this; flattened text or a drawing's linework sits in the thousands.
    _VECTOR_INK_MIN = 40

    # -- triage -------------------------------------------------------------

    def _triage(self, page, pageno: int, raw_path: Path, doc_id: str
               ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Route one page: "empty" | "code" | "hybrid" | "scanned"."""
        words = page.extract_words(extra_attrs=["size", "fontname"])
        # Rotated text (upright=False -- e.g. the 90°-set copyright /
        # confidentiality strip on every technical drawing) is dropped from
        # the flow entirely: extract_words
        # splits such glyphs into letter-sized fake words, and the top-based
        # line clustering then shuffles the letters into whatever body line
        # each one's `top` lands on -- deterministic letter soup in ~50
        # drawings. The content is rotated boilerplate, noise for RAG.
        n_rotated = sum(1 for w in words if not w.get("upright", True))
        if n_rotated:
            words = [w for w in words if w.get("upright", True)]
        n_chars = len(page.chars)
        area = float(page.width) * float(page.height) or 1.0
        # TOTAL raster coverage, not the single largest image: a flattened
        # page often carries its artwork as many small tiles (SLSC catalogue
        # p.1: 16 images, largest 3.4% but 25% combined), and taking the max
        # sent such pages down the "empty" route -- silently deleting them.
        img_frac = 0.0
        for im in page.images:
            frac = (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) / area
            img_frac += max(0.0, frac)
        text_frac = sum((w["x1"] - w["x0"]) * (w["bottom"] - w["top"])
                        for w in words) / area
        # Vector ink: text flattened to glyph outlines, or a drawing's
        # linework. Such pages show thousands of curve/rect objects (SLSC
        # p.1: 8,964 curves + 1,590 rects) while chars and rasters stay near
        # zero -- without this signal the page is indistinguishable from a
        # genuinely blank one.
        n_vector = len(page.curves) + len(page.rects) + len(page.lines)

        tables: list = []
        diverted: list[dict[str, Any]] = []
        gutter: float | None = None
        png: bytes | None = None
        if n_chars < 20 and img_frac < 0.2 and n_vector >= self._VECTOR_INK_MIN:
            # Flattened page (text drawn as vector outlines), possibly with a
            # text-layer scrap of a few real characters (a page number, a
            # stray label). Gating this on n_chars == 0 alone re-opened the
            # SLSC hole one character wide: those few chars
            # would route "code", the margin filter could eat them, and
            # thousands of vector glyphs exited as zero blocks with zero
            # notes. An inked page must never exit with zero blocks.
            route = "scanned"
        elif n_chars == 0 and img_frac < 0.2:
            # No text layer, no page-sized raster, no vector ink. "empty"
            # only when there is (nearly) no raster ink at all either --
            # a small-raster page (0.01..0.2 coverage) still has content and
            # goes "scanned" so its images survive at worst as blocks.
            route = "empty" if img_frac < 0.01 else "scanned"
        elif n_chars < 20 and img_frac >= 0.2:
            route = "scanned"  # no usable text layer over a page-sized raster
        elif img_frac > 0.8 and text_frac < 0.05:
            route = "scanned"  # thin/partial OCR layer glued onto a scan
        else:
            # find_tables() LOCATES table regions (its bboxes are reliable; it
            # only mis-reconstructs the GRID inside them -- dropping unshaded
            # rows' column-0 cell, inventing phantom columns). So every found
            # region is handed to `_build_region_tables`, which rebuilds the
            # grid deterministically from page geometry (band builder) and only
            # calls a model to referee the regions it's unsure of. pdfplumber's
            # own lattice reconstruction (`_table_to_data`) is never emitted for
            # a detected region except as a last-resort near-empty fallback.
            geom_tables = [t for t in page.find_tables()
                           if not _is_degenerate_table(t)
                           and not _covers_most_of_page(t, page)]
            if geom_tables:
                tables, diverted, png = self._build_region_tables(
                    page, geom_tables, words, pageno, doc_id)
            gutter = _find_gutter(words, float(page.width))
            route = "hybrid" if (tables or gutter is not None) else "code"
            # A page with no table candidates never runs
            # `_build_region_tables` above, so its `classify_page_regions`
            # cache is otherwise still cold once `_parse_hybrid` reaches
            # `_vlm_figure_regions` for it -- the classify call and the
            # primary-VLM read then happen back to back inside that one
            # call, exactly the interleaving `classify_only` (phase 0) exists
            # to avoid (see its docstring). Warm it here instead whenever
            # this page will actually reach `_parse_hybrid` with qualifying
            # raster images to classify; a page with no images has nothing
            # for `_vlm_figure_regions` to warm anyway, so it's skipped.
            if not geom_tables and route == "hybrid" and _page_images(page):
                visual_cfg = get_config().visual
                vlm = (get_vlm_classify_client(self._primary())
                      if visual_cfg.classify else None)
                if vlm is not None:
                    if png is None:
                        png = self._render(page)
                    classify_page_regions(png, vlm, visual_cfg)

        # Table-header recovery (see `_recover_table_headers`) has to happen
        # once here, not lazily inside `_parse_code`: the doc-wide heading-size
        # ranking pass (`_page_heading_candidates`, called on every "code" and
        # "hybrid" page before any page is actually parsed) must exclude the
        # exact same words `_parse_code` will later exclude, or the two passes
        # compute a different local `body` size for the same page and a line
        # can end up heading-sized in the real pass without its size having
        # been ranked in `heading_sizes` -- a lookup crash in `_specs_from_lines`.
        # Computing (and mutating `table_data` for) this exactly once and
        # threading the result through both consumers keeps them in sync, and
        # avoids inserting the recovered row twice into the same TableData
        # (a live risk for `_PreparedTable`, whose `.data` is one shared object).
        table_data = [_as_table_data(t) for t in tables]
        header_boxes = _recover_table_headers(words, tables, table_data)

        info = {"chars": n_chars,
                "text_coverage": round(text_frac, 3),
                "image_coverage": round(img_frac, 3),
                "vector_objects": n_vector}
        if n_rotated:
            info["rotated_words_dropped"] = n_rotated
        # `png`: the render _structure_model_tables already needed, if any --
        # _parse_hybrid reuses it instead of rendering the same page twice.
        prep = {"words": words, "tables": tables, "gutter": gutter, "png": png,
                "table_data": table_data, "header_boxes": header_boxes,
                "diverted_images": diverted}
        return route, info, prep

    # -- code path -----------------------------------------------------------

    def _parse_code(self, page, pageno: int, prep: dict, ct: _Counters,
                    heading_sizes: list[float] | None = None,
                    running_lines: frozenset[str] | None = None,
                    ) -> list[Block]:
        tables = prep.get("tables") or []
        gutter = prep.get("gutter")
        # `_triage` already computed and cached these (see its comment) so the
        # doc-wide heading-size pass and this real pass exclude the exact same
        # words; falling back to a fresh computation here only if `prep` was
        # built some other way (e.g. a future direct caller/test).
        if "table_data" in prep:
            table_data = prep["table_data"]
            header_boxes = prep["header_boxes"]
        else:
            table_data = [_as_table_data(t) for t in tables]
            header_boxes = _recover_table_headers(prep.get("words") or [], tables, table_data)
        columns = _code_columns(prep, tuple(b for b in header_boxes if b is not None))

        def col_of(x_center: float) -> int:
            return 0 if gutter is None or x_center < gutter else 1

        # (column, top, block): reading order = column-major, then top-down.
        entries: list[tuple[int, float, Block]] = []
        for ci, cwords in enumerate(columns):
            lines = [l for l in _cluster_lines(cwords)
                    if not _is_margin_noise(l, page.height, running_lines)]
            specs = _specs_from_lines(lines, heading_sizes)
            group_ids: dict[int, str] = {}
            for sp in specs:
                span = Span(page=pageno)
                runs = sp.runs if runs_have_marks(sp.runs) else []
                if sp.kind == "heading":
                    b: Block = HeadingBlock(id="", span=span, text=sp.text,
                                            level=sp.level, runs=runs,
                                            provenance=PROV_TEXT_LAYER)
                elif sp.kind == "item":
                    gid = group_ids.setdefault(sp.list_group, ct.lid())
                    b = ParagraphBlock(id="", span=span, text=sp.text,
                                       list_id=gid, list_level=sp.list_level,
                                       list_ordered=sp.list_ordered, runs=runs,
                                       provenance=PROV_TEXT_LAYER)
                else:
                    b = ParagraphBlock(id="", span=span, text=sp.text, runs=runs,
                                       provenance=PROV_TEXT_LAYER)
                entries.append((ci, sp.top, b))

        for t, data, hbox in zip(tables, table_data, header_boxes):
            b = TableBlock(id="", span=Span(page=pageno), table=data)
            if isinstance(t, _PreparedTable):
                b.provenance = t.provenance
                b.table_confidence = t.confidence
                b.table_flags = t.flags or None
            else:
                b.provenance = PROV_TEXT_LAYER
            top = hbox[1] if hbox is not None else t.bbox[1]
            entries.append((col_of((t.bbox[0] + t.bbox[2]) / 2), top, b))

        for im in _page_images(page):
            b = ImageBlock(id="", span=Span(page=pageno), image_index=0,
                           locator=_img_locator(pageno, im))
            entries.append((col_of((im["x0"] + im["x1"]) / 2), im["top"], b))

        for d in prep.get("diverted_images") or []:
            # A find_tables() region classification determined is NOT a real
            # table (see PdfParser._build_region_tables) -- placed at its own
            # bbox's reading-order position, same as a real table above.
            bbox = d["bbox"]
            b = ImageBlock(id="", span=Span(page=pageno), image_index=0,
                           locator={"page": pageno, "bbox": list(bbox),
                                    "region": "reclassified_visual"},
                           source_crop=d["crop_sha"], visual_type=d["visual_type"],
                           visual_type_source=d["visual_type_source"],
                           visual_type_confidence=d.get("visual_type_confidence"))
            entries.append((col_of((bbox[0] + bbox[2]) / 2), bbox[1], b))

        entries.sort(key=lambda e: (e[0], e[1]))
        out: list[Block] = []
        for _, _, b in entries:
            b.id = ct.bid()
            if isinstance(b, ImageBlock):
                b.image_index = ct.img()
            out.append(b)
        return out

    # -- VLM plumbing ---------------------------------------------------------

    def _read_page(self, vlm: VLMClient, png: bytes, user: str
                   ) -> list[dict[str, Any]] | None:
        max_tokens = get_config().pdf.vlm_max_tokens
        self._last_read_failure = None
        raw: str | None = None
        for attempt in range(2):
            try:
                raw = vlm.complete_vision(
                    system=_PAGE_SYSTEM, user=user, images=[("image/png", png)],
                    max_tokens=max_tokens)
            except LLMError as exc:
                self._vlm_failures += 1
                self._note_read_failure("transport", str(exc))
                if self._vlm_failures >= 2:
                    self._vlm = None  # transport is down; stop trying
                    self._vlm_disabled = ("VLM disabled after repeated "
                                          "transport errors")
                return None
            specs = _parse_vlm_blocks(raw)
            if specs is not None:
                self._read_stats["ok"] += 1
                return specs
            user = user + "\n\nReturn ONLY the JSON array, nothing else."
        # An HTTP-200 reply that still yielded nothing: "empty-reply" (blank
        # content — e.g. a reasoning model burning the budget on hidden
        # thinking) vs "bad-json" (spoke, but not the contract). These must
        # NOT feed the transport kill-switch: the server is up, disabling the
        # VLM would just mask a model/config problem.
        kind = "bad-json" if raw and raw.strip() else "empty-reply"
        self._note_read_failure(kind, raw or "")
        return None

    def _note_read_failure(self, kind: str, detail: str) -> None:
        self._read_stats[kind] += 1
        self._last_read_failure = (kind, detail.strip()[:200])

    def _reread(self, image: bytes) -> str | None:
        """Ask the secondary model to independently transcribe an image."""
        vlm2 = self._secondary()
        if vlm2 is None:
            return None
        try:
            return vlm2.complete_vision(
                system=_TRANSCRIBE_SYSTEM,
                user="Transcribe the attached image exactly.",
                images=[("image/png", image)], max_tokens=2048)
        except LLMError:
            self._vlm2 = None  # don't retry a dead endpoint per line
            return None

    def _second_reader(self, png: bytes):
        """Lazy, cached full-page second transcription -> (tokens, criticals)."""
        cache: dict[str, tuple[Counter, Counter] | None] = {}

        def get() -> tuple[Counter, Counter] | None:
            if "v" not in cache:
                text = self._reread(png)
                cache["v"] = ((_counter(_tok_list(text)), _criticals(text))
                              if text and text.strip() else None)
            return cache["v"]

        return get

    def _render(self, page) -> bytes:
        dpi = get_config().pdf.render_dpi
        img = page.to_image(resolution=dpi).original
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _structure_model_tables(self, page, png: bytes, geom_tables: list
                                ) -> list[_PreparedTable | None] | None:
        """Refine `geom_tables` (table regions) with the configured structure
        model as a REFEREE -- "structure from the model, text from the code":
        the model rebuilds a region's row/column grid, but every cell's text is
        placed from the PDF's own digital text, never from the model's reading
        of the bitmap, so a visual misread can never corrupt cell content.
        Placement is anchored per cell by whichever signal the adapter supplied:
        claimed cell content (`DetectedCell.text`, general VLMs -- see
        `_fill_cells_by_content`) or pixel boxes (TableFormer/http -- see
        `_fill_cells_by_containment`).

        Accept gate (`PDF_TABLE_STRUCT_MIN_COVERAGE`): the fraction of the
        region's real text the grid accounted for. On the content path that
        means matched-by-content (a misaligned grid can't fake it), and any
        substantial digit-bearing claim that matches nothing real (`invented`)
        rejects the grid outright -- a hallucinated number must never shape a
        grid that then looks trustworthy.

        Return contract:
          * `None` -- the model is unavailable page-wide (unconfigured, or the
            call itself failed / transport is down). The caller uses its own
            deterministic result (the band builder) for every region.
          * a list ALIGNED to `geom_tables` (same length/order) whose entry is
            a `_PreparedTable` for a region whose grid PASSED the gate, else
            `None` for a region the model skipped or whose grid was rejected.
            A rejected region is signalled with `None` -- this method NEVER
            emits pdfplumber's raw lattice geometry, so the caller stays the
            single place that decides the fallback (keeping "lattice cells are
            never emitted as output" true).
        """
        client = self._table_structure()
        if client is None:
            return None

        dpi = get_config().pdf.render_dpi
        to_px = dpi / 72.0
        to_pt = 72.0 / dpi

        table_bboxes_px = [tuple(v * to_px for v in t.bbox) for t in geom_tables]

        words = page.extract_words(extra_attrs=["size", "fontname"])
        # Passed through for adapters that match text themselves (TableFormer,
        # http). The vlm adapter ignores it -- placement happens here instead,
        # from these same words, by claimed-content matching below.
        text_cells = [
            TextCellHint(
                text=w["text"],
                bbox=(w["x0"] * to_px, w["top"] * to_px,
                     w["x1"] * to_px, w["bottom"] * to_px),
            )
            for w in words
        ]

        try:
            detected = client.detect(png, table_bboxes=table_bboxes_px,
                                     text_cells=text_cells)
        except TableStructureError:
            return None

        min_cov = get_config().pdf.table.struct_min_coverage
        result: list[_PreparedTable | None] = [None] * len(geom_tables)
        for dt in detected:
            if not dt.cells:
                continue
            if any(c.text is not None for c in dt.cells):
                data, coverage, invented = _fill_cells_by_content(words, dt, to_pt)
            else:
                data, coverage = _fill_cells_by_containment(page.chars, dt, to_pt)
                invented = 0
            region_pt = tuple(v * to_pt for v in dt.bbox)
            if coverage < min_cov or invented:
                logger.info(
                    "table struct referee: region %s rejected "
                    "(placed=%.2f invented=%d)",
                    tuple(round(v, 1) for v in region_pt), coverage, invented)
                continue  # leave None -> caller uses the band-built grid
            idx = _best_overlap_index(region_pt, geom_tables)
            if idx is None:
                continue
            result[idx] = _PreparedTable(
                bbox=region_pt, data=data, provenance=PROV_TABLE_STRUCTURE,
                confidence=round(coverage, 3))
        return result

    def _build_region_tables(self, page, geom_tables: list, words: list[dict],
                             pageno: int, doc_id: str
                             ) -> tuple[list[_PreparedTable], list[dict[str, Any]],
                                       bytes | None]:
        """Build a `_PreparedTable` for every detected region, and return
        (tables, diverted, png) -- png is the page render done for the
        referee/classifier, if any, so the hybrid path can reuse it instead
        of rendering twice.

        The deterministic band builder (parsers/table_bands.py) is the PRIMARY
        source: it rebuilds the grid from the page's own geometry with no model
        and cell text straight from the PDF, so on the born-digital corpus most
        regions never touch a model. Only regions it isn't confident about
        (`PDF_TABLE_BANDS_MIN_CONFIDENCE`, default 0.7) are sent to the
        structure-model referee; a referee rejection falls back to the band grid
        rather than to raw lattice. Every result then passes the source-agnostic
        health gate (`_apply_health`). pdfplumber's own lattice cells are emitted
        only as a last resort when the band builder found no placeable text at
        all (a near-empty region), and are flagged when they are.

        `find_tables()` locates a region reliably but says nothing about what
        it actually IS -- a block diagram or technical drawing whose frame/
        connector lines happen to close into a grid passes the same geometry
        check a real table does (the "AAF"/"Digital Input" false-positive
        class this classification step exists to fix). So, with
        `[visual].classify` on AND a VLM actually reachable, every region is
        ALSO classified before it is trusted as a table at all: a region
        classified as anything other than "table" is diverted out of
        `tables` entirely (no band/referee/health work spent on it) and
        returned in `diverted` instead, so the caller can emit it as an
        ImageBlock. `diverted` entries are `{"bbox", "visual_type",
        "visual_type_source", "crop_sha"}`. Fail-open when no VLM is
        reachable (unset/disabled/down): every region is trusted as a table
        exactly as before classification existed -- "never force unknown
        into a table" is about a genuine verdict of uncertainty, not about
        the classifier being unavailable to ask in the first place.

        The classification itself tries the cheap path first: one whole-page
        `classify_page_regions` call (cached by png sha256, and reused as a
        cache hit by `_parse_hybrid`'s own page-region call later, so this
        never costs a second real VLM call per page) is checked for a region
        confidently overlapping each candidate (`_best_region_match`,
        `_TABLE_REGION_OVERLAP_MIN`). Only a candidate with no confident
        match falls through to the old, more expensive per-candidate
        `classify_crop` call -- so a real table is never diverted just
        because the VLM's coarse page-fraction bbox doesn't line up with
        pdfplumber's own coordinates; that disagreement is a "go ask more
        precisely," never a verdict.
        """
        band = [build_band_table(page, tuple(t.bbox), words) for t in geom_tables]
        min_conf = get_config().pdf.table.bands_min_confidence
        low = [i for i, bt in enumerate(band)
               if bt is None or bt.confidence < min_conf]

        png: bytes | None = None
        referee: dict[int, _PreparedTable] = {}
        if low and self._table_structure() is not None:
            png = self._render(page)
            regions = [geom_tables[i] for i in low]
            refined = self._structure_model_tables(page, png, regions)
            if refined is not None:
                for j, i in enumerate(low):
                    if refined[j] is not None:
                        referee[i] = refined[j]

        visual_cfg = get_config().visual
        vlm = get_vlm_classify_client(self._primary()) if visual_cfg.classify else None
        # Fail-open when there's no classifier to actually ask (VLM unset/
        # disabled/down): trust the geometry as before, rather than diverting
        # every real table to "unknown" just because no one could confirm it
        # -- "never force unknown into a table" is about a genuine verdict of
        # uncertainty, not about the classifier being unreachable.
        classify_active = visual_cfg.classify and vlm is not None
        dpi_scale = get_config().pdf.render_dpi / 72.0

        # One page-region call (cached by png sha256 -- see
        # classify_page_regions) covers every candidate on this page instead
        # of one classify_crop call each; `_parse_hybrid`'s own page-region
        # call later reuses this exact cache entry (same png bytes) rather
        # than paying for a second one. A candidate with no confidently
        # overlapping region (`_best_region_match` returns None) still falls
        # back to the per-crop call below -- coordinate disagreement between
        # the VLM's coarse page-fraction bbox and pdfplumber's own geometry
        # must never be read as "not a table".
        page_regions: list = []
        if classify_active and geom_tables:
            if png is None:
                png = self._render(page)
            page_regions = classify_page_regions(png, vlm, visual_cfg)

        out: list[_PreparedTable] = []
        diverted: list[dict[str, Any]] = []
        for i, t in enumerate(geom_tables):
            if i in referee:
                pt = referee[i]
            elif band[i] is not None:
                bt = band[i]
                pt = _PreparedTable(bbox=bt.bbox, data=bt.data,
                                    provenance=PROV_TABLE_BANDS,
                                    confidence=bt.confidence, flags=list(bt.flags))
            else:
                # No text the band builder could place and no referee grid --
                # last resort is pdfplumber's own reconstruction of this region,
                # flagged so it's visible. A region with no words is near-empty
                # anyway, so this loses nothing a better path would have kept.
                pt = _PreparedTable(bbox=tuple(t.bbox), data=_table_to_data(t),
                                    provenance=PROV_TABLE_BANDS, confidence=0.0,
                                    flags=["lattice-fallback"])

            if classify_active:
                match = _best_region_match(pt.bbox, page, page_regions)
                if match is not None:
                    region, _frac = match
                    if region.visual_type != "table":
                        if png is None:
                            png = self._render(page)
                        crop = _crop_png(png, tuple(v * dpi_scale for v in pt.bbox))
                        crop_sha = _store_crop(crop)
                        diverted.append({
                            "bbox": pt.bbox, "visual_type": region.visual_type,
                            "visual_type_source": region.source,
                            "visual_type_confidence": region.confidence,
                            "crop_sha": crop_sha,
                        })
                        continue
                    # Matched and agrees "table" -- trusted without a
                    # separate classify_crop call.
                else:
                    if png is None:
                        png = self._render(page)
                    crop = _crop_png(png, tuple(v * dpi_scale for v in pt.bbox))
                    verdict = classify_crop(crop, "image/png", vlm, visual_cfg)
                    if verdict.visual_type != "table":
                        crop_sha = _store_crop(crop)
                        if verdict.visual_type == "unknown":
                            log_unknown(doc_id, f"pdf:p{pageno}", crop_sha,
                                       {"page": pageno, "bbox": list(pt.bbox)})
                        diverted.append({
                            "bbox": pt.bbox, "visual_type": verdict.visual_type,
                            "visual_type_source": verdict.source,
                            "visual_type_confidence": verdict.confidence,
                            "crop_sha": crop_sha,
                        })
                        continue

            _apply_health(pt, words)
            if not _table_worth_emitting(pt.data):
                # A false-positive region (empty ruled box, single-column strip
                # of wrapped prose, drawing-as-grid). Dropping it here removes it
                # from `tables`, so its words are no longer excluded from the
                # text flow and rejoin it as ordinary paragraphs -- no text lost.
                logger.info("table drop: region %s not tabular (%dx%d), "
                            "reflowing as text",
                            tuple(round(v, 1) for v in pt.bbox),
                            pt.data.n_rows, pt.data.n_cols)
                continue
            out.append(pt)
        return out, diverted, png

    def _blocks_from_vlm(self, specs: list[dict[str, Any]], pageno: int,
                         ct: _Counters, geom_tables: list | None = None,
                         geom_images: list[dict] | None = None,
                         emit_figures: bool = False,
                         ) -> list[Block]:
        """VLM block specs -> IR blocks (ids assigned; VLM order = reading order).

        `geom_tables` (hybrid path only): pdfplumber tables detected on this
        page, pre-sorted into reading order by the caller. The VLM has no
        coordinates for its own "table" specs, so each table spec is paired
        positionally with the front of `geom_tables` -- matched entries are
        popped in place, exactly like `geom_images`, so the caller can see
        which geometric tables the VLM never called out and recover them (see
        _parse_hybrid's unpaired-table pass). When a pairing exists the
        geometric, merge-aware `TableData` (real text-layer chars,
        deterministic rowspan/colspan) replaces the VLM's flattened JSON grid
        outright, rather than trusting the model's guess. A table spec beyond
        the last geometric table falls back to the VLM's own rows.

        `geom_images`/`emit_figures` (hybrid and scanned paths): when
        `emit_figures` is set, a `"figure"` spec becomes an `ImageBlock` at
        its true reading-order position, positionally paired with the front
        of `geom_images` the same way tables are paired above -- matched
        entries are popped from the caller's list in place, so the caller
        can tell what is left unpaired and append it separately. A
        `"figure"` spec with nothing left to pair (the VLM saw more figures
        than pdfplumber's object model has raster entries for, e.g. a vector
        drawing, or -- on a scanned page -- a sub-figure baked into the
        page's own background raster) still becomes an `ImageBlock`, just
        with no `bbox` in its locator -- the caller marks those unverified
        with an audit crop. If the spec carried its own (ungrounded) `bbox`
        estimate -- normalized [x0, y0, x1, y1] fractions of the page --
        it's kept in the locator as `vlm_bbox` so the caller can crop the
        audit image tight to the figure instead of dumping the whole page;
        it never promotes the block out of "unverified" since it's the
        model's own guess, not an independent geometric source. `emit_figures`
        defaults to False so a caller that doesn't pass `geom_images` never
        has to think about figures at all.
        """
        geom_tables = geom_tables if geom_tables is not None else []
        geom_images = geom_images if geom_images is not None else []
        blocks: list[Block] = []
        cur_list: str | None = None
        for sp in specs:
            span = Span(page=pageno)
            if sp["type"] == "list_item":
                if cur_list is None:
                    cur_list = ct.lid()
                blocks.append(ParagraphBlock(
                    id=ct.bid(), span=span, text=sp["text"], list_id=cur_list,
                    list_level=sp["level"], list_ordered=sp["ordered"]))
                continue
            cur_list = None
            if sp["type"] == "heading":
                blocks.append(HeadingBlock(id=ct.bid(), span=span,
                                           text=sp["text"], level=sp["level"]))
            elif sp["type"] == "paragraph":
                blocks.append(ParagraphBlock(id=ct.bid(), span=span,
                                             text=sp["text"]))
            elif sp["type"] == "figure":
                if not emit_figures:
                    continue
                visual_type = visual_type_source = visual_type_confidence = None
                if geom_images:
                    im = geom_images.pop(0)
                    locator = _img_locator(pageno, im)
                    # Set only when `im` came from _vlm_figure_regions (already
                    # classified by the same call that grounded its bbox) --
                    # `_page_images` dicts have no such keys, so this is a
                    # no-op fallback path when the geometric detector was used.
                    visual_type = im.get("visual_type")
                    visual_type_source = im.get("visual_type_source")
                    visual_type_confidence = im.get("visual_type_confidence")
                else:
                    locator = {"page": pageno, "region": "vlm_figure"}
                    if "bbox" in sp:
                        locator["vlm_bbox"] = list(sp["bbox"])
                blocks.append(ImageBlock(id=ct.bid(), span=span,
                                         image_index=ct.img(), locator=locator,
                                         visual_type=visual_type,
                                         visual_type_source=visual_type_source,
                                         visual_type_confidence=visual_type_confidence))
            elif geom_tables:  # table, grounded in a detected geometric region
                tb = TableBlock(id=ct.bid(), span=span)
                gt = geom_tables.pop(0)
                tb.table = _as_table_data(gt)
                # A geometric/band/model-built table is grounded in the
                # PDF's own text; carry its provenance + confidence so the
                # hybrid verification loop leaves it alone (see _parse_hybrid)
                # instead of re-deriving a weaker text-containment label.
                if isinstance(gt, _PreparedTable):
                    tb.provenance = gt.provenance
                    tb.table_confidence = gt.confidence
                    tb.table_flags = gt.flags or None
                else:
                    tb.provenance = PROV_TEXT_LAYER
                blocks.append(tb)
            else:
                # table, UNGROUNDED: the VLM says "table" but pdfplumber found
                # no matching geometric region at all -- the weakest-evidence
                # table candidate in the whole pipeline (no ruling lines, no
                # crop to classify by vision, only the VLM's own transcribed
                # cell text). One more check before this becomes a real
                # TableBlock: a text-only classification pass (no image
                # available here, unlike the geometric-candidate gate in
                # `_build_region_tables`) over the cell text itself. Fail-open
                # (verdict is None, no text LLM reachable) keeps today's
                # behavior -- this path already goes through the hybrid loop's
                # own containment verification, so it's not left unchecked.
                rows = sp["rows"]
                n_cols = max(len(r) for r in rows)
                text_rows = [[r[i] if i < len(r) else "" for i in range(n_cols)]
                            for r in rows]
                visual_cfg = get_config().visual
                llm = self._text_llm() if visual_cfg.classify else None
                verdict = (classify_table_text(text_rows, llm, visual_cfg)
                          if llm is not None else None)
                if verdict is not None and verdict.visual_type != "table":
                    locator: dict[str, Any] = {
                        "page": pageno, "region": "vlm_table_reclassified"}
                    if "bbox" in sp:
                        locator["vlm_bbox"] = list(sp["bbox"])
                    blocks.append(ImageBlock(
                        id=ct.bid(), span=span, image_index=ct.img(),
                        locator=locator, visual_type=verdict.visual_type,
                        visual_type_source=verdict.source,
                        visual_type_confidence=verdict.confidence))
                else:
                    tb = TableBlock(id=ct.bid(), span=span)
                    cells = [[text_cell(c) for c in row] for row in text_rows]
                    tb.table = TableData(n_rows=len(rows), n_cols=n_cols,
                                         cells=cells)
                    # unpaired: the VLM's own ungrounded grid -- left without a
                    # provenance so the hybrid loop verifies it by containment.
                    blocks.append(tb)
        return blocks

    # -- hybrid path -----------------------------------------------------------

    def _parse_hybrid(self, page, pageno: int, prep: dict, ct: _Counters,
                      meta: dict[str, Any],
                      heading_sizes: list[float] | None = None,
                      running_lines: frozenset[str] | None = None,
                      declared: frozenset[str] = frozenset(),
                      ) -> list[Block]:
        vlm = self._primary()
        if vlm is None:
            meta["used"] = "code"
            meta["note"] = (f"{self._vlm_disabled or 'no VLM configured'};"
                            " code-path fallback")
            return self._parse_code(page, pageno, prep, ct, heading_sizes,
                                    running_lines)

        png = prep.get("png") or self._render(page)
        # Whole-figure bboxes from a single VLM_CLASSIFY page-region call
        # (see _vlm_figure_regions) replace `_page_images`' raw raster
        # fragments when available -- a technical drawing exported as many
        # small tiles/arrowheads/dimension marks becomes one region instead
        # of one ImageBlock per fragment. Falls back to `_page_images` when
        # unavailable (no VLM_CLASSIFY configured, or no regions came back).
        vlm_classify = get_vlm_classify_client(vlm) if get_config().visual.classify else None
        images = _vlm_figure_regions(page, png, vlm_classify)
        if images is None:
            images = _page_images(page)
        # Sorted so positional pairing with the VLM's own reading-order
        # "figure" specs (see _blocks_from_vlm) lines up top-to-bottom;
        # computed before the read (not just for that pairing, further
        # below) so `_mask_figures` can blank these same regions in the
        # image sent to the model -- see its docstring.
        images = sorted(images, key=lambda im: (im["top"], im["x0"]))
        masked = _mask_figures(page, [(im["x0"], im["top"], im["x1"], im["bottom"])
                                      for im in images])
        grounding = page.extract_text() or ""
        specs = self._read_page(vlm, masked or png, _hybrid_user(pageno, grounding))
        if specs is None:
            meta["used"] = "code"
            kind, detail = self._last_read_failure or ("unknown", "")
            meta["note"] = f"VLM page read failed ({kind}); code-path fallback"
            if detail:
                meta["vlm_error"] = detail
            return self._parse_code(page, pageno, prep, ct, heading_sizes,
                                    running_lines)

        page_tokens = _counter(_tok_list(grounding))
        page_crit = _criticals(grounding)
        contain = get_config().pdf.containment
        second = self._second_reader(png)
        tboxes = _table_boxes(prep)
        word_stream = _word_marks_stream(prep, tboxes)
        align_pos = [0]

        # Reading-order copy _blocks_from_vlm pops paired tables out of in
        # place; whatever it leaves behind is a geometric table the VLM never
        # tagged as a "table" (it transcribed the region as paragraphs), which
        # the unpaired-table pass below folds back in. A copy so prep["tables"]
        # -- shared with the code path / header recovery -- stays intact.
        unpaired_tables = sorted(prep.get("tables") or [],
                                 key=lambda t: (t.bbox[1], t.bbox[0]))
        blocks = self._blocks_from_vlm(specs, pageno, ct,
                                       geom_tables=unpaired_tables,
                                       geom_images=images, emit_figures=True)
        for b in blocks:
            if isinstance(b, ImageBlock):
                if "bbox" not in b.locator:
                    # VLM claimed a figure pdfplumber's object model can't
                    # back geometrically (e.g. a vector drawing) -- unverified,
                    # with a tight audit crop only if the VLM guessed a bbox;
                    # no bbox guess means no crop rather than a full-page dump.
                    b.provenance = PROV_UNVERIFIED
                    crop = _figure_crop(png, b.locator)
                    if crop is not None:
                        b.source_crop = _store_crop(crop)
                continue
            if isinstance(b, TableBlock) and b.provenance is not None:
                # A table already grounded by the band builder / model referee
                # (paired from prep["tables"]); its cells are the PDF's own
                # words. Keep that specific provenance instead of overwriting it
                # with a coarser whole-table text-containment label.
                _flatten_newlines(b)
                continue
            text = _block_text(b)
            tokens = _tok_list(text)
            if (page_tokens and _containment(tokens, page_tokens) >= contain
                    and _crit_ok(text, page_crit)):
                b.provenance = PROV_TEXT_LAYER
            else:
                sec = second()
                if (sec and _containment(tokens, sec[0]) >= contain
                        and not (_criticals(text) - sec[1])):
                    b.provenance = PROV_CONSENSUS
                else:
                    b.provenance = PROV_UNVERIFIED
                    b.source_crop = _store_crop(png)
            _flatten_newlines(b)
            if (b.provenance == PROV_TEXT_LAYER
                    and isinstance(b, (ParagraphBlock, HeadingBlock))):
                # The block's own wording is now confirmed against this same
                # page's text layer, so its font names are a real source --
                # the same deterministic bold/italic signal the code path
                # reads, not a VLM guess. Unconfirmed blocks are skipped: their
                # wording may not line up with the text layer at all, so
                # aligning against it could attach wrong marks.
                runs = _align_runs(b.text, word_stream, align_pos)
                if runs_have_marks(runs):
                    b.runs = runs

        # Fold in every geometric table the VLM never tagged as a "table"
        # (it transcribed the region as paragraphs, so the positional pairing
        # in _blocks_from_vlm never consumed it). Without this the whole table
        # would be lost, surviving only as the VLM's flattened prose.
        blocks = _merge_unpaired_tables(blocks, unpaired_tables, pageno, ct)

        # Recover any heading the VLM dropped outright but the text layer
        # still holds (see _recover_dropped_headings) -- before the trailing
        # image append below, so a recovered heading slots into the content
        # flow by position rather than after every figure.
        recovered = _recover_dropped_headings(blocks, page, prep, pageno,
                                              running_lines, ct, declared)
        if recovered:
            meta["recovered_headings"] = recovered

        # Anything the VLM didn't call out as a figure still gets a block --
        # deterministic, from the page's own object model (or, when
        # available, the grounded regions from _vlm_figure_regions) -- just
        # without a reading-order position, since nothing anchors it in the
        # VLM's flow.
        for im in images:
            blocks.append(ImageBlock(id=ct.bid(), span=Span(page=pageno),
                                     image_index=ct.img(),
                                     locator=_img_locator(pageno, im),
                                     visual_type=im.get("visual_type"),
                                     visual_type_source=im.get("visual_type_source"),
                                     visual_type_confidence=im.get("visual_type_confidence")))

        # A find_tables() region classification determined is NOT a real
        # table (see PdfParser._build_region_tables) -- already removed from
        # prep["tables"]/geom_tables, so it never entered the VLM
        # table-pairing above; appended here for the same "no reading-order
        # position" reason as an unpaired image above.
        for d in prep.get("diverted_images") or []:
            blocks.append(ImageBlock(
                id=ct.bid(), span=Span(page=pageno), image_index=ct.img(),
                locator={"page": pageno, "bbox": list(d["bbox"]),
                        "region": "reclassified_visual"},
                source_crop=d["crop_sha"], visual_type=d["visual_type"],
                visual_type_source=d["visual_type_source"],
                visual_type_confidence=d.get("visual_type_confidence")))
        return blocks

    # -- scanned path ------------------------------------------------------------

    def _tesseract_fallback(self, page, pageno: int, ct: _Counters,
                            meta: dict[str, Any], png: bytes | None
                            ) -> list[Block]:
        """Last-resort TEXT for a scanned page whose VLM read failed.
        Tesseract is already in the tree as the
        scanned-path verifier (`TesseractDetector`); here its reading
        becomes low-trust ParagraphBlocks (provenance "unverified", noted
        "tesseract-fallback" in the page meta) so the page's text at least
        exists for search/RAG even with every model down. The caller still
        keeps the full-page ImageBlock alongside. Disable with
        PDF_TESSERACT_FALLBACK=0."""
        if not get_config().pdf.tesseract_fallback:
            return []
        detector = self._get_detector()
        if detector is None:
            return []
        if png is None:
            png = self._render(page)
        try:
            det_lines = detector.detect(png)
        except Exception:  # noqa: BLE001 -- no tesseract binary etc.: the detector is disabled permanently, no fallback text is produced (see the comment in the block)
            # Missing tesseract binary etc.: same handling as the verifier.
            self._detector = None
            return []
        paras = _det_paragraphs(det_lines)
        if not paras:
            return []
        meta["note"] = (f"{meta['note']}; tesseract-fallback text emitted"
                        " (unverified)" if meta.get("note")
                        else "tesseract-fallback text emitted (unverified)")
        return [ParagraphBlock(id=ct.bid(), span=Span(page=pageno),
                               text=text, provenance=PROV_UNVERIFIED)
                for text in paras]

    def _parse_scanned(self, page, pageno: int, ct: _Counters,
                       meta: dict[str, Any]) -> list[Block]:
        vlm = self._primary()
        specs = None
        png: bytes | None = None
        if vlm is not None:
            png = self._render(page)
            specs = self._read_page(vlm, png, _scanned_user(pageno))
        if specs is None or png is None:
            # No VLM (or the read failed): keep the page as one full-page
            # image so the images/ OCR stage can still recover its text --
            # plus tesseract's own reading as unverified text (see
            # `_tesseract_fallback`): the images stage OCRs with the SAME
            # VLM that just failed, so in a systemic outage it alone is no
            # fallback at all.
            meta["used"] = "image-fallback"
            if vlm is None:
                meta["note"] = (f"{self._vlm_disabled or 'no VLM configured'};"
                                " page kept as image")
            else:
                kind, detail = self._last_read_failure or ("unknown", "")
                meta["note"] = (f"VLM page read failed ({kind});"
                                " page kept as image")
                if detail:
                    meta["vlm_error"] = detail
            return self._tesseract_fallback(page, pageno, ct, meta, png) + [
                ImageBlock(
                    id=ct.bid(), span=Span(page=pageno), image_index=ct.img(),
                    locator={"page": pageno, "region": "full_page",
                             "bbox": [0, 0, round(float(page.width), 2),
                                      round(float(page.height), 2)]})]

        detector = self._get_detector()
        det_lines: list[DetectedLine] = []
        if detector is not None:
            try:
                det_lines = detector.detect(png)
            except Exception:  # noqa: BLE001 -- no tesseract: the detector is disabled and we drop to VLM2-only checks
                # Missing tesseract binary etc.: fall back to VLM2-only checks.
                self._detector = None
        det_text = "\n".join(dl.text for dl in det_lines)
        det_tokens = _counter(_tok_list(det_text))
        det_crit = _criticals(det_text)
        second = self._second_reader(png)

        # Real embedded raster objects, excluding ones spanning most of the
        # page: on a true scan that IS the scan itself (the whole-page
        # background raster), not a distinct callout figure, and must not be
        # allowed to "win" the positional pairing below.
        page_area = float(page.width) * float(page.height) or 1.0
        images = sorted(
            (im for im in _page_images(page)
             if (im["x1"] - im["x0"]) * (im["bottom"] - im["top"]) <= 0.6 * page_area),
            key=lambda im: (im["top"], im["x0"]))

        blocks = self._blocks_from_vlm(specs, pageno, ct,
                                       geom_images=images, emit_figures=True)
        if not blocks:
            # Defensive: every spec type now yields a block once figures are
            # emitted, so this shouldn't trigger in practice any more, but a
            # page that somehow produces nothing must still not vanish.
            meta["used"] = "image-fallback"
            meta["note"] = "VLM page read had no transcribable text; page kept as image"
            return [ImageBlock(
                id=ct.bid(), span=Span(page=pageno), image_index=ct.img(),
                locator={"page": pageno, "region": "full_page",
                         "bbox": [0, 0, round(float(page.width), 2),
                                  round(float(page.height), 2)]})]
        for b in blocks:
            if isinstance(b, ImageBlock):
                if "bbox" not in b.locator:
                    # No independent source can confirm "there is a figure
                    # here" the way the detector confirms text -- unverified,
                    # with a tight audit crop only if the VLM guessed a bbox;
                    # no bbox guess means no crop rather than a full-page dump.
                    b.provenance = PROV_UNVERIFIED
                    crop = _figure_crop(png, b.locator)
                    if crop is not None:
                        b.source_crop = _store_crop(crop)
                continue
            failed_boxes: list[tuple[float, float, float, float]] = []
            all_ok = True
            for line in _block_lines(b):
                ok, bbox = self._verify_line(line, det_lines, det_tokens,
                                             det_crit, png, second)
                if not ok:
                    all_ok = False
                    if bbox is not None:
                        failed_boxes.append(bbox)
            if all_ok:
                b.provenance = PROV_CONSENSUS
            else:
                b.provenance = PROV_UNVERIFIED
                crop = (_crop_png(png, _union_bbox(failed_boxes))
                        if failed_boxes else png)
                b.source_crop = _store_crop(crop)
            _flatten_newlines(b)

        # Completeness check: the full-page fallback above
        # fires only when the read failed outright -- a PARTIAL read (2-3
        # blocks off a dense page) used to keep only what the model
        # transcribed, with nothing measuring the loss. The detector's
        # independent reading is already in hand: when the emitted blocks
        # cover too little of it, keep the full page render as an extra
        # ImageBlock (recoverable downstream) and say so in the page note.
        if det_tokens and sum(det_tokens.values()) >= 20:
            blk_tokens = _counter([t for b in blocks
                                   for t in _tok_list(_block_text(b))])
            covered = _containment(list(det_tokens.elements()), blk_tokens)
            if covered < get_config().pdf.scanned_coverage_min:
                meta["note"] = (f"partial-vlm-read: blocks cover only "
                                f"{covered:.0%} of detector-read text;"
                                " full-page image kept")
                blocks.append(ImageBlock(
                    id=ct.bid(), span=Span(page=pageno), image_index=ct.img(),
                    locator={"page": pageno, "region": "full_page",
                             "bbox": [0, 0, round(float(page.width), 2),
                                      round(float(page.height), 2)]}))

        # Any real sub-figure the VLM didn't call out still gets a block --
        # deterministic, from the PDF's own object model -- just without a
        # reading-order position.
        for im in images:
            blocks.append(ImageBlock(id=ct.bid(), span=Span(page=pageno),
                                     image_index=ct.img(),
                                     locator=_img_locator(pageno, im)))
        return blocks

    def _verify_line(self, line: str, det_lines: list[DetectedLine],
                     det_tokens: Counter, det_crit: Counter, png: bytes,
                     second) -> tuple[bool, tuple | None]:
        """Check one VLM line against the independent sources.

        Returns (ok, bbox): `bbox` is the mapped detector region when the line
        mapped somewhere but could not be confirmed (used for the audit crop).
        No source ever "wins" a disagreement — an unconfirmed line just stays
        unverified.
        """
        if not _norm(line):
            return True, None
        line_ok = get_config().pdf.line_match
        contain = get_config().pdf.containment

        best: DetectedLine | None = None
        best_r = 0.0
        for dl in det_lines:
            r = _ratio(line, dl.text)
            if r > best_r:
                best_r, best = r, dl

        # 1. Direct consensus with the detector's own reading of that line.
        if best is not None and best_r >= line_ok and _crit_ok(line, det_crit):
            return True, None
        # 1b. Reflowed lines: containment against the detector's full text.
        if (det_tokens and _containment(_tok_list(line), det_tokens) >= contain
                and _crit_ok(line, det_crit)):
            return True, None
        # 2. Mapped but disputed region: crop-and-reread with the second model.
        if best is not None and best_r >= 0.5:
            reread = self._reread(_crop_png(png, best.bbox))
            if reread and _crit_ok(line, _criticals(reread)) and (
                    _ratio(line, reread) >= line_ok
                    or _containment(_tok_list(line),
                                    _counter(_tok_list(reread))) >= contain):
                return True, None
            return False, best.bbox
        # 3. Maps to no detector region (hallucination suspicion): only an
        #    independent full second reading can still confirm it.
        sec = second()
        if (sec and _containment(_tok_list(line), sec[0]) >= contain
                and not (_criticals(line) - sec[1])):
            return True, None
        return False, None


# ---------------------------------------------------------------------------
# Block text views used by verification
# ---------------------------------------------------------------------------


def _block_text(b: Block) -> str:
    """Flat text of a block for containment checks."""
    if isinstance(b, TableBlock):
        return " ".join(
            c.plain_text() for row in b.table.cells for c in row if c is not None)
    return getattr(b, "text", "")


def _block_lines(b: Block) -> list[str]:
    """Printed lines of a block (the unit the scanned path verifies)."""
    if isinstance(b, TableBlock):
        return [" ".join(c.plain_text() for c in row if c is not None)
                for row in b.table.cells]
    text = getattr(b, "text", "")
    return [ln for ln in text.split("\n") if ln.strip()]


def _flatten_newlines(b: Block) -> None:
    """Collapse the VLM's preserved line breaks to the IR's canonical
    single-space text (matching every other parser) once verification, which
    needed the printed lines, is done."""
    if isinstance(b, (ParagraphBlock, HeadingBlock)):
        b.text = " ".join(b.text.split())


# Fraction of a VLM paragraph/heading's tokens that must already sit inside an
# unpaired table's own cells for that block to count as the table -- itself
# transcribed as prose -- and be dropped in favour of the grid. High on
# purpose: only a block whose text is almost entirely the table's own words is
# removed, so real prose that merely mentions the same terms is never lost.
_UNPAIRED_TABLE_CONTAIN = 0.8
# Ignore blocks too short to attribute confidently (a one- or two-word line can
# sit inside a table's tokens by coincidence).
_UNPAIRED_TABLE_MIN_TOKENS = 3


def _merge_unpaired_tables(blocks: list[Block], unpaired_tables: list,
                           pageno: int, ct: _Counters) -> list[Block]:
    """Fold geometric tables the VLM never emitted a "table" spec for back into
    the block stream (hybrid path). `_build_region_tables` already built these
    regions' grids from the PDF's own geometry; the VLM just transcribed each
    region as ordinary paragraphs instead of tagging it a table, so the
    positional pairing in `_blocks_from_vlm` never consumed it -- and the whole
    table would otherwise be lost, surviving only as flattened prose.

    For each unpaired table the paragraph/heading blocks whose text is almost
    entirely contained in the table's own cells are dropped and the table takes
    the position of the first of them -- nothing is lost, since that text IS the
    table, verbatim from the PDF, and it lands in reading order rather than at
    the end. When no block matches (the VLM didn't transcribe the region at
    all) the table is appended so it can never silently disappear either way.
    Only the model's paragraph guess is ever discarded; every cell's text is
    the PDF's own characters, so this can no more hallucinate content than the
    band builder it comes from. `unpaired_tables` is consumed top-to-bottom and
    matched blocks are removed as we go, so two tables can't claim the same
    block (e.g. a header line whose terms appear in both)."""
    if not unpaired_tables:
        return blocks
    out = list(blocks)
    for t in unpaired_tables:
        tb = TableBlock(id=ct.bid(), span=Span(page=pageno))
        tb.table = _as_table_data(t)
        if isinstance(t, _PreparedTable):
            tb.provenance = t.provenance
            tb.table_confidence = t.confidence
            tb.table_flags = (list(t.flags) if t.flags else []) + ["vlm-unpaired"]
        else:
            tb.provenance = PROV_TEXT_LAYER
            tb.table_flags = ["vlm-unpaired"]
        table_counter = _counter(_tok_list(_block_text(tb)))
        hit: list[int] = []
        for i, b in enumerate(out):
            if not isinstance(b, (ParagraphBlock, HeadingBlock)):
                continue
            toks = _tok_list(_block_text(b))
            if (len(toks) >= _UNPAIRED_TABLE_MIN_TOKENS
                    and _containment(toks, table_counter) >= _UNPAIRED_TABLE_CONTAIN):
                hit.append(i)
        if hit:
            hitset = set(hit)
            kept = [b for i, b in enumerate(out) if i not in hitset]
            kept.insert(hit[0], tb)
            out = kept
        else:
            out.append(tb)
    return out
