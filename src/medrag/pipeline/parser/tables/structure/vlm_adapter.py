"""VLM-based table structure adapter for TableStructureClient.

Uses whatever vision-language model is already configured for the rest of
the pipeline (llm.get_vlm_client() -- the same VLM_* env vars that drive OCR
in images/image_handler.py and the hybrid PDF path) to infer a table's
row/column grid, instead of a dedicated structure-recognition model like
TableFormer.

No extra dependency at all beyond what the parser already needs (llm/ is
stdlib-urllib-only, Pillow is already a parser dependency) -- this is the
"don't want a heavy install (torch/docling-ibm-models)" alternative: works
with any OpenAI-compatible VLM you already run (a local Ollama model, a
hosted API, ...).

Anchoring is CONTENT-based, not coordinate-based: general-purpose VLMs are
unreliable at pixel-exact grounding (asking one for cell bboxes used to
produce grids whose boundaries cut through the middle of real tokens while
still passing a coverage check -- every character fell in *some* cell, just
the wrong one), but they are good at reading which pieces of text belong
together in a cell. So this adapter asks for the table's LOGICAL matrix --
per cell: grid position, span, and the cell's text as the model reads it --
and returns that claimed text on `DetectedCell.text`. The caller
(pdf_parser.py's `_fill_cells_by_content`) then matches each claim against
the region's real text-layer words and rebuilds every cell from the PDF's own
characters; the model's string itself is never emitted, so a misread there
costs the match (and at worst the whole grid falls back to line geometry),
never the output's fidelity. Same "structure from the model, text from the
code" split as every other adapter -- the model only ever decides shape and
grouping.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Sequence

from medrag.pipeline.parser.config import get_config

from .base import DetectedCell, DetectedTable, TableStructureError, TextCellHint

logger = logging.getLogger(__name__)

# The token budget for grid transcription of a region comes from
# config/default.toml ([table_struct] max_tokens; override with the
# TABLE_STRUCT_MAX_TOKENS env var). A large table (many pin rows) could exceed
# 4096 and get cut off mid-JSON, so the default is kept generous (8192).

_SYSTEM = (
    "You are given a cropped image of a single table from a document page. "
    "Transcribe its grid as a JSON array of cells, nothing else.\n"
    'Each cell: {"row": int, "col": int, "rowspan": int, "colspan": int, '
    '"text": "the cell\'s content exactly as printed"}\n'
    "row/col are 0-indexed grid positions. Transcribe text EXACTLY as "
    "printed -- never correct, translate, summarize or invent anything; "
    'use "" for an empty cell.\n'
    "A merged cell gets exactly one entry with rowspan/colspan > 1; do not "
    "add separate entries for the grid positions it covers.\n"
    "Output ONLY the JSON array -- no explanation, no markdown fence."
)


class VLMStructureAdapter:
    """provider="vlm". Reuses the already-configured VLM
    (llm.get_vlm_client(), same provider/base_url/api_key as VLM_*) by
    default. `model` (TABLE_STRUCT_MODEL) overrides just the model id: grid
    extraction is a different, arguably harder task than that role's
    usual OCR transcription job, so a model that's not necessarily the best
    OCR reader (but reads structure/layout well) may be worth pointing at
    separately -- same Ollama/hosted server, different model pulled there.
    Leave unset to reuse VLM_MODEL as-is. `vlm_client` overrides the client
    entirely (tests inject a fake this way)."""

    def __init__(self, vlm_client=None, model: str | None = None) -> None:
        self._vlm = vlm_client
        self._model = model

    def _ensure_client(self):
        if self._vlm is None:
            from medrag.pipeline.parser.llm import LLMError, get_vlm_client

            try:
                self._vlm = get_vlm_client(model=self._model)
            except LLMError as exc:
                raise TableStructureError(f"no VLM configured: {exc}") from exc
        return self._vlm

    def detect(self, image: bytes, *,
               table_bboxes: Sequence[tuple[float, float, float, float]] = (),
               text_cells: Sequence[TextCellHint] = ()) -> list[DetectedTable]:
        if not table_bboxes:
            # Same contract as every adapter: this reads a grid within a
            # given region, it doesn't hunt for tables on the page itself.
            return []

        vlm = self._ensure_client()

        try:
            from PIL import Image
        except ImportError as exc:
            raise TableStructureError(
                "provider='vlm' requires Pillow (already a parser dependency)"
            ) from exc

        try:
            page_img = Image.open(io.BytesIO(image)).convert("RGB")
        except Exception as exc:
            raise TableStructureError(f"could not decode page image: {exc}") from exc

        tables: list[DetectedTable] = []
        for bbox in table_bboxes:
            x0, y0, x1, y1 = bbox
            crop = page_img.crop((max(0, int(x0)), max(0, int(y0)), int(x1), int(y1)))
            cw, ch = crop.size
            if cw <= 0 or ch <= 0:
                continue

            buf = io.BytesIO()
            crop.save(buf, format="PNG")

            try:
                raw = vlm.complete_vision(
                    system=_SYSTEM, user="Transcribe this table's grid.",
                    images=[("image/png", buf.getvalue())],
                    max_tokens=get_config().table_struct.max_tokens)
            except Exception as exc:
                # A failed CALL is the model/transport being down -- that
                # affects every region equally, so raise and let the caller
                # stop trying this page (and fall back for all its regions).
                raise TableStructureError(f"VLM structure call failed: {exc}") from exc

            cell_specs = _parse_cells_json(raw)
            if cell_specs is None:
                # Unparseable response for THIS region (e.g. the JSON array was
                # truncated by the token budget). Isolate it: skip just this
                # region so the page's other, well-formed tables still refine.
                # The caller has its own per-region fallback (a deterministic
                # builder) for a region we return nothing for, so dropping it
                # here loses no data -- unlike the old behaviour, which raised
                # and discarded every table on the page for one bad region.
                logger.warning(
                    "vlm table structure: unparseable response for region %s, "
                    "skipping it: %.120r", tuple(round(v, 1) for v in bbox), raw)
                continue
            if not cell_specs:
                continue  # valid response, model explicitly found no cells here

            cells = [
                DetectedCell(
                    row=c["row"], col=c["col"],
                    rowspan=c["rowspan"], colspan=c["colspan"],
                    text=c["text"],
                )
                for c in cell_specs
            ]
            tables.append(DetectedTable(bbox=tuple(bbox), cells=cells))
        return tables


def _parse_cells_json(raw: str | None) -> list[dict] | None:
    """Tolerant JSON-array extraction -- the same fenced-code-block
    tolerance as parsers/pdf_parser.py's own _parse_vlm_blocks, duplicated
    (not imported) since tables/structure/ deliberately doesn't depend on
    parsers/."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
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

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            row, col = int(item["row"]), int(item["col"])
        except (KeyError, TypeError, ValueError):
            continue
        text_val = item.get("text")
        out.append({
            "row": row, "col": col,
            "rowspan": max(1, int(item.get("rowspan", 1) or 1)),
            "colspan": max(1, int(item.get("colspan", 1) or 1)),
            "text": "" if text_val is None else str(text_val),
        })
    return out
