"""Parser tuning configuration: `config/default.toml` → `ParserConfig`.

Type split (root `CONFIG.md`): mathematical/behavioral tuning lives HERE (TOML);
model connection identity + paths live in `.env` (see `llm/__init__.py`,
`storage_paths.py`). This module does NOT read secrets.

Precedence (last writer wins):
    default.toml  →  PARSER_CONFIG (optional partial TOML)  →  env variables

So every tuning knob can still be overridden by its UPPER_CASE env variable name
(the `_ENV_OVERRIDES` registry keeps those names exactly) — a
`pipeline/cli/config/*.env` profile that sets an env var like `PDF_VLM=0`
overrides the TOML default and keeps working. The env casters replicate the old `_env_int/_env_float/_env_bool`
semantics exactly: empty/whitespace or invalid value = the override is SKIPPED
(the TOML default stands).

Usage: library code reads `get_config().pdf.render_dpi` and friends. `get_config()`
reads the env FRESH on every call (the TOML dict is read once at module load) — so
test env monkeypatching works with no extra plumbing and file I/O is not repeated.
Validation runs on every call (fail-loud).
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class ImageOcr(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Whether visual OCR (VLM transcription) should run at all. DIFFERENT from
    # `[visual] exclude_types`: that one is type-based and requires `classify`
    # to be on, this one stops unconditionally (see the default.toml note).
    enabled: bool
    confidence_threshold: float = Field(ge=0)
    max_tokens: int = Field(ge=1)
    concurrency: int = Field(ge=0)  # guarded by max(1, ...) at the use site
    fallback: str


class PdfTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bands_min_confidence: float = Field(ge=0)
    struct_min_coverage: float = Field(ge=0)


class Pdf(BaseModel):
    model_config = ConfigDict(extra="forbid")

    render_dpi: int = Field(ge=1)
    vlm: bool
    vlm_max_tokens: int = Field(ge=1)
    heading_reconcile: bool
    fullpage_render_min_vector: int = Field(ge=0)
    broken_cmap_min: int = Field(ge=0)
    scanned_coverage_min: float = Field(ge=0)
    line_match: float = Field(ge=0)
    containment: float = Field(ge=0)
    tesseract_fallback: bool
    table: PdfTable


class DescribeContext(BaseModel):
    """Injection of the text around the described block (heading chain +
    preceding/following paragraphs) into the prompt. TYPE-INDEPENDENT: both a
    table's unit and a block diagram's subject live in the surrounding
    sentences — the concept was never table-specific."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    before: int = Field(ge=0)
    after: int = Field(ge=0)
    max_chars: int = Field(ge=0)


class Describe(BaseModel):
    """SHARED description-generation settings for all "describable block" types
    (see parsers/base.py `DescribableBlock`). Only the STRATEGY changes per type
    (table: LLM from cells; chart/SmartArt: deterministic from their own XML;
    other visuals: VLM + context) — these knobs are the same for all of them."""

    model_config = ConfigDict(extra="forbid")

    # Whether description generation should run at all. When off, `types` is not
    # consulted and `visual_check` never kicks in either (the verifier lives
    # inside description generation).
    enabled: bool
    types: list[str]
    max_tokens: int = Field(ge=1)
    concurrency: int = Field(ge=0)     # guarded by max(1, ...)
    # VLM-vision visuals: classification confidence threshold (0..1). Below it
    # the type falls to the "unknown" strategy in the describe prompt (see
    # default.toml + describe/core.py). 0.0 = off.
    min_visual_confidence: float = Field(ge=0, le=1)
    # Minimum crop edge for VLM-vision (pixels). A visual whose shortest edge is
    # below this is NEVER described (it stays type + placeholder only) — a crop
    # that small is unreadable, yet the prompt still asks for 1-4 sentences, so
    # the model fabricates (see default.toml). 0 = off.
    min_visual_edge_px: int = Field(ge=0)
    # Verify VLM-vision descriptions with a second VISION call + retry
    # (the visual counterpart of the table's [table].llm_check). It audits
    # whether the description claims something not visible in the image itself.
    # One extra call per visual on a single GPU → opt-in, off by default.
    visual_check: bool = False
    visual_check_retries: int = Field(ge=0)  # guarded by max(1, ...)
    context: DescribeContext


class Table(BaseModel):
    """Strategy knobs specific to TABLES only. The table's real difference lives
    here: because it has cell data, its render can be row-limited (max_rows),
    its description can be verified by a second LLM and retried
    (llm_check/check_retries), and its facts can pass the digit-check (facts).
    The shared knobs are in `[describe]`."""

    model_config = ConfigDict(extra="forbid")

    max_rows: int = Field(ge=0)
    llm_check: bool
    facts: bool
    header_llm: bool
    check_retries: int = Field(ge=0)   # guarded by max(1, ...)


class TableStruct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(ge=1)


class Progress(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bar: bool


class Visual(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classify: bool
    extract_charts: bool
    extract_smartart: bool
    exclude_types: list[str] = Field(default_factory=list)
    types: list[str]
    max_tokens: int = Field(ge=1)


class ParserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ocr: ImageOcr
    pdf: Pdf
    describe: Describe
    table: Table
    table_struct: TableStruct
    progress: Progress
    visual: Visual


# --- env override registry: legacy env variable name → (TOML path, caster) ----
# This table is the single documented env override for every tuning knob; the
# names are kept for backward compatibility (the pipeline profiles depend on
# them).

_SKIP = object()  # if a caster returns this, the override is not applied (TOML default stands)


def _cast_int(raw: str) -> Any:
    if not raw.strip():
        return _SKIP
    try:
        return int(raw)
    except ValueError:
        return _SKIP


def _cast_float(raw: str) -> Any:
    if not raw.strip():
        return _SKIP
    try:
        return float(raw)
    except ValueError:
        return _SKIP


def _cast_bool(raw: str) -> Any:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _cast_barbool(raw: str) -> Any:
    # PROGRESS_BAR inverted logic: only these values turn the bar OFF.
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _cast_str(raw: str) -> Any:
    return raw


_ENV_OVERRIDES: dict[str, tuple[tuple[str, ...], Any]] = {
    "IMAGE_OCR_ENABLED": (("image_ocr", "enabled"), _cast_bool),
    "IMAGE_OCR_CONFIDENCE_THRESHOLD": (("image_ocr", "confidence_threshold"), _cast_float),
    "IMAGE_OCR_MAX_TOKENS": (("image_ocr", "max_tokens"), _cast_int),
    "IMAGE_CONCURRENCY": (("image_ocr", "concurrency"), _cast_int),
    "IMAGE_OCR_FALLBACK": (("image_ocr", "fallback"), _cast_str),
    "PDF_RENDER_DPI": (("pdf", "render_dpi"), _cast_int),
    "PDF_VLM": (("pdf", "vlm"), _cast_bool),
    "PDF_VLM_MAX_TOKENS": (("pdf", "vlm_max_tokens"), _cast_int),
    "PDF_HEADING_RECONCILE": (("pdf", "heading_reconcile"), _cast_bool),
    "PDF_FULLPAGE_RENDER_MIN_VECTOR": (("pdf", "fullpage_render_min_vector"), _cast_int),
    "PDF_BROKEN_CMAP_MIN": (("pdf", "broken_cmap_min"), _cast_int),
    "PDF_SCANNED_COVERAGE_MIN": (("pdf", "scanned_coverage_min"), _cast_float),
    "PDF_LINE_MATCH": (("pdf", "line_match"), _cast_float),
    "PDF_CONTAINMENT": (("pdf", "containment"), _cast_float),
    "PDF_TESSERACT_FALLBACK": (("pdf", "tesseract_fallback"), _cast_bool),
    "PDF_TABLE_BANDS_MIN_CONFIDENCE": (("pdf", "table", "bands_min_confidence"), _cast_float),
    "PDF_TABLE_STRUCT_MIN_COVERAGE": (("pdf", "table", "struct_min_coverage"), _cast_float),
    "TABLE_MAX_ROWS": (("table", "max_rows"), _cast_int),
    "TABLE_LLM_CHECK": (("table", "llm_check"), _cast_bool),
    "TABLE_FACTS": (("table", "facts"), _cast_bool),
    "TABLE_HEADER_LLM": (("table", "header_llm"), _cast_bool),
    "TABLE_CHECK_RETRIES": (("table", "check_retries"), _cast_int),
    # `[describe]`: the shared knobs, for ALL describable block types (see
    # `Describe`) — never table-specific. These used to live under
    # `[table.context]`/`[table].concurrency` with `TABLE_CONTEXT*`/
    # `TABLE_CONCURRENCY` env names; they were consolidated under the single
    # canonical `DESCRIBE_*` names. There is deliberately NO legacy alias: two
    # names must not govern one setting.
    "DESCRIBE_ENABLED": (("describe", "enabled"), _cast_bool),
    "DESCRIBE_CONCURRENCY": (("describe", "concurrency"), _cast_int),
    "DESCRIBE_MAX_TOKENS": (("describe", "max_tokens"), _cast_int),
    "DESCRIBE_MIN_VISUAL_CONFIDENCE": (("describe", "min_visual_confidence"), _cast_float),
    "DESCRIBE_MIN_VISUAL_EDGE_PX": (("describe", "min_visual_edge_px"), _cast_int),
    "DESCRIBE_VISUAL_CHECK": (("describe", "visual_check"), _cast_bool),
    "DESCRIBE_VISUAL_CHECK_RETRIES": (("describe", "visual_check_retries"), _cast_int),
    "DESCRIBE_CONTEXT": (("describe", "context", "enabled"), _cast_bool),
    "DESCRIBE_CONTEXT_BEFORE": (("describe", "context", "before"), _cast_int),
    "DESCRIBE_CONTEXT_AFTER": (("describe", "context", "after"), _cast_int),
    "DESCRIBE_CONTEXT_MAX_CHARS": (("describe", "context", "max_chars"), _cast_int),
    "TABLE_STRUCT_MAX_TOKENS": (("table_struct", "max_tokens"), _cast_int),
    "PROGRESS_BAR": (("progress", "bar"), _cast_barbool),
    "VISUAL_CLASSIFY": (("visual", "classify"), _cast_bool),
    "VISUAL_EXTRACT_CHARTS": (("visual", "extract_charts"), _cast_bool),
    "VISUAL_EXTRACT_SMARTART": (("visual", "extract_smartart"), _cast_bool),
    "VISUAL_MAX_TOKENS": (("visual", "max_tokens"), _cast_int),
}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Builds a nested override dict from the variables defined in
    `_ENV_OVERRIDES` and PRESENT in `env`. If a caster returns `_SKIP`, that
    variable is skipped."""
    out: dict[str, Any] = {}
    for name, (path, caster) in _ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is None:
            continue
        value = caster(raw)
        if value is _SKIP:
            continue
        _set_path(out, path, value)
    return out


# default.toml is read once (immutable); the env is applied fresh in every get_config().
_DEFAULT_DATA = _read_toml(DEFAULT_CONFIG_PATH)


def load_config(env: Mapping[str, str] | None = None,
                override: str | Path | None = None) -> ParserConfig:
    """Merges and validates default.toml + the (optional) PARSER_CONFIG/`override`
    TOML + env overrides. If `env` is not given, uses `os.environ`."""
    if env is None:
        env = os.environ
    data = copy.deepcopy(_DEFAULT_DATA)
    ov_path = override if override is not None else env.get("PARSER_CONFIG")
    if ov_path:
        data = _deep_merge(data, _read_toml(Path(ov_path)))
    data = _deep_merge(data, _env_overrides(env))
    return ParserConfig.model_validate(data)


def get_config() -> ParserConfig:
    """Config freshly validated against the current env (the entry point for
    library code). Cheap — the TOML dict is cached, only env overlay + validate."""
    return load_config()
