"""Configuration loading and validation: `config/default.toml` → `ChunkerConfig`.

Rule: no number that affects splitting is hardcoded in code. This module
honors that too — pydantic fields have no defaults; every value MUST come
from TOML; deleting a key from `default.toml` causes loading to fail
loudly ("fail at the boundary" principle, same as first_parse). Unknown
keys are also rejected (`extra="forbid"`).

Flex allowance formula (per-block-type ratio):

    flex(block)    = target_tokens × flex.ratios[kind]
    effective_limit = min(target_tokens + flex, hard_max_tokens)

The formula itself lives here (`ChunkerConfig.effective_limit`) so that
the coefficients and the computation stay together; the splitting engine
calls only this method.

Override story: `load_config(override=...)` accepts a partial TOML with
only the keys you want to change and deep-merges it on top of the
defaults. Secrets do not live in config — they belong in `.env`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Config folder shipped with the package. Config.py lives directly in the
# package now (it used to live two directories up as `chunker/chunker/config.py`
# with `chunker/config/` as a sibling).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class Limits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: int = Field(ge=1)
    hard_max_tokens: int = Field(ge=1)

    @model_validator(mode="after")
    def _tavan_hedefin_altinda_olamaz(self) -> Limits:
        if self.hard_max_tokens < self.target_tokens:
            raise ValueError(
                f"hard_max_tokens ({self.hard_max_tokens}) < target_tokens "
                f"({self.target_tokens}) is not allowed")
        return self


class FlexRatios(BaseModel):
    """Per-block-type flex ratio.

    Field names match the inner model's block `kind` values exactly
    (`chunker/core/document.py`); when a new block type is added, both
    here and in `default.toml` get a ratio.
    """

    model_config = ConfigDict(extra="forbid")

    table: float = Field(ge=0)
    list: float = Field(ge=0)
    code: float = Field(ge=0)
    image: float = Field(ge=0)
    paragraph: float = Field(ge=0)
    heading: float = Field(ge=0)

    def ratio_for(self, kind: str) -> float:
        """Ratio for `kind`; unknown block types are rejected loudly."""
        if kind not in type(self).model_fields:
            raise KeyError(
                f"unknown block type {kind!r}; valid types: "
                f"{sorted(type(self).model_fields)}")
        return getattr(self, kind)


class Flex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ratios: FlexRatios


class Packing(BaseModel):
    """Section (heading) boundary strategy.

    "hard": a chunk never mixes the content of two sections. "merge":
    consecutive WHOLE sections are merged into one chunk while they fit
    in the budget (no partial section overflow); the merged chunk's
    `heading_path` is the common ancestor prefix.
    """

    model_config = ConfigDict(extra="forbid")

    section_strategy: Literal["hard", "merge"]
    # Chunks below this threshold (or whose body is empty) are merged into
    # a neighbor — otherwise tiny standalone chunks fed into summarization
    # invited hallucinations. 0 = merging disabled. The "hard" strategy's
    # "sections never mix" guarantee is intentionally pierced only for
    # these sub-threshold crumbs.
    min_chunk_tokens: int = Field(ge=0)


class FallbackSplit(BaseModel):
    """Splitting blocks that don't fit even with the flex allowance, at
    natural boundaries: paragraph by sentence, list by top-level item, code
    by line. Values are the number of repeated units between consecutive
    parts (0 = no overlap)."""

    model_config = ConfigDict(extra="forbid")

    paragraph_overlap_sentences: int = Field(ge=0)
    list_overlap_items: int = Field(ge=0)
    code_overlap_lines: int = Field(ge=0)


class TableSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlap_rows: int = Field(ge=0)
    repeat_header_rows: bool
    # When the IR doesn't carry header-row info, the adapter writes this
    # default into `Table.header_rows`; once IR carries `header_rows`, the
    # adapter uses that instead.
    default_header_rows: int = Field(ge=0)
    # Injection of Table.facts into the chunk text. When enabled, facts in
    # a split table are distributed to parts by cell overlap; facts that
    # match no row (table-wide) go in every part. Only meaningful for
    # tables (facts are produced only by the table strategy — see parser
    # describe/table.py).
    inject_facts: bool


class HeadingInjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepend_to_text: bool
    separator: str
    max_levels: int = Field(ge=0)  # 0 = the whole chain


class Boilerplate(BaseModel):
    """Drop boilerplate content (Table of Contents, Revision History…)
    that matches a heading name from the chunk entirely — these sections
    add retrieval noise and carry no answer to any question."""

    model_config = ConfigDict(extra="forbid")

    # If the section's OWN heading text (casefold, trimmed) matches one
    # of these values exactly, the section (with its heading) is dropped
    # entirely. Empty list = disabled. Values are casefold-compared, so
    # case does not matter inside the list.
    drop_headings: list[str] = Field(default_factory=list)


class Images(BaseModel):
    """Render policy for an image block's OCR text in the chunk body."""

    model_config = ConfigDict(extra="forbid")

    # OCR text whose `Image.provenance == UNVERIFIED` is tagged (label
    # below). When off, the old behavior applies: plain text, indistinct
    # from the body. Default is on: unverified OCR must not be confused
    # with real table/paragraph text by the embedding or the reader.
    label_unverified_ocr: bool

    # Whether `Image.ocr_text` enters the chunk BODY. Default off: OCR
    # text is carried only in the structured channel
    # (`ChunkNode.images[].ocr_text`). Even when tagged, OCR does not
    # separate from real paragraph/table text in the body and pushes
    # retrieval toward wrong answers (e.g. a "MPLS / 10G / VLANs" word
    # salad OCR'd from a product photo reads like that product's spec).
    # true = old behavior.
    inject_ocr: bool = False


class Visual(BaseModel):
    """Chunk-time exclusion by visual-region type (visual classification plan).

    Independent from parse-time exclusion: even after a full-include
    parse, a type listed here never enters a chunk — IR content is NOT
    deleted, just not rendered. To reverse, change only this config; no
    re-parse needed.
    """

    model_config = ConfigDict(extra="forbid")

    # Image/table blocks tagged with one of these types do NOT contribute
    # content to the chunk text — a placeholder (type + reference) is
    # rendered instead. Empty list = no type is excluded (default: all
    # included).
    exclude_types: list[str] = Field(default_factory=list)
    # Whether the parser-produced description (Table.description/Image.description)
    # enters the chunk text. Both share the same flag (a table is not a
    # separate branch of this category — see parser IR's DescribableBlock).
    # When on, for tables it's prepended to every split part; for images
    # it's injected alongside OCR/alt_text.
    inject_description: bool
    # Separate gate for IMAGE description (`Image.description` only) — NARROWS
    # the flag above for the image case (it doesn't broaden it). Rationale:
    # the two descriptions' risks are NOT the same — a table description
    # summarizes a table that already sits in the chunk (commentary); an
    # image description is a VLM interpretation of a photo and reads like
    # a spec (e.g. `connector_types` = "On the left side (labeled REAR
    # PANEL), there are input ports for..."). Measured from specs.db:
    # 188 rows from image descriptions, 63 from table descriptions. Same
    # rationale as `[images] inject_ocr = false`. Requires `inject_description`
    # to be on — this field can only narrow, not broaden.
    inject_image_description: bool = False


class Enrichment(BaseModel):
    # The RAPTOR flag was removed.
    model_config = ConfigDict(extra="forbid")

    cross_refs: bool


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limits: Limits
    flex: Flex
    packing: Packing
    fallback_split: FallbackSplit
    table_split: TableSplit
    heading_injection: HeadingInjection
    boilerplate: Boilerplate
    images: Images
    visual: Visual
    enrichment: Enrichment

    def flex_allowance(self, kind: str) -> int:
        """Allowance granted to avoid splitting a block (tokens, floored):
        flex(block) = target_tokens × ratios[kind]."""
        return int(self.limits.target_tokens * self.flex.ratios.ratio_for(kind))

    def effective_limit(self, kind: str) -> int:
        """Effective upper limit for that block type:
        min(target_tokens + flex, hard_max_tokens)."""
        return min(self.limits.target_tokens + self.flex_allowance(kind),
                   self.limits.hard_max_tokens)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply `override` on top of `base`; nested tables merge key by key,
    any other value (including lists) is overwritten as-is."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(override: str | Path | None = None) -> ChunkerConfig:
    """Load the default config; if `override` is given, that TOML's keys are
    deep-merged on top of the defaults (a partial file is sufficient).

    Validation happens AFTER the merge: unknown key, missing key, or an
    out-of-range value is rejected with a pydantic ValidationError.
    """
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return ChunkerConfig.model_validate(data)