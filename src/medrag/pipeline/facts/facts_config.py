"""Configuration loader: `config/default.toml` -> `FactsConfig`.

Pattern mirrors `medrag.pipeline.parser.config` / `vectorize/vectorize/config.py` /
`chunker/chunker/config.py` (root CONFIG.md "Loader pattern"): pydantic fields
have NO Python defaults, `extra="forbid"` -- missing/unknown keys fail loudly.

O-03 (2026-08-20): the old `facts/facts/config.py` -- filename changed to
`facts_config.py` to avoid collision with the `medrag.pipeline.facts` package
name in the same directory (see 02-SERVISLESTIRME-TASKLARI.md O-03).
`DEFAULT_CONFIG_PATH` now follows the SAME pattern as `parser/config.py`:
`config/default.toml` is a sibling directory of this file (the old
`facts/facts/config.py` -> `facts/config/` had one extra `.parent`; the moved
module now lives INSIDE the package itself)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"

# The 9 fixed doc_types from document_node_schema.md (chatbot-corpus/
# document_info/config/default.toml's doc_type table is the single producer;
# it is repeated here ONLY for validation -- if a new type is added, both
# must be updated. This list is DELIBERATELY not as strict as
# `extra="forbid"`, but rejects unknown values loudly).
KNOWN_DOC_TYPES = frozenset(
    {
        "BROCHURE",
        "CATALOGUE",
        "DATASHEET",
        "CE_DECLARATION",
        "TECHNICAL_DRAWING",
        "USER_MANUAL",
        "QUICK_START_GUIDE",
        "STP",
        "PRODUCT_IMAGE",
    }
)


class Scope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_doc_types: list[str]
    include_summary_nodes: bool
    # NO LONGER READ by discover.py (PROTOCOL KARAR-058 revised, 2026-08-04)
    # -- chunk-based narrowing (chunk_ownership_index) replaced it. The
    # field is NOT deleted (backward-compat safety net); the code just
    # does not look at this value.
    max_doc_owners: int = Field(default=1, ge=1)

    @field_validator("exclude_doc_types")
    @classmethod
    def _known_doc_types(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - KNOWN_DOC_TYPES)
        if unknown:
            raise ValueError(
                f"exclude_doc_types contains unknown doc_types: {unknown} "
                f"(known: {sorted(KNOWN_DOC_TYPES)})"
            )
        return value


class Queue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(ge=1)


class Vocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    large_wave_warning_threshold: int = Field(ge=1)


class Llm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens_extract: int = Field(ge=1)


class Vlm(BaseModel):
    """Pass 4's (ROADMAP 13f) page-render + VLM call tuning values.
    Connection identity (provider/model/base_url/api_key) is NOT here --
    root CONFIG.md four-type taxonomy; that lives in the FACTS_VLM_* block
    of `.env` (2026-07-29 decision #1: kept SEPARATE from FACTS_LLM_* because
    pass 4 invokes an IMAGE model, not a text one, which may use a
    different provider/endpoint)."""

    model_config = ConfigDict(extra="forbid")

    render_dpi: int = Field(ge=1, description="Same concept as parser/images/image_handler.py::_fetch_pdf_region's pdf.render_dpi -- facts keeps its own copy (facts does NOT import parser, per root CLAUDE.md's component-independence rule).")
    timeout_s: float = Field(gt=0, description="VLM HTTP call request timeout (seconds) -- same concept as the `timeout` field of absent_scan.py/extract.py's OpenAICompat* clients.")
    max_tokens: int = Field(ge=1, description="VLM response budget for the (confirm/reject + rationale) JSON.")
    page_render_format: Literal["png", "jpeg"] = Field(description="Image format in which the rendered page will be encoded -- _fetch_pdf_region uses PNG, and facts's own render_page.py shares the same default, but is left configurable in case pass 4's VLM provider requests JPEG.")


class FactsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Scope
    queue: Queue
    vocabulary: Vocabulary
    llm: Llm
    vlm: Vlm

    def in_scope_doc_type(self, doc_type: str) -> bool:
        """KARAR-027: pass 1/pass 3's input selector calls this. This is ONLY
        the DECLARED policy -- KARAR-027's "belt-and-braces" rule makes
        pass 1's real gate the OBSERVED state of the registry
        (`parse.status == SUCCESS` and `chunk.status == SUCCESS`); if the
        two disagree (a type not in this list produced no chunks, or a
        type in the list produced chunks) that disagreement is LOGGED,
        not here in pass 1 (13d-1, not yet implemented)."""
        return doc_type not in self.scope.exclude_doc_types

    def in_scope_tree_level(self, tree_level: int) -> bool:
        """KARAR-027 (v): leaf-only default; if `include_summary_nodes` is
        enabled, summary nodes may also enter (today deliberately False).
        Same: only the declared policy -- the note above applies."""
        if tree_level == 0:
            return True
        return self.scope.include_summary_nodes


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


def load_config(override: str | Path | None = None) -> FactsConfig:
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return FactsConfig.model_validate(data)