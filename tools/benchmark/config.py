"""Benchmark configuration: `config/default.toml` -> `BenchmarkConfig`.

Kind separation (see root `CONFIG.md`): grading behavior / tuning knobs
live HERE (TOML); model connection identity (provider / model / api_key /
thinking / num_ctx) lives in `.env` (`VLM_*` / `LLM_*`, shared with
`parser/llm` -- the root README says "model in one place, parser/.env");
run inputs (folder / scope / out) are CLI flags.

Pattern (from `chunker/config.py`): pydantic, `extra="forbid"`, NO Python
defaults on fields -- every value must come from TOML; removing a key
from `default.toml` makes loading fail loudly. `load_config(override)`
deep-merges a partial TOML (only the keys you changed) on top of the
defaults.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# The `config/` data folder sits at the same level as this module
# (benchmark/config/default.toml).
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class Render(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dpi: int = Field(ge=1)


class Grading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens: int = Field(ge=1)
    assume_context: int = Field(ge=1)
    max_pages: int = Field(ge=0)  # 0 = all pages

    def max_pages_or_none(self) -> int | None:
        return self.max_pages or None


class Chunking(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    pages_per_window: int = Field(ge=0)  # 0 = auto

    def pages_per_window_or_none(self) -> int | None:
        return self.pages_per_window or None


class Weights(BaseModel):
    """Overall dimension weights for the score.

    Field names match `benchmark/judge.py::DIMENSIONS` exactly; if the
    dimension set changes, update here too (fail-loud -- unknown weight
    keys are rejected by `extra="forbid"`).
    """

    model_config = ConfigDict(extra="forbid")

    accuracy: float = Field(ge=0)
    coverage: float = Field(ge=0)
    clarity: float = Field(ge=0)

    def as_dict(self) -> dict[str, float]:
        return {"accuracy": self.accuracy, "coverage": self.coverage,
                "clarity": self.clarity}


class Scope(BaseModel):
    """Visual types excluded from grading (from the parser's `[visual]` taxonomy).

    Visuals classified into these types have NONE of their output --
    presence, OCR text, generated description -- affect the score.
    Distinct from the parser's own `[visual] exclude_types` (same name,
    different concept): THAT one stops parse-time content production,
    THIS one excludes from grading. Values come from the `[visual] types`
    taxonomy.
    """

    model_config = ConfigDict(extra="forbid")

    exclude_types: list[str]


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    render: Render
    grading: Grading
    chunking: Chunking
    weights: Weights
    scope: Scope


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Overlay `override` on `base`: nested tables merge key by key,
    every other value (including lists) is replaced wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(override: str | Path | None = None) -> BenchmarkConfig:
    """Load the default config; if `override` is given, deep-merge the
    keys in that TOML on top of the defaults (a partial file suffices).

    Validation runs AFTER the merge: unknown keys, missing keys, or
    out-of-range values are rejected with a pydantic ValidationError.
    """
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return BenchmarkConfig.model_validate(data)
