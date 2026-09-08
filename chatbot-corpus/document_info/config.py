"""document_info configuration: `config/default.toml` -> `DocumentInfoConfig`.

Type split: scanning/classification tuning lives HERE (TOML); paths live
in `.env` (`PRODUCT_INFO_DIR` / `BELGELER_DIR` / `OUTPUT_PATH`); no
secrets.

Pattern (`chunker/config.py`): pydantic, `extra="forbid"`, no Python
defaults on fields — every value must come from TOML. `load_config(override)`
deep-merges a partial TOML on top of the defaults. When the
`DOCUMENT_INFO_CONFIG` env var is set, the CLI passes that path as the
override.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class Scan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exclude_dir_names: list[str]
    exclude_file_keywords: list[str]
    junk_prefixes: list[str]
    junk_names: list[str]
    image_extensions: list[str]
    stp_extensions: list[str]

    # startswith() takes a tuple; small helpers keep the call site readable.
    def junk_prefix_tuple(self) -> tuple[str, ...]:
        return tuple(self.junk_prefixes)

    def junk_name_set(self) -> set[str]:
        return {n.lower() for n in self.junk_names}

    def exclude_dir_set(self) -> set[str]:
        return set(self.exclude_dir_names)

    def image_ext_set(self) -> set[str]:
        return {e.lower() for e in self.image_extensions}

    def stp_ext_tuple(self) -> tuple[str, ...]:
        return tuple(e.lower() for e in self.stp_extensions)


class DocType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    keywords: list[str] = Field(min_length=1)


class Codes(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str

    @field_validator("pattern")
    @classmethod
    def _must_be_compilable(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid code regex {v!r}: {exc}")
        return v

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


class DocumentInfoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan: Scan
    codes: Codes
    doc_types: list[DocType] = Field(min_length=1)


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Apply `override` on top of `base`; nested tables are merged
    key-by-key, every other value (including lists) is overwritten as-is."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(override: str | Path | None = None) -> DocumentInfoConfig:
    """Load the default config; if `override` is given, deep-merge that TOML
    on top. Validation runs AFTER the merge: unknown / missing keys, or a
    regex that fails to compile, are rejected with a pydantic
    ValidationError."""
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return DocumentInfoConfig.model_validate(data)
