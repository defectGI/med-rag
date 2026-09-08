"""Shared TOML config loading primitives: read + deep-merge.

`_read_toml`/`_deep_merge` were byte-identical (module-level docstring
language aside) across 7 files: `tools/benchmark/config.py`,
`src/medrag/api/config.py`, `chatbot-corpus/document_info/config.py`,
`src/medrag/pipeline/chunker/config.py`, `src/medrag/pipeline/parser/config.py`,
`src/medrag/pipeline/cli/pipeline_config.py`,
`src/medrag/pipeline/vectorize/config.py`.

**Scope note (same "at least two already-installed consumers" entry rule the
other core/ consolidations used):** only `chatbot` and `vectorize` are both
installed `medrag` dependents AND use the plain `load_config(override=None)`
shape (no env-var overlay). `parser` and `pipeline` use a second, incompatible
shape -- `_ENV_OVERRIDES` registry + `_env_overrides(env)` + a two-arg `load_config(env,
override)` + `get_config()` -- and neither is an installed `medrag` dependent
yet, so that shape isn't folded in here; it stays a separate future decision
(for when env-name unification happens). `benchmark` and
`chatbot-corpus/document_info` use the same simple shape as chatbot/vectorize
but aren't installed `medrag` dependents either -- left untouched pending their
own component move. `retrieval` has no TOML loader at all (each module reads
`os.getenv` directly) -- not applicable here.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Applies `override` on top of `base`; nested tables merge key by key,
    every other value (including lists) is overwritten wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
