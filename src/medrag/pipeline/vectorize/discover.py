"""Input discovery: finds chunk JSONs under `VECTORIZE_INPUT_DIR`.

Same principle as chunker's own input discovery (see the
chunker/chunker/cli.py docstring): file CONTENT is inspected, name patterns
are not trusted. chunker used two different output layouts across time, so
all three shapes are recognized:

  1. Single ChunkSet file (legacy per-document layout, e.g.
     `{doc_id}.chunks.json`): top-level `doc_id` + `nodes`.
  2. `all_chunks.json` shape: top-level `documents` dict (doc_id -> ChunkSet).
  3. `all_raptor.json` / `all_combined.json` shape: top-level `trees`
     and/or `documents` dicts.

A file matching none of the three shapes (side file, broken JSON, empty
dict) is LOGGED and skipped -- same philosophy as chunker skipping
`*.stage1.json`/`summary.json`: a visible "I skipped X" instead of a silent
"nothing found".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from medrag.pipeline.vectorize.core import ChunkSet

log = logging.getLogger("vectorize.discover")


@dataclass
class ChunkScope:
    """A single scope to vectorize: one ChunkSet + which file it came from
    (for tracing in log/error messages only)."""

    scope_id: str
    chunk_set: ChunkSet
    source_file: Path


def _is_single(data: object) -> bool:
    return isinstance(data, dict) and "doc_id" in data and isinstance(
        data.get("nodes"), list)


def _get_dict(data: dict, key: str) -> dict[str, dict]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _parse_scope(scope_id: str, raw: dict, source_file: Path) -> ChunkScope | None:
    try:
        return ChunkScope(scope_id=scope_id, chunk_set=ChunkSet.model_validate(raw),
                          source_file=source_file)
    except ValidationError as exc:
        log.warning("%s contains invalid ChunkSet scope %r, skipping: %s",
                    source_file, scope_id, exc)
        return None


def discover_chunk_scopes(input_dir: str | Path) -> list[ChunkScope]:
    """Scans `input_dir` recursively and returns every scope found (document
    + corpus/profile trees if present). Discovery order is deterministic
    (paths sorted) -- `--limit` works on top of it.

    `input_dir` may also point at a SINGLE JSON file (e.g. setups that
    collect the whole corpus in one `all_chunks.json`/`all_combined.json`) --
    in that case only that file is read, no folder scan.

    The `archive/` subtree is SKIPPED: chunker moves each previous output to
    `<output_root>/archive/<timestamp>/` on every run, and in production
    `VECTORIZE_INPUT_DIR` IS that output root -- without the skip, stale
    archived chunk files get re-indexed into Qdrant and silently pollute
    search results (an archive copy has a different mtime but identical
    scope IDs, so staleness signatures cannot catch it).

    If the root ITSELF is an archive folder it is scanned (an operator
    deliberately re-indexing an old run must not be blocked); only the
    `archive/` tree UNDER the root is skipped."""
    root = Path(input_dir)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(
            p for p in root.rglob("*.json")
            if "archive" not in p.relative_to(root).parts
        )
    else:
        raise FileNotFoundError(
            f"VECTORIZE_INPUT_DIR not found (neither file nor directory): {root}")

    scopes: list[ChunkScope] = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.info("%s unreadable/not JSON, skipping: %s", path, exc)
            continue

        if _is_single(data):
            scope = _parse_scope(data["doc_id"], data, path)
            if scope is not None:
                scopes.append(scope)
            continue

        if not isinstance(data, dict):
            log.info("%s is not a ChunkSet/ChunkSet-dict, skipping", path)
            continue

        merged = {**_get_dict(data, "documents"), **_get_dict(data, "trees")}
        if not merged:
            log.info("%s contains no documents/trees/nodes, skipping", path)
            continue
        for scope_id, raw in merged.items():
            scope = _parse_scope(scope_id, raw, path)
            if scope is not None:
                scopes.append(scope)

    return scopes