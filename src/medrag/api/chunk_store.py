"""`node_id` -> chunk TEXT (the [SQL] side of the evidence panel).

## Why this module exists

`facts/db/specs.db` stores the SOURCE of a spec value as a pointer
(`spec_value.source_chunk_id` + `chunk_sha256` in the `spec_value.evidence`
JSON) but does NOT store the chunk's TEXT -- the `raw_text` there is only the
trimmed statement of that value (`'SATA III (6 Gb/s)'`), not the paragraph the
inference was made from. The text lives in the chunker's output
(`chunker/storage/all_chunks.json`) and in the Qdrant payload; both are keyed
by the SAME `node_id` (`<doc_id>::c54`).

Since the evidence panel shows the real text behind a SQL row (user request:
"the attributes we keep from the DB should show the evidence chunks' content
and file paths"), a reader was needed to resolve that text. The chunker's
FILE was chosen as the source, not Qdrant: so the panel works even when Qdrant
is down, and retrieval's top_n store (designed for search) doesn't need a new
"fetch by id" capability.

## Architectural place

SAME pattern as `image_store.py`/`document_store.py`: **filesystem sharing, no
code dependency** -- the `chunker` package is NEVER imported, only the JSON it
produces is read with `json`+`pathlib`. The path comes from
`CHATBOT_CHUNK_CORPUS_PATH`; if unset, it falls back to `chunker/storage/
all_chunks.json` inside the repo -- same env/bundled-fallback pattern as
`factory.py::resolve_db_path_from_env`.

## Error policy

This module NEVER raises (including `load_chunk_index`): if the file is
missing, corrupt, or unexpected, it returns an EMPTY index. The evidence panel
is an ADDITION; if chunk text can't be resolved the panel simply shows less
for that row -- the turn never fails.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medrag.core.paths import ALL_CHUNKS_PATH as _BUNDLED_CORPUS_PATH

logger = logging.getLogger("medrag.api.chunk_store")

#: Operator override. So the panel can look at the same place when the
#: chunker was run with a custom root (`CHUNKER_OUTPUT_DIR`).
CHUNK_CORPUS_ENV = "CHATBOT_CHUNK_CORPUS_PATH"

# Path resolution lives in core.paths now (identical to this file's old
# `Path(__file__).resolve().parents[2] / "chunker" / "storage" /
# "all_chunks.json"` computation).


@dataclass(frozen=True)
class ChunkText:
    """Panel-displayable form of a chunk. `source_path` is DELIBERATELY NOT
    CARRIED -- in the chunker's node that field is the path of the machine
    where the chunk was PARSED (`/home/user/...`), meaningless on this
    machine. The file path shown in the panel comes from
    `document.file_path` (see `sql_evidence.py`), i.e. the DB's own
    authoritative record."""

    node_id: str
    doc_id: str | None
    text: str
    page: str | None
    section: str | None


def resolve_chunk_corpus_path() -> Path:
    """`CHATBOT_CHUNK_CORPUS_PATH` -> if absent, the chunker output inside the repo."""
    override = (os.getenv(CHUNK_CORPUS_ENV) or "").strip()
    return Path(override) if override else _BUNDLED_CORPUS_PATH


def _as_page(node: dict[str, Any]) -> str | None:
    """`page_start`/`page_end` -> "4" or "4-6". SAME format as the [DOC] side
    in `answering_model.py::extract_evidence_chunks`."""
    start = node.get("page_start")
    if start in (None, ""):
        return None
    end = node.get("page_end")
    return str(start) if end in (None, "", start) else f"{start}-{end}"


def _as_section(node: dict[str, Any]) -> str | None:
    """`heading_path` -> "A > B". If not a list (broken record), silently None."""
    heading = node.get("heading_path")
    if not isinstance(heading, list) or not heading:
        return None
    return " > ".join(str(h) for h in heading)


def load_chunk_index(path: Path | str | None = None) -> dict[str, ChunkText]:
    """`all_chunks.json` -> `node_id -> ChunkText`. EMPTY dict on error.

    NO cache, deliberately: the caller (`sql_evidence.py`) keeps a single
    index for the process lifetime (`cached_chunk_index`). This function stays
    separate so it remains pure and testable.
    """
    corpus_path = Path(path) if path is not None else resolve_chunk_corpus_path()
    try:
        with corpus_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning("chunk korpusu yok, kanıt metinleri gösterilemeyecek: %s", corpus_path)
        return {}
    except Exception:
        logger.warning("chunk korpusu okunamadı: %s", corpus_path, exc_info=True)
        return {}

    documents = data.get("documents") if isinstance(data, dict) else None
    if not isinstance(documents, dict):
        logger.warning("chunk korpusunda `documents` sözlüğü yok: %s", corpus_path)
        return {}

    index: dict[str, ChunkText] = {}
    for doc in documents.values():
        if not isinstance(doc, dict):
            continue
        nodes = doc.get("nodes")
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            node_id = node.get("node_id")
            text = node.get("text")
            if not node_id or not isinstance(text, str):
                continue
            index[str(node_id)] = ChunkText(
                node_id=str(node_id),
                doc_id=node.get("doc_id") or doc.get("doc_id"),
                text=text,
                page=_as_page(node),
                section=_as_section(node),
            )
    logger.info("chunk korpusu yüklendi: %d chunk (%s)", len(index), corpus_path)
    return index


# Process-lifetime cache: the corpus is ~5 MB / ~20k chunks, re-parsing it on
# every request is pointless. Key is the RESOLVED PATH -- so tests with
# different `tmp_path` corpora don't pollute each other (an env change
# automatically diverges the key).
_INDEX_CACHE: dict[str, dict[str, ChunkText]] = {}


def cached_chunk_index(path: Path | str | None = None) -> dict[str, ChunkText]:
    """Process-lifetime cached form of `load_chunk_index`."""
    corpus_path = Path(path) if path is not None else resolve_chunk_corpus_path()
    key = str(corpus_path)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = load_chunk_index(corpus_path)
    return _INDEX_CACHE[key]


def clear_cache() -> None:
    """Tests only -- forces a re-read when the corpus file changes."""
    _INDEX_CACHE.clear()


__all__ = [
    "CHUNK_CORPUS_ENV",
    "ChunkText",
    "cached_chunk_index",
    "clear_cache",
    "load_chunk_index",
    "resolve_chunk_corpus_path",
]
