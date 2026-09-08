"""SQL row -> the EVIDENCE CHUNKS that produced it (text + file path + page).

## Chain

`spec_value` links a value to the chunk it was inferred from via POINTERS:

    spec_value.source_chunk_id  -> "efe8a712-…::c54"        (primary source)
    spec_value.evidence         -> [{doc_id, file_name, chunk_sha256}, …]
                                   (ALL corroborating sources; the
                                   `chunk_sha256` field practically carries
                                   the same node_id form)

The TEXT is not in this DB. This module turns the pointer into a full
evidence record from three parts:

    node_id      -> chunk text/page/heading   (chunk_store.py, chunker output)
    doc_id       -> file_name + file_path      (specs.db `document` table)
    spec_value   -> which product/attribute    (the row's own metadata)

## Why `evidence` is not SELECTed in SQL

`sql_citation.py` adds only `value_id` to the generated SQL. `evidence` is a
JSON blob and `render_row_text` puts EVERY selected column into the row text
-- it would land in the answering model's prompt (flow intermediate data
must not reach the model). Instead a single integer (`value_id`) travels in
the row, and the evidence body is resolved HERE, on the SEPARATE channel to
the panel -- same principle as the `images`/`documents` channels.

## Keying and fallback

Primary key is `value_id` (PK of spec_value, single-row guarantee). If the
query didn't SELECT it (aggregations/DISTINCT/UNION -- the queries
`sql_citation.py` leaves alone), it falls back to the `(product_code, key)`
pair; in `conflicting` cases that pair can return several rows and all are
shown as evidence (deliberate: conflicting sources must not be hidden).

## Error policy

Same as `chunk_store.py`: this module NEVER raises. If the DB can't open, a
column is missing, or the JSON is broken, it returns an EMPTY list and the
panel just shows its old (column/value) rendering for that row.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from medrag.api.chunk_store import ChunkText, cached_chunk_index
from medrag.core.db.sqlite import connect as _sqlite_connect

logger = logging.getLogger("medrag.api.sql_evidence")

#: Max evidence chunks shown under a SQL row in the panel. `evidence` carries
#: dozens of sources in some rows (the same chunk can repeat many times); this
#: upper bound is applied AFTER dedup.
DEFAULT_MAX_CHUNKS = 8

#: Chunk text is clipped to this length (panel ends with "…"). LARGER than
#: `extract_evidence_chunks`'s `snippet_len` -- the goal here is to actually
#: READ the text.
DEFAULT_SNIPPET_LEN = 1200


@dataclass(frozen=True)
class EvidenceChunk:
    """A single evidence chunk destined for the panel."""

    chunk_id: str
    doc_id: str | None
    file_name: str | None
    file_path: str | None
    page: str | None
    section: str | None
    text: str | None

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "page": self.page,
            "section": self.section,
            "text": self.text,
        }


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _pointer_rows(
    conn: sqlite3.Connection,
    *,
    value_ids: Sequence[int],
    pairs: Sequence[tuple[str, str]],
) -> dict[str, list[tuple[str | None, str | None]]]:
    """`value_id`/`(product_code,key)` -> [(node_id, doc_id), …].

    Return-key is the string form of the lookup key the caller used
    (`"v:123"` / `"k:PN1099|weight"`) -- so the two different keying schemes
    can live in a single dict."""
    found: dict[str, list[tuple[str | None, str | None]]] = {}

    def _collect(key: str, source_chunk_id, evidence, source_doc_id) -> None:
        bucket = found.setdefault(key, [])
        if source_chunk_id:
            bucket.append((str(source_chunk_id), str(source_doc_id) if source_doc_id else None))
        if not evidence:
            return
        try:
            parsed = json.loads(evidence)
        except Exception:  # noqa: BLE001 - broken JSON must not drop the evidence
            return
        if not isinstance(parsed, list):
            return
        for entry in parsed:
            if not isinstance(entry, Mapping):
                continue
            chunk_id = entry.get("chunk_sha256")
            if chunk_id:
                bucket.append((str(chunk_id), str(entry.get("doc_id") or "") or None))

    if value_ids:
        sql = (
            "SELECT value_id, source_chunk_id, evidence, source_doc_id FROM spec_value "
            f"WHERE value_id IN ({_placeholders(len(value_ids))})"
        )
        for vid, chunk_id, evidence, doc_id in conn.execute(sql, list(value_ids)):
            _collect(f"v:{vid}", chunk_id, evidence, doc_id)

    for product_code, key in pairs:
        sql = (
            "SELECT source_chunk_id, evidence, source_doc_id FROM spec_value "
            "WHERE product_code = ? AND key = ?"
        )
        for chunk_id, evidence, doc_id in conn.execute(sql, (product_code, key)):
            _collect(f"k:{product_code}|{key}", chunk_id, evidence, doc_id)

    return found


def _document_paths(
    conn: sqlite3.Connection, doc_ids: Sequence[str]
) -> dict[str, tuple[str | None, str | None]]:
    """`doc_id -> (file_name, file_path)`.

    **`file_path` is DELIBERATELY surfaced to the panel**: the "NEVER expose
    file_path" rule in facts/db/DATA_DICTIONARY.md was explicitly relaxed for
    this channel -- the user wants to see which file the evidence lives in.
    The rule stays fully in force for the ANSWERING MODEL
    (`_render_context`/`_source_suffix`) and for the `doc_download` flow; the
    only thing loosened here is the evidence panel shown to the user."""
    if not doc_ids:
        return {}
    sql = (
        "SELECT doc_id, file_name, file_path FROM document "
        f"WHERE doc_id IN ({_placeholders(len(doc_ids))})"
    )
    return {
        str(doc_id): (file_name, file_path)
        for doc_id, file_name, file_path in conn.execute(sql, list(doc_ids))
    }


def _row_key(metadata: Mapping[str, object]) -> str | None:
    """Evidence lookup key for a SQL row: first `value_id` (PK), else
    `(product_code, key)`. If neither exists, this row's evidence can't be
    resolved."""
    value_id = metadata.get("value_id")
    if value_id is not None:
        return f"v:{value_id}"
    product_code = metadata.get("product_code")
    key = metadata.get("key")
    if product_code and key:
        return f"k:{product_code}|{key}"
    return None


def resolve_sql_evidence(
    db_path: str,
    rows: Sequence[Mapping[str, object]],
    *,
    chunk_index: Mapping[str, ChunkText] | None = None,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
) -> list[list[dict]]:
    """Returns a per-row evidence chunk list in the SAME ORDER as `rows`
    (metadatas of the SQL rows). Empty list for rows that can't be resolved.

    Uses a single SQLite connection + a single chunk index (doesn't reopen per
    row); if `rows` is empty it doesn't touch the DB at all."""
    if not rows:
        return []

    keys = [_row_key(meta) for meta in rows]
    value_ids = sorted({int(k[2:]) for k in keys if k and k.startswith("v:") and k[2:].isdigit()})
    pairs = sorted({tuple(k[2:].split("|", 1)) for k in keys if k and k.startswith("k:")})

    if not value_ids and not pairs:
        return [[] for _ in rows]

    try:
        conn = _sqlite_connect(db_path, readonly=True)
    except Exception:
        logger.warning("kanıt DB'si açılamadı: %s", db_path, exc_info=True)
        return [[] for _ in rows]

    try:
        pointers = _pointer_rows(conn, value_ids=value_ids, pairs=pairs)  # type: ignore[arg-type]
        wanted_docs = sorted(
            {doc_id for bucket in pointers.values() for _, doc_id in bucket if doc_id}
        )
        docs = _document_paths(conn, wanted_docs)
    except Exception:
        logger.warning("kanıt işaretçileri okunamadı (%s)", db_path, exc_info=True)
        return [[] for _ in rows]
    finally:
        conn.close()

    index = chunk_index if chunk_index is not None else cached_chunk_index()

    out: list[list[dict]] = []
    for key in keys:
        bucket = pointers.get(key or "", [])
        seen: set[str] = set()
        chunks: list[dict] = []
        for chunk_id, doc_id in bucket:
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = index.get(chunk_id)
            # `doc_id` can come from three sources; most reliable is the
            # chunk itself, then the one carried by the pointer, last the one
            # derived from the node_id's "<doc_id>::cN" prefix. With none of
            # the three, file name/path can't be shown.
            resolved_doc = (chunk.doc_id if chunk else None) or doc_id
            if not resolved_doc and "::" in chunk_id:
                resolved_doc = chunk_id.split("::", 1)[0]
            file_name, file_path = docs.get(str(resolved_doc or ""), (None, None))
            text = chunk.text if chunk else None
            if text is not None and len(text) > snippet_len:
                text = text[:snippet_len] + "…"
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    doc_id=resolved_doc,
                    file_name=file_name,
                    file_path=file_path,
                    page=chunk.page if chunk else None,
                    section=chunk.section if chunk else None,
                    text=text,
                ).as_dict()
            )
            if len(chunks) >= max_chunks:
                break
        out.append(chunks)
    return out


def attach_sql_evidence(
    chunks: Sequence[dict],
    results: Sequence[object],
    db_path: str,
    *,
    chunk_index: Mapping[str, ChunkText] | None = None,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    snippet_len: int = DEFAULT_SNIPPET_LEN,
) -> list[dict]:
    """Enriches the `extract_evidence_chunks` output by ADDING an
    `evidence_chunks` field to [SQL] rows (input dicts are copied, no
    mutation).

    `results` is the SAME sequence given to `extract_evidence_chunks` -- the
    raw metadata isn't there, it's needed here (the panel dict doesn't carry
    `value_id`). If the lengths don't match it does nothing (defensive: if the
    contract changes, the panel degrades to its old form instead of
    breaking)."""
    if len(chunks) != len(results):
        logger.warning(
            "kanıt zenginleştirme atlandı: %d chunk vs %d sonuç", len(chunks), len(results)
        )
        return list(chunks)

    sql_positions = [i for i, c in enumerate(chunks) if c.get("source_kind") == "SQL"]
    if not sql_positions:
        return list(chunks)

    metadatas = [getattr(results[i], "metadata", None) or {} for i in sql_positions]
    resolved = resolve_sql_evidence(
        db_path,
        metadatas,
        chunk_index=chunk_index,
        max_chunks=max_chunks,
        snippet_len=snippet_len,
    )

    out = [dict(c) for c in chunks]
    for position, evidence in zip(sql_positions, resolved):
        out[position]["evidence_chunks"] = evidence
    return out


__all__ = [
    "DEFAULT_MAX_CHUNKS",
    "DEFAULT_SNIPPET_LEN",
    "EvidenceChunk",
    "attach_sql_evidence",
    "resolve_sql_evidence",
]
