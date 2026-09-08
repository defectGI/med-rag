"""O-11 (I-17, K-70): "forget the source" -- common evidence-removal function.

When a source document (doc_id) or specific chunks (content_sha256) are
removed from a product, removes their entries from
`spec_value.evidence[]`:

  - If `evidence[]` becomes EMPTY, the row is DELETED -- regardless of
    `status` (present/conflicting/superseded included). This way a
    source-disappeared `conflicting` row does not silently ACCUMULATE
    OUTSIDE `sv_unique_live_uix` (see the attention note in this task).
  - If some entries are removed but AT LEAST ONE remains, the row STAYS;
    the primary-source fields (`source_doc_id`/`source_file_name`/
    `source_chunk_id`) are re-aligned to the remaining first `evidence[]`
    entry (the trust-rank order from write time is PRESERVED, see
    `load_to_db._merge_evidence`).
  - Matching happens on BOTH `evidence[].doc_id` AND `evidence[].
    chunk_sha256` -- the K-70 transition period accepts both (new
    `sha256:<hex>`, old `<uuid>::cN`) by literal string equality, WITHOUT
    format conversion (the caller already knows whether it has migrated
    or not).

This module is the spec-side counterpart of I-08 (chunk/vector side)
for the same "forget the source" concept, and N-06 (cleanup of removed
files' derivatives) will ALSO USE IT -- a separate deletion code path is
NOT opened. The same function is used by O-11a to clean up ghost
`evidence[]` entries (evidence whose text is no longer found in the
current corpus): the caller passes those values as `content_sha256s`.

KARAR-KAYDI.md (K-78): this file was added to
`db_write_audit.ALLOWED_WRITER_FILENAMES` (DELETE/UPDATE-inclusive) as
the THIRD legitimate writer -- a deliberate exception to KARAR-016's
"two writers" decision for the DELETE-only case.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from typing import Any

_HEX64_RE_CHARS = frozenset("0123456789abcdefABCDEF")


def _normalize_content_sha256s(content_sha256s: Iterable[str] | None) -> frozenset[str]:
    """Accepts the given values AS-IS; also adds the prefixed form for
    bare-64-hex-char values that do NOT start with `sha256:` -- the
    caller may have forgotten the prefix. Values already prefixed, or in
    the old `<uuid>::cN` form, are NOT TOUCHED."""
    if not content_sha256s:
        return frozenset()
    out: set[str] = set()
    for raw in content_sha256s:
        out.add(raw)
        if not raw.startswith("sha256:") and len(raw) == 64 and set(raw) <= _HEX64_RE_CHARS:
            out.add(f"sha256:{raw}")
    return frozenset(out)


def forget_source(
    con: sqlite3.Connection,
    *,
    doc_ids: Iterable[str] | None = None,
    content_sha256s: Iterable[str] | None = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Removes entries matching `doc_ids` OR `content_sha256s` from
    `spec_value.evidence[]`; rows whose evidence ends up empty are
    DROPPED (DELETE), the rest have their primary-source fields
    re-aligned (UPDATE).

    At least one of (`doc_ids` or `content_sha256s`) must be non-empty --
    when both are empty/None this raises early, so a mis-call that
    would scan the whole table and do NOTHING (yet look like "everything
    was deleted") cannot go unnoticed.

    `commit=False` lets the caller JOIN the call into its own transaction
    (e.g. N-06 updating other tables in the same operation).

    Returns: rows_scanned, rows_touched, rows_deleted, rows_updated,
    evidence_entries_before, evidence_entries_after, evidence_entries_removed.
    """
    doc_id_set = frozenset(doc_ids) if doc_ids else frozenset()
    sha_set = _normalize_content_sha256s(content_sha256s)
    if not doc_id_set and not sha_set:
        raise ValueError("forget_source: at least one of doc_ids or content_sha256s must be non-empty")

    def _entry_matches(entry: dict[str, Any]) -> bool:
        return entry.get("doc_id") in doc_id_set or entry.get("chunk_sha256") in sha_set

    cur = con.cursor()
    cur.execute("SELECT value_id, evidence FROM spec_value WHERE evidence != '[]'")
    rows = cur.fetchall()

    rows_scanned = len(rows)
    evidence_entries_before = 0
    evidence_entries_after = 0
    delete_ids: list[int] = []
    updates: list[tuple[str | None, str | None, str | None, str, int]] = []

    for value_id, evidence_json in rows:
        evidence = json.loads(evidence_json)
        evidence_entries_before += len(evidence)
        remaining = [e for e in evidence if not _entry_matches(e)]
        if len(remaining) == len(evidence):
            # This row was NOT affected -- do not touch (would create
            # audit/diff noise).
            evidence_entries_after += len(evidence)
            continue

        evidence_entries_after += len(remaining)
        if not remaining:
            delete_ids.append(value_id)
            continue

        primary = remaining[0]
        updates.append((
            primary.get("doc_id"),
            primary.get("file_name"),
            primary.get("chunk_sha256"),
            json.dumps(remaining, ensure_ascii=False),
            value_id,
        ))

    if delete_ids:
        cur.executemany("DELETE FROM spec_value WHERE value_id = ?", [(vid,) for vid in delete_ids])
    if updates:
        cur.executemany(
            "UPDATE spec_value SET source_doc_id = ?, source_file_name = ?, "
            "source_chunk_id = ?, evidence = ? WHERE value_id = ?",
            updates,
        )
    if commit and (delete_ids or updates):
        con.commit()

    return {
        "rows_scanned": rows_scanned,
        "rows_touched": len(delete_ids) + len(updates),
        "rows_deleted": len(delete_ids),
        "rows_updated": len(updates),
        "evidence_entries_before": evidence_entries_before,
        "evidence_entries_after": evidence_entries_after,
        "evidence_entries_removed": evidence_entries_before - evidence_entries_after,
    }