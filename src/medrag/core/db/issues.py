"""The shared `issues` table.

**Location decision:** this table is NOT INSIDE `facts/db/specs.db`, it lives in
a SEPARATE SQLite file (default `facts/db/issues.db`, a SIBLING file of
`specs.db`). Reason: `specs.db` is no longer frozen, so "it needs its own DB"
does not come from frozenness alone, it comes from the WRITER COUNT.
`specs.db`'s only writer is `load_to_db.py`. The panel will update the
`resolved_at`/`resolved_by` fields -- that would have made the panel a SECOND
writer of `specs.db`, breaking the single-writer rule. A separate TABLE does not
solve this (same file = same writer constraint); a separate FILE does: the
writers of `issues.db` split into the pipeline (INSERT, every field EXCEPT
`resolved_*`) and the panel (UPDATE of `resolved_*` only), and each stays the
single writer of its own file.

Schema:
  id, stage (scan/parse/chunk/facts/vectorize/panel),
  severity (error/warning/review), source_doc, entity_ref, reason (a CODE,
  NOT free text), detail, payload (JSON text), created_at,
  resolved_at, resolved_by.

`reason` is NOT free text -- it is a code from the `REASON_CODES` set, because
"reason has to be coded, otherwise filtering and counting are impossible". The
first two codes produced here are `UNRESOLVED_DOC` (the scanner's
document-level `docs_missing_text`/`docs_excluded_fanout`) and
`UNRESOLVED_CHUNK` (chunk-level unresolved items, the `skipped` list of
`catalog_chunk_ownership/carry_over_unresolved.py`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Literal

from medrag.core.db.sqlite import connect

STAGES = ("scan", "parse", "chunk", "facts", "vectorize", "panel")
SEVERITIES = ("error", "warning", "review")

Stage = Literal["scan", "parse", "chunk", "facts", "vectorize", "panel"]
Severity = Literal["error", "warning", "review"]

# The first 2 codes written here. When a new stage/mechanism starts writing to
# `issues`, it is ADDED HERE -- the closed-set-instead-of-free-text principle
# applies here too.
#
# `NEW_KEY_PROPOSAL`: `load_to_db.py`'s behavior of writing key proposals that
# are not in the dictionary to `queue/new_key_proposals.jsonl` is not a
# ratified protocol decision yet, so this code only prepares a VISIBILITY
# channel that `issues_bridge.record_new_key_proposal` may use (the JSONL is
# NOT REPLACED, this is ADDED BESIDE it); the pipeline does not currently CALL
# this hook BY DEFAULT (`load_product`'s `issues_con=None` default) -- it is to
# be activated after approval.
# `SKIPPED_FACT`: every fact `load_to_db.py::load_product` dumps into
# `report["skipped"]` (a `condition` not in the dictionary, an invalid
# list/subfield entry, a conflicting bootstrap row, ... -- one of the findings
# of the facts inventory) -- `record_skipped_facts` writes those with this code.
REASON_CODES = frozenset(
    {"UNRESOLVED_DOC", "UNRESOLVED_CHUNK", "NEW_KEY_PROPOSAL", "SKIPPED_FACT"}
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    severity TEXT NOT NULL,
    source_doc TEXT,
    entity_ref TEXT,
    reason TEXT NOT NULL,
    detail TEXT,
    payload TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    resolved_at TEXT,
    resolved_by TEXT
);
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_issues_stage ON issues(stage);",
    "CREATE INDEX IF NOT EXISTS ix_issues_reason ON issues(reason);",
    "CREATE INDEX IF NOT EXISTS ix_issues_resolved ON issues(resolved_at);",
)


def connect_issues(path: str | Path) -> sqlite3.Connection:
    """Opens a write connection to `issues.db` and creates the schema if
    missing (idempotent).

    Uses `core.db.sqlite.connect` (the shared WAL/busy_timeout policy) -- the
    SAME connection discipline as `specs.db`, a DIFFERENT file.
    """
    con = connect(path, readonly=False)
    con.execute(_SCHEMA)
    for stmt in _INDEXES:
        con.execute(stmt)
    con.commit()
    return con


def record_issue(
    con: sqlite3.Connection,
    *,
    stage: Stage,
    severity: Severity,
    reason: str,
    source_doc: str | None = None,
    entity_ref: str | None = None,
    detail: str | None = None,
    payload: str | None = None,
) -> int:
    """Adds one issue row and returns its `id`.

    Fails loudly if `stage`/`severity` does not come from the closed set
    (no silent swallowing -- same spirit as the facts scanner's
    `docs_missing_text` convention).
    """
    if stage not in STAGES:
        raise ValueError(f"bilinmeyen stage: {stage!r} (bilinenler: {STAGES})")
    if severity not in SEVERITIES:
        raise ValueError(f"bilinmeyen severity: {severity!r} (bilinenler: {SEVERITIES})")
    if reason not in REASON_CODES:
        raise ValueError(f"bilinmeyen reason kodu: {reason!r} (bilinenler: {sorted(REASON_CODES)})")
    cur = con.execute(
        "INSERT INTO issues (stage, severity, source_doc, entity_ref, reason, detail, payload) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (stage, severity, source_doc, entity_ref, reason, detail, payload),
    )
    con.commit()
    return int(cur.lastrowid)


def count_open(con: sqlite3.Connection, *, reason: str | None = None) -> int:
    """Number of unresolved (`resolved_at IS NULL`) issues, with an optional
    `reason` filter.

    Both the "open issue count" in the nightly report and the acceptance
    criterion ("every unresolved chunk can be counted") go through this
    function.
    """
    if reason is not None:
        row = con.execute(
            "SELECT COUNT(*) FROM issues WHERE resolved_at IS NULL AND reason = ?", (reason,)
        ).fetchone()
    else:
        row = con.execute("SELECT COUNT(*) FROM issues WHERE resolved_at IS NULL").fetchone()
    return int(row[0])


def resolve_issue(con: sqlite3.Connection, issue_id: int, *, resolved_by: str) -> None:
    """The panel's ONLY write permission -- updates the `resolved_*` fields
    alone (the "write permission updates only the `resolved_*` fields" rule,
    which is the reason for the location decision in the module docstring)."""
    con.execute(
        "UPDATE issues SET resolved_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
        "resolved_by = ? WHERE id = ?",
        (resolved_by, issue_id),
    )
    con.commit()


def fetch_all(con: sqlite3.Connection) -> list[dict[str, Any]]:
    """Raw rows for the panel's listing screen -- for testing/auditing."""
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM issues ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def fetch_filtered(
    con: sqlite3.Connection,
    *,
    stage: str | None = None,
    severity: str | None = None,
    source_doc: str | None = None,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    """A list filtered by `stage`/`severity`/`source_doc`/`reason` -- the
    accumulated issue queue was unusable as a flat list, which is why the panel
    narrows it through this function.

    Equality filters -- NOT free-text search (`stage`/`severity` already come
    from the closed STAGES/SEVERITIES sets and `reason` from the closed
    REASON_CODES set) -- not re-validated here either: an unknown value simply
    returns an empty result, and unlike `record_issue` this does NOT RAISE (a
    filter is a read, not a write). With no filter given it returns the same
    result as `fetch_all`."""
    clauses: list[str] = []
    args: list[Any] = []
    if stage is not None:
        clauses.append("stage = ?")
        args.append(stage)
    if severity is not None:
        clauses.append("severity = ?")
        args.append(severity)
    if source_doc is not None:
        clauses.append("source_doc = ?")
        args.append(source_doc)
    if reason is not None:
        clauses.append("reason = ?")
        args.append(reason)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT * FROM issues{where} ORDER BY id", args).fetchall()
    return [dict(r) for r in rows]


__all__ = [
    "REASON_CODES",
    "SEVERITIES",
    "STAGES",
    "connect_issues",
    "count_open",
    "fetch_all",
    "fetch_filtered",
    "record_issue",
    "resolve_issue",
]
