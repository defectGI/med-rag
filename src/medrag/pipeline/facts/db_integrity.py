"""O-08 (I-18): pre-write / post-write integrity gate for `specs.db`.

`load_to_db.py` (PHASE C) calls `verify_db_integrity()` BEFORE it starts
writing values AND AFTER it finishes. If any check fails,
`SchemaIntegrityError` is raised -- the caller (`load_to_db.main()`)
must CATCH it and stop without writing a SINGLE row, and the run must
be reported as *failed* (I-18 + I-20, the contract group N reads).

This file is DELIBERATELY SEPARATE from `load_to_db.py`: that module
currently raises `FileNotFoundError` at import-time (K-71 -- the
prompts/*.md have not been moved into the package, see
`load_to_db.py` module docstring / `tests/test_chunk_full_run_load.py`).
The gate itself needs to be testable INDEPENDENTLY of that blocker,
so it lives here; `load_to_db.py` imports and calls it.

Checks:
  1. `PRAGMA integrity_check` -> must be exactly `['ok']`.
  2. The schema is compared with the BASE tables (NOT views, `v_`
     prefixed) listed in `facts/db/schema.yaml` -- every column
     schema.yaml lists for each table must also exist in the live DB
     (extra columns are OK, missing columns are a problem).
  3. The `sv_unique_live_uix` unique index exists and is genuinely UNIQUE.

NOTE -- `document.trust_rank`: `facts/db/schema.yaml`'s `document` table
lists this column for LLM documentation purposes (the trace of
schema_rag.sql's GENERATED column) but in the LIVE specs.db no such
PHYSICAL column EXISTS (measurement of 2026-08-19, see
service/02-SERVISLESTIRME-TASKLARI.md O-08 note). `trust_rank` is
actually computed in `load_to_db.py` as a Python dict named `TRUST_RANK`.
This SINGLE, DELIBERATE exception is excluded below via
`_SCHEMA_CHECK_EXCLUDED_COLUMNS` -- otherwise this gate would ALWAYS
false-positive fail on the live DB.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/
_REPO_ROOT = HERE.parents[3]  # medrag/ (facts/pipeline/urun/src/<repo>)
DEFAULT_SCHEMA_YAML_PATH = _REPO_ROOT / "facts" / "db" / "schema.yaml"

# See the NOTE in the module docstring.
_SCHEMA_CHECK_EXCLUDED_COLUMNS = frozenset({("document", "trust_rank")})

REQUIRED_UNIQUE_INDEX = "sv_unique_live_uix"
# NOTE: the index NAME itself contains "unique" -- substring search
# would be INSUFFICIENT (a name match could WRONGLY pass). The statement
# is checked via an ANCHORED regex for `CREATE UNIQUE INDEX`.
_CREATE_UNIQUE_INDEX_RE = re.compile(r"^\s*CREATE\s+UNIQUE\s+INDEX\b", re.IGNORECASE)


class SchemaIntegrityError(RuntimeError):
    """Raised when `verify_db_integrity` fails one or more checks. The
    message LISTS all detected issues separated by `; ` (no fail-fast --
    a single run reports them all)."""


def _expected_columns_from_schema_yaml(schema_yaml_path: Path) -> dict[str, set[str]]:
    """Extracts the expected column set for BASE tables (NOT views) in
    `schema_yaml_path`. `v_`-prefixed entries are VIEWS -- no
    specs.db write flow INSERTs/UPDATEs them, they are out of scope."""
    doc = yaml.safe_load(schema_yaml_path.read_text(encoding="utf-8")) or {}
    expected: dict[str, set[str]] = {}
    for table in doc.get("tables", []):
        name = table["name"]
        if name.startswith("v_"):
            continue
        cols = {c["name"] for c in table.get("columns", [])}
        cols -= {col for (tbl, col) in _SCHEMA_CHECK_EXCLUDED_COLUMNS if tbl == name}
        expected[name] = cols
    return expected


def verify_db_integrity(con: sqlite3.Connection, *, schema_yaml_path: Path | None = None) -> None:
    """O-08 (I-18): `load_to_db.py` calls this BEFORE writing and AFTER.
    Raises `SchemaIntegrityError` on any failed check; returns silently on
    success (None). All checks are AGGREGATED (does not stop at the
    first failure) -- the message shows every issue at once."""
    problems: list[str] = []

    try:
        rows = con.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.DatabaseError as exc:
        raise SchemaIntegrityError(f"PRAGMA integrity_check could not be run: {exc}") from exc
    check_result = [r[0] for r in rows]
    if check_result != ["ok"]:
        problems.append(f"PRAGMA integrity_check failed: {check_result}")

    expected = _expected_columns_from_schema_yaml(schema_yaml_path or DEFAULT_SCHEMA_YAML_PATH)
    existing_tables = {
        row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    for table, expected_cols in sorted(expected.items()):
        if table not in existing_tables:
            problems.append(f"expected table missing: {table!r}")
            continue
        actual_cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        missing = expected_cols - actual_cols
        if missing:
            problems.append(f"{table}: expected columns that are missing: {sorted(missing)}")

    index_row = con.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (REQUIRED_UNIQUE_INDEX,),
    ).fetchone()
    if index_row is None:
        problems.append(f"expected constraint missing: {REQUIRED_UNIQUE_INDEX} (spec_value unique index)")
    elif index_row[0] is None or not _CREATE_UNIQUE_INDEX_RE.match(index_row[0]):
        problems.append(f"{REQUIRED_UNIQUE_INDEX} exists but is NOT UNIQUE")

    if problems:
        raise SchemaIntegrityError("; ".join(problems))