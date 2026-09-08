"""Shared SQLite connection helper: busy_timeout (+ WAL for writers).

The connection policy kept in ONE place -- the implementation of the "reduce
the writer count to 1, WAL as the supporting measure" decision.

`facts/db/specs.db`: the earlier WAL exemption was REVERTED -- `specs.db` is no
longer "frozen", it is a DB that its single writer (the `load_to_db.py` stage)
really writes. `PRAGMA journal_mode=WAL` was applied to the live file (backed up
first via the `sqlite3.backup()` API, to two physical locations). That DROPS the
sha256 lock on the file header (the bytes changed) -- replaced by a
pre-write/post-write integrity gate. Readers still open with ``readonly=True``
(``mode=ro`` URI), and WAL keeps a reader isolated from a writer -- a `mode=ro`
connection CANNOT CREATE the `-shm` file but can read an existing WAL file just
fine; the `-wal`/`-shm` side files only appear while an active writer connection
exists and are cleaned up by an automatic checkpoint once the last connection
closes -- verified (the migration script read the row counts of all four tables
through `mode=ro`). The `.backup()` requirement (so a backup accounts for
-wal/-shm) applies here too.

`BUSY_TIMEOUT_MS` is public: retrieval's `SQLiteExecutor` (a gap found while
this helper was extracted -- its own `_connect()` never set `busy_timeout`)
reads the same constant to reach parity, while staying a separate class that
keeps its own `is_file()` pre-check (see
`src/medrag/api/retrieval/modules/db_query/executor.py`).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.request import pathname2url

BUSY_TIMEOUT_MS = 5000


def connect(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open a SQLite connection with ``busy_timeout`` set.

    Args:
        path: Path to the SQLite file.
        readonly: ``True`` opens with the ``mode=ro`` URI and never touches
            ``journal_mode`` -- use this for frozen/shared-artifact databases
            (e.g. ``facts/db/specs.db``, see module docstring). ``False``
            opens a normal read-write connection and switches the file to
            WAL mode if it isn't already (idempotent; requires a read-write
            handle since WAL mode is written into the file's own header).
    """
    resolved = Path(path).resolve()
    if readonly:
        uri = "file:" + pathname2url(str(resolved)) + "?mode=ro"
        con = sqlite3.connect(uri, uri=True)
    else:
        con = sqlite3.connect(str(resolved))
        current = con.execute("PRAGMA journal_mode;").fetchone()[0]
        if str(current).lower() != "wal":
            con.execute("PRAGMA journal_mode=WAL;")
    con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS};")
    return con
