"""SQL execution seam for the db_query module.

The retriever generates a SQL string (via an LLM, remotely) and then needs to
*run* it against a real database. That execution is isolated behind the
:class:`SqlExecutor` protocol so that:

- a different backend (postgres, duckdb, ...) can be plugged in without pulling
  its driver into this repo's base dependencies (ARCHITECTURE.md #3), and
- tests can run the real mapping logic against a throwaway SQLite file with no
  network and no GPU (ARCHITECTURE.md #10).

The shipped default, :class:`SQLiteExecutor`, uses only the stdlib ``sqlite3``
module (no extra dependency) and opens the database **read-only** by default.
Read-only is a deliberate defense: the SQL executed here is written by an LLM,
and enforcing read-only at the engine level rejects a stray ``DROP``/``DELETE``/
``UPDATE`` far more reliably than trying to parse the SQL ourselves (which would
mishandle CTEs, comments, and quoting).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.request import pathname2url

from medrag.core.db.sqlite import BUSY_TIMEOUT_MS as _BUSY_TIMEOUT_MS

# (column_names, rows) — rows are tuples aligned to column_names.
ExecResult = tuple[list[str], list[tuple]]


class DbQueryError(RuntimeError):
    """Base error for the db_query module.

    Wraps lower-level failures (SQL execution, engine construction) so a caller
    can catch a single module-owned type.
    """


@runtime_checkable
class SqlExecutor(Protocol):
    """Runs a SQL string and returns ``(column_names, rows)``.

    Implementations must not mutate the database for a well-formed read query;
    enforcing that (e.g. a read-only connection) is the implementation's job.
    """

    def execute(self, sql: str) -> ExecResult: ...


class SQLiteExecutor:
    """Execute SQL against a SQLite file using the stdlib driver.

    Args:
        db_path: Path to the SQLite database file.
        read_only: When ``True`` (default) the connection is opened with
            ``mode=ro``; any write attempted by the generated SQL fails at the
            engine level instead of mutating the data.
    """

    def __init__(self, db_path: str | Path, *, read_only: bool = True) -> None:
        self._db_path = Path(db_path)
        self._read_only = read_only

    def _connect(self) -> sqlite3.Connection:
        if not self._db_path.is_file():
            raise DbQueryError(f"SQLite database not found: {self._db_path}")
        if self._read_only:
            # pathname2url yields a cross-platform, correctly escaped path
            # (e.g. ///C:/... on Windows, /abs/path on POSIX), which the SQLite
            # URI opener needs to locate the file.
            uri = "file:" + pathname2url(str(self._db_path.resolve())) + "?mode=ro"
            con = sqlite3.connect(uri, uri=True)
        else:
            con = sqlite3.connect(str(self._db_path))
        # Parity with medrag.core.db.sqlite.connect(): this class keeps its own
        # is_file() pre-check instead of delegating outright, but a
        # busy_timeout gap here would defeat the point of that shared helper.
        con.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS};")
        return con

    def execute(self, sql: str) -> ExecResult:
        try:
            con = self._connect()
        except sqlite3.Error as exc:  # pragma: no cover - unusual connect failure
            raise DbQueryError(f"Could not open database: {exc}") from exc
        try:
            cur = con.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            return columns, rows
        except sqlite3.Error as exc:
            raise DbQueryError(f"SQL execution failed: {exc}") from exc
        finally:
            con.close()


__all__ = ["DbQueryError", "ExecResult", "SQLiteExecutor", "SqlExecutor"]
