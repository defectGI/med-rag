"""SQLiteExecutor tests against a real throwaway SQLite file (no network/GPU)."""

from __future__ import annotations

import sqlite3

import pytest

from medrag.api.retrieval.modules.db_query import (
    DbQueryError,
    SqlExecutor,
    SQLiteExecutor,
)


def _make_db(path) -> None:
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE products (node_id TEXT, name TEXT, price REAL)")
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [("PN1309", "Sensor", 12.5), ("PN1316", "Relay", 7.0)],
    )
    con.commit()
    con.close()


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "specs.db"
    _make_db(p)
    return p


def test_executor_satisfies_protocol(db_path):
    assert isinstance(SQLiteExecutor(db_path), SqlExecutor)


def test_execute_returns_columns_and_rows(db_path):
    cols, rows = SQLiteExecutor(db_path).execute(
        "SELECT node_id, price FROM products ORDER BY price"
    )
    assert cols == ["node_id", "price"]
    assert rows == [("PN1316", 7.0), ("PN1309", 12.5)]


def test_read_only_rejects_writes(db_path):
    ex = SQLiteExecutor(db_path, read_only=True)
    with pytest.raises(DbQueryError):
        ex.execute("DELETE FROM products")
    # data untouched
    _, rows = ex.execute("SELECT COUNT(*) FROM products")
    assert rows[0][0] == 2


def test_missing_db_raises(tmp_path):
    with pytest.raises(DbQueryError):
        SQLiteExecutor(tmp_path / "nope.db").execute("SELECT 1")


def test_bad_sql_raises_dbqueryerror(db_path):
    with pytest.raises(DbQueryError):
        SQLiteExecutor(db_path).execute("SELECT * FROM no_such_table")
