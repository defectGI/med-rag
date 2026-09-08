"""DbQueryRetriever end-to-end with a fake generator + real SQLite executor.

The generator is faked (no LLM endpoint, honoring the GPU rule); the executor
runs real SQLite over a temp file, so the query -> rows -> RetrievalResult
mapping is exercised for real.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from medrag.api.retrieval.core import RetrievalResult, Retriever
from medrag.api.retrieval.modules.db_query import DbQueryRetriever, SQLiteExecutor


class _FakeGenerator:
    """Returns a fixed SQL string regardless of the query. Does NOT support
    `on_stage` -- exercises the graceful-skip path."""

    def __init__(self, sql: str) -> None:
        self._sql = sql

    def generate(self, query: str) -> str:
        return self._sql


class _StagedFakeGenerator:
    """Supports `on_stage` -- exercises the forwarding path."""

    def __init__(self, sql: str) -> None:
        self._sql = sql
        self.calls: list[str] = []

    def generate(self, query: str, *, on_stage=None) -> str:
        self.calls.append(query)
        if on_stage is not None:
            on_stage("linking_done", {"tables": ["products"]})
        return self._sql


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "specs.db"
    con = sqlite3.connect(str(p))
    con.execute("CREATE TABLE products (node_id TEXT, name TEXT, price REAL)")
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [("PN1309", "Sensor", 12.5), ("PN1316", "Relay", 7.0), ("PN1323", "Cable", 3.0)],
    )
    con.commit()
    con.close()
    return p


def _retriever(db_path, sql, **kwargs):
    return DbQueryRetriever(_FakeGenerator(sql), SQLiteExecutor(db_path), **kwargs)


def test_satisfies_retriever_protocol(db_path):
    assert isinstance(_retriever(db_path, "SELECT 1"), Retriever)


def test_rows_mapped_to_results(db_path):
    r = _retriever(db_path, "SELECT node_id, name, price FROM products ORDER BY price")
    results = asyncio.run(r.retrieve("cheapest", k=10))
    assert all(isinstance(x, RetrievalResult) for x in results)
    assert [x.id for x in results] == ["PN1323", "PN1316", "PN1309"]  # node_id auto-id
    assert all(x.score == 1.0 for x in results)  # constant score by design
    assert results[0].metadata == {"node_id": "PN1323", "name": "Cable", "price": 3.0}
    assert "name=Cable" in results[0].text


def test_k_caps_results(db_path):
    r = _retriever(db_path, "SELECT node_id FROM products")
    results = asyncio.run(r.retrieve("all", k=2))
    assert len(results) == 2


def test_id_falls_back_to_row_index_without_id_column(db_path):
    r = _retriever(db_path, "SELECT name FROM products ORDER BY name")
    results = asyncio.run(r.retrieve("names", k=10))
    assert [x.id for x in results] == ["0", "1", "2"]


def test_explicit_id_column(db_path):
    r = _retriever(
        db_path, "SELECT node_id, name FROM products ORDER BY name", id_column="name"
    )
    results = asyncio.run(r.retrieve("names", k=10))
    assert [x.id for x in results] == ["Cable", "Relay", "Sensor"]


def test_empty_result_set(db_path):
    r = _retriever(db_path, "SELECT node_id FROM products WHERE price < 0")
    assert asyncio.run(r.retrieve("none", k=10)) == []


def test_on_sql_hook_receives_final_sql(db_path):
    r = _retriever(db_path, "SELECT node_id FROM products ORDER BY price")
    captured = []
    asyncio.run(r.retrieve("cheapest", k=10, on_sql=captured.append))
    assert captured == ["SELECT node_id FROM products ORDER BY price"]


def test_on_sql_hook_optional_by_default(db_path):
    r = _retriever(db_path, "SELECT node_id FROM products")
    # on_sql omitted entirely -- must not raise.
    results = asyncio.run(r.retrieve("all", k=10))
    assert len(results) == 3


# --- on_stage forwarding -------------------------------------------------------


def test_on_stage_forwarded_when_generator_supports_it(db_path):
    generator = _StagedFakeGenerator("SELECT node_id FROM products")
    r = DbQueryRetriever(generator, SQLiteExecutor(db_path))
    events = []
    asyncio.run(r.retrieve("all", k=10, on_stage=lambda s, d: events.append((s, d))))
    assert events == [("linking_done", {"tables": ["products"]})]
    assert generator.calls == ["all"]


def test_on_stage_skipped_gracefully_when_generator_lacks_support(db_path):
    # _FakeGenerator.generate(query) has no on_stage param -- must not raise,
    # must not silently double-call generate() either.
    r = _retriever(db_path, "SELECT node_id FROM products")
    events = []
    results = asyncio.run(r.retrieve("all", k=10, on_stage=lambda s, d: events.append((s, d))))
    assert events == []
    assert len(results) == 3


def test_on_stage_omitted_by_default(db_path):
    generator = _StagedFakeGenerator("SELECT node_id FROM products")
    r = DbQueryRetriever(generator, SQLiteExecutor(db_path))
    # on_stage not passed at all -- generator.generate called without it.
    asyncio.run(r.retrieve("all", k=10))
    assert generator.calls == ["all"]
