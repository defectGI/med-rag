"""Standalone row -> RetrievalResult mapping, usable without the retriever."""

from __future__ import annotations

from medrag.api.retrieval.core import RetrievalResult
from medrag.api.retrieval.modules.db_query import (
    pick_row_id,
    render_row_text,
    row_to_result,
    rows_to_results,
)

COLS = ["node_id", "name", "price"]
ROWS = [("PN1309", "Sensor", 12.5), ("PN1316", "Relay", 7.0)]


def test_rows_to_results_basic():
    results = rows_to_results(COLS, ROWS)
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert [r.id for r in results] == ["PN1309", "PN1316"]  # node_id auto-id
    assert all(r.score == 1.0 for r in results)
    assert results[0].metadata == {"node_id": "PN1309", "name": "Sensor", "price": 12.5}


def test_score_is_overridable():
    results = rows_to_results(COLS, ROWS, score=0.5)
    assert all(r.score == 0.5 for r in results)


def test_id_column_override():
    results = rows_to_results(COLS, ROWS, id_column="name")
    assert [r.id for r in results] == ["Sensor", "Relay"]


def test_pick_row_id_falls_back_to_index():
    assert pick_row_id({"name": "x"}, index=3) == "3"


def test_pick_row_id_prefers_explicit_column():
    assert pick_row_id({"id": "A", "name": "B"}, id_column="name") == "B"


def test_render_row_text_handles_none():
    assert render_row_text({"a": 1, "b": None}) == "a=1 | b="


def test_row_to_result_single():
    r = row_to_result(COLS, ROWS[0], index=0)
    assert r.id == "PN1309"
    assert "name=Sensor" in r.text
