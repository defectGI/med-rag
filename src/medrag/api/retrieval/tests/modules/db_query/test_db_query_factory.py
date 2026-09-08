"""``build_default_retriever`` wiring tests. No engine, no network, no DB file.

Only the composition is under test: the engine build is monkeypatched away
(it would otherwise reach an OpenAI-compatible endpoint, which this machine
never does -- see ARCHITECTURE.md #10).
"""

from __future__ import annotations

import pytest

from medrag.api.retrieval.modules import db_query
from medrag.api.retrieval.modules.db_query import factory as factory_mod


@pytest.fixture
def _no_engine(monkeypatch):
    """Replace the lazy text2sql engine build with a stub."""
    monkeypatch.setattr(factory_mod, "build_default_engine", lambda **kw: object())


def _rewriters_of(retriever) -> list:
    return retriever._generator._rewriters


def test_default_keeps_only_the_sqlite_dialect_fix(_no_engine, tmp_path):
    retriever = db_query.build_default_retriever(db_path=str(tmp_path / "x.db"))
    assert _rewriters_of(retriever) == [db_query.sqlite_ilike_to_like]


def test_custom_rewriters_are_appended_after_dialect_defaults(_no_engine, tmp_path):
    """They must ADD to the built-in ILIKE fix, never replace it -- a caller
    passing an app-specific rewriter should not silently lose the dialect
    correction."""
    def mine(sql: str) -> str:
        return sql

    retriever = db_query.build_default_retriever(
        db_path=str(tmp_path / "x.db"), rewriters=[mine]
    )
    assert _rewriters_of(retriever) == [db_query.sqlite_ilike_to_like, mine]


def test_custom_rewriters_run_in_order_after_cleaning(_no_engine, tmp_path):
    order: list[str] = []

    def first(sql: str) -> str:
        order.append("first")
        return sql + " /*1*/"

    def second(sql: str) -> str:
        order.append("second")
        return sql + " /*2*/"

    retriever = db_query.build_default_retriever(
        db_path=str(tmp_path / "x.db"), rewriters=[first, second]
    )
    chain = _rewriters_of(retriever)
    sql = "SELECT 1"
    for rewrite in chain:
        sql = rewrite(sql)
    assert order == ["first", "second"]
    assert sql == "SELECT 1 /*1*/ /*2*/"


def test_non_sqlite_dialect_gets_only_the_custom_rewriters(_no_engine, tmp_path):
    def mine(sql: str) -> str:
        return sql

    retriever = db_query.build_default_retriever(
        db_path=str(tmp_path / "x.db"), dialect="postgres", rewriters=[mine]
    )
    assert _rewriters_of(retriever) == [mine]


def test_omitting_rewriters_is_byte_for_byte_the_old_behavior(_no_engine, tmp_path):
    """Backward compatibility: the new kwarg defaults to None and changes
    nothing for existing callers."""
    a = db_query.build_default_retriever(db_path=str(tmp_path / "x.db"))
    b = db_query.build_default_retriever(db_path=str(tmp_path / "x.db"), rewriters=None)
    assert _rewriters_of(a) == _rewriters_of(b) == [db_query.sqlite_ilike_to_like]
