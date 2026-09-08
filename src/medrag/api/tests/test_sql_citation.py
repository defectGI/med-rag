"""`medrag.api.sql_citation` -- citation columns added by CODE.

No test hits an LLM or the network; the rewriter is a pure `str -> str`
function. One group of tests connects READ-ONLY to the real
`facts/db/specs.db` and verifies the rewritten SQL actually runs
(SQLite is local, no GPU needed).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from medrag.api.sql_citation import CITATION_COLUMNS, add_citation_columns

_SPECS_DB = Path(__file__).resolve().parents[4] / "facts" / "db" / "specs.db"


def _selected(sql: str) -> str:
    """The projection text between SELECT and FROM (lowercased)."""
    lowered = sql.lower()
    return lowered[lowered.index("select") + 6 : lowered.index(" from ")]


# --- cases where columns ARE added -------------------------------------------


def test_duz_projeksiyona_uc_atif_kolonu_eklenir():
    out = add_citation_columns(
        "SELECT raw_text FROM spec_value WHERE product_code='PN1099' AND key='weight'"
    )
    for column in CITATION_COLUMNS:
        assert column in _selected(out), f"{column} not added"
    assert "raw_text" in _selected(out)  # the original projection is preserved


def test_alias_kullaniliyorsa_alias_ile_nitelenir():
    out = add_citation_columns("SELECT sv.raw_text FROM spec_value sv WHERE sv.key='weight'")
    assert "sv.source_file_name" in out
    assert "spec_value.source_file_name" not in out


def test_join_varken_kolonlar_spec_value_tarafina_niteleniyor():
    """In a multi-table query a bare column name would raise an ambiguity error."""
    out = add_citation_columns(
        "SELECT sv.raw_text, p.family FROM spec_value sv "
        "JOIN product p ON p.product_code=sv.product_code"
    )
    assert "sv.product_code" in out
    assert "sv.key" in out
    assert "sv.source_file_name" in out


def test_kismen_var_olan_kolonlar_tekrar_eklenmez():
    out = add_citation_columns("SELECT product_code, raw_text FROM spec_value")
    assert _selected(out).count("product_code") == 1
    assert "source_file_name" in _selected(out)


def test_hepsi_zaten_secilmisse_sorgu_aynen_doner():
    # `value_id` was added so the evidence panel can resolve the chunk text
    # behind a SQL row via this PK (see sql_evidence.py).
    sql = "SELECT product_code, key, source_file_name, value_id, raw_text FROM spec_value"
    assert add_citation_columns(sql) == sql


def test_value_id_kanit_chunk_zinciri_icin_eklenir():
    """`value_id` is PART of the citation columns -- `sql_evidence.py` resolves
    evidence chunks (source_chunk_id/evidence -> chunk text) through this PK.
    The pointers THEMSELVES are not added: `evidence` is a JSON blob, and every
    selected column ends up in the row text the conversing model sees
    (`render_row_text`)."""
    out = _selected(add_citation_columns("SELECT raw_text FROM spec_value"))
    assert "value_id" in out
    assert "evidence" not in out
    assert "source_chunk_id" not in out


def test_alias_ile_secilmis_kolon_tekrar_eklenmez():
    """If `sv.key AS attr` is written, `key` is STILL selected -- the check
    looks at the real column name, not the alias."""
    out = add_citation_columns("SELECT sv.key AS attr, sv.raw_text FROM spec_value sv")
    assert _selected(out).count("key") == 1


def test_dis_select_hedeflenir_alt_sorgu_bozulmaz():
    out = add_citation_columns(
        "SELECT raw_text FROM spec_value WHERE product_code IN "
        "(SELECT product_code FROM product WHERE family='X')"
    )
    assert "source_file_name" in _selected(out)
    # The subquery's projection must stay single-column -- don't break IN (...).
    assert "IN (SELECT product_code FROM product" in out


# --- cases deliberately LEFT UNTOUCHED ---------------------------------------
# Shared rationale: adding columns changes row IDENTITY and breaks the answer.


@pytest.mark.parametrize(
    "sql",
    [
        # DISTINCT: a "which products" query -- adding the file name would
        # return the same product once per document and inflate the list.
        "SELECT DISTINCT product_code FROM spec_value WHERE key='weight'",
        # Aggregate / GROUP BY: breaks the grouping or invalidates the SQL.
        "SELECT COUNT(*) FROM spec_value WHERE status='present'",
        "SELECT family, COUNT(*) FROM spec_value GROUP BY family",
        "SELECT AVG(num_value) FROM spec_value WHERE key='weight'",
        # Set operation: column counts wouldn't match, the query would blow up.
        "SELECT product_code FROM spec_value UNION SELECT product_code FROM product",
        # SELECT *: the citation columns already come along.
        "SELECT * FROM spec_value WHERE key='weight'",
        # A query that never touches spec_value.
        "SELECT model FROM product WHERE family LIKE '%AVIONICS%'",
        # spec_value exists in the subquery but NOT in the OUTER select -- an
        # added column would be invisible at that level.
        (
            "SELECT p.family FROM product p WHERE p.product_code IN "
            "(SELECT product_code FROM spec_value WHERE key='weight')"
        ),
    ],
)
def test_riskli_sorgular_aynen_doner(sql):
    assert add_citation_columns(sql) == sql


# --- error policy -------------------------------------------------------------


@pytest.mark.parametrize(
    "bozuk",
    ["", "   ", "bu SQL değil ki", "SELECT FROM WHERE", "DROP TABLE spec_value",
     "SELECT raw_text FROM spec_value WHERE ((("],
)
def test_ayristirilamayan_girdi_aynen_doner_ve_patlamaz(bozuk):
    assert add_citation_columns(bozuk) == bozuk


def test_hicbir_girdide_istisna_firlatmaz():
    """The three-stage `except TypeError` fallback in `flows/sql_topn.py` used
    to mistake an exception for "kwarg not supported" and run the SAME query
    2-3 times -- triple the cost under the GPU gate. Hence the rewriter is
    absolutely silent."""
    for girdi in ["", None, 123, [], {"a": 1}, "SELECT"]:
        try:
            add_citation_columns(girdi)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"raised an exception for {girdi!r}: {type(exc).__name__}: {exc}")


# --- against the real DB (local SQLite, NO inference) ----------------------


@pytest.mark.skipif(not _SPECS_DB.exists(), reason="facts/db/specs.db yok")
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT raw_text FROM spec_value WHERE key='data_rate' LIMIT 2",
        "SELECT sv.raw_text FROM spec_value sv WHERE sv.key='data_rate' LIMIT 2",
        (
            "SELECT sv.raw_text, p.family FROM spec_value sv "
            "JOIN product p ON p.product_code=sv.product_code WHERE sv.key='data_rate' LIMIT 2"
        ),
    ],
)
def test_yeniden_yazilan_sql_gercek_dbde_calisir_ve_atif_kolonlarini_getirir(sql):
    rewritten = add_citation_columns(sql)
    assert rewritten != sql
    with sqlite3.connect(f"file:{_SPECS_DB}?mode=ro", uri=True) as conn:
        cursor = conn.execute(rewritten)
        columns = [d[0] for d in cursor.description]
        row = cursor.fetchone()
    for column in CITATION_COLUMNS:
        assert column in columns
    assert row is not None
    # `source_file_name` is actually populated -- the field the evidence panel relies on.
    assert row[columns.index("source_file_name")]


@pytest.mark.skipif(not _SPECS_DB.exists(), reason="facts/db/specs.db yok")
def test_dokunulmayan_sorgular_da_gercek_dbde_calisir():
    for sql in (
        "SELECT DISTINCT product_code FROM spec_value WHERE key='data_rate' LIMIT 3",
        "SELECT COUNT(*) FROM spec_value WHERE status='present'",
    ):
        with sqlite3.connect(f"file:{_SPECS_DB}?mode=ro", uri=True) as conn:
            conn.execute(add_citation_columns(sql)).fetchone()
