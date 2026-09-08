"""Tests for the shared `issues` table.

Behavior locked in: `reason`/`stage`/`severity` raise if they don't come from
the closed set ("reason has to be coded, otherwise filtering/counting is
impossible") · unresolved issues can be counted (every unresolved chunk is
countable) · the panel's only write permission (`resolve_issue`) changes the
`resolved_*` fields alone.
"""

from __future__ import annotations

from medrag.core.db.issues import (
    REASON_CODES,
    connect_issues,
    count_open,
    fetch_all,
    fetch_filtered,
    record_issue,
    resolve_issue,
)


def _con(tmp_path):
    return connect_issues(tmp_path / "issues.db")


def test_schema_is_created_idempotently(tmp_path):
    con = _con(tmp_path)
    con.close()
    con2 = _con(tmp_path)  # the second open must not fail on CREATE TABLE
    assert con2.execute("SELECT COUNT(*) FROM issues").fetchone()[0] == 0


def test_record_issue_rejects_unknown_reason(tmp_path):
    con = _con(tmp_path)
    try:
        record_issue(con, stage="facts", severity="warning", reason="ANLAMSIZ_KOD")
    except ValueError as exc:
        assert "ANLAMSIZ_KOD" in str(exc)
    else:
        raise AssertionError("bilinmeyen reason sessizce kabul edilmemeli")


def test_record_issue_rejects_unknown_stage(tmp_path):
    con = _con(tmp_path)
    try:
        record_issue(con, stage="uydurma", severity="warning", reason="UNRESOLVED_DOC")
    except ValueError:
        pass
    else:
        raise AssertionError("bilinmeyen stage sessizce kabul edilmemeli")


def test_record_and_count_open(tmp_path):
    con = _con(tmp_path)
    record_issue(con, stage="facts", severity="warning", reason="UNRESOLVED_DOC",
                 source_doc="d1", detail="no chunk source coverage")
    record_issue(con, stage="facts", severity="review", reason="UNRESOLVED_CHUNK",
                 source_doc="d1", entity_ref="d1::c0", detail="KARMA chunk")

    assert count_open(con) == 2
    assert count_open(con, reason="UNRESOLVED_DOC") == 1
    assert count_open(con, reason="UNRESOLVED_CHUNK") == 1


def test_resolve_issue_only_touches_resolved_fields(tmp_path):
    con = _con(tmp_path)
    issue_id = record_issue(con, stage="facts", severity="review", reason="UNRESOLVED_CHUNK",
                             source_doc="d1", entity_ref="d1::c0", detail="x")

    resolve_issue(con, issue_id, resolved_by="user")

    rows = fetch_all(con)
    assert len(rows) == 1
    row = rows[0]
    assert row["resolved_by"] == "user"
    assert row["resolved_at"] is not None
    # the write-side fields are UNCHANGED
    assert row["source_doc"] == "d1" and row["entity_ref"] == "d1::c0" and row["detail"] == "x"
    assert count_open(con) == 0


def test_fetch_filtered_narrows_by_each_field(tmp_path):
    """Narrowing by stage/severity/source_doc/reason -- a filter is required
    because the flat list was unusable."""
    con = _con(tmp_path)
    record_issue(con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="d1")
    record_issue(con, stage="facts", severity="warning", reason="SKIPPED_FACT", source_doc="d2")
    record_issue(con, stage="chunk", severity="review", reason="UNRESOLVED_CHUNK", source_doc="d3")

    assert len(fetch_filtered(con)) == 3
    assert [r["source_doc"] for r in fetch_filtered(con, stage="facts")] == ["d2"]
    assert {r["source_doc"] for r in fetch_filtered(con, reason="UNRESOLVED_CHUNK")} == {"d1", "d3"}
    assert [r["source_doc"] for r in fetch_filtered(con, stage="chunk", source_doc="d3")] == ["d3"]
    assert [r["source_doc"] for r in fetch_filtered(con, severity="warning")] == ["d2"]
    assert fetch_filtered(con, stage="vectorize") == []


def test_all_reason_codes_are_writable(tmp_path):
    """EVERY code in the closed set is really accepted -- widening the set but
    forgetting to update the code raises here."""
    con = _con(tmp_path)
    for code in sorted(REASON_CODES):
        record_issue(con, stage="facts", severity="warning", reason=code)
    assert count_open(con) == len(REASON_CODES)
