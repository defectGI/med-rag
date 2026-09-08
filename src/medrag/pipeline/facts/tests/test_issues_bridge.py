"""Tests for the bridge that routes the catalog_chunk_ownership flow's
unresolved documents/chunks into the common `issues` table (O-07, I-14)."""

from __future__ import annotations

from medrag.core.db.issues import connect_issues, count_open, fetch_all
from medrag.pipeline.facts.issues_bridge import (
    record_new_key_proposal,
    record_skipped_facts,
    record_unresolved_chunks,
    record_unresolved_docs,
)


def test_record_unresolved_docs_writes_one_row_per_doc(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    docs = [
        {"doc_id": "d1", "file_name": "f1.pdf", "reason": "no chunk source coverage"},
        {"doc_id": "d2", "file_name": "f2.pdf", "reason": "no chunk source coverage"},
    ]

    n = record_unresolved_docs(con, docs)

    assert n == 2
    assert count_open(con, reason="UNRESOLVED_DOC") == 2
    rows = fetch_all(con)
    assert {r["source_doc"] for r in rows} == {"d1", "d2"}
    assert all(r["reason"] == "UNRESOLVED_DOC" for r in rows)
    assert all(r["detail"] == "no chunk source coverage" for r in rows)


def test_record_unresolved_docs_is_a_noop_for_empty_list(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    assert record_unresolved_docs(con, []) == 0
    assert count_open(con) == 0


def test_record_unresolved_chunks_carries_chunk_id_and_reason_text(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    skipped = [
        {"doc_id": "d1", "chunk_id": "d1::c16", "file_name": "f1.pdf",
         "_why": "MIXED chunk (heading_path empty) -- cannot carry"},
        {"doc_id": "d1", "chunk_id": "d1::c17", "file_name": "f1.pdf",
         "_why": "unresolvable in the previous table too"},
    ]

    n = record_unresolved_chunks(con, skipped)

    assert n == 2
    assert count_open(con, reason="UNRESOLVED_CHUNK") == 2
    rows = fetch_all(con)
    entity_refs = {r["entity_ref"] for r in rows}
    assert entity_refs == {"d1::c16", "d1::c17"}
    details = {r["entity_ref"]: r["detail"] for r in rows}
    assert details["d1::c16"] == "MIXED chunk (heading_path empty) -- cannot carry"
    # severity="review" -- NOT an error, awaiting human review
    assert all(r["severity"] == "review" for r in rows)


def test_unresolved_chunks_are_countable(tmp_path):
    """O-07 acceptance criterion: each unresolvable chunk is countable."""
    con = connect_issues(tmp_path / "issues.db")
    skipped = [{"doc_id": "d1", "chunk_id": f"d1::c{i}", "_why": "x"} for i in range(7)]

    record_unresolved_chunks(con, skipped)

    assert count_open(con, reason="UNRESOLVED_CHUNK") == 7


# --- O-14 prep: new-key proposals also VISIBLE in issues (NOT instead of JSONL) --

def test_record_new_key_proposal_writes_an_issue_row(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    fact = {"block": "core", "key": "stub_voltage", "status": "present", "num_value": 1}

    issue_id = record_new_key_proposal(con, product_code="DE1", fact=fact)

    assert issue_id > 0
    assert count_open(con, reason="NEW_KEY_PROPOSAL") == 1
    row = fetch_all(con)[0]
    assert row["entity_ref"] == "core.stub_voltage"
    assert row["detail"] == "product_code=DE1"
    assert row["severity"] == "review", "not an error, awaiting human review"
    assert '"product_code": "DE1"' in row["payload"]


# --- M-03: load_to_db.py's report["skipped"] also enters issues --

def test_record_skipped_facts_writes_one_row_per_skipped_item(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    skipped = [
        {"key": "stub_voltage", "reason": "condition not in the dictionary"},
        {"key": None, "reason": "non-object entry in facts[]: 'x'"},
    ]

    n = record_skipped_facts(con, skipped, source_doc="DE1")

    assert n == 2
    assert count_open(con, reason="SKIPPED_FACT") == 2
    rows = fetch_all(con)
    assert all(r["source_doc"] == "DE1" for r in rows)
    assert all(r["severity"] == "review" for r in rows)
    entity_refs = {r["entity_ref"] for r in rows}
    assert entity_refs == {"stub_voltage", None}


def test_record_skipped_facts_is_a_noop_for_empty_list(tmp_path):
    con = connect_issues(tmp_path / "issues.db")
    assert record_skipped_facts(con, [], source_doc="DE1") == 0
    assert count_open(con) == 0