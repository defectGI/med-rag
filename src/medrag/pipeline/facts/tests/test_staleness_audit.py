"""K-96 (N-18) + K-97 (2026-08-26 fix): acceptance tests for the staleness
audit. No real Ollama/LLM call -- only against a real sqlite schema
(`:memory:`, spec_value's real columns, the SAME CREATE TABLE as the
other test files). `compute_staleness` WRITES NOTHING -- a test that
verifies this directly with a read-only connection exists. After K-97
`stale_product_codes` is NO LONGER a product SELECTOR; it is just the
UNION of the three criteria -- so it does NOT TOUCH
`build_product_manifest`/`all_product_codes`, no fake version needed.
"""
from __future__ import annotations

import json
import sqlite3

from medrag.pipeline.facts.load_to_db import EXTRACTOR
from medrag.pipeline.facts.staleness_audit import (
    compute_staleness,
    current_chunk_sha_set,
    stale_product_codes,
)

_SHA_LIVE = "sha256:" + "a" * 64
_SHA_GONE = "sha256:" + "b" * 64


def _db():
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE spec_value (
        value_id INTEGER PRIMARY KEY, product_code TEXT, family TEXT, subfamily TEXT,
        block TEXT, key TEXT, kind TEXT, unit TEXT, condition TEXT, status TEXT,
        num_value REAL, text_value TEXT, val_min REAL, val_typ REAL, val_max REAL,
        bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
        source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT, evidence TEXT NOT NULL DEFAULT '[]',
        extractor TEXT, extractor_version TEXT, extracted_at TEXT)""")
    con.commit()
    return con


def _insert(con, value_id, product_code, key, *, evidence, extractor=EXTRACTOR,
            extractor_version="1.0.0", status="present"):
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, status, evidence, "
        "extractor, extractor_version) VALUES (?, ?, 'core', ?, ?, ?, ?, ?)",
        (value_id, product_code, key, status, json.dumps(evidence, ensure_ascii=False),
         extractor, extractor_version),
    )
    con.commit()


def _write_all_chunks(path, **documents):
    path.write_text(json.dumps({"documents": documents}, ensure_ascii=False), encoding="utf-8")


def test_fresh_row_matches_no_criterion(tmp_path):
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}])

    report = compute_staleness(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert report.dangling_evidence.row_count == 0
    assert report.empty_evidence.row_count == 0
    assert report.stale_extractor_version.row_count == 0


def test_dangling_evidence_counts_chunk_gone_from_current_set(tmp_path):
    """K-96 criterion (a): a `chunk_sha256` in `evidence[]` is MISSING
    from the current chunk set -- chunk was regenerated, text changed,
    or the document was deleted."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_GONE}])

    report = compute_staleness(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert report.dangling_evidence.row_count == 1
    assert report.dangling_evidence.product_count == 1
    assert report.dangling_evidence.examples[0].product_code == "DE1"


def test_empty_evidence_counts_extracted_rows_with_no_evidence(tmp_path):
    """K-96 criterion (b): evidence[] is EMPTY -- nothing supports it.
    Bootstrap rows (`extractor IS NULL`) are EXCLUDED from this
    criterion (separate test below)."""
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[])

    report = compute_staleness(con, current_facts_version="1.0.0",
                                all_chunks_path=tmp_path / "missing.json")

    assert report.empty_evidence.row_count == 1
    assert report.empty_evidence.product_count == 1


def test_bootstrap_rows_are_excluded_from_every_criterion(tmp_path):
    """Bootstrap (`extractor IS NULL`, `evidence='[]'`, `status='not_specified'`)
    was NEVER EXTRACTED -- "empty", not "stale". If it stayed in scope
    thousands of bootstrap rows would drown the empty_evidence criterion
    and make it useless."""
    con = _db()
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, status) "
        "VALUES (1, 'DE1', 'core', 'attr', 'not_specified')"
    )
    con.commit()

    report = compute_staleness(con, current_facts_version="1.0.0",
                                all_chunks_path=tmp_path / "missing.json")

    assert report.empty_evidence.row_count == 0
    assert report.dangling_evidence.row_count == 0
    assert report.stale_extractor_version.row_count == 0


def test_stale_extractor_version_counts_rows_behind_current(tmp_path):
    """K-96 criterion (c): extractor_version is BEHIND the current
    facts_version -- the extraction METHOD (prompt/schema/threshold)
    has changed."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}],
            extractor_version="1.0.0")

    report = compute_staleness(con, current_facts_version="1.1.0", all_chunks_path=all_chunks)

    assert report.stale_extractor_version.row_count == 1
    assert report.dangling_evidence.row_count == 0, "version diff must NOT trigger (a)"


def test_a_row_can_match_more_than_one_criterion_independently():
    """All three are COUNTED INDEPENDENTLY -- intersections are NOT
    reported; one row can match several."""
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[], extractor_version="0.9.0")

    report = compute_staleness(con, current_facts_version="1.0.0",
                                all_chunks_path="C:/path-that-will-never-exist.json")

    assert report.empty_evidence.row_count == 1
    assert report.stale_extractor_version.row_count == 1


def test_examples_capped_at_ten_but_row_count_reflects_all():
    con = _db()
    for i in range(1, 16):
        _insert(con, i, f"DE{i}", "attr", evidence=[])

    report = compute_staleness(con, current_facts_version="1.0.0",
                                all_chunks_path="C:/path-that-will-never-exist.json")

    assert report.empty_evidence.row_count == 15
    assert report.empty_evidence.product_count == 15
    assert len(report.empty_evidence.examples) == 10


def test_current_chunk_sha_set_falls_back_to_node_id_without_content_sha256():
    """Same fallback as `discover.py::build_product_manifest`: if
    `content_sha256` is missing, use `node_id`."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "all_chunks.json"
        _write_all_chunks(path, **{"doc-1": {"nodes": [{"node_id": "doc-1::c0"}]}})
        shas = current_chunk_sha_set(path)

    assert shas == frozenset({"doc-1::c0"})


def test_current_chunk_sha_set_empty_when_file_missing(tmp_path):
    assert current_chunk_sha_set(tmp_path / "missing.json") == frozenset()


def test_compute_staleness_writes_nothing(tmp_path):
    """Must work even with a read-only connection -- direct evidence
    for N-18's "WRITES NOTHING" acceptance (a write would crash sqlite3
    with 'attempt to write a readonly database')."""
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(str(db_path))
    con.execute("""CREATE TABLE spec_value (
        value_id INTEGER PRIMARY KEY, product_code TEXT, family TEXT, subfamily TEXT,
        block TEXT, key TEXT, kind TEXT, unit TEXT, condition TEXT, status TEXT,
        num_value REAL, text_value TEXT, val_min REAL, val_typ REAL, val_max REAL,
        bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
        source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT, evidence TEXT NOT NULL DEFAULT '[]',
        extractor TEXT, extractor_version TEXT, extracted_at TEXT)""")
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, status, evidence, extractor, extractor_version) "
        "VALUES (1, 'DE1', 'core', 'attr', 'present', '[]', ?, '1.0.0')",
        (EXTRACTOR,),
    )
    con.commit()
    con.close()

    ro_con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        report = compute_staleness(ro_con, current_facts_version="1.0.0",
                                    all_chunks_path=tmp_path / "missing.json")
    finally:
        ro_con.close()

    assert report.empty_evidence.row_count == 1


def test_report_to_dict_shape():
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[])

    report = compute_staleness(con, current_facts_version="1.0.0",
                                all_chunks_path="C:/path-that-will-never-exist.json")
    data = report.to_dict()
    assert set(data) == {"dangling_evidence", "empty_evidence", "stale_extractor_version"}
    for section in data.values():
        assert set(section) == {"row_count", "product_count", "examples"}


# --- stale_product_codes (after K-97: just the AUDIT union, -----------
# no longer stage_facts' selector -- see staleness_audit.py module docstring)
# ----------------------------------------------------------------------------


def test_stale_product_codes_unions_the_three_criteria(tmp_path):
    """DE1 from dangling_evidence, DE2 from empty_evidence -- they
    UNION; the fully evidenced DE3 stays OUT. Before K-97 there was
    also a FOURTH 'no-evidence-in-new-chunk' criterion -- REMOVED (the
    chunks-that-produce-no-fact root cause that marked every product
    stale forever; see module docstring); this function now does NOT
    TOUCH `build_product_manifest`/`all_product_codes` AT ALL."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_GONE}])
    _insert(con, 2, "DE2", "attr", evidence=[])
    _insert(con, 3, "DE3", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}])

    result = stale_product_codes(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert result == ["DE1", "DE2"]


def test_stale_product_codes_empty_when_nothing_stale(tmp_path):
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}])

    result = stale_product_codes(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert result == []


def test_stale_product_codes_selects_stale_extractor_version_with_no_doc_changes(tmp_path):
    """Follow-up item 2 -- acceptance test (verbatim): 'even when no
    document changed but `facts_version` was bumped, the selector must
    NOT return EMPTY.' Evidence/chunk is all GOOD (dangling/empty not
    triggered) -- ONLY `extractor_version` is BEHIND the current
    `facts_version`."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE42", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}],
            extractor_version="0.9.0")

    result = stale_product_codes(con, current_facts_version="1.2.0", all_chunks_path=all_chunks)

    assert result == ["DE42"], "no document changed but selector returned EMPTY"


def test_stale_product_codes_selects_dangling_evidence_with_no_doc_changes(tmp_path):
    """Follow-up item 3 -- acceptance test (verbatim): 'when a product's
    evidence chunk_sha256 is missing from current all_chunks, even with
    no NEW/MODIFIED documents, that product must be selected.' Re-chunk
    blind spot: the source document DID NOT CHANGE (this function does
    not look at scan_status) but the chunker's logic/version changed and
    chunk SHAs changed -- `all_chunks.json` now carries DIFFERENT SHAs."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE43", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_GONE}])

    result = stale_product_codes(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert result == ["DE43"], "no document changed but dangling evidence selector returned EMPTY"


# --- K-101: the audit must be built with the SAME shape the production ---
# extractor_version column has in real use.

def test_production_extractor_version_is_not_stale(tmp_path):
    """MEASURED BUG (2026-08-27): OTHER tests in this file use
    `extractor_version="1.0.0"` -- values that NEVER OCCUR in
    production -- so they pass while criterion (c) is triggered for
    EVERY row in production (`load_to_db` was writing `"chunk_v1:<sha12>"`
    to the column, see `load_to_db::extractor_version`'s K-101 note).
    This test sets up the column with the PRODUCTION format."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}],
            extractor_version="1.0.0+chunk_v1:a1b2c3d4e5f6")

    report = compute_staleness(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert report.stale_extractor_version.row_count == 0
    assert stale_product_codes(con, current_facts_version="1.0.0",
                                all_chunks_path=all_chunks) == []


def test_legacy_prompt_hash_version_is_not_treated_as_behind(tmp_path):
    """Rows written BEFORE K-101 (`"chunk_v1:<sha12>"`, NOT semver)
    are "incomparable", not "behind" -- otherwise the fix itself would
    re-trigger the "every night, the whole corpus" behaviour it tried
    to close. Refreshing those rows is done MANUALLY via
    `FACTS_PROCESS_ALL=1`."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}],
            extractor_version="chunk_v1:a1b2c3d4e5f6")

    report = compute_staleness(con, current_facts_version="1.0.0", all_chunks_path=all_chunks)

    assert report.stale_extractor_version.row_count == 0


def test_facts_version_bump_still_marks_production_rows_stale(tmp_path):
    """Criterion (c) must NOT BE KILLED by the fix: when `facts_version`
    is MANUALLY bumped, production-shaped rows MUST still be counted
    as stale (K-96's real purpose)."""
    all_chunks = tmp_path / "all_chunks.json"
    _write_all_chunks(all_chunks, **{
        "doc-1": {"nodes": [{"node_id": "doc-1::c0", "content_sha256": _SHA_LIVE}]},
    })
    con = _db()
    _insert(con, 1, "DE1", "attr", evidence=[{"doc_id": "doc-1", "chunk_sha256": _SHA_LIVE}],
            extractor_version="1.0.0+chunk_v1:a1b2c3d4e5f6")

    report = compute_staleness(con, current_facts_version="1.1.0", all_chunks_path=all_chunks)

    assert report.stale_extractor_version.row_count == 1
    assert report.stale_extractor_version.products == {"DE1"}