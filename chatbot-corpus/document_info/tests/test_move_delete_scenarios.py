"""Lock-down regression tests proving that `classify_documents.main()`
PRESERVES the MOVED / DELETED / MOVED->UNCHANGED behaviour against a
real scratch/fixture BELGELER folder.

These three scenarios were previously validated by hand during an
isolated scratch rehearsal of the flow. This file freezes them as
regression tests. No code change -- tests only. Production data (the
real BELGELER/ folder and chatbot-corpus/document_info/
document_nodes.json) is NEVER touched -- everything lives under
`tmp_path`, the PRODUCT_INFO_DIR / BELGELER_DIR / OUTPUT_PATH env vars
that `main()` reads are temporarily redirected to those directories,
and the original env values are restored at the end.

Run: `pytest chatbot-corpus/document_info/tests/test_move_delete_scenarios.py`
(or `python chatbot-corpus/document_info/tests/test_move_delete_scenarios.py`).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_DOC_INFO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DOC_INFO_DIR))


def _mini_product_nodes(code: str = "PN1008") -> list[dict]:
    """Single-product minimal tree, matching the _mini_product_nodes pattern
    in tests/test_config.py -- triggers find_owner tier-1 code matching."""
    return [{
        "node_id": "n1", "type": "product", "parent_id": None, "source_ref": "1",
        "family": None, "subfamily": None, "subfamily_2": None,
        "product": {"product_code": code, "variant_base": None, "display_name": "X"},
        "category": None,
    }]


def _write_product_nodes(product_info_dir: Path, code: str = "PN1008") -> None:
    product_info_dir.mkdir(parents=True, exist_ok=True)
    (product_info_dir / "product_nodes.json").write_text(
        json.dumps({"product_nodes": _mini_product_nodes(code)}), encoding="utf-8")


def _run_scan(product_info_dir: Path, urunler_dir: Path, output_path: Path) -> dict:
    """Run `main()` once with temporary env vars, read back the produced
    document_nodes.json, return it. The previous env values are restored."""
    import classify_documents as cd

    env_keys = ("PRODUCT_INFO_DIR", "BELGELER_DIR", "OUTPUT_PATH", "DOCUMENT_INFO_CONFIG")
    backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["PRODUCT_INFO_DIR"] = str(product_info_dir)
    os.environ["BELGELER_DIR"] = str(urunler_dir)
    os.environ["OUTPUT_PATH"] = str(output_path)
    os.environ.pop("DOCUMENT_INFO_CONFIG", None)
    try:
        cd.main()
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def _by_rel_path(data: dict, rel_path: str) -> dict | None:
    for rec in data["documents"]:
        if rec["location"]["rel_path"] == rel_path:
            return rec
    return None


def _by_doc_id(data: dict, doc_id: str) -> dict | None:
    for rec in data["documents"]:
        if rec["identity"]["doc_id"] == doc_id:
            return rec
    return None


def _mark_pipeline_done(output_path: Path, rel_path: str) -> None:
    """Mark a record's parse/chunk state as SUCCESS -- in production this
    is done by run_parse_pipeline.py / run_chunk_pipeline.py; we mimic it
    here so the "preserved on next scan?" question is meaningful."""
    with open(output_path, encoding="utf-8") as f:
        data = json.load(f)
    rec = _by_rel_path(data, rel_path)
    rec["parse"] = {
        "parser": "run_parse_pipeline.py", "parser_version": "1.4.0",
        "status": "SUCCESS", "parsed_from_hash": rec["scan"]["content_hash"],
        "parsed_json_path": "out/a.json", "last_parsed": "2026-08-20T10:00:00+03:00",
        "error": None, "stages": None,
    }
    rec["chunk"] = {
        "status": "SUCCESS", "chunked_at": "2026-08-20T10:05:00+03:00",
        "chunked_from_hash": rec["scan"]["content_hash"],
        "chunks_path": "out/a.chunks.json", "chunker_version": "1.0.0",
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def test_moved_preserves_doc_id_and_pipeline_state(tmp_path):
    """(a) File renamed/moved -> scan_status=MOVED, doc_id AND previously
    completed parse/chunk state PRESERVED (no re-processing triggered)."""
    product_info_dir = tmp_path / "product_info"
    urunler_dir = tmp_path / "BELGELER"
    output_path = tmp_path / "document_nodes.json"
    _write_product_nodes(product_info_dir)
    urunler_dir.mkdir(parents=True)

    old_rel = "eski_klasor/PN1008 Datasheet.pdf"
    (urunler_dir / "eski_klasor").mkdir()
    (urunler_dir / old_rel).write_bytes(b"icerik-v1")

    first = _run_scan(product_info_dir, urunler_dir, output_path)
    rec1 = _by_rel_path(first, old_rel)
    assert rec1["scan"]["scan_status"] == "NEW"
    doc_id = rec1["identity"]["doc_id"]

    # Mark parse/chunk pipeline as completed (in real flow this is separate
    # scripts' job — mimicked here so the preservation is meaningful).
    _mark_pipeline_done(output_path, old_rel)

    # MOVE the file (rename + put in another folder), content stays the SAME.
    new_rel = "yeni_klasor/PN1008 Datasheet.pdf"
    (urunler_dir / "yeni_klasor").mkdir()
    (urunler_dir / old_rel).rename(urunler_dir / new_rel)

    second = _run_scan(product_info_dir, urunler_dir, output_path)
    rec2 = _by_rel_path(second, new_rel)
    assert rec2 is not None, "moved file should appear at the new path"
    assert rec2["scan"]["scan_status"] == "MOVED"
    assert rec2["identity"]["doc_id"] == doc_id, "identity must NOT break on move"
    assert rec2["parse"]["status"] == "SUCCESS", "completed parse must not be re-run"
    assert rec2["parse"]["parsed_json_path"] == "out/a.json"
    assert rec2["chunk"]["status"] == "SUCCESS", "completed chunk must not be re-run"
    assert rec2["chunk"]["chunks_path"] == "out/a.chunks.json"
    # At the OLD path there is NO second record now (MOVED = one record, not two).
    assert _by_rel_path(second, old_rel) is None


def test_deleted_marks_sticky_and_preserves_doc_id(tmp_path):
    """(b) File is REMOVED from disk entirely -> record is NOT deleted,
    it stays sticky with scan_status=DELETED; doc_id AND derived
    parse/chunk fields remain unchanged."""
    product_info_dir = tmp_path / "product_info"
    urunler_dir = tmp_path / "BELGELER"
    output_path = tmp_path / "document_nodes.json"
    _write_product_nodes(product_info_dir)
    urunler_dir.mkdir(parents=True)

    rel = "klasor/PN1008 Datasheet.pdf"
    (urunler_dir / "klasor").mkdir()
    (urunler_dir / rel).write_bytes(b"icerik-v1")

    first = _run_scan(product_info_dir, urunler_dir, output_path)
    rec1 = _by_rel_path(first, rel)
    doc_id = rec1["identity"]["doc_id"]
    _mark_pipeline_done(output_path, rel)

    # Remove the file from disk (not a move, an actual removal).
    (urunler_dir / rel).unlink()

    second = _run_scan(product_info_dir, urunler_dir, output_path)
    rec2 = _by_doc_id(second, doc_id)
    assert rec2 is not None, "record must NOT be deleted (sticky)"
    assert rec2["scan"]["scan_status"] == "DELETED"
    assert rec2["location"]["rel_path"] == rel
    assert rec2["parse"]["status"] == "SUCCESS", "DELETED must NOT touch derived fields"
    assert rec2["chunk"]["chunks_path"] == "out/a.chunks.json"
    assert second["summary"]["scan_status_counts"]["DELETED"] == 1
    # Files counted as DELETED do NOT contribute to "scanned_files" (per summary def).
    assert second["summary"]["scanned_files"] == 0

    # Third scan (file STILL missing) -- DELETED must stay sticky, the
    # record must still hold the same doc_id (idempotent stickiness).
    third = _run_scan(product_info_dir, urunler_dir, output_path)
    rec3 = _by_doc_id(third, doc_id)
    assert rec3["scan"]["scan_status"] == "DELETED"
    assert rec3["identity"]["doc_id"] == doc_id


def test_moved_then_rescan_is_unchanged_idempotent(tmp_path):
    """(c) After a MOVED, rescanning with the same path/content must
    produce UNCHANGED (not MOVED again) -- so the move processing is
    not repeated on the next nightly run (idempotent)."""
    product_info_dir = tmp_path / "product_info"
    urunler_dir = tmp_path / "BELGELER"
    output_path = tmp_path / "document_nodes.json"
    _write_product_nodes(product_info_dir)
    urunler_dir.mkdir(parents=True)

    old_rel = "eski/PN1008 Datasheet.pdf"
    (urunler_dir / "eski").mkdir()
    (urunler_dir / old_rel).write_bytes(b"icerik-v1")

    first = _run_scan(product_info_dir, urunler_dir, output_path)
    doc_id = _by_rel_path(first, old_rel)["identity"]["doc_id"]
    _mark_pipeline_done(output_path, old_rel)

    new_rel = "yeni/PN1008 Datasheet.pdf"
    (urunler_dir / "yeni").mkdir()
    (urunler_dir / old_rel).rename(urunler_dir / new_rel)

    second = _run_scan(product_info_dir, urunler_dir, output_path)
    rec2 = _by_rel_path(second, new_rel)
    assert rec2["scan"]["scan_status"] == "MOVED"
    assert rec2["identity"]["doc_id"] == doc_id

    # Rescan the SAME directory without touching anything -- path and
    # content are both unchanged.
    third = _run_scan(product_info_dir, urunler_dir, output_path)
    rec3 = _by_rel_path(third, new_rel)
    assert rec3["scan"]["scan_status"] == "UNCHANGED", (
        "Once MOVED has been processed, rescanning the same path must "
        "NOT repeat as MOVED -- otherwise the nightly run looks like "
        "it is redoing the move work every time."
    )
    assert rec3["identity"]["doc_id"] == doc_id
    assert rec3["parse"]["status"] == "SUCCESS"
    assert rec3["chunk"]["status"] == "SUCCESS"
    # Single record -- no duplication.
    assert sum(1 for r in third["documents"] if r["identity"]["doc_id"] == doc_id) == 1


def _run_all():
    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        passed += 1
        print(f"  ok  {name}")
    print(f"{passed} tests passed")


if __name__ == "__main__":
    _run_all()
