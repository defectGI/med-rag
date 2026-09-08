"""N-03 (kuru kosu): `run_parse_pipeline.compute_pending()` salt-okunurdur --
`document_nodes.json`'i okur, `_needs_parse`/`DOC_TYPES` filtresini uygular,
ama HICBIR SEYE YAZMAZ. Bu, `run_nightly.py --dry-run`in parse asamasi icin
raporladigi kumenin temelidir.

Calistirma: `python -m pytest src/medrag/pipeline/cli/tests/test_parse_compute_pending.py`
"""

from __future__ import annotations

import hashlib
import json

from medrag.pipeline.cli import run_parse_pipeline as rpp
from medrag.pipeline.parser.parsers.base import PARSER_VERSION


def _rec(doc_id: str, *, needs: bool, doc_type: str = "DATASHEET",
        is_active: bool = True) -> dict:
    content_hash = f"sha256:{doc_id}"
    return {
        "identity": {"doc_id": doc_id, "file_name": f"{doc_id}.pdf"},
        "scan": {"content_hash": content_hash},
        "doc_type": doc_type,
        "is_active": is_active,
        "parse": {
            "parsed_from_hash": None if needs else content_hash,
            "parser_version": None if needs else PARSER_VERSION,
            "status": "PENDING" if needs else "SUCCESS",
        },
    }


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_registry(path, records: list[dict]) -> None:
    path.write_text(json.dumps({"documents": records}, ensure_ascii=False, indent=2),
                     encoding="utf-8")


def test_compute_pending_filters_needs_parse_and_writes_nothing(tmp_path, monkeypatch):
    registry = tmp_path / "document_nodes.json"
    records = [
        _rec("A", needs=True),
        _rec("B", needs=False),
        _rec("C", needs=True),
        _rec("D", needs=True, is_active=False),  # pasif -> needs_parse=False
    ]
    _write_registry(registry, records)

    monkeypatch.setattr(rpp, "DOCUMENT_NODES_PATH", str(registry), raising=False)
    monkeypatch.setattr(rpp, "DOC_TYPES", set(), raising=False)

    mtime_before = registry.stat().st_mtime_ns
    sha_before = _sha256(registry)

    todo, info = rpp.compute_pending()

    assert {r["identity"]["doc_id"] for r in todo} == {"A", "C"}
    assert info["pending"] == 2
    assert info["total"] == 4

    # Salt-okunur: dosyanin mtime/sha256'si HIC degismemis olmali.
    assert registry.stat().st_mtime_ns == mtime_before
    assert _sha256(registry) == sha_before


def test_compute_pending_respects_doc_types_filter(tmp_path, monkeypatch):
    registry = tmp_path / "document_nodes.json"
    records = [
        _rec("A", needs=True, doc_type="DATASHEET"),
        _rec("B", needs=True, doc_type="BROCHURE"),
    ]
    _write_registry(registry, records)

    monkeypatch.setattr(rpp, "DOCUMENT_NODES_PATH", str(registry), raising=False)
    monkeypatch.setattr(rpp, "DOC_TYPES", {"DATASHEET"}, raising=False)

    todo, info = rpp.compute_pending()

    assert [r["identity"]["doc_id"] for r in todo] == ["A"]
    assert info["total"] == 2


def test_compute_pending_limit_caps_todo_but_not_pending_count(tmp_path, monkeypatch):
    registry = tmp_path / "document_nodes.json"
    records = [_rec("A", needs=True), _rec("B", needs=True), _rec("C", needs=True)]
    _write_registry(registry, records)

    monkeypatch.setattr(rpp, "DOCUMENT_NODES_PATH", str(registry), raising=False)
    monkeypatch.setattr(rpp, "DOC_TYPES", set(), raising=False)

    todo, info = rpp.compute_pending(limit=1)

    assert len(todo) == 1
    assert info["pending"] == 3  # limit'ten ONCEKI gercek bekleyen sayisi korunur
