"""Tests for mark_docs_inactive.py -- the escape-hatch script that lives
at the pipeline/ root (same class as reset_parse_flags.py). SAFE core
(`mark_inactive`) with no disk/registry access; `main()`'s env/atomic-write
legs are tested via tmp_path + monkeypatch. The script lives OUTSIDE the
`medrag` package, so it is loaded FROM PATH via importlib (same pattern
run_nightly.py uses for export_products/classify_documents).

Run: `python -m pytest pipeline/tests/test_mark_docs_inactive.py`
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "mark_docs_inactive.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mark_docs_inactive", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _doc(doc_id, file_name, rel_path, is_active=True):
    return {
        "identity": {"doc_id": doc_id, "file_name": file_name},
        "location": {"rel_path": rel_path},
        "is_active": is_active,
    }


def test_mark_inactive_flags_matching_record_by_file_name():
    mod = _load_module()
    data = {"documents": [
        _doc("d1", "PN5022_TEST_DATASHEET.pdf", "TEST/PN1120/x.pdf"),
        _doc("d2", "baska.pdf", "B/baska.pdf"),
    ]}

    result = mod.mark_inactive(data, ["PN5022_TEST_DATASHEET.pdf"])

    assert data["documents"][0]["is_active"] is False
    assert data["documents"][1]["is_active"] is True
    assert [r["doc_id"] for r in result["changed"]] == ["d1"]
    assert result["already_inactive"] == []
    assert result["unmatched"] == []


def test_mark_inactive_matches_rel_path_case_and_separator_insensitive():
    """Hedef Windows ayiriciyla VE buyuk harfle yazilsa da rel_path esler --
    operator hatasi en olasi yer burasi."""
    mod = _load_module()
    data = {"documents": [_doc("d1", "schema.json", "TOOLS/attribute_schema.json")]}

    result = mod.mark_inactive(data, [r"tools\ATTRIBUTE_SCHEMA.json"])

    assert data["documents"][0]["is_active"] is False
    assert result["unmatched"] == []


def test_mark_inactive_requires_exact_match_not_substring():
    """Kismi eslesme YOK: 'attribute_schema' hedefi 'attribute_schema.json'
    kaydini ISLEMEZ -- yanlis dosya adinin sessiz gecmesi kalici veri
    kaybidir, eslesmeyen hedef iade edilir."""
    mod = _load_module()
    data = {"documents": [_doc("d1", "attribute_schema.json", "T/attribute_schema.json")]}

    result = mod.mark_inactive(data, ["attribute_schema"])

    assert data["documents"][0]["is_active"] is True
    assert result["changed"] == []
    assert result["unmatched"] == ["attribute_schema"]


def test_mark_inactive_is_idempotent_for_already_inactive_records():
    mod = _load_module()
    data = {"documents": [_doc("d1", "eski.pdf", "E/eski.pdf", is_active=False)]}

    result = mod.mark_inactive(data, ["eski.pdf"])

    assert data["documents"][0]["is_active"] is False
    assert result["changed"] == []
    assert [r["doc_id"] for r in result["already_inactive"]] == ["d1"]


def test_mark_inactive_flags_every_record_sharing_the_file_name():
    """Ayni file_name corpus'ta birden fazla kayitta cikabilir -- script
    Hepsini isaretler ve tek tek RAPORLAR (operator fazladan eslesmeyi gorup
    rel_path ile daraltir)."""
    mod = _load_module()
    data = {"documents": [
        _doc("d1", "a.pdf", "U1/a.pdf"),
        _doc("d2", "a.pdf", "U2/a.pdf"),
        _doc("d3", "b.pdf", "U1/b.pdf"),
    ]}

    result = mod.mark_inactive(data, ["a.pdf"])

    assert [r["doc_id"] for r in result["changed"]] == ["d1", "d2"]
    assert all(rec["is_active"] is False for rec in data["documents"][:2])
    assert data["documents"][2]["is_active"] is True


def test_main_writes_atomically_and_returns_zero(monkeypatch, tmp_path):
    mod = _load_module()
    nodes = tmp_path / "document_nodes.json"
    nodes.write_text(json.dumps({"documents": [
        _doc("d1", "PN5022_TEST_DATASHEET.pdf", "T/PN5022_TEST_DATASHEET.pdf"),
    ]}), encoding="utf-8")
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(nodes))

    rc = mod.main(["PN5022_TEST_DATASHEET.pdf"])

    assert rc == 0
    data = json.loads(nodes.read_text(encoding="utf-8"))
    assert data["documents"][0]["is_active"] is False
    assert list(tmp_path.glob(".tmp_*")) == []  # gecici dosya kalmadi


def test_main_writes_nothing_when_a_target_is_unmatched(monkeypatch, tmp_path):
    """Eslesmeyen hedef VARSA butun kosu basarisizdir: HICBIR kayit
    degismez (eslesenler DAHIL -- yarim uygulama, yanlis hedefin sessiz
    gecmesinden iyidir), cikis kodu 1."""
    mod = _load_module()
    nodes = tmp_path / "document_nodes.json"
    original = {"documents": [
        _doc("d1", "eslesen.pdf", "E/eslesen.pdf"),
    ]}
    nodes.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(nodes))

    rc = mod.main(["eslesen.pdf", "yok_boyle_bir_dosya.pdf"])

    assert rc == 1
    assert json.loads(nodes.read_text(encoding="utf-8")) == original


def test_main_dry_run_leaves_file_untouched(monkeypatch, tmp_path):
    mod = _load_module()
    nodes = tmp_path / "document_nodes.json"
    original = {"documents": [
        _doc("d1", "eslenecek.pdf", "E/eslenecek.pdf"),
    ]}
    nodes.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(nodes))

    rc = mod.main(["--dry-run", "eslenecek.pdf"])

    assert rc == 0
    assert json.loads(nodes.read_text(encoding="utf-8")) == original


def test_main_fails_loudly_when_env_path_undefined(monkeypatch, tmp_path, capsys):
    mod = _load_module()
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("DOCUMENT_NODES_PATH", raising=False)

    rc = mod.main(["herhangi.pdf"])

    assert rc == 2
    assert "DOCUMENT_NODES_PATH" in capsys.readouterr().err
