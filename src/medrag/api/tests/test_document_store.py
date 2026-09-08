"""`document_store.resolve_document_download` -- the SINGLE place the document
download endpoint reads `file_path`. Tested against a real sqlite file on disk
(tmp_path); never touches network/GPU.

Note on schema: `doc_id` alone is the PK of `document` (one physical file =
one row; fan-out lives in `document_owner`) and the old `source_path` column
is now `file_path` -- this fixture reflects that schema."""

from __future__ import annotations

import sqlite3

import pytest

from medrag.api.document_store import resolve_document_download


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "specs.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE document (doc_id TEXT PRIMARY KEY, doc_type TEXT, "
        "file_name TEXT, file_path TEXT, is_active INTEGER)"
    )
    conn.commit()
    conn.close()
    return path


def _insert(db_path, doc_id, doc_type, file_name, file_path, is_active=1):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO document (doc_id, doc_type, file_name, file_path, is_active) "
        "VALUES (?, ?, ?, ?, ?)",
        (doc_id, doc_type, file_name, file_path, is_active),
    )
    conn.commit()
    conn.close()


def test_resolves_doc_id_to_real_file_on_disk(db_path, tmp_path):
    real_file = tmp_path / "PN5106_Datasheet.pdf"
    real_file.write_bytes(b"%PDF-1.4 fake")
    _insert(db_path, "d1", "DATASHEET", "PN5106_Datasheet.pdf", str(real_file))

    result = resolve_document_download(str(db_path), "d1")

    assert result is not None
    assert result.path == real_file
    assert result.file_name == "PN5106_Datasheet.pdf"


@pytest.fixture()
def db_with_rel_path(tmp_path):
    """Real `specs.db` schema: `build_facts_db.py` also writes `rel_path`."""
    path = tmp_path / "specs_rel.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE document (doc_id TEXT PRIMARY KEY, doc_type TEXT, "
        "file_name TEXT, rel_path TEXT, file_path TEXT, is_active INTEGER)"
    )
    conn.execute(
        "INSERT INTO document (doc_id, doc_type, file_name, rel_path, file_path, is_active) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        ("d1", "DATASHEET", "PN5106_Datasheet.pdf",
         "AVIONICS/PN1309/PN5106_Datasheet.pdf",
         "C:/Users/biri/laptop/chatbot-corpus/BELGELER/AVIONICS/PN1309/PN5106_Datasheet.pdf"),
    )
    conn.commit()
    conn.close()
    return path


def test_prefers_corpus_root_and_rel_path_over_stale_absolute_path(
    db_with_rel_path, tmp_path, monkeypatch
):
    """The real server situation: `file_path` points to a DIFFERENT machine's
    path (that file does not exist here), but corpus root + `rel_path` resolve
    to the real file."""
    root = tmp_path / "corpus" / "BELGELER"
    real = root / "AVIONICS" / "PN1309" / "PN5106_Datasheet.pdf"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setenv("BELGELER_DIR", str(root))

    result = resolve_document_download(str(db_with_rel_path), "d1")

    assert result is not None
    assert result.path == real
    assert result.file_name == "PN5106_Datasheet.pdf"


def test_falls_back_to_file_path_when_corpus_root_unset(db_with_rel_path, tmp_path, monkeypatch):
    """When `BELGELER_DIR` is not set, behavior does NOT change -- `file_path`
    is read (here that path is also missing, hence `None`)."""
    monkeypatch.delenv("BELGELER_DIR", raising=False)
    assert resolve_document_download(str(db_with_rel_path), "d1") is None


def test_falls_back_to_file_path_when_rel_path_missing_on_disk(
    db_with_rel_path, tmp_path, monkeypatch
):
    """Root defined but the file is NOT under the corpus -> fall back to the
    old path; if `file_path` is a real file it resolves again."""
    root = tmp_path / "bos_korpus"
    root.mkdir()
    monkeypatch.setenv("BELGELER_DIR", str(root))
    real = tmp_path / "gercek.pdf"
    real.write_bytes(b"%PDF-1.4 fake")
    conn = sqlite3.connect(db_with_rel_path)
    conn.execute("UPDATE document SET file_path = ? WHERE doc_id = 'd1'", (str(real),))
    conn.commit()
    conn.close()

    result = resolve_document_download(str(db_with_rel_path), "d1")

    assert result is not None
    assert result.path == real


def test_works_on_schema_without_rel_path_column(db_path, tmp_path, monkeypatch):
    """A schema WITHOUT the `rel_path` column (legacy/reduced) must NOT error."""
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path))
    real = tmp_path / "eski_sema.pdf"
    real.write_bytes(b"%PDF-1.4 fake")
    _insert(db_path, "d9", "DATASHEET", "eski_sema.pdf", str(real))

    result = resolve_document_download(str(db_path), "d9")

    assert result is not None
    assert result.path == real


def test_returns_none_when_doc_id_unknown(db_path):
    assert resolve_document_download(str(db_path), "not-a-real-doc-id") is None


def test_returns_none_for_empty_doc_id(db_path):
    assert resolve_document_download(str(db_path), "") is None


def test_returns_none_when_row_inactive(db_path, tmp_path):
    real_file = tmp_path / "old.pdf"
    real_file.write_bytes(b"data")
    _insert(db_path, "d1", "DATASHEET", "old.pdf", str(real_file), is_active=0)

    assert resolve_document_download(str(db_path), "d1") is None


def test_returns_none_when_file_missing_on_disk(db_path, tmp_path):
    missing = tmp_path / "gone.pdf"
    _insert(db_path, "d1", "DATASHEET", "gone.pdf", str(missing))

    assert resolve_document_download(str(db_path), "d1") is None


def test_result_never_exposes_file_path_attribute():
    from dataclasses import fields

    from medrag.api.document_store import DocumentDownload

    field_names = {f.name for f in fields(DocumentDownload)}
    assert field_names == {"path", "file_name"}
    assert "file_path" not in field_names
