"""Tests for run_chunk_pipeline's registry `chunk` block (offline, no model/network).

KARAR-009 (root PROTOCOL.md): chunk stage's registry leg. The block is READ
from each document's own provenance inside all_chunks.json (KARAR-005) --
not re-derived here; these tests pin that contract exactly.

Decision (KARAR-012 second revision): one file `all_chunks.json`
(doc_id -> ChunkSet dict) instead of per-document files. RAPTOR (scope
trees, `_corpus`/`_profile.*`) was removed entirely.

Run: `python -m pytest src/medrag/pipeline/cli/tests/test_chunk_registry.py`.
`medrag.pipeline.cli` is a real installed package -- qualified import, no
sys.path trick.
"""

from __future__ import annotations

import json
from pathlib import Path

from medrag.pipeline.cli import run_chunk_pipeline as rcp


def _all_chunks_file(path: Path, **belgeler) -> Path:
    """belgeler: doc_id -> provenance dict (empty dict = no provenance, pre-v4)."""
    documents = {}
    for doc_id, provenance in belgeler.items():
        veri = {"schema_version": 4, "doc_id": doc_id, "scope": "document", "nodes": []}
        if provenance:
            veri["provenance"] = provenance
        documents[doc_id] = veri
    path.write_text(json.dumps({"documents": documents}), encoding="utf-8")
    return path


def _registry(tmp_path: Path, *doc_ids: str) -> Path:
    path = tmp_path / "document_nodes.json"
    path.write_text(json.dumps({"documents": [
        {"identity": {"doc_id": d},
         "scan": {"content_hash": "sha256:aaa"},
         "chunk": {"status": "PENDING", "chunked_at": None,
                   "chunked_from_hash": None, "chunks_path": None,
                   "chunker_version": None}}
        for d in doc_ids]}), encoding="utf-8")
    return path


def _blok(registry: Path, doc_id: str) -> dict:
    data = json.loads(registry.read_text(encoding="utf-8"))
    return next(r["chunk"] for r in data["documents"]
                if r["identity"]["doc_id"] == doc_id)


def test_blok_provenance_dan_okunur(tmp_path):
    all_chunks = tmp_path / "all_chunks.json"
    _all_chunks_file(all_chunks, d1={
        "chunker_version": "1.0.0", "generated_at": "2026-07-17T14:03:22+03:00",
        "source": {"raw_sha256": "sha256:aaa", "parser_version": "1.4.0",
                   "ir_version": 9}})
    registry = _registry(tmp_path, "d1")

    assert rcp.update_registry(registry, all_chunks) == (1, 0)
    blok = _blok(registry, "d1")
    assert blok["status"] == "SUCCESS"
    assert blok["chunked_at"] == "2026-07-17T14:03:22+03:00"
    # Registry and all_chunks.json can never disagree about which bytes the
    # chunks came from: the value is copied, not re-derived.
    assert blok["chunked_from_hash"] == "sha256:aaa"
    assert blok["chunker_version"] == "1.0.0"
    # Single shared file -- the record's own doc_id is the key inside the
    # documents dict.
    assert blok["chunks_path"].endswith("all_chunks.json")


def test_chunk_dosyasi_yoksa_pending_e_doner(tmp_path):
    """A document that was previously chunked but is no longer in all_chunks.json
    must not claim SUCCESS forever."""
    all_chunks = tmp_path / "all_chunks.json"
    _all_chunks_file(all_chunks)  # empty -- no documents
    registry = _registry(tmp_path, "d1")
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["documents"][0]["chunk"] = {
        "status": "SUCCESS", "chunked_at": "2026-07-16T10:00:00+03:00",
        "chunked_from_hash": "sha256:eski", "chunks_path": "eski/all_chunks.json",
        "chunker_version": "0.9.0"}
    registry.write_text(json.dumps(data), encoding="utf-8")

    assert rcp.update_registry(registry, all_chunks) == (0, 1)
    assert _blok(registry, "d1") == {
        "status": "PENDING", "chunked_at": None, "chunked_from_hash": None,
        "chunks_path": None, "chunker_version": None}


def test_provenance_siz_belge_tahmin_etmez(tmp_path):
    """A pre-v4 document entry: write what is known, leave the rest None."""
    all_chunks = tmp_path / "all_chunks.json"
    _all_chunks_file(all_chunks, d1={})  # no provenance
    registry = _registry(tmp_path, "d1")

    assert rcp.update_registry(registry, all_chunks) == (1, 0)
    blok = _blok(registry, "d1")
    assert blok["status"] == "SUCCESS"
    assert blok["chunked_at"] is None and blok["chunker_version"] is None
    assert blok["chunked_from_hash"] is None


def test_dosya_yok_hepsi_pending(tmp_path):
    """If all_chunks.json hasn't been produced yet (run not started/crashed)
    every record stays PENDING -- never silently assumed SUCCESS."""
    all_chunks = tmp_path / "all_chunks.json"  # not created
    registry = _registry(tmp_path, "d1", "d2")

    assert rcp.update_registry(registry, all_chunks) == (0, 2)
    assert _blok(registry, "d1")["status"] == "PENDING"
    assert _blok(registry, "d2")["status"] == "PENDING"


def test_bozuk_all_chunks_dosyasi_hepsi_pending(tmp_path):
    """A broken/unreadable all_chunks.json -- cannot tell what's inside,
    so everything goes to PENDING (same treatment as file-missing)."""
    all_chunks = tmp_path / "all_chunks.json"
    all_chunks.write_text("{bozuk", encoding="utf-8")
    registry = _registry(tmp_path, "d1")

    assert rcp.update_registry(registry, all_chunks) == (0, 1)
    assert _blok(registry, "d1")["status"] == "PENDING"


def test_atomik_yazim_diger_alanlari_korur(tmp_path):
    all_chunks = tmp_path / "all_chunks.json"
    _all_chunks_file(all_chunks, d1={
        "chunker_version": "1.0.0", "generated_at": "2026-07-17T14:03:22+03:00"})
    registry = _registry(tmp_path, "d1")
    data = json.loads(registry.read_text(encoding="utf-8"))
    data["documents"][0]["parse"] = {"status": "SUCCESS"}
    data["summary"] = {"scanned_files": 1}
    registry.write_text(json.dumps(data), encoding="utf-8")

    rcp.update_registry(registry, all_chunks)
    sonra = json.loads(registry.read_text(encoding="utf-8"))
    assert sonra["summary"] == {"scanned_files": 1}
    assert sonra["documents"][0]["parse"] == {"status": "SUCCESS"}


# --- N-21: resolve_all_chunks_path (run_nightly.py's before/after diff
# reads the SAME path) ---------------------------------------------


def test_resolve_all_chunks_path_uses_chunks_output_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHUNKS_OUTPUT_DIR", str(tmp_path / "cikti"))
    monkeypatch.delenv("CHUNKER_DIR", raising=False)

    assert rcp.resolve_all_chunks_path() == tmp_path / "cikti" / "all_chunks.json"


def test_resolve_all_chunks_path_falls_back_to_chunker_dir_storage(monkeypatch, tmp_path):
    monkeypatch.delenv("CHUNKS_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("CHUNKER_DIR", str(tmp_path / "chunker"))

    assert rcp.resolve_all_chunks_path() == tmp_path / "chunker" / "storage" / "all_chunks.json"


def test_resolve_all_chunks_path_matches_what_main_actually_writes_to(monkeypatch, tmp_path):
    """N-21 critical check: the location `main()` ACTUALLY writes to must be
    the SAME as what `resolve_all_chunks_path()` returns -- otherwise
    `run_nightly.py::stage_chunk`'s before/after diff reads the wrong file
    and always sees 0."""
    input_dir = tmp_path / "girdi"
    input_dir.mkdir()
    monkeypatch.setenv("CHUNKER_INPUT_DIR", str(input_dir))
    monkeypatch.setenv("CHUNKS_OUTPUT_DIR", str(tmp_path / "cikti"))
    monkeypatch.delenv("DOCUMENT_NODES_PATH", raising=False)
    beklenen = rcp.resolve_all_chunks_path()

    class _SahteProc:
        returncode = 0

    monkeypatch.setattr(rcp.subprocess, "run", lambda *a, **k: _SahteProc())
    monkeypatch.setattr(rcp.importlib, "import_module", lambda name: None)

    rcp.main()

    assert beklenen.parent.is_dir(), "main() must CREATE the out_root -- same root as resolve_all_chunks_path"