"""document_nodes.json read/write -- the lifecycle side.

RECORD SHAPE: a MIRROR of the shape defined by
`chatbot-corpus/document_info/classify_documents.py` (identity/location/scan/
doc_type/is_active/links + parse/chunk/facts blocks). Components do not import
each other; the shape is a file contract (this repo's general principle).

WRITERS: api (creates an upload record, removes it on delete) and worker
(writes the parse/chunk blocks). Both take `fslock.file_lock` before writing --
prevents two writers from reading at the same time and deleting each other's
record. The out-of-band scan flow (classify_documents) does not know the lock:
concurrent runs with it are not expected (nightly reconciliation, optional).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from medrag.pipeline.lifecycle.fslock import file_lock
from medrag.pipeline.lifecycle.paths import registry_path

logger = logging.getLogger("medrag.pipeline.lifecycle.registry")


class RegistryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def blank_pipeline_state() -> dict:
    """SAME shape as `classify_documents.blank_pipeline_state` --
    parse/chunk/facts blocks PENDING."""
    return {
        "parse": {
            "parser": None, "parser_version": None, "status": "PENDING",
            "parsed_from_hash": None, "parsed_json_path": None,
            "last_parsed": None, "error": None, "stages": None,
        },
        "chunk": {
            "status": "PENDING", "chunked_at": None,
            "chunked_from_hash": None, "chunks_path": None,
            "chunker_version": None,
        },
        "facts": {
            "status": "PENDING", "extracted_at": None,
            "extracted_from_hash": None, "extractor_version": None,
            "prompt_version": None,
        },
    }


def load(registry: Path | None = None) -> dict:
    path = registry or registry_path()
    if not path.is_file():
        raise RegistryError(f"registry yok: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"registry okunamadı: {path}: {exc}") from exc


def save(data: dict, registry: Path | None = None) -> None:
    path = registry or registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        _atomic_write_json(path, data)


def find(data: dict, doc_id: str) -> dict | None:
    for record in data.get("documents", []):
        if record.get("identity", {}).get("doc_id") == doc_id:
            return record
    return None


def content_hash_of(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(blok)
    return "sha256:" + digest.hexdigest()


def _belge_record(
    *,
    file_path: Path,
    rel_path: str,
    doc_type: str,
    note_title: str | None = None,
) -> dict:
    stat = file_path.stat()
    record: dict[str, Any] = {
        "identity": {
            "doc_id": str(uuid.uuid4()),
            "file_name": file_path.name,
            "extension": file_path.suffix.lower(),
        },
        "location": {
            "rel_path": rel_path,
            "file_path": str(file_path),
        },
        "scan": {
            "content_hash": content_hash_of(file_path),
            "size_bytes": stat.st_size,
            "last_modified_time": datetime.fromtimestamp(
                stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "scan_status": "NEW",
        },
        "doc_type": doc_type,
        "is_active": True,
        "links": {
            "owner_ids": [],
            "link_count": 0,
            "matched_node_id": None,
            "resolution_method": "upload",
        },
        **blank_pipeline_state(),
    }
    if note_title is not None:
        record["note"] = {"title": note_title}
    return record


def add_upload(
    *,
    file_path: Path,
    belgeler_dir: Path,
    doc_type: str = "BELGE",
    note_title: str | None = None,
    registry: Path | None = None,
) -> dict:
    """Creates a registry record for the uploaded file (NEW). If the same
    rel_path is already recorded it is an UPDATE record (change semantics, A4):
    doc_id is preserved, the scan block is overwritten with the new hash, the
    pipeline blocks do NOT revert to PENDING -- the `parsed_from_hash !=
    content_hash` gate already triggers reprocessing."""
    path = registry or registry_path()
    rel_path = file_path.relative_to(belgeler_dir).as_posix()
    with file_lock(path):
        data = load(path)
        eski = None
        for record in data.get("documents", []):
            if record.get("location", {}).get("rel_path") == rel_path:
                eski = record
                break
        if eski is None:
            record = _belge_record(
                file_path=file_path, rel_path=rel_path, doc_type=doc_type,
                note_title=note_title)
            data.setdefault("documents", []).append(record)
            durum_kaydi = "NEW"
        else:
            record = eski
            record["scan"] = {
                **_belge_record(file_path=file_path, rel_path=rel_path,
                                doc_type=doc_type)["scan"],
                # if the same file comes back into a sticky DELETED record, it returns to active.
                "scan_status": "MODIFIED",
            }
            record["is_active"] = True
            record["location"]["file_path"] = str(file_path)
            if note_title is not None and "note" in record:
                record["note"]["title"] = note_title
            durum_kaydi = "MODIFIED"
        _atomic_write_json(path, data)
    logger.info("registry kaydı: %s (%s) -> %s", rel_path,
                record["identity"]["doc_id"], durum_kaydi)
    return record


def remove(doc_id: str, registry: Path | None = None) -> dict | None:
    """Drops the record ENTIRELY (no sticky DELETED is left: in med-rag the
    delete comes from the UI, so the user's intent is explicit -- no trace is
    left). Returns the removed record, or None if absent."""
    path = registry or registry_path()
    with file_lock(path):
        data = load(path)
        records = data.get("documents", [])
        for i, record in enumerate(records):
            if record.get("identity", {}).get("doc_id") == doc_id:
                kaldirilan = records.pop(i)
                _atomic_write_json(path, data)
                logger.info("registry kaydı silindi: %s (%s)",
                            kaldirilan.get("location", {}).get("rel_path"), doc_id)
                return kaldirilan
    return None


def update_record(doc_id: str, mutate, registry: Path | None = None) -> dict | None:
    """Updates the record under lock; `mutate(record)` mutates it in place.
    Returns None if the record is absent; if mutate raises, nothing is written."""
    path = registry or registry_path()
    with file_lock(path):
        data = load(path)
        record = find(data, doc_id)
        if record is None:
            return None
        mutate(record)
        _atomic_write_json(path, data)
    return record


__all__ = [
    "RegistryError",
    "add_upload",
    "blank_pipeline_state",
    "content_hash_of",
    "find",
    "load",
    "remove",
    "save",
    "update_record",
]
