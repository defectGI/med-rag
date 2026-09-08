"""document_nodes.json okuma/yazma -- yaşam döngüsü tarafı.

KAYIT ŞEMASI: `chatbot-corpus/document_info/classify_documents.py`
tarafından tanımlanan şeklin AYNASI (identity/location/scan/doc_type/
is_active/links + parse/chunk/facts blokları). Bileşenler birbirini
import etmez; şekil dosya sözleşmesidir (bu deponun genel ilkesi).

YAZARLAR: api (upload kaydı oluşturur, silmede kaldırır) ve worker
(parse/chunk bloklarını yazar). İkisi de yazmadan `fslock.file_lock`
alır -- iki yazarın aynı an okuyup birbirinin kaydını silmesini
engeller. Sahne dışı kalan tarama akışı (classify_documents) kilidi
bilmez: onunla eşzamanlı koşum beklenmez (gecelik uzlaştırma, opsiyonel).
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
    """`classify_documents.blank_pipeline_state` ile AYNI şekil --
    parse/chunk/facts blokları PENDING."""
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
    """Yüklenen dosya için registry kaydı oluşturur (NEW). Aynı rel_path
    zaten kayıtlıysa GÜNCELLEME kaydıdır (değişiklik semantiği, A4):
    doc_id korunur, scan bloğu yeni hash ile ezilir, pipeline blokları
    PENDING'e dönmez -- `parsed_from_hash != content_hash` kapısı yeniden
    işlemeyi zaten tetikler."""
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
                # DELETED yapışkan kaydına yeniden aynı dosya geldiyse canlıya döner.
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
    """Kaydı TAMAMEN düşürür (sticky DELETED bırakılmaz: med-rag'de silme
    UI'dan geldiği için kullanıcının niyeti açık -- iz kalmaz). Silinen
    kaydı döndürür, yoksa None."""
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
    """Kaydı kilit altında günceller; `mutate(record)` yerinde değiştirir.
    Kayıt yoksa None döner, mutate istisnası yükseltilirse yazım OLMAZ."""
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
