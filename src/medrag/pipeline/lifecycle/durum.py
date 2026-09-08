"""Belge başına durum dosyaları (A5): `/durum/<doc_id>.json`.

worker yazar, api okur (SSE ile tarayıcıya akıtır). Dosya başına atomik
yazım; okuma tarafı yarım yazım görmez.

Durum makinesi:
    queued → parsing → chunking → vectorizing → ready
                          └──────────────────────► error
Silme işi için: queued → error (silme başarısızsa) | dosya tamamen kalkar.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("medrag.pipeline.lifecycle.durum")

#: Geçerli durumlar. api tarafı bunları rozete çevirir.
STATES = ("queued", "parsing", "chunking", "vectorizing", "ready", "error")


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_status(
    durum_dir: Path,
    doc_id: str,
    state: str,
    *,
    detail: str | None = None,
    **extra: Any,
) -> dict:
    """`{doc_id}.json` yaz/üzerine yaz (idempotent). `extra`: dosya adı,
    rel_path gibi api'nin liste görünümünde kullandığı sabit alanlar."""
    if state not in STATES:
        raise ValueError(f"bilinmeyen durum: {state!r} (geçerli: {STATES})")
    durum_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "doc_id": doc_id,
        "state": state,
        "detail": detail,
        "updated_at": _now(),
    }
    data.update(extra)
    _atomic_write_json(durum_dir / f"{doc_id}.json", data)
    return data


def read_status(durum_dir: Path, doc_id: str) -> dict | None:
    path = durum_dir / f"{doc_id}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_all(durum_dir: Path) -> dict[str, dict]:
    if not durum_dir.is_dir():
        return {}
    out: dict[str, dict] = {}
    for path in durum_dir.glob("*.json"):
        if path.name.startswith(".tmp_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        doc_id = data.get("doc_id")
        if doc_id:
            out[str(doc_id)] = data
    return out


def remove_status(durum_dir: Path, doc_id: str) -> bool:
    return (durum_dir / f"{doc_id}.json").unlink(missing_ok=True) or True


__all__ = ["STATES", "read_all", "read_status", "remove_status", "write_status"]
