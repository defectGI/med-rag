"""Per-document status files (A5): `/durum/<doc_id>.json`.

worker writes, api reads (streams to the browser via SSE). Per-file atomic
write; the reading side never sees a partial write.

State machine:
    queued → parsing → chunking → vectorizing → ready
                          └──────────────────────► error
For a delete job: queued → error (if delete fails) | the file disappears entirely.
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

#: Valid states. The api side turns these into a badge.
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
    """Writes/overwrites `{doc_id}.json` (idempotent). `extra`: fixed fields the
    api uses in the list view, like file name, rel_path."""
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
