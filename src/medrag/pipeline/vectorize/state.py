"""Yerel bayatlık durumu: `scope_id -> imza` (bkz. `core.ChunkSet.signature`).

Qdrant'a round-trip atıp "bu kapsam zaten var mı, aynı mı" sormak yerine
(ağ gecikmesi + Qdrant'ın kapsam-genelinde bir imza payload'ı olması
gerekirdi) yerel tek bir JSON dosyası tutulur — `chunker`ın kendi
`ChunkProvenance` + registry ikilisinin küçük ölçekli benzeri. Dosya kaybolursa
en kötü ihtimalle bir sonraki koşu her şeyi gereksiz yere yeniden embed eder
(pahalı ama YANLIŞ değil) — sessiz veri kaybı riski yok.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_state(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict[str, dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        os.unlink(tmp_path)
        raise


def is_unchanged(state: dict[str, dict[str, Any]], scope_id: str,
                 signature: dict[str, Any]) -> bool:
    return state.get(scope_id) == signature
