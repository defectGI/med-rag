"""Yaşam döngüsü dizinlerinin env'den çözümü.

api tarafı (medrag.api.library) AYNI env adlarını AYNI öncelikle okur --
iki taraf arasında paylaşılan sözleşme buradaki adlardır:

    BELGELER_DIR         korpus kökü (upload'lar buraya, notlar `<kök>/notlar/`)
    DOCUMENT_NODES_PATH  registry (document_nodes.json)
    PARSED_OUTPUT_DIR    parse çıktı kökü
    ISLER_DIR            iş kuyruğu (job dosyaları); varsayılan `<korpus>/isler`
    DURUM_DIR            durum dosyaları; varsayılan `<korpus>/durum`

Varsayılanlar yalnız ISLER_DIR/DURUM_DIR için vardır (korpus kökünün
kardeşi); BELGELER_DIR/DOCUMENT_NODES_PATH/PARSED_OUTPUT_DIR zorunludur --
eksikleri sessiz varsayılanla örtmemek, cli/ betiklerinin `_resolve`
disipliniyle aynı ilkedir.
"""

from __future__ import annotations

import os
from pathlib import Path


class LifecycleConfigError(RuntimeError):
    """Zorunlu bir env tanımsız -- yüksek sesle düş (sessiz varsayilan YOK)."""


def _required(env: str) -> Path:
    raw = (os.environ.get(env) or "").strip()
    if not raw:
        raise LifecycleConfigError(
            f"{env} tanımsız -- yaşam döngüsü dizinleri çözülemiyor (bkz. .env.example)")
    return Path(raw).expanduser().resolve()


def belgeler_dir() -> Path:
    return _required("BELGELER_DIR")


def registry_path() -> Path:
    return _required("DOCUMENT_NODES_PATH")


def parsed_output_dir() -> Path:
    return _required("PARSED_OUTPUT_DIR")


def notlar_dir() -> Path:
    """Notların (doc_type=NOTE, .md) yaşadığı klasör: `<BELGELER_DIR>/notlar`."""
    return belgeler_dir() / "notlar"


def isler_dir() -> Path:
    raw = (os.environ.get("ISLER_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return belgeler_dir().parent / "isler"


def durum_dir() -> Path:
    raw = (os.environ.get("DURUM_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return belgeler_dir().parent / "durum"


__all__ = [
    "LifecycleConfigError",
    "belgeler_dir",
    "durum_dir",
    "isler_dir",
    "notlar_dir",
    "parsed_output_dir",
    "registry_path",
]
