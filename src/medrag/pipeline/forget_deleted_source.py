"""N-06 (I-08): silinen bir kaynagin TUM turevlerini sistemden cikaran ortak
"kaynagi unut" ust-fonksiyonu.

Tarayici (`chatbot-corpus/document_info/classify_documents.py`) bir dosya
diskten kalktiginda kaydi SILMEZ, `scan_status=DELETED` ile sticky birakir
(doc_id KORUNUR -- schema karari). Ama hicbir asama o dokumanin daha once
urettigi turevleri temizlemiyordu; bu modul o eksigi kapatir.

Talimat (N-06 + O-11 docstring'i): I-08 (chunk/vektor tarafi) ile I-17
(spec kanit tarafi) AYNI "kaynagi unut" kavraminin iki ucudur, iki ayri
silme mantigi YAZILMAZ. Bu yuzden burasi spec tarafi icin
`medrag.pipeline.facts.forget_source.forget_source`i (O-11, K-70/K-78) DOGRUDAN
cagirir -- kendi DELETE/UPDATE mantigini tekrar etmez; yalniz O-11'in
kapsamadigi UC parcayi (parse ciktisi, chunk kaydi, Qdrant noktalari) ekler:

  1. parse cikti klasoru -- `PARSED_OUTPUT_DIR/<doc_type>_<doc_id>/` (bkz.
     `pipeline/cli/run_parse_pipeline.py::_group_dir`, AYNI isimlendirme).
     `doc_type` bilinmiyorsa (ya da tarama sirasinda degismisse) klasor
     `_<doc_id>` sonekiyle GLOB edilir -- isimlendirme kaymasina karsi.
  2. `all_chunks.json`daki `documents[doc_id]` kaydi -- dosya `run_chunk_
     pipeline.py`nin kullandigi ayni atomik yaz-sonra-degistir deseniyle
     geri yazilir (yarim yazimda dosya bozulmaz).
  3. Qdrant noktalari -- `vector_store.replace_scope(doc_id, [])` (mekanizma
     zaten var, KARAR-004: bos liste = yalniz sil, hic upsert etme).

Idempotentlik: her adim kendi yoklugunu sessizce KABUL eder -- klasor zaten
yoksa `removed` listesine girmez, `all_chunks.json`da kayit yoksa `False`
doner, Qdrant `replace_scope` bir filtreyle siler (eslesen nokta yoksa no-op),
`forget_source` eslesen `evidence[]` yoksa hicbir satira dokunmaz. Yani bu
fonksiyon YARIDA KESILIP TEKRAR COSTURULABILIR -- ikinci cagri ilkiyle AYNI
sonucu (bos degisiklik) uretir, hata FIRLATMAZ.

Katman kurali: bu modul `medrag.core`e baglanabilir, kendi paketi icindeki
`medrag.pipeline.facts`i import eder (pipeline-ici, ihlal degil) ama
`medrag.api`yi HIC import etmez.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from medrag.pipeline.facts.forget_source import forget_source


class ScopeVectorStore(Protocol):
    """`QdrantVectorStore.replace_scope` ile ayni imza -- gercek Qdrant
    baglantisi gerektirmeden testte sahte bir kayitci gecirilebilsin diye."""

    def replace_scope(self, scope_id: str, points: list[Any]) -> None: ...


@dataclass
class ForgetDeletedSourceResult:
    """Her adimin ne yaptigini raporlar -- N-09'un "silinen kaynagin
    turevleri" satiri ve testler bunu okur."""

    doc_id: str
    parse_dirs_removed: list[str] = field(default_factory=list)
    chunk_entry_removed: bool = False
    qdrant_scope_cleared: bool = False
    spec_evidence: dict[str, Any] = field(default_factory=dict)


def _remove_parse_output(parsed_output_dir: Path, doc_id: str,
                          doc_type: str | None) -> list[str]:
    """`PARSED_OUTPUT_DIR/<doc_type>_<doc_id>/` klasorunu (varsa) siler.
    `doc_type` verilmemisse ya da kayitli deger diskteki eski klasorle
    uyusmuyorsa `_<doc_id>` sonekli TUM klasorler bulunup silinir (birden
    fazla eslesme cikmasi normalde beklenmez, ama silme YARIM kalmasin)."""
    if not parsed_output_dir.is_dir():
        return []

    hedefler: list[Path] = []
    if doc_type:
        aday = parsed_output_dir / f"{doc_type}_{doc_id}"
        if aday.is_dir():
            hedefler.append(aday)

    sonek = f"_{doc_id}"
    for child in parsed_output_dir.iterdir():
        if child.is_dir() and child.name.endswith(sonek) and child not in hedefler:
            hedefler.append(child)

    kaldirilan: list[str] = []
    for hedef in hedefler:
        shutil.rmtree(hedef)
        kaldirilan.append(str(hedef))
    return kaldirilan


def _atomic_write_json(path: Path, data: dict) -> None:
    """Yaz-sonra-degistir -- `run_chunk_pipeline.py::_atomic_write_json` ile
    AYNI desen (yarim yazimda dosya bozulmaz)."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _remove_from_all_chunks(all_chunks_path: Path, doc_id: str) -> bool:
    """`all_chunks.json`in `documents[doc_id]` kaydini duser. Dosya yoksa,
    bozuksa ya da kayit zaten yoksa (idempotent ikinci cagri) sessizce
    `False` doner -- HATA FIRLATMAZ (chunk_store.py'nin hata politikasiyla
    ayni ilke: turev temizligi bir EK, korpus dosyasi olmadan da calismali)."""
    if not all_chunks_path.is_file():
        return False
    try:
        data = json.loads(all_chunks_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    documents = data.get("documents")
    if not isinstance(documents, dict) or doc_id not in documents:
        return False

    del documents[doc_id]
    _atomic_write_json(all_chunks_path, data)
    return True


def forget_deleted_source(
    *,
    doc_id: str,
    doc_type: str | None,
    parsed_output_dir: Path | str,
    all_chunks_path: Path | str,
    vector_store: ScopeVectorStore,
    specs_con: sqlite3.Connection,
    commit: bool = True,
) -> ForgetDeletedSourceResult:
    """Bir `doc_id`nin TUM turevlerini siler: parse ciktisi + chunk kaydi +
    Qdrant noktalari + spec kanitlari (I-08 kabulunun tam kapsami).

    Cagiran taraf (gece kosusunun N.2 silme akisi) her `scan_status=DELETED`
    kaydi icin bunu bir kez cagirir; idempotent oldugu icin yarida kesilip
    tekrar calistirilabilir (bkz. modul docstring'i)."""
    parse_dirs_removed = _remove_parse_output(Path(parsed_output_dir), doc_id, doc_type)
    chunk_entry_removed = _remove_from_all_chunks(Path(all_chunks_path), doc_id)
    # KARAR-004: bos liste = kapsamin TUM eski noktalarini sil, hic upsert etme.
    # Eslesen nokta yoksa bu bir no-op'tur (idempotent).
    vector_store.replace_scope(doc_id, [])
    spec_evidence = forget_source(specs_con, doc_ids=[doc_id], commit=commit)

    return ForgetDeletedSourceResult(
        doc_id=doc_id,
        parse_dirs_removed=parse_dirs_removed,
        chunk_entry_removed=chunk_entry_removed,
        qdrant_scope_cleared=True,
        spec_evidence=spec_evidence,
    )


__all__ = ["ForgetDeletedSourceResult", "ScopeVectorStore", "forget_deleted_source"]
