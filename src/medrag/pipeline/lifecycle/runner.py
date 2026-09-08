"""Tek-dosya pipeline koşucu (A2/A3/A4): process + delete.

Bir `doc_id` için üç aşamayı sıralar -- mevcut aşama kodları yeniden
YAZILMAZ, doğrudan çağrılır (worker pipeline paketinin İÇİNDE olduğu
için bu import'lar katman ihlali değildir):

  parse   : run_parse_pipeline.phase0/1/2_one(record) -- tek kayıt,
            registry'deki `parse` bloğunu doldurur
  chunk   : chunker.cli.calistir(env) -- tüm korpusu yeniden üretir
            (LLM'siz + ucuz, KARAR-012); yalnız bu dokümanın `chunk`
            bloğu kilit altında registry'ye yazılır
  vectorize: vectorize.cli.calistir(env) -- bayatlık kapısı sayesinde
            yalnız imzası değişen kapsam (bu doküman) gömülür

Silme: forget_deleted_source (N-06) -- parse klasörü + all_chunks kaydı
+ Qdrant noktaları + (varsa) spec kanıtları; ardından registry kaydı
kaldırılır.

Değişiklik semantiği (A4): process_document HER ZAMAN önce türevleri
unutturur (idempotent) sonra yeniden işler -- "eski hali silinmiş + yeni
hali eklenmiş" tek kod yoludur; ayrı bir "reprocess" yolu açılmaz.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

from medrag.pipeline.lifecycle import durum as durum_store
from medrag.pipeline.lifecycle import registry_rw
from medrag.pipeline.lifecycle.paths import (
    durum_dir,
    parsed_output_dir,
    registry_path,
)

logger = logging.getLogger("medrag.pipeline.lifecycle.runner")

#: durum yazım köprüsü -- worker on_status("parsing", "okuma") çağırır.
StatusFn = Callable[..., None]


def _bootstrap_env() -> None:
    """cli/.env + parser/.env yüklenir (PARSER_DIR/BELGELER_DIR/... +
    LLM/VLM ayarları). `run_parse_pipeline._bootstrap` ile aynı disiplin;
    worker girişi bunu bir kez çağırır."""
    from medrag.pipeline.cli import run_parse_pipeline as rpp

    rpp._bootstrap()


def _chunk_output_root() -> Path:
    """`run_chunk_pipeline.resolve_all_chunks_path` ile AYNI çözümleme
    (CHUNKS_OUTPUT_DIR > CHUNKER_DIR/storage) -- kod tekrarı yerine
    o modülden alınır."""
    from medrag.pipeline.cli import run_chunk_pipeline as rcp

    return rcp.resolve_all_chunks_path().parent


def _vector_store():
    """Gerçek Qdrant istemcisi -- vectorize ile AYNI yapılandırma
    (aynı koleksiyon/boyut politikası), config'i vectorize'in kendisi
    çözer."""
    from medrag.pipeline.vectorize.config import load_config
    from medrag.pipeline.vectorize.store import QdrantVectorStore

    cfg = load_config(os.environ.get("VECTORIZE_CONFIG") or None)
    url = (os.environ.get("QDRANT_URL") or "").strip()
    if not url:
        raise RuntimeError(
            "QDRANT_URL tanımsız -- silme işleminde vektör noktaları "
            "temizlenemez; worker QDRANT_URL olmadan başlamamalı")
    return QdrantVectorStore.from_env(
        collection_name=cfg.qdrant.collection_name,
        distance=cfg.qdrant.distance,
        on_dim_mismatch=cfg.qdrant.on_dim_mismatch,
        upsert_batch_size=cfg.qdrant.upsert_batch_size,
        url=url,
        api_key=(os.environ.get("QDRANT_API_KEY") or "").strip() or None,
    )


def _specs_connection() -> sqlite3.Connection:
    """specs.db VARSA gerçek bağlantı, YOKSA minimal boş şemalı bellek-içi.

    med-rag'de facts kapalı olduğundan çoğu kurulumda specs.db ya yoktur
    ya da kanıt taşımaz; `forget_source` bellek-içi boş `spec_value`
    tablosunda no-op'tur. Tablosuz gerçek bir DB'ye denk gelirse
    OperationalError yükselir -- bu durumda da bellek-içi yola düşülür
    (silme, facts tarafı olmadan da tamamlanır)."""
    from medrag.core.paths import resolve_specs_db_path

    try:
        path = resolve_specs_db_path()
    except Exception:  # noqa: BLE001 -- çözümlenemiyorsa bellek-içi
        path = None
    con = None
    if path and Path(path).is_file():
        try:
            con = sqlite3.connect(str(path))
            con.execute("SELECT 1 FROM spec_value LIMIT 1")
            return con
        except sqlite3.OperationalError:
            if con is not None:
                con.close()
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE IF NOT EXISTS spec_value "
                "(value_id TEXT PRIMARY KEY, evidence TEXT NOT NULL DEFAULT '[]')")
    return con


def forget_derivatives(record: dict) -> dict:
    """Bir kaydın TÜM türevlerini siler (idempotent) -- A3/A4 ortak yolu."""
    from medrag.pipeline.forget_deleted_source import forget_deleted_source

    doc_id = record["identity"]["doc_id"]
    store = _vector_store()
    con = _specs_connection()
    try:
        result = forget_deleted_source(
            doc_id=doc_id,
            doc_type=record.get("doc_type"),
            parsed_output_dir=parsed_output_dir(),
            all_chunks_path=_chunk_output_root() / "all_chunks.json",
            vector_store=store,
            specs_con=con,
        )
    finally:
        con.close()
    logger.info("türevler unutturuldu: %s (%s)", doc_id, result)
    return {
        "parse_dirs_removed": result.parse_dirs_removed,
        "chunk_entry_removed": result.chunk_entry_removed,
        "qdrant_scope_cleared": result.qdrant_scope_cleared,
    }


def process_document(doc_id: str, *, status: StatusFn) -> str:
    """Tek dokümanı uçtan uca işler; son durumu ("ready"|"error") döner."""
    from medrag.pipeline.cli import run_parse_pipeline as rpp

    record = registry_rw.find(registry_rw.load(registry_path()), doc_id)
    if record is None:
        durum_store.write_status(durum_dir(), doc_id, "error",
                                 detail="registry kaydı bulunamadı")
        return "error"

    # 1) Eski türevler (değişiklik/silme-sonrası-yeniden ekleme semantiği).
    status("parsing", "eski türevler temizleniyor")
    forget_derivatives(record)

    # 2) parse -- tek kayıt üzerinde mevcut faz fonksiyonları.
    rel_path = record["location"]["rel_path"]
    src = Path(os.environ["BELGELER_DIR"]) / rel_path
    if not src.is_file():
        durum_store.write_status(durum_dir(), doc_id, "error",
                                 detail=f"kaynak dosya yok: {rel_path}")
        return "error"
    status("parsing", "görsel bölge sınıflandırma")
    rpp.phase0_classify_one(record)
    status("parsing", "sayfa okuma + OCR")
    rpp.phase1_one(record)
    status("parsing", "tablo açıklama + son kayıt")
    rpp.phase2_one(record)
    registry_rw.update_record(doc_id, lambda r: r.update(
        {"parse": record["parse"]}))

    parse_status = record["parse"].get("status")
    if parse_status == "SKIPPED":
        detail = "bu format için ayrıştırıcı yok (SKIPPED)"
        durum_store.write_status(durum_dir(), doc_id, "error", detail=detail)
        return "error"
    if parse_status == "FAILED":
        detail = f"parse başarısız: {record['parse'].get('error')}"
        durum_store.write_status(durum_dir(), doc_id, "error", detail=detail)
        return "error"

    # 3) chunk -- tüm korpus yeniden üretilir (ucuz, LLM'siz); bu dokümanın
    #    bloğu provenance'tan okunup kilit altında yazılır.
    status("chunking", "chunk'lar yeniden üretiliyor")
    env = os.environ
    env["CHUNKER_INPUT_DIR"] = str(parsed_output_dir())
    # İki env, iki okuyucu: chunker'ın OutputLayout'u CHUNKER_OUTPUT_DIR'i,
    # `resolve_all_chunks_path` (registry bloğu için) CHUNKS_OUTPUT_DIR'i
    # okur -- aynı kökü göstermeli, yoksa blok yanlış dosyadan okunur.
    env["CHUNKER_OUTPUT_DIR"] = str(_chunk_output_root())
    env["CHUNKS_OUTPUT_DIR"] = str(_chunk_output_root())
    env.pop("PARSER_DIR", None)  # chunker'ın kendi PARSER_DIR'i karışmasın (bkz. run_chunk_pipeline)
    from medrag.pipeline.chunker import cli as chunker_cli

    rc = chunker_cli.calistir(env=env)
    if rc not in (0,):
        logger.warning("chunker çıkışı %s (%s) -- yine de devam", rc, doc_id)
    from medrag.pipeline.cli import run_chunk_pipeline as rcp

    all_chunks = rcp.resolve_all_chunks_path()
    doc_data = rcp._all_documents(all_chunks).get(doc_id)
    block = rcp._chunk_block(doc_data, all_chunks) or {
        "status": "PENDING", "chunked_at": None, "chunked_from_hash": None,
        "chunks_path": None, "chunker_version": None,
    }
    registry_rw.update_record(doc_id, lambda r: r.update({"chunk": block}))

    # 4) vectorize -- bayatlık kapısı yalnız değişen kapsamı gömer.
    status("vectorizing", "embedding")
    from medrag.pipeline.vectorize import cli as vec_cli

    stats = vec_cli.calistir(env=env)
    logger.info("vectorize: %s nokta (%s)", stats.points_written, doc_id)

    durum_store.write_status(
        durum_dir(), doc_id, "ready",
        detail=(f"{stats.points_written} parça gömüldü"
                if stats.points_written else "gömülecek yeni parça yok"),
        parse_status=parse_status,
    )
    return "ready"


def delete_document(doc_id: str, *, status: StatusFn) -> str:
    """Dokümanı ve TÜM türevlerini siler; "deleted" döner (ya da fırlatır)."""
    status("parsing", "siliniyor")
    data = registry_rw.load(registry_path())
    record = registry_rw.find(data, doc_id)
    if record is None:
        # registry kaydı yok ama durum dosyası asılı kalmış olabilir.
        durum_store.remove_status(durum_dir(), doc_id)
        return "deleted"
    forget_derivatives(record)
    registry_rw.remove(doc_id, registry_path())
    durum_store.remove_status(durum_dir(), doc_id)
    return "deleted"


__all__ = ["delete_document", "forget_derivatives", "process_document"]
