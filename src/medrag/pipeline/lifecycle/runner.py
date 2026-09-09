"""Single-document pipeline runner (A2/A3/A4): process + delete.

For one `doc_id` it runs the three stages in order -- the existing stage code
is NOT rewritten, it is called directly (the worker is INSIDE the pipeline
package, so these imports are not a layer violation):

  parse   : run_parse_pipeline.phase0/1/2_one(record) -- a single record,
            fills the `parse` block in the registry
  chunk   : chunker.cli.calistir(env) -- re-produces the whole corpus
            (LLM-free + cheap, KARAR-012); only this document's `chunk`
            block is written to the registry under lock
  vectorize: vectorize.cli.calistir(env) -- thanks to the staleness gate only
            the scope whose signature changed (this document) is embedded

Delete: forget_deleted_source (N-06) -- parse folder + all_chunks record
+ Qdrant points + (if any) spec evidence; then the registry record is removed.

Change semantics (A4): process_document ALWAYS forgets the derivatives first
(idempotent) then reprocesses -- "old version removed + new version added" is
the single code path; a separate "reprocess" path is not opened.
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

#: status-write bridge -- worker calls on_status("parsing", "okuma").
StatusFn = Callable[..., None]


def _bootstrap_env() -> None:
    """Loads cli/.env + parser/.env (PARSER_DIR/BELGELER_DIR/... +
    LLM/VLM settings). Same discipline as `run_parse_pipeline._bootstrap`;
    the worker entry point calls this once."""
    from medrag.pipeline.cli import run_parse_pipeline as rpp

    rpp._bootstrap()


def _chunk_output_root() -> Path:
    """SAME resolution as `run_chunk_pipeline.resolve_all_chunks_path`
    (CHUNKS_OUTPUT_DIR > CHUNKER_DIR/storage) -- taken from that module rather
    than duplicating the code."""
    from medrag.pipeline.cli import run_chunk_pipeline as rcp

    return rcp.resolve_all_chunks_path().parent


def _reset_staleness(doc_id: str) -> None:
    """Clears vectorize's local staleness signature for this document.

    `process_document`, before processing, DELETES the Qdrant vectors with
    `forget_derivatives` but vectorize's `state.json` signature remains. Then
    `vectorize.cli.calistir` says "this scope is unchanged" and skips the
    embedding -> the document stays "ready" with 0 vectors and is NEVER found
    in retrieval (a note or a re-loaded file). Deleting the signature forces
    calistir to re-embed."""

    from medrag.pipeline.vectorize.layout import OutputLayout
    from medrag.pipeline.vectorize.state import load_state, save_state

    root = (os.environ.get("VECTORIZE_OUTPUT_DIR") or "").strip() or "./storage"
    out = OutputLayout(root)
    state = load_state(out.state_file)
    if doc_id not in state:
        return
    state.pop(doc_id, None)
    save_state(state, out.state_file)


def _vector_store():
    """The real Qdrant client -- SAME configuration as vectorize (same
    collection/size policy); the config is resolved by vectorize itself."""
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
    """REAL connection if specs.db exists, else a minimal empty-schema in-memory one.

    Since facts is off in med-rag, in most setups specs.db either does not exist
    or carries no evidence; `forget_source` no-ops in the in-memory empty
    `spec_value` table. If it encounters a real table-less DB it raises
    OperationalError -- in that case it also falls back to the in-memory path
    (the delete completes without the facts side)."""
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
    """Deletes ALL derivatives of a record (idempotent) -- the shared A3/A4 path."""
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
    """Processes a single document end-to-end; returns the final state ("ready"|"error")."""
    from medrag.pipeline.cli import run_parse_pipeline as rpp

    record = registry_rw.find(registry_rw.load(registry_path()), doc_id)
    if record is None:
        durum_store.write_status(durum_dir(), doc_id, "error",
                                 detail="registry kaydı bulunamadı")
        return "error"

    # 1) Old derivatives (change/delete-then-re-add semantics).
    status("parsing", "eski türevler temizleniyor")
    forget_derivatives(record)

    # 2) parse -- existing phase functions on a single record.
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

    # 3) chunk -- the whole corpus is re-produced (cheap, LLM-free); this
    #    document's block is read from provenance and written under lock.
    status("chunking", "chunk'lar yeniden üretiliyor")
    env = os.environ
    env["CHUNKER_INPUT_DIR"] = str(parsed_output_dir())
    # Two env vars, two readers: chunker's OutputLayout reads CHUNKER_OUTPUT_DIR,
    # `resolve_all_chunks_path` (for the registry block) reads CHUNKS_OUTPUT_DIR
    # -- both must point at the same root, or the block is read from the wrong file.
    env["CHUNKER_OUTPUT_DIR"] = str(_chunk_output_root())
    env["CHUNKS_OUTPUT_DIR"] = str(_chunk_output_root())
    env.pop("PARSER_DIR", None)  # keep chunker's own PARSER_DIR out of the way (see run_chunk_pipeline)
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

    # 4) vectorize -- the staleness gate embeds only the changed scope.
    status("vectorizing", "embedding")
    from medrag.pipeline.vectorize import cli as vec_cli

    # forget_derivatives DELETED this document's Qdrant vectors; vectorize's
    # local staleness signature (state.json) is still there, so it does NOT
    # re-embed, saying "unchanged, skipping" -> the document stays "ready" with
    # 0 vectors and is never found in retrieval (the notes-not-in-answer bug).
    # Clear this document's signature so calistir re-embeds it.
    _reset_staleness(doc_id)

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
    """Deletes the document and ALL its derivatives; returns "deleted" (or raises)."""
    status("parsing", "siliniyor")
    data = registry_rw.load(registry_path())
    record = registry_rw.find(data, doc_id)
    if record is None:
        # no registry record, but a status file may be left hanging.
        durum_store.remove_status(durum_dir(), doc_id)
        return "deleted"
    forget_derivatives(record)
    registry_rw.remove(doc_id, registry_path())
    durum_store.remove_status(durum_dir(), doc_id)
    return "deleted"


__all__ = ["delete_document", "forget_derivatives", "process_document"]
