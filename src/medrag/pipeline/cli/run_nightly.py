"""Single entry point: `medrag-nightly` (N-01).

Runs the `scan -> parse -> chunk -> ownership -> facts -> load -> vectorize`
order (N-22: `ownership`/`load` were both NOT PREVIOUSLY connected to the chain
-- `ownership` comes BEFORE `facts` because it produces the chunk-ownership
table (`catalog_chunk_ownership/build_all.py`) and facts' multi-owner fan-out
narrowing reads this table; `load` comes AFTER `facts` because it WRITES the
JSONs facts produces into `specs.db` -- before these two were connected the
nightly run did NOT write a SINGLE ROW to specs.db, see the `stage_ownership`/
`stage_load` docstrings below). Each
stage calls the PROGRAMMATIC function of the existing entry points --
NO subprocess (this file's only job is orchestration; the business logic
already lives in the existing modules):

    scan       chatbot-corpus/document_info/classify_documents.py::main()
               (OUTSIDE the `medrag` package -- it has not yet entered Group D's
               migration scope; because it imports the plain `config` module in
               its own dir under the same name, a path-based lazy-import + a
               temporary sys.path insertion is used instead of a normal
               `import`, the same pattern the parser used BEFORE D-39 -- it only
               runs when stage_scan() is actually called, no side effect at
               module import time)
    parse      medrag.pipeline.cli.run_parse_pipeline.main(argv=[])
               (argv=[] MANDATORY: if not given empty, argparse reads the
               orchestrator's own sys.argv and rejects --stage/--from/--doc as
               "unknown flag")
    chunk      medrag.pipeline.cli.run_chunk_pipeline.main() (it already
               subprocesses `python -m medrag.pipeline.chunker` internally --
               that is the chunker's OWN env-only contract, NOT N-01's scope;
               what changes here is only that the ORCHESTRATOR calls this script
               as a function)
    ownership  medrag.pipeline.facts.catalog_chunk_ownership.build_all.main(argv=[])
               (N-22: comes AFTER chunk because it needs `all_chunks.json`, and
               BEFORE facts because facts' multi-owner fan-out narrowing reads
               the `chunk_ownership_all.json` this stage produces -- if the table
               is absent facts EXCLUDES multi-owner documents entirely. `--doc`
               is a NO-OP in this stage: build_all.py does not support a
               doc-level filter; EVERY run scans the whole corpus (single+multi
               owner) -- it has its own checkpoint/resume mechanism
               (`chunk_owner_progress.jsonl`), similar to the second version of
               KARAR-012.)
    facts      medrag.pipeline.facts.run_full.run_products(...) -- NOT `main()`:
               run_full.main() calls `argparse.parse_args()` without argv (it
               would read the orchestrator's own argv). run_full.py's own
               docstring defines `run_products` for exactly this purpose: "The
               programmatic entry point Group N's nightly incremental run (re-
               processing the products affected when a doc_id changes) can call
               directly, independent of the CLI."
    load       medrag.pipeline.facts.load_to_db.main(argv=[...]) (N-22: WRITES
               the `results/<CODE>.json`s facts produces into `specs.db` --
               since `run_products` wrote NOTHING to specs.db, without this
               stage the nightly run changed no row in the DB. It RE-resolves the
               SAME product set `facts` processed with `_resolve_incremental_codes()`
               (specs.db did not change during facts, so the same set) and calls
               with `--models <set> --force` -- this is EXACTLY EQUIVALENT to
               `load_product_incremental` (O-12) (see that function's docstring).)
    vectorize  medrag.pipeline.vectorize.cli.calistir(...) -- NOT `main()`:
               main() only wraps argv parsing + logging.basicConfig +
               .env loading; the real work is in `calistir()`.

Rule (I-26's preserved acceptance sentence): in a chain started with
`--stage`/`--from`, the functions of stages NOT INCLUDED in the plan are NEVER
CALLED. This module resolves the `stage_*` functions via `globals()`, ONLY for
the names that enter the plan at run time -- a `stage_*` name outside the plan
is never referenced even once in this process (see `tests/test_run_nightly.py`,
which proves with a mock that `--from chunk` never calls `stage_scan`/`stage_parse`;
parse can take hours, and if it runs silently you don't notice but it occupies
the GPU).

Flags:
    --stage <stage>   run ONLY this stage (manual-intervention shortcut)
    --from <stage>    run this stage through the end of the chain
Environment variables (repo convention: settings in env, see CONFIG.md):
    FACTS_PROCESS_ALL=1    bypass the staleness gate, process ALL products
    FACTS_SKIP_EXISTING=1  manually to resume a half-finished run: skip
                           products whose `results/<CODE>.json` is COMPLETE
                           (DEFAULT OFF -- leaving it on forever would let
                           stale products go untainted)
    FACTS_PENDING_REWORK_PATH  path to the JSONL queue of doc_ids whose
                           facts/load step still has work pending
                           (default `facts/pending_rework.jsonl`)

    --doc <doc_id>    passed as `--doc-id` to the `facts` AND `load` stages
                      (reprocesses the products using that document -- the
                      other stages don't take a doc filter)
                      the entry points UNDER scan/parse/chunk/ownership/vectorize
                      do NOT support a doc-level filter (scan: KARAR-001 -- a
                      partial scan wrongly drops unseen files to DELETED;
                      parse: processes the whole pending corpus; chunk/vectorize:
                      the second version of KARAR-012 -- each run re-processes the
                      whole scope; ownership: build_all.py has its own
                      checkpoint/resume mechanism, no doc-level filter) -- in these
                      stages `--doc` is not silently ignored, a warning is printed
                      and it becomes a NO-OP (known limitation, separate task later).
    --dry-run         (N-03) writes no data and makes no network calls (Ollama/VLM/
                      Qdrant/embedding) -- for each stage a `_dry_run_*` reporter is
                      called instead of the real `stage_*`; it reads the SAME
                      resources on disk (document_nodes.json, DOCUMENT_NODES_PATH,
                      the chunk input folder, specs.db, chunk scopes) and prints the
                      set "what would this stage really process if it ran" then stops.

Usage:
    python -m medrag.pipeline.cli.run_nightly                  # full chain
    python -m medrag.pipeline.cli.run_nightly --from chunk      # chunk->vectorize
    python -m medrag.pipeline.cli.run_nightly --stage facts --doc PN1057
    python -m medrag.pipeline.cli.run_nightly --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

STAGES = ["scan", "parse", "chunk", "ownership", "facts", "load", "vectorize"]

# src/medrag/pipeline/cli/run_nightly.py -> repo root (4 levels up, the SAME
# convention as the other pipeline scripts -- see run_chunk_pipeline.py's
# CHUNKER_DIR comment).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCUMENT_INFO_DIR = _REPO_ROOT / "chatbot-corpus" / "document_info"
def _load_classify_documents():
    """Lazy-imports `classify_documents.py` from a path. The module imports its
    own plain `config.py` with `from config import load_config` -- so at import
    time the `document_info/` dir is temporarily added to `sys.path` (only while
    this function runs; removed in finally, leaving no permanent side effect)."""
    if not _DOCUMENT_INFO_DIR.is_dir():
        raise RuntimeError(
            f"chatbot-corpus/document_info bulunamadi: {_DOCUMENT_INFO_DIR}")
    ekliydi = str(_DOCUMENT_INFO_DIR) not in sys.path
    if ekliydi:
        sys.path.insert(0, str(_DOCUMENT_INFO_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "medrag_nightly_classify_documents",
            _DOCUMENT_INFO_DIR / "classify_documents.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if ekliydi:
            sys.path.remove(str(_DOCUMENT_INFO_DIR))


def _doc_noop(stage: str, reason: str, doc_id: str | None) -> None:
    if doc_id:
        print(f"{stage}: --doc no-op ({reason})")


def stage_scan(*, doc_id: str | None = None) -> dict:
    """N-21: the `output` dict `module.main()` returns (`summary`/`documents`)
    is no longer THROWN AWAY, it is returned to the caller -- this is the ONLY
    real source of the nightly report's Source-diff/Skipped sections
    (`_build_nightly_report` -- no number is INVENTED, scan already computes it)."""
    _doc_noop("scan", "classify_documents.py dokuman-bazli filtre "
              "desteklemiyor, KARAR-001 -- tum BELGELER agaci taranir", doc_id)
    module = _load_classify_documents()
    return module.main()


def _read_document_nodes() -> dict | None:
    """K-99 (2026-08-26 fix, follow-up inspection item 6 -- cleanup):
    the ONE shared point of ever place reading `document_nodes.json`
    (`_forgettable_scan_records`/`_incremental_doc_ids`/`_parse_failed_docs`) --
    all three carried their own COPY of `_bootstrap()` + "does file exist" +
    `json.loads`. Returns `None` if the file DOES NOT EXIST (the caller returns
    an empty list/set).

    DELIBERATELY does NOT cache -- re-reads the file EVERY call: all three are
    called at DIFFERENT points of the run (`_forgettable_scan_records`: BEFORE
    parse; `_parse_failed_docs`: AFTER parse; `_incremental_doc_ids`: in the
    facts stage, in a separate `stage_*` call) and the file REALLY changes between
    those points (`run_parse_pipeline.main()` writes after every document) -- a
    single cache would return stale/wrong data. Uses the `DOCUMENT_NODES_PATH`
    `run_parse_pipeline._bootstrap()` already resolves -- it points at the SAME
    file as classify_documents.py's own `OUTPUT_PATH` (both are the single record
    scan writes/parse reads); no second path-resolution logic is added."""
    from medrag.pipeline.cli import run_parse_pipeline

    run_parse_pipeline._bootstrap()
    document_nodes_path = Path(run_parse_pipeline.DOCUMENT_NODES_PATH)
    if not document_nodes_path.is_file():
        return None
    return json.loads(document_nodes_path.read_text(encoding="utf-8"))


def _forgettable_scan_records() -> list[tuple[str, str | None, str, str | None]]:
    """N-19 (K-96 step 2) + K-97 (2026-08-26 fix) + K-100 (2026-08-27 fix):
    the list of `(doc_id, doc_type, scan_status, content_hash)` for the
    `scan_status IN ('DELETED', 'MODIFIED')` records in `document_nodes.json` --
    `scan_status` is carried as the THIRD field so the caller
    (`_forget_deleted_sources`) can write to the pending-rework queue; `content_hash`
    is carried as the FOURTH field so the caller can write that hash to
    `scan.forgotten_hash` and not try to forget the ALSO same record again the
    next night (the K-100 note below).
    `is_active=false` records are NOT HERE -- they are not "deleted" but
    "editorially disabled" (a different intent, K-96); the scan_status field
    does not make them DELETED anyway.

    MODIFIED was ADDED to DELETED (previously only DELETED was processed): user
    decision -- a source CHANGE is modeled in code as "DELETE + ADD", not
    "update in place". When a document becomes MODIFIED, ALL derivatives of that
    document's OLD content (parse output, chunk record, Qdrant points, spec
    evidence) are removed with `forget_deleted_source`; the parse/chunk/facts
    chain re-produces them from the NEW content that same night, within the SAME
    run (the document never fell out of the "pending" list, it only CHANGED what
    was deleted -- see where `_forget_deleted_sources` is called, at the VERY START
    of `stage_parse`). MOVED is NOT INCLUDED HERE (N-07: a file's location change
    does NOT count as delete+add, its derivatives are preserved).

    K-100 (2026-08-27 fix): `scan_status=DELETED` is STICKY (the folder scanner
    never clears it, a schema decision) -- the natural consequence: this filter,
    with NO gate, returns "every record currently DELETED", so records already
    forgotten in the PAST (their derivatives removed) would be selected again
    EVERY NIGHT. In the real world there is a second cost: when a source
    disappears ONLY TEMPORARILY (an AGM network drive scanned HALF-way during a
    move/copy -- the 2026-08-27 incident) forget_deleted_source REALLY runs, and
    when the file comes BACK (SAME rel_path + SAME content_hash) the scanner
    silently turns it to UNCHANGED -- but because `parse.parsed_from_hash` still
    carries the SAME hash (forget never touched this field) parse/chunk/facts
    would NEVER re-produce it (see the `parsed_from_hash != content_hash` gate in
    `run_parse_pipeline.py`). `_forget_deleted_sources` now writes
    `scan.forgotten_hash = content_hash` to every record it successfully forgets
    and resets the `parse` block to PENDING (see that function's note) -- and this
    filter SKIPS records whose `forgotten_hash == content_hash` (i.e. whose
    content has not changed since it was forgotten). So: (1) the same record is
    not processed forever (the buildup ends), (2) if the content CHANGES again
    (forgotten_hash no longer matches the current content_hash) it is naturally
    re-selected, (3) since `parse` is reset, if the file comes back (UNCHANGED)
    the hash-gate still fires correctly (silent data loss ends)."""
    data = _read_document_nodes()
    if data is None:
        return []
    result: list[tuple[str, str | None, str, str | None]] = []
    for rec in data.get("documents", []):
        scan = rec.get("scan", {})
        status = scan.get("scan_status")
        if status not in ("DELETED", "MODIFIED"):
            continue
        content_hash = scan.get("content_hash")
        forgotten_hash = scan.get("forgotten_hash")
        if forgotten_hash is not None and forgotten_hash == content_hash:
            continue  # K-100: already forgotten with this content before, skip
        result.append((rec["identity"]["doc_id"], rec.get("doc_type"), status, content_hash))
    return result


#: K-100 (2026-08-27 fix): the SAME copy of the `parse` sub-block of
#: `classify_documents.py::blank_pipeline_state()` -- that module is OUTSIDE the
#: `medrag` package (requires a path-based lazy-import, see the note at the very
#: TOP of the module file), so here it is cheaper to re-define this ONE block than
#: to import the whole module, AND it does not break the layer rule. Deliberately
#: ONLY `parse` is reset -- `chunk` per KARAR-012 already re-produces the whole
#: corpus every run (no hash-gate of its own, no need to reset), `facts` is
#: covered via the pending-rework queue (below).
_BLANK_PARSE_STATE = {
    "parser": None, "parser_version": None, "status": "PENDING",
    "parsed_from_hash": None, "parsed_json_path": None,
    "last_parsed": None, "error": None, "stages": None,
}


def _forget_deleted_sources(*, dry_run: bool = False) -> list:
    """N-19 (K-96 step 2): connects `forget_deleted_source` (N-06, written but
    never called anywhere) to the nightly run -- runs AFTER scan, BEFORE parse
    (at the VERY START of `stage_parse`) so parse does not uselessly process a
    gone/changed file with its OLD derivatives.

    `dry_run=True`: deletes NOTHING, only prints which doc_ids would be forgotten
    (the SAME pattern as the other `_dry_run_*` reporters).

    K-100 (2026-08-27 fix, replaces K-98): on the real path (dry_run=False) BOTH
    MODIFIED and DELETED doc_ids are written 'pending' to the PERSISTENT queue via
    `_append_pending_rework_events` -- previously DELETED was deliberately EXCLUDED
    (with the assumption "the source is gone forever"), but the 2026-08-27 incident
    showed this assumption could be WRONG: a source can appear DELETED TEMPORARILY
    (when a move/copy was cut off MIDWAY), be genuinely forgotten, then come BACK
    with the SAME content. Since such a doc_id is in the pending-rework queue,
    `facts` still retries it even if its scan_status turned back to UNCHANGED (see
    its merge with `_incremental_doc_ids`, `_settle_pending_rework`). For a source
    REALLY gone forever this is HARMLESS: `resolve_codes` returns empty,
    `_settle_pending_rework` drops it from the queue as 'orphaned' (see that
    function's docstring) -- it cleans itself up, never retried forever.

    K-100 (2026-08-27 fix, continued): `scan.forgotten_hash = content_hash` is
    written in `document_nodes.json` to every record successfully forgotten (so the
    next night `_forgettable_scan_records` does not select the SAME record again,
    see that function's K-100 note) AND the `parse` block is reset to
    `_BLANK_PARSE_STATE` (so when the file comes back and is counted UNCHANGED,
    the `parsed_from_hash != content_hash` gate fires correctly -- forget_deleted_source
    ITSELF never touched these fields, and that was the real source of the silent
    data loss). The log is also cleaned within the same run: records that really
    removed something are printed one by one; records where nothing was found
    (already clean) are grouped into a single summary line -- before, EVERY record
    (mostly no-op) printed its own line, flooding the log with hundreds of
    meaningless lines every night."""
    from medrag.pipeline.cli import run_parse_pipeline

    forgettable = _forgettable_scan_records()
    if not forgettable:
        print("forget_deleted_source: scan_status=DELETED/MODIFIED kayit yok "
              "(ya da hepsi zaten daha once unutulmustu)")
        return []

    if dry_run:
        doc_ids = [d for d, _, _, _ in forgettable]
        print(f"[dry-run] forget_deleted_source: {len(forgettable)} silinmis/degismis "
              f"kaynak unutulacakti -> {doc_ids[:5]}{', ...' if len(doc_ids) > 5 else ''} "
              "-> YAZILMAYACAK")
        return forgettable

    from medrag.pipeline.facts.discover import ALL_CHUNKS_PATH, DB_PATH
    from medrag.pipeline.forget_deleted_source import forget_deleted_source
    from medrag.pipeline.vectorize.config import load_config as _load_vectorize_config
    from medrag.pipeline.vectorize.store import QdrantVectorStore

    vec_cfg = _load_vectorize_config(os.environ.get("VECTORIZE_CONFIG") or None)
    qdrant_url = (os.environ.get("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise RuntimeError(
            "forget_deleted_source: QDRANT_URL tanimsiz -- silinen/degisen kaynaklarin "
            "Qdrant temizligi yapilamaz (bkz. .env.example)")
    store = QdrantVectorStore.from_env(
        collection_name=vec_cfg.qdrant.collection_name, distance=vec_cfg.qdrant.distance,
        on_dim_mismatch=vec_cfg.qdrant.on_dim_mismatch,
        upsert_batch_size=vec_cfg.qdrant.upsert_batch_size,
        url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)

    data = _read_document_nodes()
    records_by_id = {
        rec["identity"]["doc_id"]: rec for rec in (data.get("documents", []) if data else [])
    }

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        results = []
        pending_events = []
        already_clean = 0
        for source_doc_id, doc_type, scan_status, content_hash in forgettable:
            result = forget_deleted_source(
                doc_id=source_doc_id, doc_type=doc_type,
                parsed_output_dir=Path(run_parse_pipeline.PARSED_OUTPUT_DIR),
                all_chunks_path=ALL_CHUNKS_PATH, vector_store=store, specs_con=con,
            )
            results.append(result)
            spec_rows_deleted = result.spec_evidence.get("rows_deleted", 0)
            if result.parse_dirs_removed or result.chunk_entry_removed or spec_rows_deleted:
                print(f"forget_deleted_source: {source_doc_id} -> "
                      f"parse_dirs_removed={len(result.parse_dirs_removed)} "
                      f"chunk_entry_removed={result.chunk_entry_removed} "
                      f"spec_rows_deleted={spec_rows_deleted}")
            else:
                already_clean += 1
            pending_events.append({"doc_id": source_doc_id, "status": "pending"})
            rec = records_by_id.get(source_doc_id)
            if rec is not None:
                rec.setdefault("scan", {})["forgotten_hash"] = content_hash
                rec["parse"] = dict(_BLANK_PARSE_STATE)
        if already_clean:
            print(f"forget_deleted_source: {already_clean} kayit zaten temizdi (no-op, "
                  "tek tek basilmadi)")
        _append_pending_rework_events(pending_events)
        if data is not None:
            run_parse_pipeline._atomic_write_json(run_parse_pipeline.DOCUMENT_NODES_PATH, data)
        return results
    finally:
        con.close()


def stage_parse(*, doc_id: str | None = None) -> dict:
    """N-21: `processed_count`/`duration_seconds` are really MEASURED --
    `run_parse_pipeline.main()` now returns the number of documents processed
    (before it returned `None`, the number was DISCARDED); the duration is this
    function's own `time.monotonic()` measurement (EXCLUDING the
    forget_deleted_source step, it only covers the real parse call).
    `model_call_count` is NOT HERE -- the parser has no counter at any of its
    LLM/VLM call points, it falls into `nightly_report.py::ParseSection.model_call_count`'s
    default of `None` ("not measured") (N-21 end note -- leave it explicitly
    missing rather than inventing it).

    N-10 fix (2026-08-26): `failed_docs` -- `run_parse_pipeline.main()` did NOT
    carry the per-document failures (a VLM/OCR/LLM blowup on a page sets
    `parse.status='FAILED'`) in its return value AT ALL, it only returned the
    TOTAL processed count (the `counts` dict was computed INSIDE the module and
    only PRINTED). `_build_nightly_report` reads this list and turns each row
    into a `FailureEntry` -- so even though the stage itself does NOT raise an
    exception (parse `main()` SWALLOWS the per-document errors and keeps
    processing the whole corpus), a single FAILED document now drops `outcome()`
    from `full_success`.

    K-99 fix: `failed_docs` only carries FAILED records whose
    `parse.last_parsed` was UPDATED this run (`since=run_started_at`) --
    previously every FAILED record in `document_nodes.json` (including
    "tombstones" from previous nights -- e.g. a source file later DELETED
    but the record stuck at FAILED forever) entered the report every
    night, so even on a perfectly clean night `outcome()` could NEVER be
    `full_success` -- the INVERSE of what N-10 fixed (the report says
    "always bad", carries no information)."""
    _doc_noop("parse", "run_parse_pipeline.py does not support a doc-level filter "
              "-- the whole pending corpus is processed", doc_id)
    _forget_deleted_sources(dry_run=False)
    from datetime import datetime
    from time import monotonic

    from medrag.pipeline.cli import run_parse_pipeline
    run_started_at = datetime.now().astimezone()
    t0 = monotonic()
    processed_count = run_parse_pipeline.main(argv=[])
    duration_seconds = monotonic() - t0
    return {"processed_count": processed_count, "duration_seconds": duration_seconds,
            "failed_docs": _parse_failed_docs(since=run_started_at)}


def _parse_failed_docs(*, since) -> list[dict]:
    """N-10/K-99 fix (2026-08-26): the `(doc_id, file_name, error)` list of the
    records in `document_nodes.json` with `parse.status == 'FAILED'` AND
    `parse.last_parsed >= since` -- called RIGHT AFTER the `stage_parse` run
    (`run_parse_pipeline.main()` writes the file to disk after every document, see
    that module's `_run_phase`). Without the `since` filter (run start,
    `datetime.now().astimezone()` -- the SAME tz-aware form as
    `run_parse_pipeline.py::_now()`) the FAILED "tombstone" records left over from
    PREVIOUS nights (a source file deleted, a record stuck at FAILED forever) would
    enter the report EVERY NIGHT and `outcome()` could NEVER say `full_success` --
    see `stage_parse`'s K-99 note. A FAILED record with NO `last_parsed` (or an
    unparseable one) -- since its validity period is UNKNOWN -- stays on the SAFE
    side and does NOT enter the report."""
    from datetime import datetime

    data = _read_document_nodes()
    if data is None:
        return []
    failed = []
    for rec in data.get("documents", []):
        parse = rec.get("parse", {})
        if parse.get("status") != "FAILED":
            continue
        last_parsed = parse.get("last_parsed")
        if not last_parsed:
            continue
        try:
            when = datetime.fromisoformat(last_parsed)
        except ValueError:
            continue
        if when < since:
            continue
        failed.append({
            "doc_id": rec["identity"]["doc_id"],
            "file_name": rec["identity"].get("file_name", "?"),
            "error": parse.get("error") or "parse.status=FAILED",
        })
    return failed


def _all_chunks_node_counts(path: Path) -> dict[str, int]:
    """`{doc_id: node_count}` -- an empty dict if the file is missing/unreadable
    (KARAR-012: each run writes `all_chunks.json` FROM SCRATCH, so the
    "before/after" difference means a diff of THAT whole file)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {doc_id: len(cs.get("nodes") or []) for doc_id, cs in (data.get("documents") or {}).items()}


def stage_chunk(*, doc_id: str | None = None) -> dict:
    """N-21: `produced_count`/`per_document_counts`/`removed_count` are a REAL
    MEASUREMENT -- `run_chunk_pipeline.py` (which subprocesses the chunker) never
    RETURNS these numbers, so `all_chunks.json`'s pre-run/AFTER-run state is
    compared here (with `resolve_all_chunks_path()` the SAME file, no second
    path-resolution logic -- see that function's docstring). `produced_count` =
    the total chunk (node) count AFTER the run; `removed_count` = the sum of the
    chunk counts of documents present BEFORE the run but not visible after (deleted
    / no longer chunkable)."""
    _doc_noop("chunk", "run_chunk_pipeline.py her kosuda tum korpusu "
              "yeniden isler (KARAR-012 ikinci surumu)", doc_id)
    from medrag.pipeline.cli import run_chunk_pipeline

    all_chunks_path = run_chunk_pipeline.resolve_all_chunks_path()
    before = _all_chunks_node_counts(all_chunks_path)
    run_chunk_pipeline.main()
    after = _all_chunks_node_counts(all_chunks_path)

    removed_count = sum(count for doc_id_, count in before.items() if doc_id_ not in after)
    return {
        "produced_count": sum(after.values()),
        "per_document_counts": after,
        "removed_count": removed_count,
    }


def _env_flag(name: str) -> bool:
    """`1/true/yes/on` -> True (whitespace and case irrelevant). The SINGLE parsing
    point for the nightly run's env gates (`FACTS_PROCESS_ALL`, `FACTS_SKIP_EXISTING`)
    -- if the two were parsed separately, one might accept "on" while the other
    didn't and the difference would SILENTLY reach the operator."""
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _incremental_doc_ids() -> list[str]:
    """K-97 (2026-08-26 fix): the PRIMARY SOURCE of tonight's `facts` work -- the
    doc_ids of the `scan_status IN ('NEW', 'MODIFIED')` records in
    `document_nodes.json`. Reads the SAME file as `_forgettable_scan_records` (via
    `_read_document_nodes()`), but wants a different set: DELETED not here (its
    source is gone, nothing left to produce a product -- `_forget_deleted_sources`
    already deleted its derivatives), MOVED not here (N-07: a location change is
    not delete+add, products/facts not affected), UNCHANGED not here (nothing
    changed -- this function's ENTIRE purpose is to make facts process ZERO
    products on a night where nothing changed)."""
    data = _read_document_nodes()
    if data is None:
        return []
    return [
        rec["identity"]["doc_id"]
        for rec in data.get("documents", [])
        if rec.get("scan", {}).get("scan_status") in ("NEW", "MODIFIED")
    ]


def _pending_rework_doc_ids() -> list[str]:
    """K-98 (2026-08-26 fix, follow-up 1): the doc_ids in the pending-rework queue
    (append-only JSONL, `core/paths.py::resolve_pending_rework_path`) whose LAST
    status is 'pending' -- the SAME discipline as `chunk_owner_progress.jsonl`:
    the file is read LINE BY LINE, and for each doc_id the LAST record wins (if
    'pending' is seen then 'done' -- or the reverse -- the last written one is
    valid)."""
    from medrag.core.paths import resolve_pending_rework_path

    path = resolve_pending_rework_path()
    if not path.is_file():
        return []
    latest: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[rec["doc_id"]] = rec["status"]
    return sorted(doc_id for doc_id, status in latest.items() if status == "pending")


def _append_pending_rework_events(events: list[dict]) -> None:
    """K-98: appends an append-only event to the pending-rework queue -- the file is
    NEVER OVERWRITTEN/TRUNCATED (the SAME discipline as `chunk_owner_progress.jsonl`),
    the reading side (`_pending_rework_doc_ids`) bases on the LAST record."""
    if not events:
        return
    from medrag.core.paths import resolve_pending_rework_path

    path = resolve_pending_rework_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _settle_pending_rework(con, load_report: dict) -> None:
    """K-98 (2026-08-26 fix, follow-up 1): the pending doc_ids whose products were
    LOADED successfully this run drop from the queue (append-only 'done' event). If
    ANY ONE of the products of a doc_id resolved via `run_full.resolve_codes(con, doc_id=...)`
    was not attempted this run, or ended with an error (the `error` field in
    `load_to_db.main()`'s `reports[]`) the doc_id STAYS on the queue -- partial success is NOT
    sufficient, the next night retries. An empty `load_report` (e.g.
    `stage_load` had an empty `codes` and so never called
    `load_to_db.main()`) drops NOTHING.

    Ownerless docs: a pending doc_id whose `resolve_codes` returns empty
    CANNOT be satisfied -- the queue exists to "reprocess this doc's
    products", and if there are none, there's nothing to do. Previously
    `continue`'d, the record stayed on the queue forever: re-tried every
    night with zero work, harmless but the queue grows unbounded. Now
    settled with an 'orphaned' event (NOT 'done' -- no products were
    processed); if the doc's content later changes, scan marks it MODIFIED
    and `_forget_deleted_sources` re-enqueues 'pending', so the protection
    is preserved. `stage_load` calls settle even in the empty-codes branch
    for this reason."""
    from medrag.pipeline.facts import run_full

    pending = _pending_rework_doc_ids()
    if not pending:
        return
    reports = load_report.get("reports") or []
    attempted = {r.get("model_code") for r in reports if r.get("model_code")}
    errored = {r.get("model_code") for r in reports if r.get("error")}

    events = []
    for pending_doc_id in pending:
        products = run_full.resolve_codes(con, doc_id=pending_doc_id)
        if not products:
            events.append({
                "doc_id": pending_doc_id, "status": "orphaned",
                "reason": "resolve_codes bos -- bu dokumana ait urun yok, islenecek sey yok",
            })
            continue
        if all(p in attempted and p not in errored for p in products):
            events.append({"doc_id": pending_doc_id, "status": "done"})
    _append_pending_rework_events(events)


def _resolve_incremental_codes(con, *, doc_id: str | None) -> tuple[list[str], bool]:
    """N-22 + K-97/K-98 (2026-08-26 fix): the SINGLE resolution point for the
    `--doc` + `FACTS_PROCESS_ALL` escape hatch, AND for product selection on the
    default (no escape hatch) path -- `stage_facts` AND `stage_load` (which writes
    the JSONs facts produces into specs.db for the SAME product set) SHARE it.
    Since `run_products` writes NOTHING to specs.db (see run_full.py's module
    docstring) the DB state does not change between `facts` and `load` -- the two
    calls MUST give the same set, so the code lives in ONE place (if two separate
    copies silently diverged, `load` would write a set DIFFERENT from the products
    facts processed).

    On the default path (no escape hatch) `codes` is the UNION OF THREE SOURCES:
      1. **Primary** -- products owned by NEW + MODIFIED documents
         (`_incremental_doc_ids()`, K-97). This covers the large majority.
      2. **K-98 (ADDED this run, follow-up 1)** -- products owned by the doc_ids
         in the pending-rework queue (`_pending_rework_doc_ids()`). K-97's
         assumption that "the next night the same document is STILL MODIFIED and
         fixes itself" turned out WRONG: `classify_documents.py::scan_status_for`
         decides MODIFIED/UNCHANGED against the hash a PREVIOUS scan saved, and the
         scan saves the NEW hash IMMEDIATELY -- so if the run CRASHES between
         `_forget_deleted_sources` (parse start) and where this function is called
         (`facts`/`load`, the end of the chain), the document appears UNCHANGED the
         NEXT NIGHT and is NEVER SELECTED AGAIN: silent, PERMANENT data loss (the
         2026-08-26 run was stopped exactly in this range, at the `facts` stage --
         not hypothetical). The queue keeps this range in a place that SURVIVES the
         run (see `_forget_deleted_sources`'s K-98 note, `_settle_pending_rework`).
      3. **K-96's audit endpoints RE-ADDED as the SECONDARY net (this run,
         follow-up 2+3)** -- `staleness_audit.stale_product_codes`
         (`dangling_evidence`/`empty_evidence`/`stale_extractor_version`).
         These THREE criteria together are the "reconciliation net", NOT the primary
         selector: (a) without `stale_extractor_version`, after `facts_version`
         (prompt/schema/threshold) advanced NO product was ever re-produced --
         even if no document changed, this selector must NOT return EMPTY in that
         setup (regression, follow-up 2). (b) without
         `dangling_evidence`/`empty_evidence`, if the chunker logic/version changes
         WITHOUT the source changing and chunk sha's change (documents stay
         UNCHANGED), evidence now points at chunks that no longer exist and NEVER
         REFRESHES -- this is the "re-chunking blind spot" (follow-up 3). The
         fourth rule K-97 REMOVED ("products with a new chunk that has no evidence
         at all") was NOT RE-ADDED HERE -- it saw EVERY chunk that produces no fact
         as "unprocessed" forever and marked ~ALL the corpus (218 products) EVERY
         NIGHT, a FUNDAMENTALLY DIFFERENT and wrong criterion (K-97's root cause);
         `stale_product_codes`'s three criteria do not include it, staying narrow
         and bounded.

    Acceptance: on a night where no document changed, the pending-rework queue is
    empty, and `facts_version`/the chunk set did not change, this function returns
    an EMPTY list and facts processes ZERO products.

    Returns: `(codes, process_all)`."""
    from medrag.pipeline.facts import run_full

    process_all = _env_flag("FACTS_PROCESS_ALL")
    if doc_id or process_all:
        return run_full.resolve_codes(con, doc_id=doc_id), process_all

    from medrag.pipeline.facts.staleness_audit import stale_product_codes
    from medrag.pipeline.nightly_report import current_pipeline_versions

    codes: set[str] = set()
    for changed_doc_id in _incremental_doc_ids():
        codes.update(run_full.resolve_codes(con, doc_id=changed_doc_id))
    for pending_doc_id in _pending_rework_doc_ids():
        codes.update(run_full.resolve_codes(con, doc_id=pending_doc_id))
    facts_version = current_pipeline_versions()["facts"]
    codes.update(stale_product_codes(con, current_facts_version=facts_version))
    return sorted(codes), process_all


def stage_ownership(*, doc_id: str | None = None) -> dict | None:
    """N-22: runs via the documented programmatic entry point of
    `catalog_chunk_ownership/build_all.py` (`main(argv=[])`) -- see the module
    docstring. Without being connected to the chain, `facts`' multi-owner fan-out
    narrowing (`chunk_ownership_index`) was NEVER filled; documents like
    catalog/brochure were EXCLUDED entirely (`docs_excluded_fanout`).

    `--doc` is a NO-OP in this stage: build_all.py does not support a doc-level
    filter; EVERY run scans the whole corpus (single+multi-owner documents) -- it
    has its own checkpoint/resume mechanism (if interrupted with Ctrl+C the NEXT
    run continues from where it left off, `chunk_ownership_all.json` remains
    intact)."""
    _doc_noop("ownership", "build_all.py dokuman-bazli filtre desteklemiyor, "
              "tum korpus (tek+cok-sahipli dokuman) taranir", doc_id)
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    return build_all.main(argv=[])


def stage_facts(*, doc_id: str | None = None) -> dict:
    """Runs via the documented programmatic entry point of `run_full.py`
    (`run_products`) -- see the module docstring.

    K-97/K-98 (2026-08-26 fix, replaces the old N-20/K-96 step 3): product
    selection is NOT `resolve_codes(con)` (ALL products in specs.db) -- it is the
    UNION of the NEW+MODIFIED document set (primary) + the pending-rework queue
    (K-98) + `staleness_audit.stale_product_codes`'s three criteria (secondary
    reconciliation net) (see `_resolve_incremental_codes`'s docstring, the whole
    rationale is there). **CRITICAL CONSTRAINT**: this only NARROWS which products
    are processed -- for the selected product the write path does NOT change;
    `run_products` -> `load_product_incremental` still does a FULL reset via
    `_reset_product` (force) (K-96, 2026-08-06 measurement: pinpoint/partial
    deletion must NOT be attempted).

    Escape hatches (`--all` from K-96, also kept in K-97): `--doc`
    (manual intervention, always existed) AND `FACTS_PROCESS_ALL=1`
    (env var -- repo convention: settings in env, no new CLI flag) fully
    bypass the gate and revert to the old "all products" behavior.

    `FACTS_SKIP_EXISTING=1` (a SEPARATE gate, affects `run_products`'s
    inner loop, not product selection): products whose
    `results/<CODE>.json` is COMPLETE (not `partial`) are SKIPPED. MUST
    stay OFF by default -- if left on, a complete JSON from a previous
    night also counts as "already there", so a product whose chunk changed
    (looks stale) would never be refreshed (silent staleness, opposite of
    what this gate is for). The only legitimate use is manual resume: if
    the nightly run was killed mid-`facts`, the next run sees the same set
    as stale and would redo all the LLM work.
    to resume from that point.

    N-22: product selection (`codes`) now goes through `_resolve_incremental_codes()`
    -- `stage_load` (the stage that writes this stage's JSONs into specs.db) calls
    the SAME function, so the product set NEVER silently diverges between the two
    stages."""
    from dotenv import load_dotenv

    from medrag.pipeline.facts import run_full

    load_dotenv(run_full.ENV_PATH)
    provider, model, base_url, api_key = run_full._resolve_llm_env(os.environ)
    system = run_full.build_system_message()
    con = run_full._connect_ro()
    (all_chunks_index, atoms_bridge_index, exclude_doc_types,
     chunk_ownership_index, chunk_anchor_index) = run_full.load_indexes(
        run_full.DEFAULT_ATOMS_BRIDGE)
    owner_counts = run_full.owner_count_by_doc(con)

    codes, process_all = _resolve_incremental_codes(con, doc_id=doc_id)
    if doc_id:
        print(f"facts: --doc-id={doc_id} -> {len(codes)} urun etkileniyor")
    elif process_all:
        print(f"facts: FACTS_PROCESS_ALL=1 -- bayatlik kapisi atlaniyor, "
              f"{len(codes)} urun (TUMU) isleniyor")
    else:
        print(f"facts: bayatlik kapisi -> {len(codes)} urun etkileniyor")

    skip_existing = _env_flag("FACTS_SKIP_EXISTING")
    if skip_existing:
        print("facts: FACTS_SKIP_EXISTING=1 -- TAM (`partial` OLMAYAN) "
              "`results/<CODE>.json` dosyasi olan urunler ATLANIYOR")

    return run_full.run_products(
        codes, con=con, provider=provider, model=model, base_url=base_url,
        api_key=api_key, all_chunks_index=all_chunks_index,
        atoms_bridge_index=atoms_bridge_index,
        exclude_doc_types=exclude_doc_types, owner_counts=owner_counts,
        chunk_ownership_index=chunk_ownership_index,
        chunk_anchor_index=chunk_anchor_index, system=system,
        skip_existing=skip_existing)


def stage_load(*, doc_id: str | None = None) -> dict:
    """N-22: runs via the documented programmatic entry point of `load_to_db.py`
    (`main(argv=[...])`). Without this stage connected, `stage_facts` only
    PRODUCED `results/<CODE>.json` -- `run_products` writes NOTHING to specs.db
    (see that function's docstring), so the nightly run changed not even a single
    row in the DB.

    If `--doc <id>` is given it is called with `load_to_db.py --doc-id <id>` --
    the `load_product_incremental` (O-12) path, which processes the products USING
    that doc_id with REPLACE semantics (see `load_to_db.py::load_product_incremental`
    docstring).

    If not given: the SAME product set `stage_facts` processed is RE-resolved with
    `_resolve_incremental_codes()` (facts did not write to specs.db, so the DB state
    did not change; same input gives the same set) and called with `--models <set>
    --force`. **CRITICAL**: without `--force`, `load_to_db.py` SKIPS already-loaded
    products as "already loaded" -- the products passing facts' staleness gate are
    ALREADY loaded every night, so WITHOUT `--force` this stage would write NOTHING
    (the SAME rationale as K-96's "full reset" principle, see `stage_facts` docstring).
    `--force` + `--models`,
    `load_product_incremental`'s doc_id-less branch (`load_product(force=
    True) is EXACTLY EQUIVALENT to `load_product_incremental` (O-12) --
    see that function's docstring. There is NO way to call
    `load_product_incremental` without doc_id.

    If no products are due (the staleness gate selected none),
    `load_to_db.main()` is NEVER called -- empty `--models ""` would error;
    instead `n_products=0` is returned.

    After `load_to_db.main()` returns successfully, `_settle_pending_rework`
    is called -- the pending-rework doc_ids whose products were loaded
    successfully this run drop from the queue (see that function's
    docstring). Both branches (manual `--doc` AND automatic `codes`) do
    this -- both answer the same "did the product load OK?" question.
    The empty-`codes` branch also calls settle
    (2026-08-26 nightly run inspection): orphaned pending records only drop
    this way, even on a night without `load_to_db.main()`."""
    from medrag.pipeline.facts import load_to_db, run_full

    if doc_id:
        print(f"load: --doc-id={doc_id} -> load_to_db.py'ye devrediliyor")
        result = load_to_db.main(argv=["--doc-id", doc_id])
        _settle_pending_rework(run_full._connect_ro(), result)
        return result

    con = run_full._connect_ro()
    codes, process_all = _resolve_incremental_codes(con, doc_id=None)

    if not codes:
        print("load: islenecek urun yok (facts asamasi hicbir urunu bayat bulmadi)")
        # Orphaned pending records (doc_ids where resolve_codes returns EMPTY) can
        # NEVER enter any run's `codes` set -- until settled they stay in the queue
        # forever (see `_settle_pending_rework`'s ownerless-doc note). An empty
        # `reports` is given: the WITH-products pending docs naturally STAY in the
        # queue because they were not attempted; only the ownerless ones drop with
        # 'orphaned'.
        _settle_pending_rework(con, {"reports": []})
        return {"n_products": 0}

    print(f"load: {'FACTS_PROCESS_ALL=1 -- ' if process_all else ''}"
          f"{len(codes)} urun specs.db'ye yazilacak (facts ile AYNI kume)")
    result = load_to_db.main(argv=["--models", ",".join(codes), "--force"])
    _settle_pending_rework(con, result)
    return result


def stage_vectorize(*, doc_id: str | None = None):
    """N-21: the return value is no longer a plain `int` -- `calistir()` (see its
    `VectorizeRunStats` dataclass) also carries `points_written`/`embedding_model`,
    and the nightly report uses those as REAL numbers. `points_deleted`/
    `total_points_after` are NOT in that dataclass (known limitation, see the
    dataclass docstring) -- the report writes both as `None` ("not measured").

    2026-08-27 fix: `vectorize_cli.calistir()` does NOT set up its own
    `logging.basicConfig` (that is only in `main()`, see cli.py) -- since
    `run_nightly.py` calls this function DIRECTLY (not main()) the INFO/ERROR logs
    were written NOWHERE (in the 2026-08-26 run there was not a SINGLE line after
    the `=== vectorize ===` header -- even if QDRANT_URL was missing, the embedder
    could not be built, or no scope was found, it stayed silent). ALSO,
    previously `VectorizeRunStats.exit_code` was NEVER CHECKED -- even with
    `exit_code=2` (config/connection error) or `exit_code=1` (a scope failed) the
    stage returned successfully, `run()`'s `except Exception` catch (see that
    function's N-21 note) never triggered, and the nightly report counted it as a
    PART of `full_success`. Now logging is set up AND `exit_code != 0` is
    explicitly turned into an exception -- so a stage FAILURE shows up in the
    report's `failures[]`, no longer silently swallowed."""
    _doc_noop("vectorize", "kapsam-bazli calisir, dokuman-bazli filtre yok",
              doc_id)
    from medrag.pipeline.vectorize import cli as vectorize_cli
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = vectorize_cli.calistir(env=os.environ)
    if stats.exit_code != 0:
        raise RuntimeError(f"vectorize basarisiz (exit_code={stats.exit_code}) -- "
                           "yukaridaki loglarda ilk ERROR/exception satirina bakin")
    return stats


# --- N-03: dry-run reporters -------------------------------------------
#
# Each writes NO FILE and makes NO NETWORK CALL (Ollama/VLM/Qdrant/embedding) --
# it only reads the current disk state and computes+prints the set "what would
# this stage process if it ran". `run()` calls these instead of `stage_*` when
# dry_run=True (see `_DRY_RUN_FUNC_NAMES` below).


def _dry_run_scan(*, doc_id: str | None = None) -> None:
    _doc_noop("scan", "classify_documents.py dokuman-bazli filtre "
              "desteklemiyor, KARAR-001 -- tum BELGELER agaci taranir", doc_id)
    module = _load_classify_documents()
    # classify_documents.main(dry_run=True): the SAME scan/match/scan_status diff
    # is computed (NEW/UNCHANGED/MODIFIED/MOVED/DELETED), but NOTHING is written to
    # output_path -- this is N-03's "lists the same sets as a real run" acceptance
    # criterion.
    module.main(dry_run=True)


def _dry_run_parse(*, doc_id: str | None = None) -> None:
    _doc_noop("parse", "run_parse_pipeline.py dokuman-bazli filtre "
              "desteklemiyor, tum bekleyen korpus islenir", doc_id)
    from medrag.pipeline.cli import run_parse_pipeline
    run_parse_pipeline._bootstrap()  # only resolves paths/DOC_TYPES from .env, no network call
    _forget_deleted_sources(dry_run=True)
    if run_parse_pipeline.DOC_TYPES:
        print(f"parse: doc_type filter: {', '.join(sorted(run_parse_pipeline.DOC_TYPES))}")
    todo, info = run_parse_pipeline.compute_pending()
    print(f"[dry-run] parse: {len(todo)} / {info['total']} dokuman islenecekti "
          f"({', '.join(r['identity']['file_name'] for r in todo[:5])}"
          f"{', ...' if len(todo) > 5 else ''}) -> YAZILMAYACAK")


def _dry_run_chunk(*, doc_id: str | None = None) -> None:
    """Per KARAR-012, the chunk stage re-processes the WHOLE corpus every run --
    "the set to process" and "all documents in the input folder" are the same thing.
    The input folder is resolved from the SAME two env vars
    (`CHUNKER_INPUT_DIR`/`PARSED_OUTPUT_DIR`) that `run_chunk_pipeline.main()` uses,
    but the subprocess (`python -m medrag.pipeline.chunker`) is NEVER started."""
    _doc_noop("chunk", "run_chunk_pipeline.py her kosuda tum korpusu "
              "yeniden isler (KARAR-012 ikinci surumu)", doc_id)
    from medrag.pipeline.cli import run_chunk_pipeline
    input_raw = os.environ.get("CHUNKER_INPUT_DIR") or os.environ.get("PARSED_OUTPUT_DIR")
    if not input_raw:
        print("[dry-run] chunk: CHUNKER_INPUT_DIR de PARSED_OUTPUT_DIR de "
              "tanimsiz, hesaplanamadi")
        return
    input_dir = run_chunk_pipeline._resolve(input_raw)
    if not input_dir.is_dir():
        print(f"[dry-run] chunk: girdi klasoru yok: {input_dir}")
        return
    doc_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    print(f"[dry-run] chunk: KARAR-012 geregi butun korpus yeniden islenir "
          f"-> {len(doc_dirs)} dokuman klasoru ({input_dir}) -> YAZILMAYACAK")


def _dry_run_ownership(*, doc_id: str | None = None) -> None:
    """Calls NOTHING besides `build_all.split_ownership_work` -- that function's OWN
    docstring (O-06 acceptance criterion) guarantees it never IMPORTS/CALLS
    `_chat_ollama`/`_tag_chunk_llm`, so an LLM/Ollama request here is STRUCTURALLY
    impossible -- no need for a mock/monkeypatch discipline either."""
    _doc_noop("ownership", "build_all.py dokuman-bazli filtre desteklemiyor, "
              "tum korpus (tek+cok-sahipli dokuman) taranir", doc_id)
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    con = build_all._connect_ro()
    try:
        config = build_all.load_config()
        exclude_doc_types = frozenset(config.scope.exclude_doc_types)
        docs = build_all._load_documents(con, exclude_doc_types)
        primary = build_all._load_chunks_from_all_chunks_json()
        fallback = build_all._load_chunk_bridge()

        def _chunks_for(did: str):
            return primary.get(did) or fallback.get(did)

        all_codes: set[str] = set()
        for d in docs:
            all_codes.update(d["product_codes"])
        product_info = build_all._load_product_info(con, all_codes)
        _rows, _missing, llm_jobs, single_docs, multi_docs = build_all.split_ownership_work(
            docs, _chunks_for, product_info, verbose=False)
    finally:
        con.close()

    print(f"[dry-run] ownership: {len(single_docs)} tek-sahipli dokuman (LLM'siz, "
          f"deterministik), {len(multi_docs)} cok-sahipli dokuman -> {len(llm_jobs)} "
          "chunk LLM'e gonderilecekti -> YAZILMAYACAK")


def _dry_run_facts(*, doc_id: str | None = None) -> None:
    """Calls NOTHING besides `resolve_codes` -- `run_products` (and therefore an
    LLM/Ollama request) never triggers."""
    from dotenv import load_dotenv

    from medrag.pipeline.facts import run_full

    load_dotenv(run_full.ENV_PATH)
    con = run_full._connect_ro()
    codes = run_full.resolve_codes(con, doc_id=doc_id)
    print(f"[dry-run] facts: {len(codes)} urun islenecekti"
          + (f" (--doc-id={doc_id})" if doc_id else "")
          + f" ({', '.join(codes[:5])}{', ...' if len(codes) > 5 else ''})"
          + " -> YAZILMAYACAK")


def _dry_run_load(*, doc_id: str | None = None) -> None:
    """Calls NOTHING besides `_resolve_incremental_codes` -- `load_to_db.main` (and
    therefore REAL writing to specs.db) never triggers. DIFFERENT from the `facts`
    dry-run report (which simply uses `resolve_codes`), here the REAL result of the
    K-96 staleness gate is used -- it reflects the set `stage_load` would process in
    a real run MORE ACCURATELY."""
    from medrag.pipeline.facts import run_full

    if doc_id:
        print(f"[dry-run] load: --doc-id={doc_id} -> load_to_db.py --doc-id "
              "yoluna gidecekti -> YAZILMAYACAK")
        return

    con = run_full._connect_ro()
    codes, process_all = _resolve_incremental_codes(con, doc_id=None)
    print(f"[dry-run] load: {'FACTS_PROCESS_ALL=1 -- ' if process_all else ''}"
          f"{len(codes)} urun specs.db'ye yazilacakti "
          f"({', '.join(codes[:5])}{', ...' if len(codes) > 5 else ''}) -> YAZILMAYACAK")


def _dry_run_vectorize(*, doc_id: str | None = None) -> None:
    """The embedder/Qdrant client is NEVER built (network-call risk) -- only the
    chunk scopes on disk are counted; which are stale (the staleness state
    comparison depends on the embedder) is OUT OF SCOPE for this report (known
    limitation)."""
    _doc_noop("vectorize", "kapsam-bazli calisir, dokuman-bazli filtre yok",
              doc_id)
    from medrag.pipeline.vectorize.discover import discover_chunk_scopes

    input_dir_ham = os.environ.get("VECTORIZE_INPUT_DIR")
    if not input_dir_ham:
        print("[dry-run] vectorize: VECTORIZE_INPUT_DIR tanimsiz, hesaplanamadi")
        return
    burasi = Path(__file__).resolve().parent.parent
    input_dir = (burasi / input_dir_ham).resolve() if not Path(input_dir_ham).is_absolute() \
        else Path(input_dir_ham)
    kapsamlar = discover_chunk_scopes(input_dir)
    print(f"[dry-run] vectorize: {len(kapsamlar)} kapsam bulundu ({input_dir}); "
          "hangisinin bayat oldugu (gercekten islenecek alt kume) embedder "
          "kurulumu gerektirir, bu rapor kapsami DEGIL -> YAZILMAYACAK")


# name -> module-level function name (NOT the function OBJECT): `run()` resolves
# this via `globals()` at each call, so when tests monkeypatch/Mock a name like
# `run_nightly.stage_parse` `run()` really sees the patched version (if it were a
# dict frozen at import time the patch would be invisible).
_STAGE_FUNC_NAMES = {
    "scan": "stage_scan",
    "parse": "stage_parse",
    "chunk": "stage_chunk",
    "ownership": "stage_ownership",
    "facts": "stage_facts",
    "load": "stage_load",
    "vectorize": "stage_vectorize",
}

# N-03: on dry_run=True its counterpart is called instead of each `stage_*` --
# the same `globals()` resolution pattern, the same reason (tests can monkeypatch
# single functions).
_DRY_RUN_FUNC_NAMES = {
    "scan": "_dry_run_scan",
    "parse": "_dry_run_parse",
    "chunk": "_dry_run_chunk",
    "ownership": "_dry_run_ownership",
    "facts": "_dry_run_facts",
    "load": "_dry_run_load",
    "vectorize": "_dry_run_vectorize",
}


def _stage_index(name: str) -> int:
    try:
        return STAGES.index(name)
    except ValueError:
        raise SystemExit(
            f"bilinmeyen asama: {name!r} (gecerli: {', '.join(STAGES)})"
        ) from None


def plan(*, stage: str | None = None, from_stage: str | None = None) -> list[str]:
    """The list of stages to run. `--stage`/`--from` cannot be given together
    (one means "only this", the other "from here to the end" -- both at once is
    meaningless)."""
    if stage and from_stage:
        raise SystemExit("--stage ve --from birlikte verilemez")
    if stage:
        _stage_index(stage)
        return [stage]
    if from_stage:
        i = _stage_index(from_stage)
        return STAGES[i:]
    return list(STAGES)


def run(*, stage: str | None = None, from_stage: str | None = None,
        doc_id: str | None = None, dry_run: bool = False,
        collect: dict | None = None) -> list[str]:
    """The orchestrator's body. Returns the list of planned stages (also useful for
    tests to verify). There is NO reference inside this function to any `stage_*`
    outside the plan -- only the names in the `asamalar` list are resolved from
    `globals()` and called.

    N-21: if `collect` (default `None`) is given, each completed stage's return
    value is written to `collect["stages"][ad]` -- this is the ONLY real data
    source of the nightly report (see `_build_nightly_report`). If a stage CRASHES
    it is written to `collect["failure"]` (stage/reason/error_text) and the
    exception is STILL RAISED (not swallowed) -- the caller (`main()`) catches it
    and writes the PARTIAL report anyway, satisfying K-62's "if it hits the ceiling
    it stops in a controlled way and reports partially" intent. `collect=None`
    (the default, the path ALL existing tests use) changes behavior NOT AT ALL --
    the exception raises directly to the caller, as before."""
    asamalar = plan(stage=stage, from_stage=from_stage)
    print(f"medrag-nightly: plan = {' -> '.join(asamalar)}"
          + (f" (doc={doc_id})" if doc_id else ""))
    if dry_run:
        print("--dry-run: hicbir asama gercekten calistirilmiyor, yalniz "
              "her asamanin isleyecegi kume hesaplanip raporlanacak (N-03)")
        for name in asamalar:
            fn = globals()[_DRY_RUN_FUNC_NAMES[name]]
            print(f"\n=== {name} (dry-run) ===")
            fn(doc_id=doc_id)
        return asamalar
    for name in asamalar:
        fn = globals()[_STAGE_FUNC_NAMES[name]]
        print(f"\n=== {name} ===")
        try:
            result = fn(doc_id=doc_id)
        except Exception as exc:
            if collect is not None:
                collect["failure"] = {
                    "stage": name, "reason": type(exc).__name__, "error_text": str(exc),
                }
            raise
        if collect is not None:
            collect.setdefault("stages", {})[name] = result
    return asamalar


def _scan_paths_by_status(scan_output: dict) -> dict[str, list[str]]:
    """Groups the `output["documents"]` that `stage_scan` returns by `scan_status`
    -- `location.rel_path` (else `identity.file_name`) is used as the PATH label.
    If `scan_output` is empty/None (scan did not run this run) all groups are empty
    -- the caller reads that, together with an accompanying `FailureEntry`, as
    "not measured"."""
    out: dict[str, list[str]] = {}
    for rec in (scan_output or {}).get("documents", []):
        status = rec.get("scan", {}).get("scan_status")
        etiket = rec.get("location", {}).get("rel_path") or rec.get("identity", {}).get("file_name", "?")
        out.setdefault(status, []).append(etiket)
    return out


def _build_nightly_report(
    *, collect: dict, started_at, finished_at, backup_manifest, restore_note: str = "",
):
    """N-21: converts the RAW stage results in `collect` (see `run()`) into a
    single `NightlyReport`. Does not INVENT NUMBERS -- if a stage never ran (not in
    collect) that section's numeric fields fall to their empty/0 default (list/dict
    fields) or to the model's already-Optional (`None` = not measured) fields (see
    the section docstrings in `nightly_report.py` -- model_call_count,
    points_deleted/total_points_after are STILL never computed).
    N-22: both ends of `ownership` and `product_features.features_added` are now
    REALLY computed when `stage_ownership`/`stage_load` are connected to the chain
    (the relevant block below) -- `evidence_added`/`evidence_removed`/
    `features_removed_for_no_evidence` are STILL not measured, `forget_source` is
    not used in this flow)."""
    from medrag.pipeline.nightly_report import (
        ChunkSection,
        FailureEntry,
        IntegritySection,
        NightlyReport,
        OwnershipSection,
        ParseSection,
        ProductFeaturesSection,
        SkippedSection,
        SourceDiffSection,
        VectorizeSection,
    )

    stages = collect.get("stages", {})
    failure = collect.get("failure")

    by_status = _scan_paths_by_status(stages.get("scan"))
    source_diff = SourceDiffSection(
        changed=by_status.get("MODIFIED", []), added=by_status.get("NEW", []),
        removed=by_status.get("DELETED", []), moved=by_status.get("MOVED", []),
    )
    skipped = SkippedSection(
        unchanged_count=len(by_status.get("UNCHANGED", [])),
        unchanged_paths=by_status.get("UNCHANGED", []),
    )

    parse_out = stages.get("parse") or {}
    parse = ParseSection(
        processed_count=parse_out.get("processed_count") or 0,
        duration_seconds=parse_out.get("duration_seconds") or 0.0,
        model_call_count=None,
    )

    chunk_out = stages.get("chunk") or {}
    chunk = ChunkSection(
        produced_count=chunk_out.get("produced_count") or 0,
        per_document_counts=chunk_out.get("per_document_counts") or {},
        removed_count=chunk_out.get("removed_count") or 0,
    )

    vec_stats = stages.get("vectorize")
    vectorize = VectorizeSection(
        points_written=getattr(vec_stats, "points_written", 0) or 0,
        points_deleted=None,
        total_points_after=None,
        embedding_model=getattr(vec_stats, "embedding_model", None) or "olculmedi",
    )

    # N-22: `ownership` is now connected to the chain -- the REAL chunk numbers
    # are derived from the `build_all.main()` output (meta + rows) that
    # `stage_ownership` returns. If `stages.get("ownership")` is `None`/empty (the
    # stage never ran, was killed by Ctrl+C, or a path like `--stage`/`--from` did
    # not include it in the PLAN) NO NUMBER IS INVENTED, all three stay None --
    # "not measured" is NOT confused with "there are no unresolved chunks".
    ownership_out = stages.get("ownership")
    ownership = OwnershipSection(deterministic_count=None, model_routed_count=None,
                                 unresolved_count=None)
    if isinstance(ownership_out, dict) and not ownership_out.get("interrupted") \
            and isinstance(ownership_out.get("rows"), list):
        rows = ownership_out["rows"]
        deterministic_count = sum(1 for r in rows if r.get("source") == "deterministic")
        llm_chunk_ids: set[tuple] = set()
        accepted_chunk_ids: set[tuple] = set()
        for r in rows:
            if r.get("source") != "llm":
                continue
            key = (r.get("doc_id"), r.get("chunk_id"))
            llm_chunk_ids.add(key)
            if r.get("accepted"):
                accepted_chunk_ids.add(key)
        ownership = OwnershipSection(
            deterministic_count=deterministic_count,
            model_routed_count=ownership_out.get("meta", {}).get("n_llm_chunks", len(llm_chunk_ids)),
            unresolved_count=len(llm_chunk_ids - accepted_chunk_ids),
        )

    # N-22: `load` is now connected to the chain -- the `n_present` in the
    # `load_to_db.main()` summary that `stage_load` returns (the number of spec_value
    # rows written with `present` status this run) maps DIRECTLY to "features added"
    # (`features_added`) (`load_product_incremental` first returns every product to
    # not_specified, then re-writes it, so this run's `n_present` is the number of
    # features that are present in it). The OTHER three fields
    # (evidence_added/evidence_removed/features_removed_for_no_evidence) are NOT in
    # `load_to_db.py`'s report -- they need `forget_source` (NOT USED in this flow,
    # see `load_product_incremental` docstring); NO NUMBER IS INVENTED, all three
    # stay None.
    load_out = stages.get("load")
    product_features = ProductFeaturesSection(
        features_added=load_out.get("n_present") if isinstance(load_out, dict) else None,
        evidence_added=None, evidence_removed=None,
        features_removed_for_no_evidence=None,
    )

    # N-10 fix (2026-08-26): before, `failures` only filled when a stage raised an
    # exception -- the individual failures INSIDE a stage (a product's `load_to_db`
    # write died, a document's parse FAILED) never REACHED the report because the
    # stage itself returned SUCCESSFULLY (no exception, only an error row in its
    # report) -- in the 2026-08-26 run 34 products could not be written + 1 document
    # parse FAILED yet the report said `full_success`. In-stage failures are added
    # here, BEFORE the top-level exception check, as separate rows.
    failures: list[FailureEntry] = []

    for failed_doc in parse_out.get("failed_docs") or []:
        failures.append(FailureEntry(
            file=failed_doc.get("file_name") or failed_doc.get("doc_id", "-"),
            stage="parse", reason="document_parse_failed",
            error_text=f"{failed_doc.get('doc_id', '?')}: "
                       f"{failed_doc.get('error') or 'parse.status=FAILED'}",
        ))

    for product_report in (stages.get("load") or {}).get("reports") or []:
        if product_report.get("error"):
            failures.append(FailureEntry(
                file=product_report.get("model_code", "-"), stage="load",
                reason="product_write_failed", error_text=product_report["error"],
            ))

    if failure is not None:
        failures.append(FailureEntry(
            file="-", stage=failure["stage"], reason=failure["reason"],
            error_text=failure["error_text"],
        ))
        # the planned stages AFTER the failed stage never ran -- a separate record,
        # so "0 measured" is not confused with "never ran".
        idx = STAGES.index(failure["stage"])
        for name in STAGES[idx + 1:]:
            failures.append(FailureEntry(
                file="-", stage=name, reason="skipped_upstream_failure",
                error_text=f"{failure['stage']} basarisiz oldugu icin bu asamaya hic gelinmedi",
            ))

    integrity = IntegritySection(
        # O-08 integrity gate (verify_db_integrity) is NOT YET connected to
        # run_nightly.py -- the SAME convention N-05 uses for the backup_completed
        # field in the same situation: False = "not yet integrated", NOT "check ran
        # and FAILED" (see IntegritySection docstring).
        pre_write_check_passed=False, post_write_check_passed=False,
        backup_completed=True, backup_location=str(backup_manifest.root),
        notes=("O-08 butunluk kapisi henuz run_nightly'e baglanmadi (ayri gorev, "
               "N-21 kapsami disi)" + (f" | {restore_note}" if restore_note else "")),
    )

    return NightlyReport(
        run_id=started_at.strftime("%Y-%m-%dT%H-%M-%SZ"),
        started_at=started_at, finished_at=finished_at,
        source_diff=source_diff, skipped=skipped, parse=parse, chunk=chunk,
        vectorize=vectorize, ownership=ownership, product_features=product_features,
        failures=failures, integrity=integrity,
    )


def _nightly_email_subject(report) -> str:
    return f"[medrag-nightly] {report.run_id} -- {report.outcome()}"


def build_nightly_email_summary(report) -> str:
    """Body text -- a few lines of numeric summary, NOT the report (per
    operator's preference: short numeric summary, not a wall of prose).
    The full report is attached as `.md` (see `_send_nightly_email`).

    K-101: `sd.removed` (`scan_status=DELETED`) is STICKY -- it means
    "every record currently DELETED", not "deleted tonight" (may have
    accumulated over previous nights). The label says so explicitly so a
    reader doesn't misread "638 removed" as "638 files deleted tonight" --
    the exact confusion that caused the K-101 incident."""
    sd, sk, ch, ve = report.source_diff, report.skipped, report.chunk, report.vectorize
    pf = report.product_features
    return (
        f"Outcome: {report.outcome()}\n"
        f"Kaynak: +{len(sd.added)} yeni, {len(sd.changed)} degisen, "
        f"{len(sd.removed)} DELETED (sistemde toplam, sticky -- bu gece degil), "
        f"{len(sd.moved)} tasinan, {sk.unchanged_count} degismedi\n"
        f"Chunk: {ch.produced_count} produced, {ch.removed_count} removed\n"
        f"Vector: {ve.points_written} points written\n"
        f"Product features: {pf.features_added if pf.features_added is not None else 'not measured'} added\n"
        f"Failures: {len(report.failures)}\n"
    )


def build_nightly_email(*, report, markdown_text: str, md_path: Path,
                        from_addr: str, to_addrs: list[str]):
    """Builds the `EmailMessage` to send (no network call, pure -- kept
    separate from `_send_nightly_email` so tests can verify without SMTP).

    Body is ALWAYS a short numeric summary (`build_nightly_email_summary`) --
    the full report is NEVER embedded in the body, it's always a `.md`
    attachment."""
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = _nightly_email_subject(report)
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.set_content(build_nightly_email_summary(report))
    msg.add_attachment(markdown_text.encode("utf-8"), maintype="text",
                      subtype="plain", filename=md_path.name)
    return msg


def _send_nightly_email(*, report, markdown_text: str, md_path: Path) -> None:
    """Nightly report email (repo convention: settings in env):

        NIGHTLY_REPORT_EMAIL_TO     comma-separated recipient list. EMPTY/
                                    UNSET -- email is NEVER sent (default off,
                                    no surprise mail).
        NIGHTLY_REPORT_EMAIL_FROM   sender address. Default: nightly@<hostname>.
        SMTP_HOST / SMTP_PORT       default 127.0.0.1 / 25 (local postfix relay).

    Even if mail delivery FAILS, the nightly run is NOT brought down -- the report
    was already written to disk (JSON+MD), the mail is only a notification channel;
    an SMTP error is SWALLOWED here, only a warning is printed."""
    to_addrs_raw = (os.environ.get("NIGHTLY_REPORT_EMAIL_TO") or "").strip()
    if not to_addrs_raw:
        print("nightly-email: NIGHTLY_REPORT_EMAIL_TO tanimsiz, mail atlaniyor")
        return
    to_addrs = [a.strip() for a in to_addrs_raw.split(",") if a.strip()]

    import socket

    from_addr = os.environ.get("NIGHTLY_REPORT_EMAIL_FROM") or f"nightly@{socket.gethostname()}"
    smtp_host = os.environ.get("SMTP_HOST") or "127.0.0.1"
    smtp_port = int(os.environ.get("SMTP_PORT") or "25")

    msg = build_nightly_email(
        report=report, markdown_text=markdown_text, md_path=md_path,
        from_addr=from_addr, to_addrs=to_addrs,
    )

    import smtplib

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.send_message(msg)
        print(f"nightly-email: gonderildi -> {', '.join(to_addrs)} "
              f"(ozet govdede, tam rapor ek olarak -- {md_path.name})")
    except OSError as exc:
        print(f"nightly-email: gonderilemedi ({exc}) -- rapor diskte kaldi ({md_path})")


def _run_full_chain_with_report(*, doc_id: str | None, backup_manifest) -> None:
    """N-21: wraps `run()` for the full chain (NO `--stage`/`--from`) -- whether the
    run completes successfully OR a stage CRASHES halfway, IN EITHER CASE a report
    (JSON + markdown) is written (K-62 intent: "if it hits the ceiling it stops in
    a controlled way and reports partially" -- before, when a stage crashed it just
    exited with a traceback, leaving NO trace at all).

    Scope DELIBERATELY kept narrow: only the full chain (NO `--stage`/`--from`)
    produces a report -- expecting a report after K-60's "manual intervention" tools
    (like `--stage facts`) is OUTSIDE this task's (N-21) scope, see the end note.
    `--dry-run` NEVER reaches this function (N-03: it writes no data, `main()` returns
    EARLY on dry-run).

    N-25: if a stage CRASHES halfway, BEFORE the report is written it returns to
    this run's own backup (`backup_manifest.root`) via `restore_backup()` --
    EVERYTHING that run wrote so far (including the successfully finished stages)
    is rolled back, so no "halfway but progressed" broken state REMAINS on disk (see
    `nightly_backup.py`'s N-25 note -- the only protection; it does not collide with
    K-63 because the backed-up-to state is ALWAYS the one BEFORE this run started).
    If the restore itself crashes (`RestoreError`) it is NOT swallowed -- the data may
    now be in an undefined state, this must stop LOUDLY."""
    from datetime import UTC, datetime

    from medrag.core.paths import resolve_reports_dir
    from medrag.pipeline.cli.nightly_backup import restore_backup
    from medrag.pipeline.nightly_report import render_markdown, save_nightly_report

    started_at = datetime.now(UTC)
    collect: dict = {}
    stage_exc: Exception | None = None
    restore_note = ""
    try:
        run(doc_id=doc_id, dry_run=False, collect=collect)
    except Exception as exc:  # noqa: BLE001 -- kismi bile olsa RAPOR yazilsin, sonra yeniden firlatilir
        stage_exc = exc
        print(f"medrag-nightly: asama patladi ({type(exc).__name__}: {exc}) -- "
              f"kosu-oncesi yedege ({backup_manifest.root}) geri donuluyor")
        restored = restore_backup(backup_manifest.root)
        restore_note = (f"asama basarisiz oldugu icin bu kosunun yazdigi veri "
                         f"{backup_manifest.root} yedegine GERI ALINDI "
                         f"({len(restored)} kalem)")
        print(f"medrag-nightly: geri yukleme tamam -- {restore_note}")
    finished_at = datetime.now(UTC)

    report = _build_nightly_report(collect=collect, started_at=started_at,
                                    finished_at=finished_at, backup_manifest=backup_manifest,
                                    restore_note=restore_note)
    reports_dir = resolve_reports_dir()
    json_path = save_nightly_report(report, reports_dir=reports_dir)
    md_path = json_path.with_suffix(".md")
    markdown_text = render_markdown(report)
    md_path.write_text(markdown_text, encoding="utf-8")
    print(f"medrag-nightly: rapor yazildi ({json_path}, {md_path}) -- "
          f"outcome={report.outcome()}")

    # K-101: `build_nightly_email_summary` used to only go to the email body.
    # When the run was triggered via cron's MAILTO (which posts the process's
    # raw stdout without SMTP), the short summary was never visible, and the
    # recipient had to read hundreds of lines of raw log (including the
    # forget_deleted_source dump). The summary is now ALSO printed to the
    # console INDEPENDENTLY of the SMTP path -- whatever channel posts the
    # log, a searchable summary block sits at the very end.
    print("\n=== SUMMARY ===")
    print(build_nightly_email_summary(report).rstrip())
    print("=== END SUMMARY ===\n")

    # Whether the run finishes cleanly OR a stage CRASHES mid-run -- in
    # BOTH cases the email is tried (report is already on disk, before
    # `stage_exc` is re-raised below so failed nights still notify).
    _send_nightly_email(report=report, markdown_text=markdown_text, md_path=md_path)

    if stage_exc is not None:
        # Traceback printed EXPLICITLY: `SystemExit` does NOT print a
        # traceback (Python default), and the `from stage_exc` chain is
        # also invisible. The report's `reason`/`error_text` carries the
        # exception TEXT but not the traceback -- two separate incidents
        # (VLM canary,
        # facts'in mkdir'i) teshisi dogrudan log'daki traceback'in son
        # last line in the log. Losing the traceback means gaining the
        # report but losing diagnosis.
        import traceback
        traceback.print_exception(type(stage_exc), stage_exc, stage_exc.__traceback__)
        raise SystemExit(1) from stage_exc


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="medrag-nightly", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="run only this stage (manual intervention)")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default=None,
                    help="run this stage through the end of the chain")
    ap.add_argument("--doc", dest="doc_id", default=None,
                    help="only this doc_id (currently only effective in the "
                         "facts stage, passed as --doc-id)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run no stage")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_argv(argv)
    if args.dry_run:
        # N-03: a dry run writes no data -- no need for the lock (N-02); running
        # concurrently with a real run is harmless.
        run(stage=args.stage, from_stage=args.from_stage, doc_id=args.doc_id,
            dry_run=True)
        return

    from medrag.pipeline.cli.nightly_backup import (
        BackupError,
        RestoreError,
        clear_in_progress,
        mark_in_progress,
        read_in_progress,
        restore_backup,
        run_backup,
    )
    from medrag.pipeline.cli.nightly_lock import NightlyLock, NightlyLockHeld

    # N-21: the report isn't generated for the full chain only (NO --stage/--from)
    # -- see `_run_full_chain_with_report`'s docstring (scope deliberately narrow,
    # K-60's manual-intervention tools don't expect a report).
    is_full_chain = args.stage is None and args.from_stage is None

    try:
        with NightlyLock() as lock:
            # N-25: the marker file is placed NEXT TO the lock file (shares the same
            # `NIGHTLY_LOCK_PATH` resolution) -- no need to manage a separate env
            # var; the `NIGHTLY_LOCK_PATH` tests already set also isolates the marker
            # automatically.
            inprogress_path = lock.path.with_name(lock.path.stem + ".inprogress.json")

            # If the previous run did NOT exit cleanly (kill -9/crash -- the process
            # died before any except/finally could run) an "in progress" marker stays
            # on disk (see nightly_backup.py's N-25 note). Check this BEFORE taking a
            # NEW backup: if it exists, return to that run's own backup -- otherwise
            # this run would lay on top of the previous run's HALF/BROKEN data.
            pending = read_in_progress(path=inprogress_path)
            if pending is not None:
                prev_root = pending.get("backup_root")
                print(f"medrag-nightly: onceki kosu TEMIZ CIKMAMIS (kill/cokme "
                      f"supheli) -- yarim kalan veri, o kosunun yedegine "
                      f"({prev_root}) geri donuluyor")
                restore_backup(prev_root)
                clear_in_progress(path=inprogress_path)
                print("medrag-nightly: onceki yarim kosunun verisi geri yuklendi")

            # N-05 (I-40): no writes START before a successful backup is taken --
            # right AFTER the lock, BEFORE the first `stage_*`. If the backup fails
            # (BackupError) no `stage_*` is CALLED and the night must be reported
            # *failed* (K-63's only protection -- the single safety net against the
            # nightly run overwriting manually-corrected data).
            manifest = run_backup()
            print(f"medrag-nightly: yedek tamam ({manifest.root}, "
                  f"{manifest.total_size_bytes} byte, "
                  f"{manifest.duration_seconds:.1f}s)")
            # N-25: write our marker right before starting to call `stage_*` -- if
            # the process disappears via kill/crash after this point, the next call
            # sees this marker above and does the restore.
            mark_in_progress(manifest.root, path=inprogress_path)
            try:
                if is_full_chain:
                    _run_full_chain_with_report(doc_id=args.doc_id, backup_manifest=manifest)
                else:
                    run(stage=args.stage, from_stage=args.from_stage,
                        doc_id=args.doc_id, dry_run=False)
            except RestoreError:
                # The restore itself CRASHED -- the data is in an undefined state;
                # the marker is DELIBERATELY left on disk so the next call retries
                # the recovery; swallowing/clearing it here would make this protection
                # single-use.
                print("medrag-nightly: GERI YUKLEME BASARISIZ -- veri belirsiz "
                      "durumda, isaretci elle mudahale icin diskte birakildi")
                raise
            except BaseException:
                # The process is STILL alive, exiting in a controlled way (a stage
                # error that was already rolled back -- SystemExit(1) -- or the
                # `--stage`/`--from` manual-intervention path's own error, where no
                # automatic restore happens under K-60 but this is NOT a crash):
                # clear the marker, then re-raise WHATEVER exception it is, unchanged.
                clear_in_progress(path=inprogress_path)
                raise
            else:
                clear_in_progress(path=inprogress_path)
    except NightlyLockHeld as exc:
        print(f"medrag-nightly: {exc}")
        raise SystemExit(1) from exc
    except BackupError as exc:
        print(f"medrag-nightly: BASARISIZ -- yedek alinamadi, hicbir asama "
              f"calistirilmadi: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
