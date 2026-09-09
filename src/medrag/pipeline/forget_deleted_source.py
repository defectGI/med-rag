"""N-06 (I-08): the shared "forget the source" top-level function that removes
ALL derivatives of a deleted source from the system.

When the scanner (`chatbot-corpus/document_info/classify_documents.py`) sees a
file disappear from disk it does NOT delete the record, it leaves it sticky with
`scan_status=DELETED` (doc_id is PRESERVED -- a schema decision). But no stage
cleaned up the derivatives that document had produced before; this module fills
that gap.

Instruction (N-06 + O-11 docstring): I-08 (chunk/vector side) and I-17 (spec
evidence side) are the two ends of the SAME "forget the source" concept; two
separate delete logics are NOT written. So for the spec side this calls
`medrag.pipeline.facts.forget_source.forget_source` (O-11, K-70/K-78) DIRECTLY
-- it does not repeat its OWN DELETE/UPDATE logic; it only adds the THREE pieces
O-11 does not cover (parse output, chunk record, Qdrant points):

  1. the parse output folder -- `PARSED_OUTPUT_DIR/<doc_type>_<doc_id>/` (see
     `pipeline/cli/run_parse_pipeline.py::_group_dir`, SAME naming).
     If `doc_type` is unknown (or changed during scanning) the folder is GLOBbed
     by the `_<doc_id>` suffix -- to guard against naming drift.
  2. the `documents[doc_id]` record in `all_chunks.json` -- the file is written
     back with the same atomic write-then-replace pattern `run_chunk_pipeline.py`
     uses (a half-write does not corrupt the file).
  3. Qdrant points -- `vector_store.replace_scope(doc_id, [])` (the mechanism
     already exists, KARAR-004: an empty list = delete only, do not upsert).

Idempotency: each step silently ACCEPTS its own absence -- if the folder does
not exist it is not added to `removed`, if there is no record in `all_chunks.json`
it returns `False`, Qdrant's `replace_scope` deletes by a filter (no-op if no
matching point), `forget_source` touches no row if there is no matching
`evidence[]`. So this function can be CUT OFF MIDWAY AND RE-RUN -- the second
call yields the SAME result (empty change) as the first, and raises no error.

Layer rule: this module may depend on `medrag.core` and imports
`medrag.pipeline.facts` within its own package (in-pipeline, not a violation)
but NEVER imports `medrag.api`.
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
    """Same signature as `QdrantVectorStore.replace_scope` -- so a fake recorder
    can be passed in a test without a real Qdrant connection."""

    def replace_scope(self, scope_id: str, points: list[Any]) -> None: ...


@dataclass
class ForgetDeletedSourceResult:
    """Reports what each step did -- N-09's "deleted source derivatives" row and
    the tests read this."""

    doc_id: str
    parse_dirs_removed: list[str] = field(default_factory=list)
    chunk_entry_removed: bool = False
    qdrant_scope_cleared: bool = False
    spec_evidence: dict[str, Any] = field(default_factory=dict)


def _remove_parse_output(parsed_output_dir: Path, doc_id: str,
                          doc_type: str | None) -> list[str]:
    """Deletes the `PARSED_OUTPUT_DIR/<doc_type>_<doc_id>/` folder (if present).
    If `doc_type` is not given, or the recorded value does not match the old
    folder on disk, ALL folders with the `_<doc_id>` suffix are found and deleted
    (more than one match is normally not expected, but the delete must not be
    left HALF-done)."""
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
    """Write-then-replace -- SAME pattern as `run_chunk_pipeline.py::_atomic_write_json`
    (a half-write does not corrupt the file)."""
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _remove_from_all_chunks(all_chunks_path: Path, doc_id: str) -> bool:
    """Drops the `documents[doc_id]` record of `all_chunks.json`. If the file is
    missing, corrupt, or the record is already absent (idempotent second call) it
    silently returns `False` -- does NOT raise (same principle as chunk_store.py's
    error policy: derivative cleanup is an ADDITION, it must work without the
    corpus file)."""
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
    """Deletes ALL derivatives of one `doc_id`: parse output + chunk record +
    Qdrant points + spec evidence (the full I-08 acceptance scope).

    The caller (the nightly run's N.2 delete flow) calls this once for each
    `scan_status=DELETED` record; being idempotent, it can be cut off midway and
    re-run (see the module docstring)."""
    parse_dirs_removed = _remove_parse_output(Path(parsed_output_dir), doc_id, doc_type)
    chunk_entry_removed = _remove_from_all_chunks(Path(all_chunks_path), doc_id)
    # KARAR-004: an empty list = delete ALL the scope's old points, do not upsert.
    # If no matching point exists this is a no-op (idempotent).
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
