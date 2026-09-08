"""`doc_id` -> the real file on disk.

This is the ONLY place that READS the `file_path` column of the `document`
table (see facts/db/DATA_DICTIONARY.md, "CRITICAL RULE -- NEVER expose
file_path"). The retrieval path that feeds the answering model
(`flows/doc_download.py::DocumentLookup`) never sees this column -- the
`DocumentRow` type doesn't carry it (a type-level guarantee, see that
module's docstring). Only THIS module + `webapp.py`'s
`GET /api/documents/<doc_id>` endpoint read `file_path` and turn it into
bytes; every channel reaching the user/model shows ONLY `file_name`.

Design note: `document`'s PK is `doc_id` ALONE -- one physical file = one
row, and the fan-out lives in `document_owner` (M:N). Historically this query
could meet several rows via a `(doc_id, model)` composite PK; now `doc_id`
always means a SINGLE row.

This is DIFFERENT from the image-serving endpoint's ARCHITECTURAL EXCEPTION
(`image_store.py`): there, reaching parser's OWN storage
(`STORAGE_IMAGES_DIR`) required INVENTING a separate chatbot env variable
(`CHATBOT_IMAGE_STORAGE_DIR`) because image_id is content-addressed and the
full path is not in the DB. Here NO new storage-access env is invented:
`document.file_path` already stores the full path inside chatbot's OWN
`specs.db` (the same `CHATBOT_DB_QUERY_DB_PATH` sql_topn uses; it was
populated by `pipeline/facts/build_facts_db.py` from `location.file_path` in
`document_nodes.json`). So document download REUSES the same storage-access
pattern/env (one env-var-gated, code-import-free file access) and the resolver
with bundled fallback that sql_topn uses (`factory.py::
resolve_db_path_from_env`) -- no second env variable invented.

NO `parser`/`chunker`/`vectorize` CODE IS IMPORTED -- only stdlib `sqlite3`
against chatbot's OWN `specs.db`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from medrag.core.db.sqlite import connect as _sqlite_connect


@dataclass(frozen=True)
class DocumentDownload:
    path: Path
    file_name: str


#: Corpus root (`BELGELER/`). The name the pipeline side ALREADY uses
#: (`compose.yaml`: `BELGELER_DIR`) -- NO new env is invented here, same
#: principle as the module docstring's "no second env variable invented".
_BELGELER_DIR_ENV = "BELGELER_DIR"


def _corpus_root() -> Path | None:
    raw = (os.getenv(_BELGELER_DIR_ENV) or "").strip()
    return Path(raw) if raw else None


def resolve_document_download(db_path: str, doc_id: str) -> DocumentDownload | None:
    """`doc_id` (PK on its own) -> the `is_active=1` row's file + `file_name`.
    `doc_id` is only ever a bound value of a parameterized SELECT (never
    joined in as part of a file path) -- path traversal does NOT require the
    hex-regex guard used in `image_store.py`, because the real path on disk is
    never DERIVED from `doc_id`; it is read from trusted DB columns.

    **Path resolution is two-step.** `document.file_path` is an ABSOLUTE path
    and `build_facts_db.py` fills it from the scan-time path in
    `document_nodes.json` -- i.e. the DB carries the path of whatever machine
    it was built on. Manually editing that column is NOT the fix: `specs.db`
    is a derived product and `build_facts_db.py` rebuilds it from scratch on
    every run -- a hand-written value disappears at the next build.

    So the machine-independent path is tried FIRST: `BELGELER_DIR` (corpus
    root) + `document.rel_path` (already in the same table, `build_facts_db.py`
    writes it too; its separator is `/` on every platform, see
    `classify_documents.py` `rel_key`). If the root is unset or that file
    doesn't exist on disk, it FALLS BACK to the old behavior (`file_path`) --
    nothing changes on dev machines or deployments without `BELGELER_DIR`.

    Missing row / file not on disk -> `None` (webapp.py turns it into 404)."""
    if not doc_id:
        return None
    conn = _sqlite_connect(db_path, readonly=True)
    try:
        # `rel_path` may be absent in old/trimmed schemas -- its absence is
        # NOT an error, it just disables the first step.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(document)")}
        selected = "file_path, file_name" + (", rel_path" if "rel_path" in columns else "")
        row = conn.execute(
            # Column names are CODE CONSTANTS (the `selected` above); doc_id is parameterized.
            f"SELECT {selected} FROM document "
            "WHERE doc_id = ? AND is_active = 1 LIMIT 1",
            (doc_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    file_path, file_name = row[0], row[1]
    rel_path = row[2] if len(row) > 2 else None

    root = _corpus_root()
    if root is not None and rel_path:
        candidate = root / rel_path
        if candidate.is_file():
            return DocumentDownload(path=candidate, file_name=file_name)

    if not file_path:
        return None
    path = Path(file_path)
    if not path.is_file():
        return None
    return DocumentDownload(path=path, file_name=file_name)


__all__ = ["DocumentDownload", "resolve_document_download"]
