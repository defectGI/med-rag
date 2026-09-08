"""Writes the unresolved documents/chunks from the `catalog_chunk_ownership`
flow into the common `issues` table (O-07, I-14).

**Scope note (deliberate narrowing):** O-07's task description mentions
two sources: "the document-level `unresolved` list (from the scanner)"
and "the chunk-level unresolvables (from `carry_over_unresolved.py`)".
M-01 (the inventory of existing marking mechanisms) has NOT yet been
done, so it is UNCERTAIN which concrete `unresolved` list the "scanner"
(scan phase, registry) produces within this task's scope -- we did not
invent an assumption. This module currently writes the FACTS phase's
OWN document-level unresolvable list (`build_all.py::docs_without_chunk_text`
-- "no chunk source coverage") and chunk-level unresolvables
(`carry_over_unresolved.py::plan()`'s `skipped` list). If the scan
phase's own `unresolved` list later needs to enter the table (after the
M-01 inventory), the same function can feed it via `stage="scan"` --
the function accepts a `stage` parameter and is NOT locked to facts.
"""

from __future__ import annotations

import json
import sqlite3

from medrag.core.db.issues import Stage, record_issue
from medrag.core.paths import ISSUES_DB_PATH as DEFAULT_ISSUES_DB_PATH

# SIBLING file to specs.db (K-75) -- NOT a table inside specs.db.
#
# 2026-08-26 fix: `DEFAULT_ISSUES_DB_PATH` is now an alias for the
# constant default in `core/paths.py` (unchanged) -- in the container,
# this path lives INSIDE the image and is LOST on every redeploy. In the
# real container runtime `ISSUES_DB_PATH` env var must be set -- direct
# users of this constant (callers that import this module outside
# `main()`) should PREFER `core.paths.resolve_issues_db_path()` (see
# that function's docstring, same "read at call time" rationale as
# `resolve_reports_dir()`). The `main()` functions of `load_to_db.py` /
# `build_all.py` / `carry_over_unresolved.py` already DO this.


def record_unresolved_docs(
    con: sqlite3.Connection,
    docs: list[dict],
    *,
    stage: Stage = "facts",
) -> int:
    """Writes rows shaped `{"doc_id", "file_name", "reason"}` into
    `issues` with reason code `UNRESOLVED_DOC`. Returns: row count
    written.

    Source: today `catalog_chunk_ownership/build_all.py::docs_without_chunk_text`
    ("no chunk source coverage" -- documents whose text could not be
    found in either `all_chunks.json` or the atoms bridge). No silent
    drops -- each entry becomes one row."""
    n = 0
    for d in docs:
        record_issue(
            con,
            stage=stage,
            severity="warning",
            reason="UNRESOLVED_DOC",
            source_doc=d.get("doc_id"),
            entity_ref=None,
            detail=d.get("reason"),
            payload=json.dumps(d, ensure_ascii=False),
        )
        n += 1
    return n


def record_unresolved_chunks(
    con: sqlite3.Connection,
    skipped: list[dict],
    *,
    stage: Stage = "facts",
) -> int:
    """Writes `carry_over_unresolved.py::plan()`'s `skipped` list (each
    entry contains `chunk_id`, `doc_id`, `_why` -- see that module's
    docstring) into `issues` with reason code `UNRESOLVED_CHUNK`.

    `severity="review"`: these are NOT errors but cases that need human
    review (mixed chunks / chunks already unresolvable in a previous run)."""
    n = 0
    for s in skipped:
        record_issue(
            con,
            stage=stage,
            severity="review",
            reason="UNRESOLVED_CHUNK",
            source_doc=s.get("doc_id"),
            entity_ref=s.get("chunk_id"),
            detail=s.get("_why"),
            payload=json.dumps(s, ensure_ascii=False),
        )
        n += 1
    return n


def record_skipped_facts(
    con: sqlite3.Connection,
    skipped: list[dict],
    *,
    source_doc: str | None,
    stage: Stage = "facts",
) -> int:
    """Writes `load_to_db.py::load_product`'s `report["skipped"]` (each
    entry has at least `key`/`reason`, see `load_product`'s
    `report["skipped"].append(...)` calls) into `issues` with reason code
    `SKIPPED_FACT`. Returns: row count written.

    The M-01 inventory's "facts dropped by load_to_db.py" entry -- up to
    now this list only went to `_load_report.json` (per-run, overwritten,
    not countable / filterable). `source_doc` here is `product_code`
    (NOT the fact's own source document -- `report["skipped"]` entries
    do not carry `source_doc_id`, only `key`/`reason`); tracking which
    product's load dropped them is the purpose, the chunk/document-level
    evidence reference is the job of `UNRESOLVED_CHUNK`/`UNRESOLVED_DOC`.

    **NOTE (unlike O-14, NO user approval REQUIRED):** unlike
    `NEW_KEY_PROPOSAL`, this code only MAKES VISIBLE a decision the
    pipeline was already making silently today (dropping the fact); it
    is not a new behaviour/policy decision that requires approval, so it
    was activated directly under M-03 (see `load_to_db.py::load_product`'s
    `issues_con` parameter)."""
    n = 0
    for item in skipped:
        record_issue(
            con,
            stage=stage,
            severity="review",
            reason="SKIPPED_FACT",
            source_doc=source_doc,
            entity_ref=item.get("key"),
            detail=item.get("reason"),
            payload=json.dumps(item, ensure_ascii=False),
        )
        n += 1
    return n


def record_new_key_proposal(
    con: sqlite3.Connection, *, product_code: str, fact: dict, stage: Stage = "facts",
) -> int:
    """O-14 prep (the visibility leg of decision (1), not yet logged as
    KARAR-NNN in PROTOCOL): a key proposal NOT in the dictionary is
    ALSO written to `issues` with reason code `NEW_KEY_PROPOSAL` --
    alongside `load_to_db.py`'s existing-only record,
    `queue/new_key_proposals.jsonl` -- NOT INSTEAD OF it
    append-only JSONL behaviour is UNCHANGED.

    Why needed (O-14 note): the 12 MB `no_key_match.jsonl` is proof that
    no one is looking at it -- an append-only file does NOT SHOW UP in
    the panel / M-08's open-issue count. `severity="review"` (not an
    error, needs human review -- whether to add to the dictionary or
    whether it's a mis-spelling of an existing key).

    **Not currently called by ANY production call path** (`load_product`'s
    default `issues_con=None`) -- pending user approval of O-14's
    decision (1); the hook is just ready."""
    return record_issue(
        con, stage=stage, severity="review", reason="NEW_KEY_PROPOSAL",
        source_doc=None, entity_ref=f"{fact.get('block')}.{fact.get('key')}",
        detail=f"product_code={product_code}",
        payload=json.dumps({"product_code": product_code, "fact": fact}, ensure_ascii=False),
    )


__all__ = [
    "DEFAULT_ISSUES_DB_PATH",
    "record_new_key_proposal",
    "record_skipped_facts",
    "record_unresolved_chunks",
    "record_unresolved_docs",
]