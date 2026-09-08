"""Editorial deactivation: marks selected documents `is_active=false` in
document_nodes.json.

Why it exists: a document that is re-tried every run and keeps failing
has its `parse.last_parsed` refreshed every night -- so the `since=`
filter on `_parse_failed_docs` (the "don't include previous-night FAILED
tombstones in the report" rule) never eliminates it, and the nightly
report can NEVER reach `full_success`. Example: `PN5022_TEST_DATASHEET.pdf`
and two `attribute_schema*.json` (same nightly run, every run). This is
NOT a code bug: the source file is genuinely broken/unwanted and must be
shut off by hand. The right axis is `is_active`: scan PRESERVES this
field (`classify_documents.py::merge_record` -- the scanner doesn't know
about editorial decisions), `run_parse_pipeline._needs_parse` does NOT
re-try a passive record, so `last_parsed` goes stale and the tombstone
filter does its job.

Reverse direction is deliberately ABSENT (no re-activation): the user
who wants to undo it edits `document_nodes.json` by hand -- this script
is a one-way "retirement" tool; accidentally making re-activation easy
would be a bug.

Same pattern as `reset_parse_flags.py` (same D-58 escape-hatch class):
reads DOCUMENT_NODES_PATH from the cli/.env that `run_parse_pipeline.py`
uses (if defined in env, `load_dotenv` does not override it -- in a
container: `DOCUMENT_NODES_PATH=/corpus/document_nodes.json python
mark_docs_inactive.py ...` runs against the live copy), atomic write
(tmp + os.replace, same discipline as
`run_parse_pipeline._atomic_write_json`).

Usage (from pipeline/):
    python mark_docs_inactive.py --dry-run PN5022_TEST_DATASHEET.pdf
    python mark_docs_inactive.py PN5022_TEST_DATASHEET.pdf attribute_schema.json

Match: exact equality (case- and path-separator-insensitive) against
either `identity.file_name` OR `location.rel_path`. A target that
matches no record is NOT silently ignored: the script exits with 1
having written nothing -- the "probably processed" assumption for a
wrong file name is worse than silent permanent data loss. The same
`file_name` may appear in multiple records (e.g. the same PDF in two
product folders); the script flags every matching record individually so
the operator can narrow with `rel_path` if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
# script stayed in pipeline/ but DOCUMENT_NODES_PATH has one home now, not
# a duplicated copy -- read it from the same .env run_parse_pipeline.py
# uses, and resolve the path relative to THAT file's own directory
# (paths inside it are relative to cli/, not to pipeline/).
_ENV_DIR = BASE_DIR.parent / "src" / "urun" / "pipeline" / "cli"


def _norm(value: str) -> str:
    """Match key: strip quotes/whitespace, normalize backslashes, lowercase."""
    return value.strip().replace("\\", "/").lower()


def mark_inactive(data: dict, targets: list[str]) -> dict:
    """SAFE core (no disk access, `data` is mutated IN PLACE).

    Returns: `{"changed": [...], "already_inactive": [...], "unmatched": [...]}`.
    `changed`/`already_inactive` entries are `{doc_id, file_name, rel_path}`
    for the operator report. A record already inactive is NOT counted
    as changed (idempotent re-runs don't bloat the report); an unmatched
    target is returned to the caller, not silently dropped.
    """
    by_norm = {_norm(t): t for t in targets}
    changed: list[dict] = []
    already_inactive: list[dict] = []
    matched_norm: set[str] = set()
    for record in data.get("documents", []):
        file_name = _norm((record.get("identity") or {}).get("file_name") or "")
        rel_path = _norm((record.get("location") or {}).get("rel_path") or "")
        hits = {file_name, rel_path} & set(by_norm)
        if not hits:
            continue
        matched_norm.update(hits)
        if record.get("is_active", True):
            record["is_active"] = False
            changed.append({
                "doc_id": (record.get("identity") or {}).get("doc_id"),
                "file_name": (record.get("identity") or {}).get("file_name"),
                "rel_path": (record.get("location") or {}).get("rel_path"),
            })
        else:
            already_inactive.append({
                "doc_id": (record.get("identity") or {}).get("doc_id"),
                "file_name": (record.get("identity") or {}).get("file_name"),
                "rel_path": (record.get("location") or {}).get("rel_path"),
            })
    return {
        "changed": changed,
        "already_inactive": already_inactive,
        "unmatched": [t for n, t in by_norm.items() if n not in matched_norm],
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", metavar="FILE",
                    help="exact match against identity.file_name OR location.rel_path "
                         "in document_nodes.json (case- and path-separator-insensitive)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report matches only; writes NOTHING")
    args = ap.parse_args(argv)

    load_dotenv(_ENV_DIR / ".env")
    raw = os.environ.get("DOCUMENT_NODES_PATH")
    if not raw:
        print("DOCUMENT_NODES_PATH is not defined (see src/medrag/pipeline/cli/.env)",
              file=sys.stderr)
        return 2
    path = (_ENV_DIR / raw).resolve()

    data = json.loads(path.read_text(encoding="utf-8"))
    result = mark_inactive(data, args.targets)

    for rec in result["changed"]:
        print(f"  to be deactivated: {rec['file_name']}  ({rec['rel_path']}, doc_id={rec['doc_id']})")
    for rec in result["already_inactive"]:
        print(f"  already inactive:  {rec['file_name']}  ({rec['rel_path']})")
    for target in result["unmatched"]:
        print(f"  NO MATCH:          {target}", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] {len(result['changed'])} records would be deactivated -> NOT WRITTEN: {path}")
        return 1 if result["unmatched"] else 0
    if result["unmatched"]:
        print("Nothing is written while any target is unmatched -- fix the targets "
              "and re-run (a wrong file name silently slipping through is permanent "
              "data loss).",
              file=sys.stderr)
        return 1
    if not result["changed"]:
        print(f"No change needed (matched records already inactive): {path}")
        return 0

    _atomic_write_json(path, data)
    print(f"{len(result['changed'])} records deactivated -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())