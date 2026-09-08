"""Data for the panel's **landing page** -- last night's outcome + four counters.

A page hidden in a submenu informs no one, so the data this module produces is
shown on the panel's `/` root page (see `blueprint.py::_index`) on the first
screen, with no click required.

**The panel does NOT compute, it reads the report** (same rationale as the
"single producer" principle): the `outcome`/`derivative_status`/
`ownership.unresolved_count` fields are already computed and written by
`medrag.pipeline.nightly_report` to `reports/nightly_YYYY-MM-DD.json`. This module
ONLY reads the latest file and COUNTS the precomputed statuses (it derives no
new business rule -- `derive_derivative_status` is not called, only its string
output is counted).

**Layering note:** it does NOT import `medrag.pipeline.nightly_report` (enforced
by `src/medrag/tests/test_import_contracts.py` +
`tests/test_blueprint.py::test_pipeline_not_imported_by_blueprint_module`) --
the report file is read directly via `json.load` as a raw dict, the
`NightlyReport` class is never constructed. This yields the same result (raw
dict) as `load_nightly_report_dict` but is an independent panel-side
implementation -- the layer boundary prevents the two sides from sharing that
function, so the file-name convention (`nightly_YYYY-MM-DD.json`) is known HERE
as well; if one changes, both must be updated together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_FILENAME_PREFIX = "nightly_"
_FILENAME_SUFFIX = ".json"

# Two of the four derivative statuses count as "stale": they need DIFFERENT
# interventions, so they get separate counters -- "kaynak_degisti" needs a source
# fix, "asama_degisti" needs a bulk re-process.
STALE_STATUSES = ("kaynak_degisti", "asama_degisti")
MISSING_STATUS = "uretilmemis"


def latest_report_path(reports_dir: Path) -> Path | None:
    """Finds the latest nightly report under `reports_dir`.

    File name is `nightly_YYYY-MM-DD.json` -- ISO-date string ordering ==
    chronological ordering, so the greatest name is the newest (the
    "keep-unbounded" rule: old files are never deleted, so this stays correct as
    the directory grows)."""
    if not reports_dir.exists():
        return None
    candidates = sorted(
        p for p in reports_dir.glob(f"{_FILENAME_PREFIX}*{_FILENAME_SUFFIX}") if p.is_file()
    )
    return candidates[-1] if candidates else None


def load_latest_report(reports_dir: Path) -> dict[str, Any] | None:
    """Reads the latest report as a RAW dict -- `None` if absent (no nightly run
    may have happened yet; the panel should show it silently as empty, no
    exception)."""
    path = latest_report_path(reports_dir)
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def stale_breakdown(derivative_status: dict[str, dict[str, str]]) -> dict[str, int]:
    """COUNTS the precomputed derivative statuses, does NOT re-derive them.

    `derivative_status` -- `doc_id -> stage_name -> status` (a field of the
    report schema itself). Returned dict: `{"kaynak_degisti": N,
    "asama_degisti": N, "missing": N}` -- `missing` here is the count of the
    `uretilmemis` status (the "missing" counter)."""
    counts: dict[str, int] = {status: 0 for status in STALE_STATUSES}
    missing = 0
    for stage_map in derivative_status.values():
        for status in stage_map.values():
            if status in counts:
                counts[status] += 1
            elif status == MISSING_STATUS:
                missing += 1
    counts["missing"] = missing
    return counts


def build_dashboard(*, reports_dir: Path, open_issues_count: int) -> dict[str, Any]:
    """The single assembly point: last-night summary + four counters.

    `open_issues_count` is computed and passed in by the caller
    (`core.db.issues.count_open`) -- this function does not touch `issues.db`
    (the blueprint already opens and closes its own connection; no second
    connection is opened here)."""
    report = load_latest_report(reports_dir)
    if report is None:
        return {
            "last_night": None,
            "counters": {
                "stale_source_changed": 0,
                "stale_stage_changed": 0,
                "ownerless": 0,
                "missing": 0,
                "open_issues": open_issues_count,
            },
        }

    breakdown = stale_breakdown(report.get("derivative_status", {}))
    ownership = report.get("ownership") or {}
    return {
        "last_night": {
            "run_id": report.get("run_id"),
            "started_at": report.get("started_at"),
            "finished_at": report.get("finished_at"),
            "outcome": report.get("outcome"),
        },
        "counters": {
            "stale_source_changed": breakdown["kaynak_degisti"],
            "stale_stage_changed": breakdown["asama_degisti"],
            "ownerless": ownership.get("unresolved_count", 0),
            "missing": breakdown["missing"],
            "open_issues": open_issues_count,
        },
    }


__all__ = [
    "MISSING_STATUS",
    "STALE_STATUSES",
    "build_dashboard",
    "latest_report_path",
    "load_latest_report",
    "stale_breakdown",
]
