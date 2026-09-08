"""Data for the panel's landing page.

Locked behavior: the latest (by date) nightly report is read · the "stale"
counter is kept SEPARATE for the `kaynak_degisti`/`asama_degisti` statuses · if
there is no report the panel does NOT crash, it returns zeroed counters · no
derivation function such as `derive_derivative_status` is ever called here --
only already-computed strings are counted (the panel-does-not-compute rule).
"""

from __future__ import annotations

import json

from medrag.api.panel.dashboard import (
    build_dashboard,
    latest_report_path,
    load_latest_report,
    stale_breakdown,
)


def _write_report(reports_dir, date: str, **overrides) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = {
        "run_id": f"{date}T23-00-00Z",
        "started_at": f"{date}T23:00:00+00:00",
        "finished_at": f"{date}T23:45:00+00:00",
        "outcome": "full_success",
        "ownership": {"deterministic_count": 10, "model_routed_count": 2, "unresolved_count": 0},
        "derivative_status": {},
    }
    base.update(overrides)
    (reports_dir / f"nightly_{date}.json").write_text(json.dumps(base), encoding="utf-8")


def test_latest_report_path_picks_newest_by_filename_date(tmp_path):
    _write_report(tmp_path, "2026-08-18")
    _write_report(tmp_path, "2026-08-20")
    _write_report(tmp_path, "2026-08-19")

    assert latest_report_path(tmp_path).name == "nightly_2026-08-20.json"


def test_latest_report_path_missing_dir_returns_none(tmp_path):
    assert latest_report_path(tmp_path / "does-not-exist") is None


def test_load_latest_report_returns_none_without_any_report(tmp_path):
    assert load_latest_report(tmp_path) is None


def test_stale_breakdown_separates_source_changed_from_stage_changed():
    derivative_status = {
        "doc-1": {"parse": "kaynak_degisti", "chunk": "kaynak_degisti"},
        "doc-2": {"parse": "asama_degisti"},
        "doc-3": {"parse": "taze"},
        "doc-4": {"parse": "uretilmemis"},
        "doc-5": {"parse": "kaynak_yok"},
    }
    breakdown = stale_breakdown(derivative_status)
    assert breakdown == {"kaynak_degisti": 2, "asama_degisti": 1, "missing": 1}


def test_build_dashboard_without_any_report_is_all_zero(tmp_path):
    result = build_dashboard(reports_dir=tmp_path, open_issues_count=0)
    assert result == {
        "last_night": None,
        "counters": {
            "stale_source_changed": 0,
            "stale_stage_changed": 0,
            "ownerless": 0,
            "missing": 0,
            "open_issues": 0,
        },
    }


def test_build_dashboard_intentionally_failed_night_surfaces_on_first_read(tmp_path):
    """When the night is intentionally failed (fixture), the result shows
    `failed` -- the panel does NOT recompute it, it reads the report's own
    `outcome` field."""
    _write_report(
        tmp_path,
        "2026-08-19",
        outcome="failed",
        derivative_status={
            "doc-a": {"parse": "kaynak_degisti"},
            "doc-b": {"chunk": "asama_degisti", "facts": "asama_degisti"},
            "doc-c": {"facts": "uretilmemis"},
        },
        ownership={"deterministic_count": 0, "model_routed_count": 0, "unresolved_count": 7},
    )

    result = build_dashboard(reports_dir=tmp_path, open_issues_count=3)

    assert result["last_night"]["outcome"] == "failed"
    assert result["last_night"]["run_id"] == "2026-08-19T23-00-00Z"
    assert result["counters"] == {
        "stale_source_changed": 1,
        "stale_stage_changed": 2,
        "ownerless": 7,
        "missing": 1,
        "open_issues": 3,
    }


def test_build_dashboard_picks_the_most_recent_night_only(tmp_path):
    _write_report(tmp_path, "2026-08-18", outcome="failed")
    _write_report(tmp_path, "2026-08-19", outcome="full_success")

    result = build_dashboard(reports_dir=tmp_path, open_issues_count=0)
    assert result["last_night"]["outcome"] == "full_success"
