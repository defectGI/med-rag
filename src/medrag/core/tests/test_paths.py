"""Locking tests for `resolve_reports_dir`.

The problem: in the container `REPORTS_DIR` (`_REPO_ROOT / "reports"`)
EVAPORATES on every redeploy -- the `NIGHTLY_REPORTS_DIR` env variable overrides
it, and `run_nightly.py` (the writer) and `panel/blueprint.py` (the reader) call
THIS SAME function. The tests lock down that the env is read at CALL time (not
at import time) -- otherwise `monkeypatch.setenv` wouldn't work in tests."""
from __future__ import annotations

from pathlib import Path

from medrag.core import paths


def test_resolves_to_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(paths.NIGHTLY_REPORTS_DIR_ENV, raising=False)
    assert paths.resolve_reports_dir() == paths.REPORTS_DIR


def test_env_override_takes_precedence(monkeypatch, tmp_path):
    override = tmp_path / "kalici" / "reports"
    monkeypatch.setenv(paths.NIGHTLY_REPORTS_DIR_ENV, str(override))
    assert paths.resolve_reports_dir() == override


def test_resolution_happens_at_call_time_not_import_time(monkeypatch, tmp_path):
    """The env variable can change AFTER the function is DEFINED (or between
    different tests) -- `resolve_reports_dir` must READ on every call, not hold
    a value FROZEN while the module was imported."""
    monkeypatch.delenv(paths.NIGHTLY_REPORTS_DIR_ENV, raising=False)
    assert paths.resolve_reports_dir() == paths.REPORTS_DIR

    override = tmp_path / "sonradan"
    monkeypatch.setenv(paths.NIGHTLY_REPORTS_DIR_ENV, str(override))
    assert paths.resolve_reports_dir() == override


def test_env_value_returned_as_path_object(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.NIGHTLY_REPORTS_DIR_ENV, str(tmp_path))
    result = paths.resolve_reports_dir()
    assert isinstance(result, Path)


# --- resolve_issues_db_path: the in-container issues.db path problem
# (`/app/facts/db/issues.db`, evaporating on every redeploy), fixed with the
# SAME pattern as `resolve_reports_dir()`.


def test_issues_db_path_resolves_to_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(paths.ISSUES_DB_PATH_ENV, raising=False)
    assert paths.resolve_issues_db_path() == paths.ISSUES_DB_PATH


def test_issues_db_path_env_override_takes_precedence(monkeypatch, tmp_path):
    override = tmp_path / "kalici" / "issues.db"
    monkeypatch.setenv(paths.ISSUES_DB_PATH_ENV, str(override))
    assert paths.resolve_issues_db_path() == override


def test_issues_db_path_resolution_happens_at_call_time_not_import_time(monkeypatch, tmp_path):
    monkeypatch.delenv(paths.ISSUES_DB_PATH_ENV, raising=False)
    assert paths.resolve_issues_db_path() == paths.ISSUES_DB_PATH

    override = tmp_path / "sonradan"
    monkeypatch.setenv(paths.ISSUES_DB_PATH_ENV, str(override))
    assert paths.resolve_issues_db_path() == override


# --- resolve_pending_rework_path (follow-up review item 1) ------------------


def test_pending_rework_path_resolves_to_default_when_env_unset(monkeypatch):
    monkeypatch.delenv(paths.PENDING_REWORK_PATH_ENV, raising=False)
    assert paths.resolve_pending_rework_path() == paths.PENDING_REWORK_PATH


def test_pending_rework_path_env_override_takes_precedence(monkeypatch, tmp_path):
    override = tmp_path / "kalici" / "pending_rework.jsonl"
    monkeypatch.setenv(paths.PENDING_REWORK_PATH_ENV, str(override))
    assert paths.resolve_pending_rework_path() == override
