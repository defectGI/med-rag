"""Persistent logging tests: is the file created, does the text2sql log
(despite propagate=False) reach the file, does the trace bridge log sql_error
at ERROR with raw_text. No network.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import medrag.api.logging_setup as ls


def _fresh_module():
    # `_configured` is module-global; reload the module so each test can do a
    # fresh setup, clearing previous handlers first.
    logging.getLogger().handlers.clear()
    logging.getLogger("text2sql").handlers.clear()
    return importlib.reload(ls)


def test_creates_log_file(tmp_path: Path):
    mod = _fresh_module()
    log_file = mod.configure_logging(log_dir=tmp_path, console=False)
    assert log_file == tmp_path / "medrag.api.log"
    logging.getLogger("medrag.api.test").info("merhaba")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "merhaba" in log_file.read_text(encoding="utf-8")


def test_text2sql_logs_reach_file_despite_no_propagate(tmp_path: Path):
    mod = _fresh_module()
    log_file = mod.configure_logging(log_dir=tmp_path, text2sql_level="DEBUG", console=False)
    t2s = logging.getLogger("text2sql.pipeline.linking")
    assert logging.getLogger("text2sql").propagate is False
    t2s.debug("linking picked 2 tables")
    for h in logging.getLogger("text2sql").handlers:
        h.flush()
    assert "linking picked 2 tables" in log_file.read_text(encoding="utf-8")


def test_idempotent_no_duplicate_handlers(tmp_path: Path):
    mod = _fresh_module()
    mod.configure_logging(log_dir=tmp_path, console=False)
    n_root = len(logging.getLogger().handlers)
    n_t2s = len(logging.getLogger("text2sql").handlers)
    mod.configure_logging(log_dir=tmp_path, console=False, level="DEBUG")
    assert len(logging.getLogger().handlers) == n_root
    assert len(logging.getLogger("text2sql").handlers) == n_t2s
    assert logging.getLogger().level == logging.DEBUG


def _read(log_file: Path) -> str:
    for h in logging.getLogger().handlers + logging.getLogger("text2sql").handlers:
        h.flush()
    return log_file.read_text(encoding="utf-8")


def test_log_trace_step_sql_error_is_error_with_raw_text(tmp_path: Path):
    mod = _fresh_module()
    log_file = mod.configure_logging(log_dir=tmp_path, console=False)
    log = logging.getLogger("medrag.api.webapp")
    mod.log_trace_step(
        log, "sql_error",
        {"error_type": "StructuredOutputError", "message": "boş", "raw_text": "<<HAM>>"},
        prefix="ab12: ",
    )
    text = _read(log_file)
    # At ERROR level, with full raw_text, with the req tag.
    assert "ERROR" in text
    assert "<<HAM>>" in text
    assert "ab12:" in text


def test_log_trace_step_results_are_debug(tmp_path: Path):
    mod = _fresh_module()
    log_file = mod.configure_logging(log_dir=tmp_path, console=False, level="DEBUG")
    log = logging.getLogger("medrag.api.webapp")
    mod.log_trace_step(log, "sql_rows", [{"id": "PN1309", "score": 0.9, "text": "x"}])
    assert "1 sonuç" in _read(log_file)


def test_sql_error_survives_warning_level(tmp_path: Path):
    # Privacy dial: at level=WARNING the query/INFO is hidden but the SQL ERROR
    # is still written -- the user's privacy-vs-visibility tradeoff.
    mod = _fresh_module()
    log_file = mod.configure_logging(log_dir=tmp_path, console=False, level="WARNING")
    log = logging.getLogger("medrag.api.webapp")
    mod.log_trace_step(log, "resolved_query", "everest zirvesi ürünü")  # INFO -> hidden
    mod.log_trace_step(log, "sql_error", {"raw_text": "<<HAM>>"})        # ERROR -> visible
    text = _read(log_file)
    assert "everest zirvesi" not in text
    assert "<<HAM>>" in text
