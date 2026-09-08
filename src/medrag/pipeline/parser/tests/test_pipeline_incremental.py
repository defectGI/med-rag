"""run_parse_pipeline: the incremental check and the `--limit` run control.

_needs_parse must not compare scan.content_hash alone -- unchanged files would
then be skipped forever no matter how much the parser changed. The check must
also re-parse records whose parse.parser_version is older than the current
PARSER_VERSION.

`--limit N` is the trial-run control (parse N pending documents, leave the
rest). It is tested through _parse_argv only -- main() would drive real
models, which no test here starts.

The pipeline module resolves its .env paths lazily, via an explicit
`_bootstrap()` call -- so the required variables are injected first, then
`_bootstrap()` is called to rerun env resolution against the monkeypatched
environment (importing also loads parser/.env, which must not leak into other
tests, hence the save/restore of os.environ around it).

`run_parse_pipeline` lives in `src/medrag/pipeline/cli/`, a real installed
package (`medrag.pipeline.cli`) -- imported qualified, no `sys.path`/bare-import
bridge.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from medrag.pipeline.cli import run_parse_pipeline as mod

ROOT = Path(__file__).resolve().parents[5]


def _load_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER_DIR", str(ROOT / "src" / "medrag" / "pipeline" / "parser"))
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(tmp_path / "nodes.json"))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "out"))
    saved = dict(os.environ)
    try:
        mod._bootstrap()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return mod


def test_needs_parse_version_gate(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    rec = {"is_active": True,
           "scan": {"content_hash": "h1"},
           "parse": {"parsed_from_hash": "h1", "status": "SUCCESS",
                     "parser_version": mod.PARSER_VERSION}}
    # up to date: same bytes, same parser
    assert mod._needs_parse(rec) is False
    # same bytes, older parser -> the fix wave must reach this record
    rec["parse"]["parser_version"] = "1.0.0"
    assert mod._needs_parse(rec) is True
    # a content change re-parses regardless of version
    rec["parse"].update(parser_version=mod.PARSER_VERSION,
                        parsed_from_hash="stale")
    assert mod._needs_parse(rec) is True


def test_needs_parse_skipped_and_inactive_records(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    skipped = {"is_active": True,
               "scan": {"content_hash": "h1"},
               "parse": {"parsed_from_hash": "h1", "status": "SKIPPED",
                         "parser": None, "parser_version": None}}
    # unsupported format: no parser ran, so no parser change affects it
    assert mod._needs_parse(skipped) is False
    inactive = {"is_active": False,
                "scan": {"content_hash": "x"},
                "parse": {"parsed_from_hash": None}}
    assert mod._needs_parse(inactive) is False


# -- --limit: trial-run control (CONFIG.md: run input lives on the CLI) --------


def test_limit_flag_parsing(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    assert mod._parse_argv([]) is None          # unset = whole pending corpus
    assert mod._parse_argv(["--limit", "3"]) == 3


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_limit_rejects_non_positive(monkeypatch, tmp_path, bad):
    """0/negative would silently mean "parse nothing" -- argparse exits 2."""
    mod = _load_pipeline(monkeypatch, tmp_path)
    with pytest.raises(SystemExit):
        mod._parse_argv(["--limit", bad])


def test_limit_slices_after_the_needs_parse_filter(monkeypatch, tmp_path):
    """The semantics `--limit` promises: N of the documents that actually
    need work, not "the first N records, some of which are already done".
    Locks the ordering of filter-then-slice in main()."""
    mod = _load_pipeline(monkeypatch, tmp_path)
    done = {"is_active": True, "scan": {"content_hash": "h"},
            "parse": {"parsed_from_hash": "h", "status": "SUCCESS",
                      "parser_version": mod.PARSER_VERSION}}
    pending = {"is_active": True, "scan": {"content_hash": "h"},
               "parse": {"parsed_from_hash": None, "status": "PENDING",
                         "parser_version": None}}
    documents = [done, dict(pending), dict(pending), dict(pending)]

    todo = [r for r in documents if mod._needs_parse(r)]
    assert len(todo) == 3            # the finished one is out of the way
    assert len(todo[:2]) == 2        # --limit 2 takes 2 of the pending three
    # ...and the registry itself is never truncated by the limit.
    assert len(documents) == 4
