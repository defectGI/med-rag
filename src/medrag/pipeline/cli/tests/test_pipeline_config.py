"""Tests for pipeline_config + profile resolution (offline, no model/network).

Run: `python -m pytest src/medrag/pipeline/cli/tests/test_pipeline_config.py`.
`medrag.pipeline.cli` is a real installed package -- `pipeline_config` and
`run_e2e_test` (profile_path) are imported qualified, no sys.path trick.
NOTE: run_e2e_test is imported here but parser modules are NOT called at
import time (profile_path is pure), and there is no module-level os.environ
read -- safe.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from medrag.pipeline.cli import pipeline_config as pc


def test_defaults_load():
    c = pc.load_config(env={})
    assert c.run.doc_concurrency == 1  # sequential on the GPU box, to avoid contention
    assert c.health.enabled is True
    assert c.health.fail_streak == 3


def test_env_override_wins():
    c = pc.load_config(env={"DOC_CONCURRENCY": "8", "HEALTH_FAIL_STREAK": "5"})
    assert c.run.doc_concurrency == 8
    assert c.health.fail_streak == 5


def test_health_check_semantics():
    # Old _health_on() semantics: only "0"/"false" disable it.
    assert pc.load_config(env={"HEALTH_CHECK": "0"}).health.enabled is False
    assert pc.load_config(env={"HEALTH_CHECK": "false"}).health.enabled is False
    assert pc.load_config(env={"HEALTH_CHECK": "1"}).health.enabled is True
    assert pc.load_config(env={"HEALTH_CHECK": "anything"}).health.enabled is True


def test_invalid_int_falls_to_default():
    assert pc.load_config(env={"DOC_CONCURRENCY": "x"}).run.doc_concurrency == 1
    assert pc.load_config(env={"DOC_CONCURRENCY": ""}).run.doc_concurrency == 1


def test_unknown_key_rejected(tmp_path):
    from pydantic import ValidationError
    ov = tmp_path / "bad.toml"
    ov.write_text("[run]\nbogus = 1\n", encoding="utf-8")
    try:
        pc.load_config(env={}, override=ov)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for unknown key")


def test_profile_path_resolution():
    from medrag.pipeline.cli import run_e2e_test as e2e
    assert e2e.profile_path("cfg_default").name == "cfg_default.env"
    assert e2e.profile_path("cfg_default").parent.name == "config"
    # No PIPELINE_PROFILE/arg given: default:
    assert e2e.resolve_profile(None) == e2e.CONFIG_PATH
    assert e2e.resolve_profile("cfg_default").name == "cfg_default.env"


def _run_all():
    passed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        if fn.__code__.co_argcount:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        passed += 1
        print(f"  ok  {name}")
    print(f"{passed} tests passed")


if __name__ == "__main__":
    _run_all()