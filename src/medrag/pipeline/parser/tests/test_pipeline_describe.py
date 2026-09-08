"""run_parse_pipeline phase1_one/phase2_one: the describe pass is wired into
the two-phase model-affinity split (see module docstring) -- visual
description (VLM) runs in phase 1 next to image OCR, table description (LLM)
runs in phase 2 next to the OCR check. No network: a fake VLM/LLM replace the
real clients.

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

import json
import os
from pathlib import Path

from medrag.pipeline.cli import run_parse_pipeline as mod

ROOT = Path(__file__).resolve().parents[5]


def _load_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("PARSER_DIR", str(ROOT / "src" / "medrag" / "pipeline" / "parser"))
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path))
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(tmp_path / "nodes.json"))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setenv("PROGRESS_BAR", "0")
    saved = dict(os.environ)
    try:
        mod._bootstrap()
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return mod


def _record(doc_id: str, rel_path: str) -> dict:
    return {
        "identity": {"doc_id": doc_id},
        "location": {"rel_path": rel_path},
        "scan": {"content_hash": "h1"},
        "doc_type": "TEST",
        "parse": {"status": None},
    }


class _RecordingParse(dict):
    """A dict with `.update` recorded, matching how the pipeline mutates
    `record["parse"]` in place."""


def test_phase1_calls_describe_blocks_with_vlm_stage(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nSome text.\n", encoding="utf-8")

    calls: list[str] = []

    def _fake_describe(doc, *, stage, llm=None, vlm=None):
        calls.append(stage)
        return doc

    monkeypatch.setattr(mod, "describe_blocks", _fake_describe)
    record = _record("d1", "doc.md")

    mod.phase1_one(record)

    assert calls == ["vlm"]
    ckpt = mod._stage1_path(record)
    assert ckpt.is_file()
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert saved["visual_desc_stage"]["status"] == "SUCCESS"


def test_phase1_visual_desc_failure_is_recorded_not_raised(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nSome text.\n", encoding="utf-8")

    def _raising(doc, *, stage, llm=None, vlm=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "describe_blocks", _raising)
    record = _record("d1", "doc.md")

    mod.phase1_one(record)  # must not raise

    ckpt = mod._stage1_path(record)
    saved = json.loads(ckpt.read_text(encoding="utf-8"))
    assert saved["visual_desc_stage"]["status"] == "FAILED"
    assert "boom" in saved["visual_desc_stage"]["error"]


def test_phase2_calls_describe_blocks_with_llm_stage_for_tables(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\n| a | b |\n| - | - |\n| 1 | 2 |\n", encoding="utf-8")

    stage1_calls: list[str] = []
    stage2_calls: list[str] = []

    def _fake_describe(doc, *, stage, llm=None, vlm=None):
        (stage1_calls if stage == "vlm" else stage2_calls).append(stage)
        return doc

    monkeypatch.setattr(mod, "describe_blocks", _fake_describe)
    record = _record("d2", "doc.md")
    mod.phase1_one(record)
    assert stage1_calls == ["vlm"]

    mod.phase2_one(record)

    assert stage2_calls == ["llm"]
    assert record["parse"]["status"] == "SUCCESS"
    out_path = Path(record["parse"]["parsed_json_path"])
    assert out_path.is_file()


def test_phase2_skips_table_stage_when_document_has_no_tables(monkeypatch, tmp_path):
    mod = _load_pipeline(monkeypatch, tmp_path)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nJust a paragraph, no tables.\n", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(mod, "describe_blocks",
                        lambda doc, *, stage, llm=None, vlm=None: (calls.append(stage), doc)[1])
    record = _record("d3", "doc.md")
    mod.phase1_one(record)
    calls.clear()

    mod.phase2_one(record)

    assert calls == []  # no tables -> the llm stage is never invoked
    assert record["parse"]["stages"]["table_desc"]["status"] == "SKIPPED"


def test_phase2_reads_old_checkpoint_without_visual_desc_stage(monkeypatch, tmp_path):
    """A checkpoint written before this pass existed has no `visual_desc_stage`
    key -- phase2_one must still finish the document rather than crash on it."""
    mod = _load_pipeline(monkeypatch, tmp_path)
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nText only.\n", encoding="utf-8")
    record = _record("d4", "doc.md")

    monkeypatch.setattr(mod, "describe_blocks",
                        lambda doc, *, stage, llm=None, vlm=None: doc)
    mod.phase1_one(record)
    ckpt_path = mod._stage1_path(record)
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    del data["visual_desc_stage"]  # simulate a pre-existing pass checkpoint
    ckpt_path.write_text(json.dumps(data), encoding="utf-8")

    mod.phase2_one(record)  # must not raise

    assert record["parse"]["status"] == "SUCCESS"
    assert "visual_desc" not in record["parse"]["stages"]
