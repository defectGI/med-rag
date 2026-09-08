"""Interrupt-resilience for the chunk stage.

MEASURED BUG (the old code): `calistir()` wrote the two fixed files
(`all_chunks.json`, `all_combined.json`) directly via `open(..., "w")`
to their final paths — NOT write-then-replace. Worse: BEFORE writing,
the previous run's output was MOVED OUT (`_archive_existing_output`) — so
an interruption DURING writing left neither an old (archived) nor a new
(half-written) complete file.

Fix: both files are FIRST written COMPLETELY into separate TEMP files;
ONLY after both finish successfully is the previous run archived and the
temp files renamed into place with `os.replace`. The "neither old nor
new" window shrinks to a few rename calls (not json.dump time).

This test uses the approach recommended by the task note (a testable
breakpoint, not real timing): it makes `json.dump` raise
`KeyboardInterrupt` on its second call to simulate a kill mid-write.
"""

from __future__ import annotations

import json

import pytest

from medrag.pipeline.chunker.cli import calistir
from medrag.pipeline.chunker.layout import OutputLayout


def _ir(doc_id):
    return {
        "ir_version": 7, "doc_id": doc_id, "source_path": f"{doc_id}.md",
        "fmt": "markdown",
        "blocks": [
            {"type": "heading", "id": "b0", "text": "Kurulum", "level": 1,
             "anchor_id": "kurulum"},
            {"type": "paragraph", "id": "b1", "text": "Once paketi indirin.",
             "heading_path": ["Kurulum"]},
        ],
    }


@pytest.fixture
def girdi(tmp_path):
    d = tmp_path / "girdi"
    d.mkdir()
    (d / "a.json").write_text(json.dumps(_ir("a")), encoding="utf-8")
    return d


def _env(girdi, tmp_path):
    return {"CHUNKER_INPUT_DIR": str(girdi),
            "CHUNKER_OUTPUT_DIR": str(tmp_path / "cikti"),
            "TOKENIZER": "fake", "CHUNKER_VIZ": "off"}


def _out(tmp_path) -> OutputLayout:
    return OutputLayout(tmp_path / "cikti")


def test_kill_mid_write_leaves_previous_output_intact(girdi, tmp_path, monkeypatch):
    env = _env(girdi, tmp_path)

    # Run 1: v1 written successfully.
    assert calistir(env) == 0
    out = _out(tmp_path)
    v1_chunks = out.all_chunks_file.read_text(encoding="utf-8")
    v1_combined = out.all_combined_file.read_text(encoding="utf-8")
    json.loads(v1_chunks)
    json.loads(v1_combined)

    # Run 2: the second temp file's json.dump is interrupted — the first
    # temp file (all_chunks) was already written COMPLETELY.
    call_count = {"n": 0}
    orijinal_dump = json.dump

    def _sayan_dump(obj, fp, **kw):
        call_count["n"] += 1
        if call_count["n"] == 2:
            fp.write('{"generated_at": "YARIM')
            raise KeyboardInterrupt("simulated kill mid-write")
        return orijinal_dump(obj, fp, **kw)

    with monkeypatch.context() as mp:
        mp.setattr("medrag.pipeline.chunker.cli.json.dump", _sayan_dump)
        with pytest.raises(KeyboardInterrupt):
            calistir(env)

    # AFTER the interruption: archive/ was NOT created (archiving only
    # happens once BOTH temp files are SUCCESSFULLY written) and the v1
    # files are not corrupted/changed — write-then-replace + "write first,
    # archive after" ordering keeps the old pair fully intact.
    assert not (out.root / "archive").exists(), (
        "archiving must only happen after BOTH temp files are written SUCCESSFULLY")
    assert out.all_chunks_file.read_text(encoding="utf-8") == v1_chunks, (
        "the old all_chunks.json must be UNAFFECTED by the interruption"
    )
    assert out.all_combined_file.read_text(encoding="utf-8") == v1_combined, (
        "the old all_combined.json must be UNAFFECTED by the interruption"
    )
    # Real/permanent file names (all_chunks.json, all_combined.json) NEVER
    # saw half-written data.
    for ad in ("all_chunks.json", "all_combined.json"):
        json.loads((out.root / ad).read_text(encoding="utf-8"))
    # Temp (.tmp_*) files are also cleaned up — `except BaseException` cleanup
    # runs on KeyboardInterrupt too, no leftover on disk.
    assert not list(out.root.glob(".tmp_*")), "temp files must be cleaned up"

    # --- The next (real) run: the layout (consolidated two-file output)
    # requires the full corpus to be reprocessed — this is not "resume
    # from where it stopped" but a CONSISTENT full recovery (the only
    # semantics the chunk stage promises).
    assert calistir(env) == 0
    v2_chunks = json.loads(out.all_chunks_file.read_text(encoding="utf-8"))
    assert set(v2_chunks["documents"]) == {"a"}
    arsiv_klasorleri = list((out.root / "archive").iterdir())
    assert len(arsiv_klasorleri) == 1, "the successful run after the interruption must have archived v1"