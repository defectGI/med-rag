"""scripts/to_markdown.py stage split: parse (VLM half, checkpoint) then
finish (LLM half, final json+md) must equal the single-process "all" flow.

Offline: input is a plain markdown file (MarkdownParser needs no models), the
image is a data: URI, and the VLM/LLM are fakes injected by monkeypatching the
image_handler client getters -- no network, no Ollama.
"""

from __future__ import annotations

import base64
import json

import pytest

import medrag.pipeline.parser.images.image_handler as ih
from medrag.pipeline.parser.scripts.to_markdown import (
    stage_all,
    stage_finish,
    stage_parse,
)

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


class FakeVLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls += 1
        return self.text


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, *, system, user, max_tokens=1024):
        self.calls += 1
        return self.reply


@pytest.fixture
def doc_md(tmp_path):
    encoded = base64.b64encode(_PNG_1PX).decode()
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nSome prose.\n\n"
                   f"![alt](data:image/png;base64,{encoded})\n",
                   encoding="utf-8")
    return src


@pytest.fixture
def storage(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("STORAGE_OUTPUT_DIR", str(out))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    return out


def _fake_clients(monkeypatch):
    vlm = FakeVLM("Serial: 42")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "Serial: 42"}')
    monkeypatch.setattr(ih, "get_vlm_client", lambda *a, **k: vlm)
    monkeypatch.setattr(ih, "get_client", lambda *a, **k: llm)
    return vlm, llm


def test_parse_then_finish_equals_all(doc_md, storage, monkeypatch):
    vlm, llm = _fake_clients(monkeypatch)
    stage_all(doc_md, "single")
    assert vlm.calls == 2 and llm.calls == 1  # classify (A.5) + OCR per image

    stage_parse(doc_md, "phased")
    # +1 (OCR only): "phased" embeds the SAME image bytes as "single", so its
    # classify call is a cache hit (see visual_classify.py) -- proof the
    # cache is keyed by crop content, not by document.
    assert vlm.calls == 3
    assert llm.calls == 1              # the parse stage never touches the LLM
    ckpt = json.loads((storage / "phased.stage1.json").read_text(encoding="utf-8"))
    assert list(ckpt["raw_ocr"].values()) == ["Serial: 42"]
    assert not (storage / "phased.md").exists()   # no premature final output

    stage_finish("phased")
    assert llm.calls == 2

    single = json.loads((storage / "single.json").read_text(encoding="utf-8"))
    phased = json.loads((storage / "phased.json").read_text(encoding="utf-8"))
    single["doc_id"] = phased["doc_id"] = "x"
    # "phased" embeds the same image bytes as "single" (same source doc_md),
    # so its classify call is a cache hit -- visual_type_source legitimately
    # differs ("vlm" vs "cache") even though visual_type itself is identical.
    # Normalized away like doc_id above; not a real single-vs-split divergence.
    for doc in (single, phased):
        for block in doc["blocks"]:
            block.pop("visual_type_source", None)
    assert phased == single
    assert ((storage / "phased.md").read_text(encoding="utf-8")
            == (storage / "single.md").read_text(encoding="utf-8"))


def test_finish_is_rerunnable_from_kept_checkpoint(doc_md, storage, monkeypatch):
    # finish keeps the checkpoint -> a second finish (e.g. after an LLM-side
    # fix) works without re-running any VLM work.
    vlm, _llm = _fake_clients(monkeypatch)
    stage_parse(doc_md, "d")
    stage_finish("d")
    stage_finish("d")
    assert vlm.calls == 2  # classify (A.5) + OCR, only from stage_parse
    assert (storage / "d.stage1.json").exists()
    assert (storage / "d.md").exists()


def test_stage_parse_checkpoint_has_no_tmp_leftover(doc_md, storage, monkeypatch):
    _fake_clients(monkeypatch)
    stage_parse(doc_md, "d")
    leftovers = [p.name for p in storage.iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
