"""images/visual_classify.py tests: cache hit/miss, unknown fallback on
malformed replies / no VLM / classification disabled, and that a cache hit
never re-calls the classifier. No network: VLM/LLM are fakes.
"""

from __future__ import annotations

import json

import pytest

from medrag.pipeline.parser.config import Visual
from medrag.pipeline.parser.images.visual_classify import (
    classify_crop,
    classify_page_regions,
    classify_table_text,
    log_unknown,
)
from medrag.pipeline.parser.llm import LLMError


class FakeVLM:
    def __init__(self, reply: str | None = None, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls += 1
        if self.raises:
            raise LLMError("boom")
        return self.reply


class FakeLLM:
    def __init__(self, reply: str | None = None, raises: bool = False) -> None:
        self.reply = reply
        self.raises = raises
        self.calls = 0

    def complete(self, *, system, user, max_tokens=1024):
        self.calls += 1
        if self.raises:
            raise LLMError("boom")
        return self.reply


def _cfg(**overrides) -> Visual:
    base = {
        "classify": True, "extract_charts": True, "extract_smartart": True,
        "exclude_types": [],
        "types": ["table", "chart", "block_diagram", "technical_drawing",
                  "flowchart", "product_photo", "decorative", "unknown"],
        "max_tokens": 512,
    }
    base.update(overrides)
    return Visual.model_validate(base)


@pytest.fixture(autouse=True)
def _isolate_labels_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))


def test_classify_crop_calls_vlm_and_caches():
    vlm = FakeVLM(json.dumps({"type": "block_diagram", "confidence": 0.9}))
    cfg = _cfg()
    result = classify_crop(b"crop-bytes", "image/png", vlm, cfg)
    assert result.visual_type == "block_diagram"
    assert result.source == "vlm"
    assert result.confidence == 0.9
    assert vlm.calls == 1

    # Second call, identical bytes: cache hit, VLM never called again.
    result2 = classify_crop(b"crop-bytes", "image/png", vlm, cfg)
    assert result2.visual_type == "block_diagram"
    assert result2.source == "cache"
    assert vlm.calls == 1


def test_classify_crop_different_bytes_not_cached_together():
    vlm = FakeVLM(json.dumps({"type": "chart", "confidence": 0.8}))
    cfg = _cfg()
    classify_crop(b"crop-a", "image/png", vlm, cfg)
    classify_crop(b"crop-b", "image/png", vlm, cfg)
    assert vlm.calls == 2


def test_classify_crop_malformed_json_is_unknown():
    vlm = FakeVLM("not json at all")
    result = classify_crop(b"x", "image/png", vlm, _cfg())
    assert result.visual_type == "unknown"
    assert result.source == "vlm"


def test_classify_crop_type_outside_taxonomy_is_unknown():
    vlm = FakeVLM(json.dumps({"type": "spaceship", "confidence": 0.99}))
    result = classify_crop(b"x", "image/png", vlm, _cfg())
    assert result.visual_type == "unknown"


def test_classify_crop_no_vlm_is_unavailable_and_not_cached():
    cfg = _cfg()
    result = classify_crop(b"x", "image/png", None, cfg)
    assert result.visual_type == "unknown"
    assert result.source == "unavailable"

    # Not cached: a later call with a real VLM still gets classified.
    vlm = FakeVLM(json.dumps({"type": "table", "confidence": 1.0}))
    result2 = classify_crop(b"x", "image/png", vlm, cfg)
    assert result2.visual_type == "table"
    assert result2.source == "vlm"
    assert vlm.calls == 1


def test_classify_crop_disabled_is_unavailable():
    vlm = FakeVLM(json.dumps({"type": "table", "confidence": 1.0}))
    cfg = _cfg(classify=False)
    result = classify_crop(b"x", "image/png", vlm, cfg)
    assert result.visual_type == "unknown"
    assert result.source == "unavailable"
    assert vlm.calls == 0


def test_classify_crop_llm_error_is_unknown():
    vlm = FakeVLM(raises=True)
    result = classify_crop(b"x", "image/png", vlm, _cfg())
    assert result.visual_type == "unknown"
    assert result.source == "unavailable"


def test_classify_table_text_calls_llm_and_caches():
    llm = FakeLLM(json.dumps({"type": "block_diagram", "confidence": 0.7}))
    cfg = _cfg()
    rows = [["AAF", "AAF", "AAF"], ["", "", ""]]
    result = classify_table_text(rows, llm, cfg)
    assert result.visual_type == "block_diagram"
    assert result.source == "text-llm"
    assert llm.calls == 1

    result2 = classify_table_text(rows, llm, cfg)
    assert result2.source == "cache"
    assert llm.calls == 1


def test_classify_table_text_no_llm_is_unavailable():
    result = classify_table_text([["a", "b"]], None, _cfg())
    assert result.visual_type == "unknown"
    assert result.source == "unavailable"


def test_classify_page_regions_calls_vlm_and_caches():
    vlm = FakeVLM(json.dumps({"regions": [
        {"type": "technical_drawing", "bbox": [0.1, 0.2, 0.9, 0.8], "confidence": 0.9},
    ]}))
    cfg = _cfg()
    regions = classify_page_regions(b"page-png", vlm, cfg)
    assert len(regions) == 1
    assert regions[0].visual_type == "technical_drawing"
    assert regions[0].bbox == (0.1, 0.2, 0.9, 0.8)
    assert regions[0].confidence == 0.9
    assert regions[0].source == "vlm"
    assert vlm.calls == 1

    # Second call, identical page bytes: cache hit, VLM never called again.
    regions2 = classify_page_regions(b"page-png", vlm, cfg)
    assert len(regions2) == 1
    assert regions2[0].source == "cache"
    assert vlm.calls == 1


def test_classify_page_regions_merges_are_left_to_the_model():
    """The prompt asks the model to merge fragments into one region -- this
    only checks the parser trusts however many regions it's given back
    (here two genuinely independent ones), not that it does any merging of
    its own (there is none; see classify_page_regions' docstring)."""
    vlm = FakeVLM(json.dumps({"regions": [
        {"type": "product_photo", "bbox": [0.0, 0.0, 0.4, 0.3], "confidence": 0.8},
        {"type": "table", "bbox": [0.5, 0.0, 1.0, 0.3], "confidence": 0.7},
    ]}))
    regions = classify_page_regions(b"page-png", vlm, _cfg())
    assert {r.visual_type for r in regions} == {"product_photo", "table"}


def test_classify_page_regions_drops_bad_entries():
    vlm = FakeVLM(json.dumps({"regions": [
        {"type": "chart", "bbox": [0.1, 0.1, 0.2, 0.2]},        # ok, no confidence
        {"type": "spaceship", "bbox": [0.0, 0.0, 1.0, 1.0]},    # bad type
        {"type": "chart", "bbox": [0.5, 0.5, 0.5, 0.5]},        # degenerate bbox
        {"type": "chart"},                                     # missing bbox
    ]}))
    regions = classify_page_regions(b"page-png", vlm, _cfg())
    assert len(regions) == 1
    assert regions[0].bbox == (0.1, 0.1, 0.2, 0.2)


def test_classify_page_regions_no_vlm_returns_empty_and_not_cached():
    cfg = _cfg()
    assert classify_page_regions(b"page-png", None, cfg) == []

    vlm = FakeVLM(json.dumps({"regions": [
        {"type": "table", "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 1.0},
    ]}))
    regions = classify_page_regions(b"page-png", vlm, cfg)
    assert len(regions) == 1
    assert vlm.calls == 1


def test_classify_page_regions_disabled_returns_empty():
    vlm = FakeVLM(json.dumps({"regions": [{"type": "table", "bbox": [0, 0, 1, 1]}]}))
    regions = classify_page_regions(b"page-png", vlm, _cfg(classify=False))
    assert regions == []
    assert vlm.calls == 0


def test_classify_page_regions_malformed_json_is_empty():
    vlm = FakeVLM("not json at all")
    assert classify_page_regions(b"page-png", vlm, _cfg()) == []


def test_classify_page_regions_llm_error_is_empty():
    vlm = FakeVLM(raises=True)
    assert classify_page_regions(b"page-png", vlm, _cfg()) == []


def test_log_unknown_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    log_unknown("doc1", "b24", "abc123", {"page": 3})
    log_unknown("doc1", "b51", "def456", {"page": 7})
    lines = (tmp_path / "labels" / "unknown.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["doc_id"] == "doc1"
    assert first["block_id"] == "b24"
    assert first["ref"] == "abc123"
    assert first["page"] == 3
    assert "classified_at" in first
