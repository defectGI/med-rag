"""base.py boundary hardening: a malformed serialized IR fails loudly at
`from_dict`/`from_json` (IRParseError naming the path) instead of leaking a None
inward; a file newer than this reader is rejected; optional-but-absent fields are
NOT errors."""

from __future__ import annotations

import json

import pytest

from medrag.pipeline.parser.parsers.base import (
    IR_VERSION,
    ImageBlock,
    IRParseError,
    ParagraphBlock,
    ParsedDocument,
    migrate_dict,
)


def _min_doc_dict(**over) -> dict:
    d = {"ir_version": IR_VERSION, "doc_id": "d", "source_path": "x",
         "fmt": "markdown", "blocks": []}
    d.update(over)
    return d


# --- happy path ---------------------------------------------------------------


def test_valid_roundtrip():
    doc = ParsedDocument(doc_id="d", source_path="x", fmt="markdown",
                         blocks=[ParagraphBlock(id="b0", text="hi")])
    back = ParsedDocument.from_json(doc.to_json())
    assert back.doc_id == "d"
    assert back.blocks[0].text == "hi"


def test_visual_type_confidence_roundtrips():
    # v10: the classifier's confidence in visual_type survives to_dict/from_dict
    # and is a real float, not dropped like it used to be.
    doc = ParsedDocument(doc_id="d", source_path="x", fmt="pdf",
                         blocks=[ImageBlock(id="b0", image_index=1,
                                            visual_type="block_diagram",
                                            visual_type_source="vlm",
                                            visual_type_confidence=0.42)])
    back = ParsedDocument.from_json(doc.to_json())
    assert back.blocks[0].visual_type_confidence == 0.42


def test_visual_type_confidence_absent_is_none():
    # Purely optional: a block without it (older file / deterministic source)
    # reads back as None, not an error.
    doc = ParsedDocument.from_dict(_min_doc_dict(
        blocks=[{"type": "image", "id": "b0", "image_index": 1,
                 "visual_type": "chart", "visual_type_source": "structural"}]))
    assert doc.blocks[0].visual_type_confidence is None


def test_absent_optional_field_is_not_an_error():
    # heading_path/provenance/etc. omitted -> None, not a failure
    doc = ParsedDocument.from_dict(_min_doc_dict(
        blocks=[{"type": "paragraph", "id": "b0", "text": "hi"}]))
    assert doc.blocks[0].heading_path is None
    assert doc.blocks[0].provenance is None


# --- missing required fields fail with a located error ------------------------


def test_missing_doc_field_raises_with_path():
    bad = _min_doc_dict()
    del bad["doc_id"]
    with pytest.raises(IRParseError) as ei:
        ParsedDocument.from_dict(bad)
    assert "doc_id" in str(ei.value)


def test_block_missing_type_names_index():
    bad = _min_doc_dict(blocks=[{"id": "b0", "text": "hi"}])
    with pytest.raises(IRParseError) as ei:
        ParsedDocument.from_dict(bad)
    assert ei.value.path == "blocks[0]"
    assert "type" in ei.value.reason


def test_block_missing_required_field_names_index():
    # a heading with no 'id' -> _base_kwargs KeyError, surfaced as blocks[1]
    bad = _min_doc_dict(blocks=[
        {"type": "paragraph", "id": "b0", "text": "ok"},
        {"type": "heading", "text": "no id", "level": 1}])
    with pytest.raises(IRParseError) as ei:
        ParsedDocument.from_dict(bad)
    assert ei.value.path == "blocks[1]"
    assert "id" in ei.value.reason


def test_unknown_block_type_rejected():
    bad = _min_doc_dict(blocks=[{"type": "sparkle", "id": "b0"}])
    with pytest.raises(IRParseError) as ei:
        ParsedDocument.from_dict(bad)
    assert "sparkle" in str(ei.value)


# --- version gate / migration seam --------------------------------------------


def test_newer_version_rejected():
    with pytest.raises(IRParseError) as ei:
        migrate_dict(_min_doc_dict(ir_version=IR_VERSION + 1))
    assert "ir_version" in str(ei.value)


def test_non_int_version_rejected():
    with pytest.raises(IRParseError):
        migrate_dict(_min_doc_dict(ir_version="4"))


def test_from_json_rejects_future_file():
    text = json.dumps(_min_doc_dict(ir_version=IR_VERSION + 5))
    with pytest.raises(IRParseError):
        ParsedDocument.from_json(text)


def test_migrate_passes_current_version_through():
    d = _min_doc_dict()
    assert migrate_dict(d) is d  # already current -> nothing to rewrite


# --- v8 -> v9: type-specific description fields become the shared set ----------


def _v8_table(**over) -> dict:
    d = {"type": "table", "id": "b0",
         "table": {"n_rows": 1, "n_cols": 1,
                   "cells": [[{"blocks": []}]], "merges": []}}
    d.update(over)
    return d


def test_v8_table_description_becomes_shared_description():
    doc = ParsedDocument.from_dict(migrate_dict(_min_doc_dict(
        ir_version=8,
        blocks=[_v8_table(table_description="A spec table.",
                          table_facts=["The range is 0-40 C."])])))
    block = doc.blocks[0]
    assert block.description == "A spec table."
    assert block.facts == ["The range is 0-40 C."]
    # the old name implied the producer: cells -> LLM
    assert block.description_source == "llm-cells"


def test_v8_chart_description_becomes_shared_description():
    doc = ParsedDocument.from_dict(migrate_dict(_min_doc_dict(
        ir_version=8,
        blocks=[{"type": "image", "id": "b0", "image_index": 1, "locator": {},
                 "visual_type": "chart",
                 "chart_description": "Bar chart. Series 'X': A=1."}])))
    block = doc.blocks[0]
    assert block.description == "Bar chart. Series 'X': A=1."
    assert block.description_source == "structural"  # chart XML, no model call
    assert block.visual_type == "chart"  # untouched by the migration


def test_v8_migration_reaches_blocks_nested_in_table_cells():
    """A cell body is a block list like any other -- a migration that only
    walked top-level blocks would leave old field names alive one level down."""
    nested = _v8_table(id="nested", table_description="Inner table.")
    outer = _v8_table(id="b0", table={
        "n_rows": 1, "n_cols": 1,
        "cells": [[{"blocks": [nested]}]], "merges": []})
    doc = ParsedDocument.from_dict(migrate_dict(
        _min_doc_dict(ir_version=8, blocks=[outer])))
    inner = doc.blocks[0].table.cells[0][0].blocks[0]
    assert inner.description == "Inner table."
    assert inner.description_source == "llm-cells"


def test_v8_migration_stamps_the_new_version():
    """`ir_version` rides through from_dict into to_dict, so a migration that
    renames fields MUST restamp it -- otherwise a migrated-then-resaved file
    claims v8 while holding v9 field names."""
    doc = ParsedDocument.from_dict(migrate_dict(_min_doc_dict(
        ir_version=8, blocks=[_v8_table(table_description="x")])))
    assert doc.ir_version == IR_VERSION
    assert json.loads(doc.to_json())["ir_version"] == IR_VERSION


def test_v8_block_without_a_description_gains_no_fields():
    """Absent stays absent: a table that was never described must not acquire
    an empty `description`/`description_source` out of the migration."""
    doc = ParsedDocument.from_dict(migrate_dict(
        _min_doc_dict(ir_version=8, blocks=[_v8_table()])))
    block = doc.blocks[0]
    assert block.description is None
    assert block.description_source is None
    assert block.facts is None
    assert "description" not in block.to_dict()


def test_v8_migration_drops_the_old_names_entirely():
    doc = ParsedDocument.from_dict(migrate_dict(_min_doc_dict(
        ir_version=8,
        blocks=[_v8_table(table_description="d", table_facts=["f"]),
                {"type": "image", "id": "b1", "image_index": 1, "locator": {},
                 "chart_description": "c"}])))
    text = doc.to_json()
    for old in ("table_description", "table_facts", "chart_description"):
        assert old not in text
