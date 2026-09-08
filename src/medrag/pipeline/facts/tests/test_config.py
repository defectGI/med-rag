"""O-04 (2026-08-20): restored from the
`facts/tests/test_config.py` in the `facts-kodu-son-hali` tag -- only the
import path was updated from `facts.config` to
`medrag.pipeline.facts.facts_config` (O-03 changed the filename; see
facts_config.py header comment). Content/asserts UNCHANGED."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from medrag.pipeline.facts.facts_config import (
    KNOWN_DOC_TYPES,
    FactsConfig,
    Vlm,
    load_config,
)


def test_default_config_loads_and_matches_karar_027():
    cfg = load_config()
    assert cfg.scope.exclude_doc_types == ["PRODUCT_IMAGE", "STP"]
    assert cfg.scope.include_summary_nodes is False
    assert cfg.queue.batch_size >= 1
    assert cfg.vocabulary.large_wave_warning_threshold >= 1
    assert cfg.vlm.render_dpi >= 1
    assert cfg.vlm.timeout_s > 0
    assert cfg.vlm.max_tokens >= 1
    assert cfg.vlm.page_render_format in ("png", "jpeg")


def test_in_scope_doc_type():
    cfg = load_config()
    assert cfg.in_scope_doc_type("DATASHEET") is True
    assert cfg.in_scope_doc_type("CE_DECLARATION") is True  # KARAR-027: deliberately included
    assert cfg.in_scope_doc_type("TECHNICAL_DRAWING") is True  # KARAR-027: deliberately included
    assert cfg.in_scope_doc_type("PRODUCT_IMAGE") is False
    assert cfg.in_scope_doc_type("STP") is False


def test_in_scope_tree_level_leaf_only_by_default():
    cfg = load_config()
    assert cfg.in_scope_tree_level(0) is True
    assert cfg.in_scope_tree_level(1) is False
    assert cfg.in_scope_tree_level(2) is False


def test_unknown_doc_type_rejected():
    with pytest.raises(ValidationError, match="unknown doc_type"):
        FactsConfig.model_validate(
            {
                "scope": {"exclude_doc_types": ["NOT_A_REAL_TYPE"], "include_summary_nodes": False},
                "queue": {"batch_size": 10},
                "vocabulary": {"large_wave_warning_threshold": 5},
            }
        )


def test_extra_key_rejected():
    with pytest.raises(ValidationError):
        FactsConfig.model_validate(
            {
                "scope": {"exclude_doc_types": [], "include_summary_nodes": False},
                "queue": {"batch_size": 10},
                "vocabulary": {"large_wave_warning_threshold": 5},
                "surprise": {"whatever": 1},
            }
        )


def test_all_nine_doc_types_known():
    # KARAR-027 rationale count: 9 fixed doc_types.
    assert len(KNOWN_DOC_TYPES) == 9


def test_vlm_section_round_trip():
    # ROADMAP 13f pass 4: page-render + VLM tuning section.
    vlm = Vlm(render_dpi=150, timeout_s=180.0, max_tokens=512, page_render_format="png")
    assert vlm.render_dpi == 150
    assert vlm.page_render_format == "png"


def test_vlm_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Vlm.model_validate(
            {"render_dpi": 150, "timeout_s": 180.0, "max_tokens": 512, "page_render_format": "png", "surprise": 1}
        )


def test_vlm_page_render_format_enum_locked():
    with pytest.raises(ValidationError):
        Vlm.model_validate(
            {"render_dpi": 150, "timeout_s": 180.0, "max_tokens": 512, "page_render_format": "bmp"}
        )


def test_vlm_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        FactsConfig.model_validate(
            {
                "scope": {"exclude_doc_types": [], "include_summary_nodes": False},
                "queue": {"batch_size": 10},
                "vocabulary": {"large_wave_warning_threshold": 5},
                "llm": {"max_tokens_extract": 4096},
                # vlm missing -- FactsConfig's required 5th field, must fail loudly with extra="forbid".
            }
        )