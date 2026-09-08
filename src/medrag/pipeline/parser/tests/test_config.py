"""Parser config loader tests (offline): TOML defaults + env override +
validation. The critical one: env overrides beat the TOML defaults -- the
`pipeline/cli/config/*.env` profiles (e.g. `PDF_VLM=0`) depend on that
behavior.
"""

from __future__ import annotations

import pytest

from medrag.pipeline.parser.config import ParserConfig, get_config, load_config


def test_defaults_match_old_hardcoded():
    """The TOML defaults match the old in-code hardcoded defaults exactly."""
    c = load_config(env={})
    assert c.image_ocr.confidence_threshold == 0.7
    assert c.image_ocr.max_tokens == 1024
    assert c.image_ocr.concurrency == 4
    assert c.image_ocr.fallback == ""
    assert c.pdf.render_dpi == 150
    assert c.pdf.vlm is True
    assert c.pdf.vlm_max_tokens == 8192
    assert c.pdf.heading_reconcile is True
    assert c.pdf.fullpage_render_min_vector == 250
    assert c.pdf.broken_cmap_min == 5
    assert c.pdf.scanned_coverage_min == 0.5
    assert c.pdf.line_match == 0.8
    assert c.pdf.containment == 0.9
    assert c.pdf.tesseract_fallback is True
    assert c.pdf.table.bands_min_confidence == 0.7
    assert c.pdf.table.struct_min_coverage == 0.9
    assert c.table.max_rows == 150
    assert c.table.llm_check is False
    assert c.table.facts is True
    assert c.table.header_llm is True
    assert c.table.check_retries == 3
    # Shared (type-independent) describe knobs -- they used to live under [table].
    assert c.describe.types == ["table", "chart", "block_diagram",
                                "technical_drawing", "flowchart"]
    assert c.describe.max_tokens == 512
    assert c.describe.concurrency == 4
    assert c.describe.context.enabled is False
    assert c.describe.context.before == 1
    assert c.describe.context.after == 1
    assert c.describe.context.max_chars == 400
    assert c.table_struct.max_tokens == 8192
    assert c.progress.bar is True
    assert c.visual.classify is True
    assert c.visual.extract_charts is True
    assert c.visual.extract_smartart is True
    assert c.visual.exclude_types == []
    assert c.visual.types == ["table", "chart", "block_diagram",
                              "technical_drawing", "flowchart",
                              "product_photo", "decorative", "unknown"]
    assert c.visual.max_tokens == 512


def test_env_override_wins():
    """An env variable beats the TOML default (the basis of the .env profiles)."""
    c = load_config(env={
        "PDF_VLM": "0",
        "TABLE_MAX_ROWS": "5",
        "IMAGE_OCR_CONFIDENCE_THRESHOLD": "0.95",
        "DESCRIBE_CONTEXT": "1",
        "TABLE_STRUCT_MAX_TOKENS": "16384",
        "VISUAL_CLASSIFY": "0",
        "VISUAL_EXTRACT_CHARTS": "0",
        "VISUAL_EXTRACT_SMARTART": "0",
        "VISUAL_MAX_TOKENS": "256",
    })
    assert c.pdf.vlm is False
    assert c.table.max_rows == 5
    assert c.image_ocr.confidence_threshold == 0.95
    assert c.describe.context.enabled is True
    assert c.table_struct.max_tokens == 16384
    assert c.visual.classify is False
    assert c.visual.extract_charts is False
    assert c.visual.extract_smartart is False
    assert c.visual.max_tokens == 256


def test_describe_env_knobs():
    """`[table.context]`/`[table].concurrency` moved to `[describe]` (the concept
    was never table-specific); `DESCRIBE_*` is the single canonical env family."""
    c = load_config(env={
        "DESCRIBE_MAX_TOKENS": "128",
        "DESCRIBE_CONCURRENCY": "9",
        "DESCRIBE_CONTEXT": "1",
        "DESCRIBE_CONTEXT_BEFORE": "3",
        "DESCRIBE_CONTEXT_AFTER": "2",
        "DESCRIBE_CONTEXT_MAX_CHARS": "800",
    })
    assert c.describe.max_tokens == 128
    assert c.describe.concurrency == 9
    assert c.describe.context.enabled is True
    assert c.describe.context.before == 3
    assert c.describe.context.after == 2
    assert c.describe.context.max_chars == 800


def test_legacy_table_context_and_concurrency_env_names_no_longer_exist():
    """`TABLE_CONTEXT*`/`TABLE_CONCURRENCY` were retired, not aliased: their
    users were migrated to `DESCRIBE_CONTEXT*`, and the "two names, one
    setting" ambiguity was deliberately left out. They are now silently ignored
    (SKIP), so the default.toml value stays untouched."""
    c = load_config(env={
        "TABLE_CONTEXT": "1", "TABLE_CONTEXT_BEFORE": "3",
        "TABLE_CONTEXT_AFTER": "2", "TABLE_CONTEXT_MAX_CHARS": "800",
        "TABLE_CONCURRENCY": "2",
    })
    assert c.describe.context.enabled is False   # default.toml, not overridden
    assert c.describe.context.before == 1
    assert c.describe.context.after == 1
    assert c.describe.context.max_chars == 400
    assert c.describe.concurrency == 4


def test_progress_bar_barbool_semantics():
    assert load_config(env={"PROGRESS_BAR": "off"}).progress.bar is False
    assert load_config(env={"PROGRESS_BAR": "0"}).progress.bar is False
    assert load_config(env={"PROGRESS_BAR": "1"}).progress.bar is True
    assert load_config(env={"PROGRESS_BAR": "anything-else"}).progress.bar is True


def test_bool_env_semantics():
    assert load_config(env={"PDF_VLM": "yes"}).pdf.vlm is True
    assert load_config(env={"PDF_VLM": "garbage"}).pdf.vlm is False  # non-truthy -> False


@pytest.mark.parametrize("bad", ["notanint", "", "   "])
def test_invalid_or_empty_env_falls_to_default(bad):
    """An empty/whitespace/invalid value SKIPS the override — the TOML default
    stands (the _env_int/_env_float semantics)."""
    assert load_config(env={"TABLE_MAX_ROWS": bad}).table.max_rows == 150
    assert load_config(env={"PDF_RENDER_DPI": bad}).pdf.render_dpi == 150


def test_get_config_reads_process_env(monkeypatch):
    """get_config() reads the env fresh (so test monkeypatching works)."""
    monkeypatch.setenv("TABLE_MAX_ROWS", "7")
    assert get_config().table.max_rows == 7
    monkeypatch.delenv("TABLE_MAX_ROWS", raising=False)
    assert get_config().table.max_rows == 150


def test_unknown_toml_key_rejected(tmp_path):
    from pydantic import ValidationError
    ov = tmp_path / "bad.toml"
    ov.write_text("[pdf]\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(env={}, override=ov)


def test_missing_section_fails():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ParserConfig.model_validate({"pdf": {}})


def test_parser_config_override_file_merges(tmp_path):
    """A PARSER_CONFIG partial TOML merges over the defaults; env still wins."""
    ov = tmp_path / "ov.toml"
    ov.write_text("[table]\nmax_rows = 20\n", encoding="utf-8")
    c = load_config(env={}, override=ov)
    assert c.table.max_rows == 20        # from the override file
    assert c.table.check_retries == 3    # untouched default
    # env beats the override file too:
    c2 = load_config(env={"TABLE_MAX_ROWS": "99"}, override=ov)
    assert c2.table.max_rows == 99


def test_visual_exclude_types_only_via_override_file(tmp_path):
    """`exclude_types` is a list field
    (same pattern as the chunker's `boilerplate.drop_headings`): no env caster
    at all, only a PARSER_CONFIG partial TOML can change it."""
    ov = tmp_path / "ov.toml"
    ov.write_text(
        '[visual]\nexclude_types = ["block_diagram", "decorative"]\n',
        encoding="utf-8")
    c = load_config(env={}, override=ov)
    assert c.visual.exclude_types == ["block_diagram", "decorative"]
    assert c.visual.classify is True  # the untouched default stands


def test_visual_unknown_key_rejected(tmp_path):
    from pydantic import ValidationError
    ov = tmp_path / "bad.toml"
    ov.write_text("[visual]\nbogus = 1\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(env={}, override=ov)


def test_gorsel_maliyet_anahtarlari_env_ile_kapanir():
    """The four switches an operator needs to be able to turn off from Coolify.
    All are ON by default; all must be closable from the env (so no file has to
    be edited outside Coolify's env screen)."""
    varsayilan = load_config(env={})
    assert varsayilan.image_ocr.enabled is True
    assert varsayilan.describe.enabled is True
    assert varsayilan.describe.visual_check is True
    assert varsayilan.visual.classify is True

    kapali = load_config(env={
        "IMAGE_OCR_ENABLED": "0",
        "DESCRIBE_ENABLED": "0",
        "DESCRIBE_VISUAL_CHECK": "0",
        "VISUAL_CLASSIFY": "0",
    })
    assert kapali.image_ocr.enabled is False
    assert kapali.describe.enabled is False
    assert kapali.describe.visual_check is False
    assert kapali.visual.classify is False
