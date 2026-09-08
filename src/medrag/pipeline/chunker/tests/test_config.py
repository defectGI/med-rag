"""Configuration (`chunker.config`) tests — offline, dependency-free.

The DEFAULT values themselves are NOT asserted (so a number change in
TOML doesn't break tests); relationships and contracts are asserted.
Where a specific number is needed, the test writes its own override.
"""

import pytest
from pydantic import ValidationError

from medrag.pipeline.chunker.config import (
    DEFAULT_CONFIG_PATH,
    ChunkerConfig,
    load_config,
)


def override_yaz(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return dosya


# -- Default file -----------------------------------------------------------------


def test_varsayilan_yuklenir_ve_tutarli():
    cfg = load_config()
    assert cfg.limits.target_tokens >= 1
    assert cfg.limits.hard_max_tokens >= cfg.limits.target_tokens


def test_varsayilan_dosya_repoda():
    assert DEFAULT_CONFIG_PATH.is_file(), DEFAULT_CONFIG_PATH


# -- Visual-region type exclusion (visual classification plan) -------------------


def test_visual_exclude_types_varsayilan_bos():
    assert load_config().visual.exclude_types == []


def test_visual_exclude_types_override_ile_ayarlanir(tmp_path):
    cfg = load_config(override_yaz(
        tmp_path, '[visual]\nexclude_types = ["block_diagram", "decorative"]\n'))
    assert cfg.visual.exclude_types == ["block_diagram", "decorative"]


def test_visual_bilinmeyen_anahtar_reddedilir(tmp_path):
    with pytest.raises(ValidationError):
        load_config(override_yaz(tmp_path, "[visual]\nbogus = 1\n"))


# -- Flex allowance formula -------------------------------------------------------


def test_flex_allowance_formulu():
    cfg = load_config()
    target = cfg.limits.target_tokens
    for kind in ("table", "list", "code", "image", "paragraph", "heading"):
        beklenen = int(target * cfg.flex.ratios.ratio_for(kind))
        assert cfg.flex_allowance(kind) == beklenen


def test_effective_limit_hard_tavanini_asamaz(tmp_path):
    # Force target + flex > hard: 1000 + 500 > 1100.
    cfg = load_config(override_yaz(tmp_path, """
[limits]
target_tokens = 1000
hard_max_tokens = 1100
[flex.ratios]
table = 0.5
"""))
    assert cfg.flex_allowance("table") == 500
    assert cfg.effective_limit("table") == 1100


def test_effective_limit_tavana_takilmayan(tmp_path):
    cfg = load_config(override_yaz(tmp_path, """
[limits]
target_tokens = 100
hard_max_tokens = 1000
[flex.ratios]
list = 0.35
"""))
    assert cfg.effective_limit("list") == 135


def test_sifir_oran_esneme_yok(tmp_path):
    cfg = load_config(override_yaz(tmp_path, "[flex.ratios]\nheading = 0.0\n"))
    assert cfg.flex_allowance("heading") == 0
    assert cfg.effective_limit("heading") == cfg.limits.target_tokens


def test_bilinmeyen_blok_tipi_reddedilir():
    cfg = load_config()
    with pytest.raises(KeyError, match="unknown block type"):
        cfg.flex.ratios.ratio_for("footnote")


# -- Override / merge ----------------------------------------------------------


def test_kismi_override_merge(tmp_path):
    varsayilan = load_config()
    cfg = load_config(override_yaz(tmp_path, """
[limits]
target_tokens = 256
"""))
    # Changed keys changed...
    assert cfg.limits.target_tokens == 256
    # ...untouched keys stayed at default (including siblings in the same section).
    assert cfg.limits.hard_max_tokens == varsayilan.limits.hard_max_tokens
    assert cfg.table_split == varsayilan.table_split


# -- Boundary validation ---------------------------------------------------------


def test_bilinmeyen_anahtar_reddedilir(tmp_path):
    with pytest.raises(ValidationError):
        load_config(override_yaz(tmp_path, "[limits]\ntraget_tokens = 512\n"))


def test_bilinmeyen_bolum_reddedilir(tmp_path):
    with pytest.raises(ValidationError):
        load_config(override_yaz(tmp_path, "[splitting]\nx = 1\n"))


def test_hard_max_hedefin_altinda_reddedilir(tmp_path):
    with pytest.raises(ValidationError, match="hard_max_tokens"):
        load_config(override_yaz(tmp_path, "[limits]\nhard_max_tokens = 1\n"))


def test_negatif_oran_reddedilir(tmp_path):
    with pytest.raises(ValidationError):
        load_config(override_yaz(tmp_path, "[flex.ratios]\ntable = -0.1\n"))


def test_negatif_min_chunk_tokens_reddedilir(tmp_path):
    with pytest.raises(ValidationError):
        load_config(override_yaz(tmp_path, "[packing]\nmin_chunk_tokens = -1\n"))


def test_yeni_anahtarlar_yuklenir():
    # New keys present in default.toml with correct types.
    cfg = load_config()
    assert cfg.packing.min_chunk_tokens >= 0


def test_eksik_anahtar_reddedilir():
    # No code-side default: if a section is missing from TOML, loading must fail.
    with pytest.raises(ValidationError):
        ChunkerConfig.model_validate({"limits": {"target_tokens": 512,
                                                 "hard_max_tokens": 1024}})