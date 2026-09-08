"""VLM canary'sinin gercekten VLM'e is dustugunde kostugu (ve dusmediginde
kosmadigi) testleri.

Baglam (2026-08-25 gercek olay): gorsel maliyet anahtarlari kapatilip
`PDF_VLM=0` verildiginde parse yine `_probe_or_abort("VLM")`e giriyor,
Ollama dejenere cikti dondurunce de kosuyu durduruyordu -- VLM'e TEK bir
cagri gitmeyecek olsa bile. Tek kacis `HEALTH_CHECK=0`di, o da LLM
canary'sini de kapattigi icin korelticiydi.
"""

from __future__ import annotations

import pytest

from medrag.pipeline.cli.run_parse_pipeline import _vlm_work_configured
from medrag.pipeline.parser.config import load_config


@pytest.fixture(autouse=True)
def _temiz_config(monkeypatch):
    """Her test kendi env'inden config kursun (parser config'i onbellekli)."""
    import medrag.pipeline.parser.config as pc
    monkeypatch.setattr(pc, "_CONFIG", None, raising=False)
    yield
    monkeypatch.setattr(pc, "_CONFIG", None, raising=False)


def _cfg(monkeypatch, **env):
    import medrag.pipeline.parser.config as pc
    cfg = load_config(env=env)
    monkeypatch.setattr(pc, "get_config", lambda: cfg)
    import medrag.pipeline.cli.run_parse_pipeline as rpp
    monkeypatch.setattr(rpp, "_parser_config", lambda: cfg)
    return cfg


def test_varsayilan_kurulumda_canary_kosar(monkeypatch):
    """Hicbir sey kapatilmamissa VLM'e is duser -- probe ATLANMAMALI.
    Bu testin asil isi: kapiyi eklerken korumayi kazara kapatmadigimiz."""
    _cfg(monkeypatch)
    assert _vlm_work_configured() is True


@pytest.mark.parametrize("acik", [
    "PDF_VLM",
    "IMAGE_OCR_ENABLED",
    "DESCRIBE_ENABLED",
    "VISUAL_CLASSIFY",
])
def test_tek_bir_tuketici_bile_acikken_canary_kosar(monkeypatch, acik):
    """Dort tuketiciden biri acikken VLM'e is dusebilir, probe kosmali.
    Muhafazakar taraf bilincli: yanlis tarafa dusmek bozuk bir VLM'i fark
    etmemek demek."""
    env = {k: "0" for k in
           ("PDF_VLM", "IMAGE_OCR_ENABLED", "DESCRIBE_ENABLED", "VISUAL_CLASSIFY")}
    env[acik] = "1"
    _cfg(monkeypatch, **env)
    assert _vlm_work_configured() is True


def test_dordu_de_kapaliyken_canary_atlanir(monkeypatch):
    """Kullanicinin senaryosu: digital-born korpus, VLM tamamen kapali.
    VLM'e tek cagri gitmeyecek, dolayisiyla bozuk bir VLM'in verebilecegi
    zarar da yok -- probe kosmamali ve kosuyu durdurmamali."""
    _cfg(monkeypatch, PDF_VLM="0", IMAGE_OCR_ENABLED="0",
         DESCRIBE_ENABLED="0", VISUAL_CLASSIFY="0")
    assert _vlm_work_configured() is False

TOML_TABLE_CHART = '[describe]\ntypes = ["table", "chart"]\n'
TOML_TABLE_TECHDRAW = '[describe]\ntypes = ["table", "technical_drawing"]\n'


def test_yalniz_tablo_ve_chart_aciklamasi_vlm_isi_sayilmaz(monkeypatch, tmp_path):
    """Kullanicinin gercek senaryosu: tablo aciklamasi KORUNSUN ama VLM hic
    kosmasin. Tablo hucrelerden LLM ile, chart XML'den deterministik uretilir
    -- ikisi de VLM'siz, dolayisiyla canary kosmamali. DESCRIBE_ENABLED ACIK
    kaliyor; kaba kapi bu senaryoda yanlis pozitif veriyordu."""
    ov = tmp_path / "ov.toml"
    ov.write_text(TOML_TABLE_CHART, encoding="utf-8")
    _cfg(monkeypatch, PARSER_CONFIG=str(ov), PDF_VLM="0",
         IMAGE_OCR_ENABLED="0", VISUAL_CLASSIFY="0")
    assert _vlm_work_configured() is False


def test_listede_tek_bir_vlm_tipi_varsa_canary_kosar(monkeypatch, tmp_path):
    """`technical_drawing` VLM-vision ile aciklanir -- listede oldugu surece
    VLM'e is duser ve canary kosmali. Inceltmenin korumayi delmedigini
    kilitler."""
    ov = tmp_path / "ov.toml"
    ov.write_text(TOML_TABLE_TECHDRAW, encoding="utf-8")
    _cfg(monkeypatch, PARSER_CONFIG=str(ov), PDF_VLM="0",
         IMAGE_OCR_ENABLED="0", VISUAL_CLASSIFY="0")
    assert _vlm_work_configured() is True
