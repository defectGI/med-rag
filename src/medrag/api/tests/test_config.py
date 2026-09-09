from pathlib import Path

import pytest
from pydantic import ValidationError

from medrag.api.config import ChatbotConfig, load_config


def test_load_default_config():
    cfg = load_config()
    assert cfg.routing.default_flow == "default_topn"
    # med-rag (C5): yapılandırılmış spec yolu kapalı; her tıbbi intent tek
    # vektör yoluna (default_topn) düşer -- out_of_scope hariç.
    assert cfg.routing.intents["medical_fact"] == "default_topn"
    assert cfg.routing.intents["clinical_decision"] == "default_topn"
    assert cfg.routing.intents["out_of_scope"] == "no_retrieval"
    assert cfg.flow.top_n_k >= 1
    assert cfg.flow.sql_k >= 1
    assert cfg.flow.sql_topn_k >= 1
    assert cfg.session.decay_window >= 1
    assert 0.0 <= cfg.session.hedge_confidence_threshold <= 1.0
    assert isinstance(cfg.session.fuse_intent, bool)
    assert cfg.session.sql_concurrency >= 1
    # Visual visibility filter, default empty (everything is shown).
    assert cfg.visual.exclude_types == []
    # Quality-analysis JSON conversation dump (see conversation_log.py), enabled by default.
    assert cfg.conversation_log.enabled is True
    assert cfg.conversation_log.dir == "logs/conversations"
    # Routing of background lines into the conversation folder is enabled BY DEFAULT.
    assert cfg.conversation_log.capture_background is True
    # FlowContext.result_shape threshold.
    assert cfg.result_shape.few_max >= 1


def test_override_deep_merges(tmp_path: Path):
    override = tmp_path / "override.toml"
    override.write_text('[flow]\ntop_n_k = 3\n', encoding="utf-8")
    cfg = load_config(override=override)
    assert cfg.flow.top_n_k == 3
    # untouched keys come from the defaults
    assert cfg.flow.sql_k == load_config().flow.sql_k
    assert cfg.routing.default_flow == "default_topn"


def test_unknown_key_rejected():
    with pytest.raises(ValidationError):
        ChatbotConfig.model_validate({
            "routing": {"default_flow": "default_topn"},
            "flow": {"top_n_k": 5, "sql_k": 5, "sql_topn_k": 5},
            "unknown_section": {},
        })


def test_missing_key_rejected():
    with pytest.raises(ValidationError):
        ChatbotConfig.model_validate({"routing": {"default_flow": "default_topn"}})


# --- preamble ---------------------------------------------------------------


def test_preamble_varsayilan_olarak_KAPALI():
    """The mechanism was disabled via a toggle; the code remains intact.
    Flipping `enabled = true` in this test would turn the feature back on --
    so this line is a deliberate lock, not an accident."""
    cfg = load_config()
    assert cfg.preamble.enabled is False
    assert "whatsapp" in cfg.preamble.channels
    assert cfg.preamble.watchdog_seconds > 0
    # Measured fast flows (see the comment in config/default.toml [preamble]).
    assert "no_retrieval" in cfg.preamble.fast_flows


def test_preamble_kapaliyken_hicbir_kanalda_uretilmez():
    """Default (disabled) config: EVERY channel gets `None` -> the Orchestrator
    produces no bubble at all, behavior identical to before the feature was disabled."""
    from medrag.api.factory import preamble_settings_from_config

    cfg = load_config()
    for kanal in ("whatsapp", "web", "bilinmeyen_kanal"):
        assert preamble_settings_from_config(cfg, kanal) is None


def test_preamble_acilinca_kanal_bazli_calisir(tmp_path):
    """When the toggle is re-enabled, the channel filter works as before --
    the mechanism was DISABLED, not BROKEN."""
    from medrag.api.factory import preamble_settings_from_config

    override = tmp_path / "acik.toml"
    override.write_text("[preamble]\nenabled = true\n", encoding="utf-8")
    cfg = load_config(override)
    assert preamble_settings_from_config(cfg, "whatsapp") is not None
    assert preamble_settings_from_config(cfg, "web") is not None
    assert preamble_settings_from_config(cfg, "bilinmeyen_kanal") is None


def test_preamble_kapaliyken_hicbir_kanal_acilmaz(tmp_path):
    from medrag.api.factory import preamble_settings_from_config

    override = tmp_path / "kapali.toml"
    override.write_text("[preamble]\nenabled = false\n", encoding="utf-8")
    cfg = load_config(override)
    assert preamble_settings_from_config(cfg, "whatsapp") is None
    assert preamble_settings_from_config(cfg, "web") is None
