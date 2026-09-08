"""Two-layer config loading tests."""

from __future__ import annotations

import pytest

from text2sql_native.config import load_settings
from text2sql_native.errors import ConfigError


def _write_config(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[general]
sql_dialect = "mysql"
log_level = "DEBUG"
strict_json = false

[linking]
temperature = 0.3
max_tables = 4

[generation]
max_tokens = 2048

[retry]
max_attempts = 5
""",
        encoding="utf-8",
    )
    return cfg


def test_load_settings_merges_both_layers(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path)
    monkeypatch.setenv("SCHEMA_PATH", "examples/schema.yaml")
    monkeypatch.setenv("PROMPT_DIR", "text2sql_native/prompts")
    monkeypatch.setenv("LINKING_PROVIDER", "openai")
    monkeypatch.setenv("LINKING_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SQL_PROVIDER", "anthropic")
    monkeypatch.setenv("SQL_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    settings = load_settings(config_path=cfg)

    # .env layer
    assert settings.linking_selection.provider == "openai"
    assert settings.linking_selection.api_key == "sk-openai"
    assert settings.sql_selection.provider == "anthropic"
    assert settings.sql_selection.api_key == "sk-anthropic"
    # config.toml layer
    assert settings.general.sql_dialect == "mysql"
    assert settings.general.strict_json is False
    assert settings.linking.temperature == 0.3
    assert settings.linking.max_tables == 4
    assert settings.generation.max_tokens == 2048
    assert settings.retry.max_attempts == 5
    # default fallthrough for an unspecified key
    assert settings.generation.temperature == 0.0
    # num_ctx not set anywhere in _write_config's [linking]/[generation] -> None
    assert settings.linking.num_ctx is None
    assert settings.generation.num_ctx is None


def test_num_ctx_reads_from_toml(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        """
[linking]
num_ctx = 16384

[generation]
num_ctx = 8192
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCHEMA_PATH", "examples/schema.yaml")
    monkeypatch.setenv("LINKING_PROVIDER", "ollama")
    monkeypatch.setenv("LINKING_MODEL", "gemma4:26b")
    monkeypatch.setenv("SQL_PROVIDER", "ollama")
    monkeypatch.setenv("SQL_MODEL", "qwen2.5-coder:32b")

    settings = load_settings(config_path=cfg)

    assert settings.linking.num_ctx == 16384
    assert settings.generation.num_ctx == 8192


def test_missing_required_env_raises(tmp_path, monkeypatch):
    cfg = _write_config(tmp_path)
    for var in (
        "SCHEMA_PATH",
        "PROMPT_DIR",
        "LINKING_PROVIDER",
        "LINKING_MODEL",
        "SQL_PROVIDER",
        "SQL_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ConfigError) as exc:
        load_settings(env_file="/nonexistent/.env", config_path=cfg)
    assert "SCHEMA_PATH" in str(exc.value)


def _set_min_env(monkeypatch):
    monkeypatch.setenv("SCHEMA_PATH", "examples/schema.yaml")
    monkeypatch.setenv("PROMPT_DIR", "text2sql_native/prompts")
    monkeypatch.setenv("LINKING_PROVIDER", "openai")
    monkeypatch.setenv("LINKING_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SQL_PROVIDER", "openai")
    monkeypatch.setenv("SQL_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")


def test_output_mode_defaults_to_structured(tmp_path, monkeypatch):
    _set_min_env(monkeypatch)
    cfg = _write_config(tmp_path)  # no output_mode set
    settings = load_settings(config_path=cfg)
    assert settings.generation.output_mode == "structured"
    assert settings.generation.prompt_template == "generation.txt"


def test_raw_mode_defaults_to_raw_template(tmp_path, monkeypatch):
    _set_min_env(monkeypatch)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[generation]\noutput_mode = "raw"\n', encoding="utf-8")
    settings = load_settings(config_path=cfg)
    assert settings.generation.output_mode == "raw"
    assert settings.generation.prompt_template == "generation_raw.txt"


def test_invalid_output_mode_raises(tmp_path, monkeypatch):
    _set_min_env(monkeypatch)
    cfg = tmp_path / "config.toml"
    cfg.write_text('[generation]\noutput_mode = "banana"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config_path=cfg)


def test_missing_config_toml_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEMA_PATH", "examples/schema.yaml")
    monkeypatch.setenv("PROMPT_DIR", "text2sql_native/prompts")
    monkeypatch.setenv("LINKING_PROVIDER", "openai")
    monkeypatch.setenv("LINKING_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SQL_PROVIDER", "openai")
    monkeypatch.setenv("SQL_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    with pytest.raises(ConfigError):
        load_settings(config_path=tmp_path / "missing.toml")
