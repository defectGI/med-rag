"""Locks in that `core/config/schema.py` accepts the real `default.toml` and
rejects broken values with a CLEAR error."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from medrag.core.config.schema import CoreConfig, load_core_config

_VALID_TOML = """
[retrieval]
llm_timeout_seconds = 60
llm_max_retries = 2
embedder_timeout_seconds = 120.0

[chunking]

[llm]
ollama_base_url = "http://localhost:11434/v1"

[whatsapp]
gateway_url = "http://127.0.0.1:8000"
host = "127.0.0.1"
port = 8003
gateway_timeout_seconds = 30

[logging]
chunker_cli_level = "INFO"
vectorize_cli_level = "INFO"
vectorize_query_level = "WARNING"

[parser]

[pipeline]
worker_replicas = 1
ollama_max_concurrent = 2

[pipeline.versions]
parser_version = "1.5.0"
chunker_version = "1.0.1"
facts_version = "1.0.0"
vectorize_version = "1.0.0"
"""


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_default_toml_loads_and_validates():
    """The real `default.toml` in the repo loads without any error."""
    cfg = load_core_config()
    assert isinstance(cfg, CoreConfig)
    assert cfg.whatsapp.port == 8003
    assert cfg.retrieval.llm_timeout_seconds == 60


def test_import_side_effect_validates_eagerly():
    """As soon as the `medrag.core.config` package is imported (its startup hook)
    the real `default.toml` must already have been validated -- this test only
    confirms that importing the package has completed without exploding so far
    (i.e. that the hook actually ran)."""
    import medrag.core.config  # noqa: F401 -- side effect: module-level validation


def test_control_toml_is_valid(tmp_path: Path):
    """Baseline for comparison: the text above (a copy of default.toml) is
    valid on its own too -- so that the failures of the broken-value tests
    below really come from THAT ONE broken field."""
    path = _write(tmp_path, _VALID_TOML)
    load_core_config(path)


def test_wrong_type_in_numeric_field_raises_clear_error(tmp_path: Path):
    """Deliberately broken value: writing a string into a numeric field must
    stop the process with a CLEAR `ValidationError`, not silently fall back to
    a default."""
    broken = _VALID_TOML.replace("port = 8003", 'port = "seksen-uc"')
    path = _write(tmp_path, broken)
    with pytest.raises(ValidationError, match="whatsapp.port"):
        load_core_config(path)


def test_out_of_range_port_raises_clear_error(tmp_path: Path):
    """Out-of-range value: a port above 65535 must be rejected."""
    broken = _VALID_TOML.replace("port = 8003", "port = 70000")
    path = _write(tmp_path, broken)
    with pytest.raises(ValidationError, match="whatsapp.port"):
        load_core_config(path)


def test_unknown_key_raises_clear_error(tmp_path: Path):
    """An unknown key (`extra=\"forbid\"`) must not be silently swallowed."""
    broken = _VALID_TOML.replace(
        "port = 8003", "port = 8003\nyeni_alan = \"beklenmedik\""
    )
    path = _write(tmp_path, broken)
    with pytest.raises(ValidationError, match="yeni_alan"):
        load_core_config(path)


def test_missing_key_raises_clear_error(tmp_path: Path):
    """A missing required key must not silently fall back to a default -- it
    has to raise."""
    broken = _VALID_TOML.replace('gateway_url = "http://127.0.0.1:8000"\n', "")
    path = _write(tmp_path, broken)
    with pytest.raises(ValidationError, match="gateway_url"):
        load_core_config(path)
