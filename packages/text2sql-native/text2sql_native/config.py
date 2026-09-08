"""Typed config.

Two separate sources merge into one typed :class:`Settings`:

* ``.env``        -> secrets, paths, provider/model selection.
* ``config.toml`` -> behavioral parameters. No magic numbers in code.

Loading fails fast with a clear :class:`~text2sql_native.errors.ConfigError` when a
required ``.env`` variable is missing or ``config.toml`` is broken.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .errors import ConfigError

# .env keys, declared once so they are easy to audit and document.
ENV_SCHEMA_PATH = "SCHEMA_PATH"
ENV_PROMPT_DIR = "PROMPT_DIR"
ENV_CONFIG_PATH = "CONFIG_PATH"

ENV_LINKING_PROVIDER = "LINKING_PROVIDER"
ENV_LINKING_MODEL = "LINKING_MODEL"
ENV_LINKING_BASE_URL = "LINKING_BASE_URL"

ENV_SQL_PROVIDER = "SQL_PROVIDER"
ENV_SQL_MODEL = "SQL_MODEL"
ENV_SQL_BASE_URL = "SQL_BASE_URL"

# API-key variable per provider. A provider needs its key only if a stage uses it.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


@dataclass
class StageSettings:
    """Sampling and behavior for one stage."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    prompt_template: str = ""
    timeout_seconds: float = 60.0
    # Used by linking only. Generation ignores it.
    max_tables: int = 8
    # Used by generation only. "structured" forces JSON (sql + explanation).
    # "raw" asks for plain SQL text. Linking ignores it.
    output_mode: str = "structured"
    # Ollama-only context window to force via native /api/chat (see
    # providers/ollama_provider.py) -- None means "don't force one" (other
    # providers, or an Ollama deployment relying on its own Modelfile
    # default). Not a magic default here: an unset value is a real, distinct
    # choice from "force some specific number".
    num_ctx: int | None = None


@dataclass
class RetrySettings:
    """Retry/backoff, shared by both stages."""

    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_factor: float = 2.0


@dataclass
class GeneralSettings:
    """Top-level behavior."""

    sql_dialect: str = "postgres"
    log_level: str = "INFO"
    strict_json: bool = True


@dataclass
class ProviderSelection:
    """Which provider/model/endpoint a stage uses. Comes from ``.env``."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


@dataclass
class Settings:
    """Full config for a run."""

    # Paths (.env)
    schema_path: Path
    # None -> use the prompt templates packaged inside text2sql_native.prompts.
    prompt_dir: Path | None

    # Provider selection (.env)
    linking_selection: ProviderSelection
    sql_selection: ProviderSelection

    # Behavior (config.toml)
    general: GeneralSettings = field(default_factory=GeneralSettings)
    linking: StageSettings = field(default_factory=StageSettings)
    generation: StageSettings = field(default_factory=StageSettings)
    retry: RetrySettings = field(default_factory=RetrySettings)


def _require_env(key: str) -> str:
    """Return an env var, or raise a clear ConfigError if it is missing."""
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        raise ConfigError(
            f"Required environment variable '{key}' is missing or empty. "
            f"See .env.example for the full list."
        )
    return value


def _optional_env(key: str) -> str | None:
    value = os.environ.get(key)
    if value is None or value.strip() == "":
        return None
    return value


def _load_toml(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ConfigError(f"config.toml not found at '{config_path}'.")
    try:
        with config_path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Failed to parse config.toml at '{config_path}': {exc}") from exc


def _stage_from_toml(
    table: dict[str, Any],
    *,
    default_template: str,
    include_max_tables: bool,
) -> StageSettings:
    """Build a StageSettings. Falls back to dataclass defaults per key."""
    defaults = StageSettings()
    return StageSettings(
        temperature=float(table.get("temperature", defaults.temperature)),
        top_p=float(table.get("top_p", defaults.top_p)),
        max_tokens=int(table.get("max_tokens", defaults.max_tokens)),
        prompt_template=str(table.get("prompt_template", default_template)),
        timeout_seconds=float(table.get("timeout_seconds", defaults.timeout_seconds)),
        max_tables=(
            int(table.get("max_tables", defaults.max_tables))
            if include_max_tables
            else defaults.max_tables
        ),
        output_mode=str(table.get("output_mode", defaults.output_mode)).lower(),
        num_ctx=(int(table["num_ctx"]) if table.get("num_ctx") else defaults.num_ctx),
    )


def _selection_for_stage(
    provider_key: str,
    model_key: str,
    base_url_key: str,
) -> ProviderSelection:
    provider = _require_env(provider_key).strip().lower()
    model = _require_env(model_key)
    base_url = _optional_env(base_url_key)

    # Pick the API key for the selected provider, if we know one for it.
    key_env = PROVIDER_KEY_ENV.get(provider)
    api_key = _optional_env(key_env) if key_env else None

    return ProviderSelection(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
    )


def load_settings(
    env_file: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> Settings:
    """Load and merge ``.env`` and ``config.toml`` into a :class:`Settings`.

    Args:
        env_file: Optional path to a ``.env`` file. If ``None``, dotenv's default
            lookup is used (nearest ``.env`` walking up from CWD). Real process
            env vars always win over the file.
        config_path: Optional path to ``config.toml``. If ``None``, uses the
            ``CONFIG_PATH`` env var, else ``./config.toml``.

    Returns:
        A full, typed settings object.

    Raises:
        ConfigError: If a required variable is missing or a file is invalid.
    """
    # override=False -> real environment wins over the file. Safer for secrets.
    load_dotenv(dotenv_path=env_file, override=False)

    schema_path = Path(_require_env(ENV_SCHEMA_PATH)).expanduser()
    # PROMPT_DIR is optional. If unset, packaged prompt templates are used.
    prompt_dir_raw = _optional_env(ENV_PROMPT_DIR)
    prompt_dir = Path(prompt_dir_raw).expanduser() if prompt_dir_raw else None

    linking_selection = _selection_for_stage(
        ENV_LINKING_PROVIDER, ENV_LINKING_MODEL, ENV_LINKING_BASE_URL
    )
    sql_selection = _selection_for_stage(
        ENV_SQL_PROVIDER, ENV_SQL_MODEL, ENV_SQL_BASE_URL
    )

    resolved_config_path = Path(
        config_path
        or _optional_env(ENV_CONFIG_PATH)
        or "config.toml"
    ).expanduser()
    toml_data = _load_toml(resolved_config_path)

    general_tbl = toml_data.get("general", {})
    general = GeneralSettings(
        sql_dialect=str(general_tbl.get("sql_dialect", GeneralSettings.sql_dialect)),
        log_level=str(general_tbl.get("log_level", GeneralSettings.log_level)),
        strict_json=bool(general_tbl.get("strict_json", GeneralSettings.strict_json)),
    )

    linking = _stage_from_toml(
        toml_data.get("linking", {}),
        default_template="linking.txt",
        include_max_tables=True,
    )

    gen_tbl = toml_data.get("generation", {})
    gen_output_mode = str(gen_tbl.get("output_mode", "structured")).lower()
    if gen_output_mode not in ("structured", "raw"):
        raise ConfigError(
            f"[generation] output_mode must be 'structured' or 'raw', "
            f"got '{gen_output_mode}'."
        )
    # In raw mode, default to the raw template unless the user set one.
    gen_default_template = "generation.txt"
    if gen_output_mode == "raw" and "prompt_template" not in gen_tbl:
        gen_default_template = "generation_raw.txt"
    generation = _stage_from_toml(
        gen_tbl,
        default_template=gen_default_template,
        include_max_tables=False,
    )

    retry_tbl = toml_data.get("retry", {})
    retry = RetrySettings(
        max_attempts=int(retry_tbl.get("max_attempts", RetrySettings.max_attempts)),
        backoff_seconds=float(retry_tbl.get("backoff_seconds", RetrySettings.backoff_seconds)),
        backoff_factor=float(retry_tbl.get("backoff_factor", RetrySettings.backoff_factor)),
    )

    return Settings(
        schema_path=schema_path,
        prompt_dir=prompt_dir,
        linking_selection=linking_selection,
        sql_selection=sql_selection,
        general=general,
        linking=linking,
        generation=generation,
        retry=retry,
    )
