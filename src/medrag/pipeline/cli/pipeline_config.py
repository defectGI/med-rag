"""pipeline tuning yapılandırması: `pipeline_config.toml` → `PipelineConfig`.

Tür ayrımı (kök `CONFIG.md`): pipeline'ın kendi orkestrasyon tuning'i BURADA
(TOML); yollar/girdi filtresi `.env`'de; model bağlantısı
`src/medrag/pipeline/parser/.env`'de.

İSİM NOTU (tarihsel, D-39 Faz B'den önce): bu modül bilinçli "config" DEĞİL
"pipeline_config" adını taşırdı — `run_parse_pipeline` `sys.path.insert(0,
PARSER_DIR)` yapıp parser modüllerini niteliksiz import ediyordu; bir
`config` modülü/dizini burada olsaydı sys.path'te parser'ın kendi
`config.py`'sini gölgelerdi. D-39 Faz B'den sonra `run_parse_pipeline`
`medrag.pipeline.parser`'ı gerçek (nitelikli) paket olarak import ediyor,
sys.path hilesi yok — gölgeleme riski artık YOK. İsim yine de değiştirilmedi
(gereksiz churn, iki isim de zaten doğru anlatıyor).

D-58 (2026-08-19): bu modülün kendisi de `pipeline/` kökünden
`src/medrag/pipeline/cli/` altına taşındı, gerçek kurulu paketin (`medrag.pipeline.cli`)
bir parçası oldu. `run_parse_pipeline.py` artık bunu
`from medrag.pipeline.cli.pipeline_config import get_config as _pipeline_config`
ile nitelikli import ediyor (eski bare `from pipeline_config import ...` değil).

Precedence: pipeline_config.toml → env (env kazanır). Her knob hâlâ eski env
adıyla override edilebilir (`_ENV_OVERRIDES`); `get_config()` her çağrıda taze
env okur (TOML dict cache'li). Caster'lar eski os.getenv semantiğini korur.
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "pipeline_config.toml"


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_concurrency: int = Field(ge=0)  # kullanım yerinde max(1, ...) ile korunur


class Health(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    fail_streak: int = Field(ge=0)  # max(1, ...) ile korunur


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run: Run
    health: Health


_SKIP = object()


def _cast_int(raw: str) -> Any:
    if not raw.strip():
        return _SKIP
    try:
        return int(raw)
    except ValueError:
        return _SKIP


def _cast_health(raw: str) -> Any:
    # Old _health_on() semantics: only "0"/"false" turns it off (anything else enables it).
    return raw.strip().lower() not in ("0", "false")


_ENV_OVERRIDES: dict[str, tuple[tuple[str, ...], Any]] = {
    "DOC_CONCURRENCY": (("run", "doc_concurrency"), _cast_int),
    "HEALTH_CHECK": (("health", "enabled"), _cast_health),
    "HEALTH_FAIL_STREAK": (("health", "fail_streak"), _cast_int),
}


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = data
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def _env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (path, caster) in _ENV_OVERRIDES.items():
        raw = env.get(name)
        if raw is None:
            continue
        value = caster(raw)
        if value is _SKIP:
            continue
        _set_path(out, path, value)
    return out


_DEFAULT_DATA = _read_toml(DEFAULT_CONFIG_PATH)


def load_config(env: Mapping[str, str] | None = None,
                override: str | Path | None = None) -> PipelineConfig:
    if env is None:
        env = os.environ
    data = copy.deepcopy(_DEFAULT_DATA)
    ov_path = override if override is not None else env.get("PIPELINE_CONFIG")
    if ov_path:
        data = _deep_merge(data, _read_toml(Path(ov_path)))
    data = _deep_merge(data, _env_overrides(env))
    return PipelineConfig.model_validate(data)


def get_config() -> PipelineConfig:
    return load_config()
