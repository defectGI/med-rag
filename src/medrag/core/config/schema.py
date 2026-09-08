"""Type/range validation for `core/config/default.toml`.

Why it lives here: `default.toml` is the SINGLE tabulated representation of the
env-fallback values that were merged/inventoried alongside it (see the file's
own header comment -- every line is annotated with the `<file>:<line>` whose
`os.getenv(...)` it mirrors). There is NO code path reading the live
`os.environ` through it yet (the "single read point: `os.getenv` only inside
core/config/" step is still pending). That is why a SEPARATE live-env schema is
NOT defined here: validating `default.toml` already validates the same shape
that was merged (WhatsApp port, Ollama URL, thinning thresholds, etc.) -- two
parallel schemas (one TOML, one env) would drift apart easily and it would stop
being clear which one is right.

Behavior: `extra="forbid"` + fields with no Python defaults (root CONFIG.md
"loader pattern", same principle as `api/config.py`/`vectorize/config.py`) --
a value missing from/wrongly typed/out of range in `default.toml` does not
silently fall back to a default, it fails LOUDLY with a
`pydantic.ValidationError`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from medrag.core.config.loader import read_toml as _read_toml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default.toml"

_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")


class RetrievalTuning(BaseModel):
    """Settings in `src/medrag/api/retrieval/` that today carry NO TOML loader at
    all and live as `os.getenv(...)` fallbacks (see the `default.toml`
    [retrieval] comment -> `modules/intent_classification/factory.py`,
    `modules/top_n/embedder.py`)."""

    model_config = ConfigDict(extra="forbid")

    llm_timeout_seconds: float = Field(gt=0)
    llm_max_retries: int = Field(ge=0)
    embedder_timeout_seconds: float = Field(gt=0)


class ChunkingTuning(BaseModel):
    """Deliberately empty (see the `default.toml` [chunking] comment) -- the
    chunker's own `config/default.toml` is already complete. Staying empty
    under `extra="forbid"` is LOCKED: if a key were ever added here silently
    (here, not in the chunker's own file), that would quietly undo this
    deliberate decision."""

    model_config = ConfigDict(extra="forbid")


class LlmTuning(BaseModel):
    """The single last-resort in-code fallback of the shared `core/llm/`
    client."""

    model_config = ConfigDict(extra="forbid")

    ollama_base_url: str = Field(min_length=1)


class WhatsappTuning(BaseModel):
    """`src/medrag/api/wa_bot.py` -- where that module's
    "to be moved into core/config" TODO ended up."""

    model_config = ConfigDict(extra="forbid")

    gateway_url: str = Field(min_length=1)
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65535)
    gateway_timeout_seconds: float = Field(gt=0)


class LoggingTuning(BaseModel):
    """Only the two pipeline CLIs that still run a bare
    `logging.basicConfig(...)` + `vectorize/query.py` (the chatbot's own
    [logging] block lives SEPARATELY in `api/config.py::LoggingTuning`, not
    repeated here)."""

    model_config = ConfigDict(extra="forbid")

    chunker_cli_level: str = Field(pattern="|".join(_LOG_LEVELS))
    vectorize_cli_level: str = Field(pattern="|".join(_LOG_LEVELS))
    vectorize_query_level: str = Field(pattern="|".join(_LOG_LEVELS))


class ParserTuning(BaseModel):
    """Deliberately almost empty (see the `default.toml` [parser] comment) --
    the geometry/threshold constants are exempt from this round by their own
    decision, and the parser already has its own `config/default.toml`.
    Staying empty under `extra="forbid"` is locked for the same reason as
    `ChunkingTuning`."""

    model_config = ConfigDict(extra="forbid")


class PipelineVersionsTuning(BaseModel):
    """The current version of every stage in ONE place (see the `default.toml`
    [pipeline.versions] comment). The nightly report compares a record's
    `*_version` against this table and derives "the stage changed"."""

    model_config = ConfigDict(extra="forbid")

    parser_version: str = Field(min_length=1)
    chunker_version: str = Field(min_length=1)
    facts_version: str = Field(min_length=1)
    vectorize_version: str = Field(min_length=1)


class PipelineTuning(BaseModel):
    """Two guards so the nightly run does not take the service down
    (`worker_replicas`, `ollama_max_concurrent`, see the `default.toml`
    [pipeline] comment) + the per-stage version sub-table."""

    model_config = ConfigDict(extra="forbid")

    worker_replicas: int = Field(ge=1)
    ollama_max_concurrent: int = Field(ge=1)
    versions: PipelineVersionsTuning


class CoreConfig(BaseModel):
    """The full schema of `core/config/default.toml`."""

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalTuning
    chunking: ChunkingTuning
    llm: LlmTuning
    whatsapp: WhatsappTuning
    logging: LoggingTuning
    parser: ParserTuning
    pipeline: PipelineTuning


def load_core_config(path: Path | None = None) -> CoreConfig:
    """Reads `default.toml` and validates it against `CoreConfig`.

    An invalid value (wrong type, out of range, unknown/missing key) raises
    `pydantic.ValidationError` -- with a CLEAR message showing which field
    broke which constraint -- nothing silently falls back to a default (see
    the module docstring).
    """
    data = _read_toml(path or DEFAULT_CONFIG_PATH)
    return CoreConfig.model_validate(data)


__all__ = [
    "ChunkingTuning",
    "CoreConfig",
    "LlmTuning",
    "LoggingTuning",
    "ParserTuning",
    "PipelineTuning",
    "PipelineVersionsTuning",
    "RetrievalTuning",
    "WhatsappTuning",
    "load_core_config",
]
