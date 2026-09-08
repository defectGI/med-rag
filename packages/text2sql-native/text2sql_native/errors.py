"""Exceptions used across the library.

Everything subclasses ``Text2SQLError``. Catch that to catch all of them.
"""

from __future__ import annotations


class Text2SQLError(Exception):
    """Base class for every error this library raises."""


class ConfigError(Text2SQLError):
    """Config is missing or invalid.

    Example: a required ``.env`` variable is not set, or ``config.toml`` is
    broken, or a provider/model selection is incomplete.
    """


class SchemaError(Text2SQLError):
    """Schema metadata cannot be loaded or is not valid."""


class ProviderError(Text2SQLError):
    """An LLM provider failed to build or a call failed.

    Wraps SDK/transport errors so callers do not deal with SDK-specific types.
    """


class StructuredOutputError(Text2SQLError):
    """LLM output could not be parsed as the expected JSON.

    Raised only after all retries are used.
    """


class EmptyLinkingError(Text2SQLError):
    """Stage 1 linked no tables or columns.

    A valid but empty match means we cannot generate SQL.
    """
