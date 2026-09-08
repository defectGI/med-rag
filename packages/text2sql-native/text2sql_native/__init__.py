"""text2sql_native — schema-agnostic, config-driven Text-to-SQL library.

Independent fork (2026-08-06): fully decoupled from the ``text2sql-engine``
PyPI package -- different distribution name AND different import name
(``text2sql_native``, not ``text2sql``), so the two can never collide or
silently shadow each other in the same environment. A plain
``pip install text2sql-engine`` can pull the stale upstream package over
this actively-maintained one because both still share the same import
name (``text2sql``); the rename makes that misroute impossible.

Entry point::

    from text2sql_native import Text2SQL
    engine = Text2SQL.from_config()
    result = engine.run("get me the top 5 best-selling products last month")
    print(result.sql)
    print(result.linked_schema)
    print(result.metadata)

The library only produces a SQL string. It never connects to or runs against a
database.
"""

from __future__ import annotations

from .errors import (
    ConfigError,
    EmptyLinkingError,
    ProviderError,
    SchemaError,
    StructuredOutputError,
    Text2SQLError,
)
from .pipeline import Text2SQL
from .prompts import PromptTemplates
from .providers import LLMProvider, register_provider
from .schema import FileSchemaProvider, Schema, SchemaProvider
from .types import (
    LinkedColumn,
    LinkedJoin,
    LinkedSchema,
    RunMetadata,
    StageMetadata,
    Text2SQLResult,
    TokenUsage,
)

__version__ = "0.2.0"

__all__ = [
    "Text2SQL",
    # Schema
    "Schema",
    "SchemaProvider",
    "FileSchemaProvider",
    # Providers / DI
    "LLMProvider",
    "register_provider",
    "PromptTemplates",
    # Result types
    "Text2SQLResult",
    "LinkedSchema",
    "LinkedColumn",
    "LinkedJoin",
    "RunMetadata",
    "StageMetadata",
    "TokenUsage",
    # Errors
    "Text2SQLError",
    "ConfigError",
    "SchemaError",
    "ProviderError",
    "StructuredOutputError",
    "EmptyLinkingError",
]
