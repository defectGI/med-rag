"""Production wiring: build the real text2sql engine and a DbQueryRetriever.

Kept apart from the retriever/generator so those stay provider-agnostic and
unit-testable without ``text2sql-engine-native`` installed. ``text2sql_native``
is imported lazily *inside* the function, so importing this module never
requires the ``db_query`` extra.

The import name is ``text2sql_native``, NOT ``text2sql``: a real, unrelated
``text2sql-engine`` PyPI package exists (an older, unmaintained public release
by this fork's own author) that provides the SAME ``text2sql`` import name.
Depending on that name let a plain ``pip install text2sql-engine`` silently
shadow the intended vendored fork, so the import was renamed to remove the
collision structurally instead of relying on a version pin to make it merely
improbable.

GPU/inference rule (ARCHITECTURE.md #10): the engine built here points at a
remote OpenAI-compatible endpoint (configured in text2sql_native's own
``.env``). Calling ``retriever.retrieve(...)`` reaches that endpoint for the
SQL-generation step — do that on the inference machine, not here. The SQL
*execution* step is local SQLite and does no inference.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from .executor import DbQueryError, SQLiteExecutor
from .generator import SqlEngine, Text2SqlGenerator, sqlite_ilike_to_like
from .retriever import DbQueryRetriever


def build_default_engine(
    *,
    env_file: str | None = None,
    config_path: str | None = None,
) -> SqlEngine:
    """Construct a ``text2sql_native.Text2SQL`` engine from its own config files.

    ``text2sql-engine-native`` carries a self-contained config layer (its
    ``.env`` + ``config.toml``: schema path, per-stage provider/model/base_url,
    dialect). This just points at those files. Reads ``DB_QUERY_TEXT2SQL_ENV``
    / ``DB_QUERY_TEXT2SQL_CONFIG`` when the paths are not passed explicitly.
    """
    try:
        from text2sql_native import Text2SQL
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "text2sql-engine-native is required to build the default db_query "
            "engine. Install the vendored fork: "
            "pip install -e path/to/packages/text2sql-native"
        ) from exc

    env_file = env_file or os.getenv("DB_QUERY_TEXT2SQL_ENV")
    config_path = config_path or os.getenv("DB_QUERY_TEXT2SQL_CONFIG")
    try:
        return Text2SQL.from_config(env_file=env_file, config_path=config_path)
    except Exception as exc:
        # Wrap text2sql's own errors (ConfigError/SchemaError/...) in the
        # module's error type so callers catch a single DbQueryError.
        raise DbQueryError(f"Failed to build text2sql engine: {exc}") from exc


def build_default_retriever(
    *,
    db_path: str | None = None,
    env_file: str | None = None,
    config_path: str | None = None,
    dialect: str | None = None,
    id_column: str | None = None,
    rewriters: list[Callable[[str], str]] | None = None,
) -> DbQueryRetriever:
    """Wire a text2sql engine + a read-only SQLite executor into a retriever.

    Env fallbacks: ``DB_QUERY_DB_PATH`` (the SQLite file the generated SQL runs
    against) and ``DB_QUERY_DIALECT`` (default ``sqlite``). For a SQLite target
    an ``ILIKE`` -> ``LIKE`` rewriter is added, since models may emit Postgres
    syntax.

    Args:
        rewriters: Extra ``str -> str`` steps applied to the generated SQL
            before it is executed, appended *after* the dialect defaults (they
            add to, never replace, the built-in ``ILIKE`` fix). This exposes
            the seam :class:`~.generator.Text2SqlGenerator` already had but the
            factory kept private, so a caller can post-process the SQL without
            hand-wiring the whole chain. Deliberately generic: a rewriter that
            knows about a specific application's tables belongs in that
            application, not here. Omitting it (the default) keeps behavior
            byte-for-byte identical.
    """
    db_path = db_path or os.getenv("DB_QUERY_DB_PATH")
    if not db_path:
        raise DbQueryError(
            "db_path is required (pass it, or set DB_QUERY_DB_PATH). It is the "
            "SQLite database the generated SQL is executed against."
        )
    dialect = (dialect or os.getenv("DB_QUERY_DIALECT") or "sqlite").lower()

    engine = build_default_engine(env_file=env_file, config_path=config_path)
    chain = [sqlite_ilike_to_like] if dialect == "sqlite" else []
    chain.extend(rewriters or [])
    generator = Text2SqlGenerator(engine, rewriters=chain)
    executor = SQLiteExecutor(db_path, read_only=True)
    return DbQueryRetriever(generator, executor, id_column=id_column)


__all__ = ["build_default_engine", "build_default_retriever"]
