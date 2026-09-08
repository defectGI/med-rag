"""Natural-language -> SQL string, isolated behind a protocol.

The SQL is produced by an LLM. Under this repo's GPU rule (ARCHITECTURE.md #10)
that inference runs on a *separate* machine behind an OpenAI-compatible
endpoint; this module only sends the request. Keeping generation behind the
:class:`SqlGenerator` protocol is what lets the retriever be tested with a fake
that returns a canned SQL string — the default test run never reaches an
endpoint.

The production implementation, :class:`Text2SqlGenerator`, wraps a
``text2sql.Text2SQL`` engine (the ``text2sql-engine`` PyPI package). That engine
only *produces* a SQL string; it never touches a database. Running the SQL is
the executor's job (see :mod:`.executor`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

_FENCE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)
_LEADING_SQL = re.compile(r"^sql\s*\n", re.IGNORECASE)
_ILIKE = re.compile(r"\bILIKE\b", re.IGNORECASE)

# Stage-observability callback: (stage_name, detail_dict) -> None. Purely
# observational -- never changes behavior, only surfaces what the multi-stage
# engine already computed (linking tables/rationale, per-stage latency/tokens,
# or -- on failure -- the stage name + raw model text that didn't parse). Not
# part of the core `SqlGenerator` protocol (fakes/other engines don't need to
# support it); `Text2SqlGenerator.generate` accepts it as an optional kwarg, and
# `DbQueryRetriever` detects support via `inspect.signature` before passing it.
OnStage = Callable[[str, dict], None]


@runtime_checkable
class SqlGenerator(Protocol):
    """Turns a natural-language query into a single SQL string."""

    def generate(self, query: str) -> str: ...


@runtime_checkable
class SqlResult(Protocol):
    """The minimal shape :class:`Text2SqlGenerator` reads off an engine run.

    ``text2sql.Text2SQLResult`` satisfies this, but so does any object exposing
    a ``.sql`` string — so this module never has to import ``text2sql`` just to
    be type-checked, and a fake result works in tests.
    """

    sql: str


@runtime_checkable
class SqlEngine(Protocol):
    """Minimal contract of a text2sql-style engine: ``run(request) -> .sql``.

    Depending on this Protocol (instead of the concrete ``text2sql.Text2SQL``)
    keeps ``text2sql-engine`` an optional, injectable dependency and documents
    exactly what an alternative engine must provide.
    """

    def run(self, request: str) -> SqlResult: ...


def clean_sql(sql: str) -> str:
    """Strip markdown fences / stray prefixes and a trailing semicolon.

    Dialect-neutral: only removes wrapping that models add around the SQL. It
    does not rewrite the SQL itself — see :func:`sqlite_ilike_to_like` for an
    opt-in dialect fix.
    """
    sql = sql.strip()
    fence = _FENCE.match(sql)
    if fence:
        sql = fence.group(1).strip()
    sql = _LEADING_SQL.sub("", sql)
    return sql.strip().rstrip(";").strip()


def sqlite_ilike_to_like(sql: str) -> str:
    """Rewrite ``ILIKE`` to ``LIKE`` for SQLite.

    SQLite has no ``ILIKE``; its ``LIKE`` is already case-insensitive for ASCII.
    Pass this as a rewriter to :class:`Text2SqlGenerator` when the target is
    SQLite and the model may emit Postgres-style ``ILIKE``.
    """
    return _ILIKE.sub("LIKE", sql)


class Text2SqlGenerator:
    """Adapt a ``text2sql.Text2SQL`` engine to the :class:`SqlGenerator` seam.

    Args:
        engine: A constructed ``text2sql.Text2SQL`` instance. Injected rather
            than built here so this class carries no hard dependency on
            ``text2sql-engine`` (the factory builds the real one lazily).
        rewriters: Optional post-processing steps applied to the cleaned SQL,
            e.g. ``[sqlite_ilike_to_like]`` for a SQLite target.
    """

    def __init__(
        self,
        engine: SqlEngine,
        *,
        rewriters: list[Callable[[str], str]] | None = None,
    ) -> None:
        self._engine = engine
        self._rewriters = list(rewriters or [])

    def generate(self, query: str, *, on_stage: OnStage | None = None) -> str:
        """`on_stage` is purely observational: if given, it is called with the
        linking/generation stage detail the underlying engine already computes
        (tables/rationale, latency, tokens) -- or, on failure, with the stage
        name + raw model text that didn't parse. Never changes what is returned
        or raised; the caller can always drop this kwarg and get identical prior
        behavior.

        Deliberately catches bare ``Exception`` (not a specific text2sql
        error type): this module never hard-imports ``text2sql`` (see
        ``SqlEngine``/``SqlResult`` -- optional dependency via duck typing),
        so it cannot name the real exception classes. ``getattr(exc,
        "raw_text", "")`` reads text2sql's optional ``raw_text`` attribute
        without needing to import its type."""
        try:
            result = self._engine.run(query)
        except Exception as exc:
            if on_stage is not None:
                on_stage("sql_error", {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "raw_text": getattr(exc, "raw_text", ""),
                })
            raise

        if on_stage is not None:
            self._emit_stage_details(result, on_stage)

        sql = clean_sql(result.sql)
        for rewrite in self._rewriters:
            sql = rewrite(sql)
        return sql

    @staticmethod
    def _emit_stage_details(result: Any, on_stage: OnStage) -> None:
        """Reads `linked_schema`/`metadata`/`explanation` off `result` via
        `getattr` (not required by the minimal `SqlResult` protocol, which
        only guarantees `.sql` -- a fake test result without these is fine,
        stage events are just skipped)."""
        linked = getattr(result, "linked_schema", None)
        metadata = getattr(result, "metadata", None)
        if linked is not None and metadata is not None:
            linking_meta = metadata.linking
            on_stage("linking_done", {
                "tables": list(linked.tables),
                "columns": [f"{c.table}.{c.column}" for c in linked.columns],
                "rationale": linked.rationale,
                "latency_ms": round(linking_meta.latency_seconds * 1000),
                "tokens": linking_meta.usage.total_tokens,
                "attempts": linking_meta.attempts,
            })
            generation_meta = metadata.generation
            on_stage("generation_done", {
                "explanation": getattr(result, "explanation", ""),
                "latency_ms": round(generation_meta.latency_seconds * 1000),
                "tokens": generation_meta.usage.total_tokens,
                "attempts": generation_meta.attempts,
            })


__all__ = [
    "OnStage",
    "SqlEngine",
    "SqlGenerator",
    "SqlResult",
    "Text2SqlGenerator",
    "clean_sql",
    "sqlite_ilike_to_like",
]
