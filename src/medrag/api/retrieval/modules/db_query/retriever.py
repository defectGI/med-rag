"""db_query retriever: query -> SQL -> rows -> ``RetrievalResult`` list.

Implements the core :class:`~retrieval.core.Retriever` protocol by composing two
seams: a :class:`~.generator.SqlGenerator` (NL -> SQL, remote LLM) and a
:class:`~.executor.SqlExecutor` (SQL -> rows). Both are injected, so the module
stays testable without a live endpoint or a specific DB driver.

This class is a *convenience composition*, not the only entry point. Every piece
it uses is independently importable: run SQL yourself with a
:class:`~.executor.SqlExecutor` and map the rows with
:func:`~.mapping.rows_to_results`; clean model SQL with
:func:`~.generator.clean_sql`; or swap either seam. Nothing here is hidden behind
the pipeline.

Boundary note (ARCHITECTURE.md #7): this module is a *retrieval building block*.
It does not decide when db_query should be used — that routing (e.g. "the
``product_fact`` intent goes to db_query") belongs to the consuming project.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable

from medrag.api.retrieval.core import RetrievalResult

from .executor import SqlExecutor
from .generator import OnStage, SqlGenerator
from .mapping import rows_to_results

logger = logging.getLogger("retrieval.db_query")


def _supports_on_stage(generator: SqlGenerator) -> bool:
    """Detects `on_stage` support via `inspect.signature` rather than a
    try/except TypeError around the whole call -- a TypeError raised for an
    unrelated internal reason inside `generate()` would otherwise be misread as
    "kwarg unsupported" and silently trigger a second, redundant network
    round-trip without `on_stage`."""
    try:
        return "on_stage" in inspect.signature(generator.generate).parameters
    except (TypeError, ValueError):
        return False


class DbQueryRetriever:
    """Retrieve structured rows by generating and running SQL.

    Every returned :class:`RetrievalResult` carries ``score`` (default ``1.0``):
    a SQL result is an exact match set, not a similarity ranking, so ordering is
    left to the query's own ``ORDER BY``. The full row is exposed in ``metadata``
    (column -> value) and rendered into ``text``.

    Args:
        generator: Produces the SQL string for a query.
        executor: Runs the SQL and returns ``(columns, rows)``.
        id_column: Column to use as ``RetrievalResult.id``. When ``None`` a
            common identifier column is auto-detected, falling back to the row's
            position in the result set.
        score: Score assigned to every row (see above).
    """

    def __init__(
        self,
        generator: SqlGenerator,
        executor: SqlExecutor,
        *,
        id_column: str | None = None,
        score: float = 1.0,
    ) -> None:
        self._generator = generator
        self._executor = executor
        self._id_column = id_column
        self._score = score

    @property
    def generator(self) -> SqlGenerator:
        """The injected SQL generator (reach the seam without rebuilding)."""
        return self._generator

    @property
    def executor(self) -> SqlExecutor:
        """The injected SQL executor."""
        return self._executor

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        *,
        on_sql: Callable[[str], None] | None = None,
        on_stage: OnStage | None = None,
    ) -> list[RetrievalResult]:
        """``on_sql``/``on_stage`` are optional, purely observational hooks.
        ``on_sql`` is called with the final (cleaned + rewritten) SQL string
        right before execution. ``on_stage`` is forwarded to the generator IF it
        supports it (``Text2SqlGenerator`` does) -- surfaces linking/generation
        stage detail (tables, rationale, latency, tokens) or, on failure, the
        stage + raw model text. Neither is part of the core ``Retriever``
        protocol -- a consuming project (e.g. a UI showing "what SQL ran"/"why
        did linking fail") can pass them; nothing here depends on either being
        set.
        """
        def _generate() -> str:
            if on_stage is not None and _supports_on_stage(self._generator):
                return self._generator.generate(query, on_stage=on_stage)
            return self._generator.generate(query)

        # Both steps are blocking (LLM SDK call + sqlite). Offload them so the
        # event loop is never blocked, honoring the async Retriever contract.
        sql = await asyncio.to_thread(_generate)
        logger.info("db_query generated SQL: %s", sql)
        if on_sql is not None:
            on_sql(sql)
        columns, rows = await asyncio.to_thread(self._executor.execute, sql)

        results = rows_to_results(
            columns, rows, id_column=self._id_column, score=self._score
        )
        # k is a cap: the SQL decides ordering; we trim to the requested max.
        return results[:k] if k is not None and k >= 0 else results


__all__ = ["DbQueryRetriever"]
