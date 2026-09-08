"""Map raw SQL rows to core ``RetrievalResult`` items — standalone.

These are plain functions, usable **without** :class:`DbQueryRetriever`. If you
run SQL your own way (your own driver, a connection pool, an ORM cursor) you can
still get the same ``RetrievalResult`` shape by calling :func:`rows_to_results`
on the ``(columns, rows)`` you already have. The retriever is just one caller of
these.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from medrag.api.retrieval.core import RetrievalResult

# Columns that make a sensible stable id for a row, tried in order. These match
# the ingestion-contract identifiers (core.schemas) a DB is typically built from.
DEFAULT_ID_CANDIDATES: tuple[str, ...] = (
    "node_id",
    "id",
    "product_code",
    "acme_code",
    "doc_id",
)


def pick_row_id(
    record: dict,
    *,
    index: int = 0,
    id_column: str | None = None,
    id_candidates: Sequence[str] = DEFAULT_ID_CANDIDATES,
) -> str:
    """Choose a stable id for a row.

    Order: an explicit ``id_column`` if present and non-null, then the first
    populated column in ``id_candidates``, then the row's ``index`` as a string.
    """
    if id_column and record.get(id_column) is not None:
        return str(record[id_column])
    for candidate in id_candidates:
        value = record.get(candidate)
        if value is not None:
            return str(value)
    return str(index)


def render_row_text(record: dict) -> str:
    """Render a row as ``col=value | col=value`` for display/citation."""
    return " | ".join(
        f"{col}={'' if val is None else val}" for col, val in record.items()
    )


def row_to_result(
    columns: Sequence[str],
    row: Sequence,
    *,
    index: int = 0,
    id_column: str | None = None,
    score: float = 1.0,
    id_candidates: Sequence[str] = DEFAULT_ID_CANDIDATES,
) -> RetrievalResult:
    """Map one ``(columns, row)`` pair to a :class:`RetrievalResult`.

    The full row is preserved in ``metadata`` (column -> value); ``text`` is the
    rendered row. ``score`` defaults to ``1.0`` (a SQL row is an exact match, not
    a similarity rank) but is overridable for callers that rank rows themselves.
    """
    record = dict(zip(columns, row))
    return RetrievalResult(
        id=pick_row_id(
            record, index=index, id_column=id_column, id_candidates=id_candidates
        ),
        score=score,
        text=render_row_text(record),
        metadata=record,
    )


def rows_to_results(
    columns: Sequence[str],
    rows: Iterable[Sequence],
    *,
    id_column: str | None = None,
    score: float = 1.0,
    id_candidates: Sequence[str] = DEFAULT_ID_CANDIDATES,
) -> list[RetrievalResult]:
    """Map a whole result set to ``RetrievalResult`` items, order preserved."""
    return [
        row_to_result(
            columns,
            row,
            index=i,
            id_column=id_column,
            score=score,
            id_candidates=id_candidates,
        )
        for i, row in enumerate(rows)
    ]


__all__ = [
    "DEFAULT_ID_CANDIDATES",
    "pick_row_id",
    "render_row_text",
    "row_to_result",
    "rows_to_results",
]
