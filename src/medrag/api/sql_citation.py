"""Rewriter that adds citation columns to generated SQL in CODE.

Design rationale: bookkeeping like this shouldn't be told to the model -- it
isn't information the model needs; it's something to be handled entirely in
code so the model doesn't get confused.

## Why code instead of a prompt

The evidence panel shows, for every row, "which product, which attribute,
which file" (`answering_model.extract_evidence_chunks`). These three fields
already exist in the `spec_value` table
(`product_code`/`key`/`source_file_name`) but only enter a row's metadata if
the SQL SELECTs them
(`retrieval.modules.db_query.mapping.row_to_result`: metadata = selected
column names -> values).

This was previously solved by adding a "CITATION COLUMNS" rule to the
generation prompt. Wrong layer: (1) it made the model do bookkeeping, (2) the
panel silently emptied whenever the model forgot, (3) it pulled the model's
attention away from the actual job (correct WHERE/JOIN). The rule was
reverted; the job moved here.

It plugs into `retrieval`'s EXISTING rewriter seam
(`db_query/generator.py::Text2SqlGenerator(rewriters=...)`) -- a `str -> str`
chain applied AFTER `clean_sql` and BEFORE the query runs. This module
deliberately lives on the `chatbot` side: `spec_value`/`source_file_name` are
product-specific names and the `retrieval` package must not depend on any
product project (only the generic `rewriters=` parameter was added there).

## Why some queries are never touched

Adding columns to the SELECT list is harmless in most queries (row count
doesn't change), but in some it BREAKS the answer:

- `SELECT DISTINCT product_code ...` -- a "which products" query. Adding
  `source_file_name` would repeat the same product per document and inflate
  the list.
- `GROUP BY` / aggregates (`COUNT`/`AVG`/...) -- breaks the grouping or makes
  the SQL invalid.
- `UNION`/`INTERSECT`/`EXCEPT` -- column counts stop matching and the query
  dies.

In those cases the query is returned VERBATIM and the panel simply shows less
for that turn. Deliberate limit: a missing citation is better than a wrong
answer.

## Error policy

This function NEVER raises -- on any input it can't parse it returns the
original SQL unchanged. In particular it must not raise `TypeError`: the
three-stage `except TypeError` fallback in `flows/sql_topn.py` would misread
it as "this retriever doesn't support the `on_sql` kwarg" and run the same
query 2-3 times (triple cost under the GPU gate).
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp

logger = logging.getLogger("medrag.api.sql_citation")

#: Columns wanted for citation -- all sit DENORMALIZED in `spec_value`, so no
#: JOIN is required (see facts/db/schema.yaml `spec_value` description:
#: "family/subfamily/unit/source_file_name are ALREADY joined in").
#: `value_id` (PK) was ADDED: the evidence panel shows the chunk TEXT behind a
#: SQL row, and the pointers to that text (`source_chunk_id`/`evidence`) are
#: resolved via the row's PK (see `sql_evidence.py`). The pointers THEMSELVES
#: are NOT added to the SELECT -- `evidence` is a JSON blob and
#: `render_row_text` puts every selected column into the row text seen by the
#: answering model; carrying a single integer is enough.
CITATION_COLUMNS = ("product_code", "key", "source_file_name", "value_id")

#: Table the citation columns live in. If a query never touches this table
#: (e.g. only a `product` or `document` query) the rewriter doesn't engage.
CITATION_TABLE = "spec_value"

_DIALECT = "sqlite"


def _outer_select(tree: exp.Expression) -> exp.Select | None:
    """The outermost SELECT. If the query is a set operation
    (UNION/INTERSECT/EXCEPT) -> `None` -- adding columns that way desyncs the
    column counts of the two branches."""
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        return None
    if isinstance(tree, exp.Select):
        return tree
    return None


def _citation_source(select: exp.Select) -> exp.Table | None:
    """Finds `spec_value` in the FROM/JOIN chain and returns that table node
    (the node itself is needed for alias resolution). `spec_value` in
    SUBQUERIES does NOT count -- only this level's sources are walked via
    `select.args`, because a column added to the outer SELECT must be visible
    at this level."""
    sources: list[exp.Expression] = []
    # sqlglot 30.x keeps this argument as `from_`; older versions used `from`.
    # Try both -- otherwise the rewriter can silently become a no-op on a
    # version surprise: when `from` returns empty, every query is treated as
    # "no spec_value" and passed through untouched, without raising any error.
    from_ = select.args.get("from_") or select.args.get("from")
    if from_ is not None:
        sources.append(from_.this)
    for join in select.args.get("joins") or []:
        sources.append(join.this)
    for src in sources:
        if isinstance(src, exp.Table) and src.name.lower() == CITATION_TABLE:
            return src
    return None


def _has_aggregate(select: exp.Select) -> bool:
    """Whether an aggregate exists in the SELECT list or in HAVING.
    `select.find` would walk the ENTIRE tree (subqueries included); only this
    level's projections are checked here -- a COUNT in a subquery doesn't
    prevent adding columns to the outer SELECT."""
    for projection in select.expressions:
        if projection.find(exp.AggFunc):
            return True
    having = select.args.get("having")
    return having is not None


def _selected_names(select: exp.Select) -> set[str]:
    """Which column names already exist in the projection (the REAL column
    name is checked, not the alias -- if `sv.key AS attr` was written, `key`
    still counts as selected)."""
    names: set[str] = set()
    for projection in select.expressions:
        target = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(target, exp.Column):
            names.add(target.name.lower())
    return names


def add_citation_columns(sql: str) -> str:
    """Adds the missing citation columns to a plain projection selecting from
    `spec_value`.

    Untouched cases (rationale in the module docstring): set operations,
    `SELECT DISTINCT`, `GROUP BY`, aggregates/HAVING, `SELECT *`, queries not
    containing `spec_value`, and unparseable text. All return the input
    VERBATIM.
    """
    try:
        return _add_citation_columns(sql)
    except Exception:
        logger.warning("citation kolonları eklenemedi, SQL aynen kullanılıyor", exc_info=True)
        return sql


def _add_citation_columns(sql: str) -> str:
    tree = sqlglot.parse_one(sql, dialect=_DIALECT)
    select = _outer_select(tree)
    if select is None:
        return sql
    if select.args.get("distinct") or select.args.get("group"):
        return sql
    if _has_aggregate(select):
        return sql
    # `SELECT *` already brings the citation columns too.
    if any(isinstance(p, exp.Star) for p in select.expressions):
        return sql
    table = _citation_source(select)
    if table is None:
        return sql

    # In a multi-table query don't leave column names ambiguous: qualify with
    # `spec_value`'s alias (or its name if unaliased). `generation_raw.txt`
    # already asks the model for qualified columns in JOINed queries; the same
    # rule is applied to the columns added here.
    qualifier = table.alias_or_name
    already = _selected_names(select)
    eklenecek = [c for c in CITATION_COLUMNS if c not in already]
    if not eklenecek:
        return sql

    for column in eklenecek:
        select.select(exp.column(column, table=qualifier), append=True, copy=False)
    return select.sql(dialect=_DIALECT)


__all__ = ["CITATION_COLUMNS", "CITATION_TABLE", "add_citation_columns"]
