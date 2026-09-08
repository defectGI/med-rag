"""db_query — retrieve structured rows by generating and running SQL.

A retrieval strategy backed by a Text-to-SQL step: a natural-language query is
turned into a SQL string (remote LLM, via the ``text2sql-engine`` package), that
SQL is run against a ready-made database, and the rows come back as core
``RetrievalResult`` items. The natural target of structured intents
(``product_fact``, ``aggregation``, ``doc_download``, ``visual_request``) in a
consuming project — though this module never decides that mapping itself
(ARCHITECTURE.md #7).

Two seams keep it testable and dependency-isolated:
- :class:`SqlGenerator` (NL -> SQL) — a fake is injected in unit tests so the
  default test run never reaches an LLM endpoint (ARCHITECTURE.md #10).
- :class:`SqlExecutor` (SQL -> rows) — the stdlib :class:`SQLiteExecutor` ships
  as the default (read-only); other DB drivers stay out of base dependencies.

The full pipeline is a convenience, not a requirement — every piece is usable
on its own::

    from medrag.api.retrieval.modules.db_query import build_default_retriever

    # 1. The whole pipeline (on the inference machine):
    retriever = build_default_retriever(db_path="specs.db")
    results = await retriever.retrieve("operating temperature of PN1309", k=5)

    # 2. À la carte — run your own SQL, reuse only the row mapping:
    from medrag.api.retrieval.modules.db_query import SQLiteExecutor, rows_to_results
    cols, rows = SQLiteExecutor("specs.db").execute("SELECT * FROM products LIMIT 5")
    results = rows_to_results(cols, rows)               # -> list[RetrievalResult]

    # 3. Or just clean a model's SQL, or map a single row, etc.
    from medrag.api.retrieval.modules.db_query import clean_sql, row_to_result

In tests, construct ``DbQueryRetriever`` directly with a fake generator and a
``SQLiteExecutor`` over a temp file.
"""

from medrag.api.retrieval.modules.db_query.executor import (
    DbQueryError,
    ExecResult,
    SqlExecutor,
    SQLiteExecutor,
)
from medrag.api.retrieval.modules.db_query.factory import (
    build_default_engine,
    build_default_retriever,
)
from medrag.api.retrieval.modules.db_query.generator import (
    OnStage,
    SqlEngine,
    SqlGenerator,
    SqlResult,
    Text2SqlGenerator,
    clean_sql,
    sqlite_ilike_to_like,
)
from medrag.api.retrieval.modules.db_query.mapping import (
    DEFAULT_ID_CANDIDATES,
    pick_row_id,
    render_row_text,
    row_to_result,
    rows_to_results,
)
from medrag.api.retrieval.modules.db_query.retriever import DbQueryRetriever

# RUF022: sorted alphabetically; grouping comments dropped (each name's source
# module is already clear from the import blocks above).
__all__ = [
    "DEFAULT_ID_CANDIDATES",
    "DbQueryError",
    "DbQueryRetriever",
    "ExecResult",
    "OnStage",
    "SQLiteExecutor",
    "SqlEngine",
    "SqlExecutor",
    "SqlGenerator",
    "SqlResult",
    "Text2SqlGenerator",
    "build_default_engine",
    "build_default_retriever",
    "clean_sql",
    "pick_row_id",
    "render_row_text",
    "row_to_result",
    "rows_to_results",
    "sqlite_ilike_to_like",
]
