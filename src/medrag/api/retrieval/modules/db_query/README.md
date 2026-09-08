# db_query

Status: **usable** (v1).

Retrieval by generating and running SQL. A natural-language query is turned into
a SQL string by the [`text2sql-engine`](https://pypi.org/project/text2sql-engine/)
package (a 2-stage LLM pipeline: schema linking → SQL generation), that SQL is
executed against a **ready-made** database, and the rows come back as core
`RetrievalResult` items.

The natural target of structured intents (`product_fact`, `aggregation`,
`doc_download`, `visual_request`) in a consuming project — but this module never
decides that routing itself (ARCHITECTURE.md #7). It only depends on
`retrieval.core`, no other module.

## Design

```
query ──▶ SqlGenerator ──▶ SQL string ──▶ SqlExecutor ──▶ rows ──▶ RetrievalResult[]
          (text2sql, remote LLM)          (SQLite, read-only)      (score = 1.0)
```

Two injected seams keep it testable and dependency-isolated:

- **`SqlGenerator`** — NL → SQL. The production `Text2SqlGenerator` wraps a
  `text2sql.Text2SQL` engine, which calls a remote OpenAI-compatible endpoint.
  Under the GPU rule (ARCHITECTURE.md #10) no inference runs on the dev machine;
  unit tests inject a fake generator that returns a canned SQL string.
- **`SqlExecutor`** — SQL → rows. The shipped `SQLiteExecutor` uses only stdlib
  `sqlite3` and opens the DB **read-only** by default, so a stray write in the
  generated SQL fails at the engine level. Other DB drivers can implement the
  protocol without landing in this repo's base dependencies.

Notes:
- **Score is a constant `1.0`.** A SQL result is an exact match set, not a
  similarity ranking; ordering is the query's own `ORDER BY`. `k` is applied as
  a post-cap on the row count.
- The full row is exposed in `RetrievalResult.metadata` (column → value) and
  rendered into `.text` as `col=value` pairs. `id` auto-detects a common
  identifier column (`node_id`, `id`, `product_code`, ...) or falls back to the
  row index; override with `id_column`.
- **The DB is built elsewhere.** This module does not create or populate the
  database — it connects to an existing one (ARCHITECTURE.md).

## Use

```python
from retrieval.modules.db_query import build_default_retriever

# On the inference machine (reaches the LLM endpoint for SQL generation):
retriever = build_default_retriever(db_path="specs.db")
results = await retriever.retrieve("operating temperature of PN1309", k=5)
```

`build_default_retriever` reads `DB_QUERY_DB_PATH` and the text2sql config
pointers from the environment (see `.env.example`). SQL generation itself is
configured by `text2sql-engine`'s own `.env` + `config.toml`.

Extra: `pip install -e ".[db_query]"`
