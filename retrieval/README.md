# mkd-retriever

**A modular, project-agnostic RAG retrieval layer.** Multiple retrieval
strategies — intent classification, text-to-SQL, dense vector search, and
more — coexist around a small shared core **without dependencies between
them**.

> **Note:** This directory (`retrieval/`) now contains only the
> config/docs artifacts. The live code lives at `src/medrag/api/retrieval/`;
> the import path is **`medrag.api.retrieval.xxx`**, not `retrieval.xxx`. The
> PyPI distribution name is still `mkd-retriever` (see the root
> `pyproject.toml`), but install/packaging in this repo is driven by the
> root `pyproject.toml`. The `retrieval/pyproject.toml` in this directory
> is stale and kept only as a historical reference — do not install from
> it.

## Where this layer sits in the pipeline

It is the chatbot/RAG answering side of `medrag` — the
"how do I answer the question" layer of the `chatbot` API:

```
user query
   │
   ▼
intent_classification  →  which strategy? (returns only the label; routing
   │                       is the consumer's -- chatbot -- job)
   ▼
top_n (vector search, Qdrant)  /  db_query (text-to-SQL, specs.db)  / ...
   │
   ▼
RetrievalResult[]  →  chatbot feeds these to the LLM as context
```

- **Previous step:** `vectorize` (writes the Qdrant collection) and `facts`
  (produces `specs.db`) — these prepare the data this layer READS, but
  this layer never imports them (only the format/location contract binds
  it).
- **Next step:** `medrag.api.chatbot` — takes the `RetrievalResult` list and
  produces the answer.
- Code location: `src/medrag/api/retrieval/` (under `api/` alongside
  `chatbot`).

## Architecture rule

1. **A module never imports another module** (`modules/<x>` →
   `modules/<y>` is forbidden). Inter-module wiring is the consuming
   project's job.
2. **Anything shared lives in `core/`, and `core/` stays small** — only
   minimal interfaces + common data schemas, never any module's
   implementation detail. The dependency arrow always goes *module →
   core*.
3. **Dependencies are isolated per module.** Each module has its own
   optional-dependency group; a library one module needs does not leak
   into another module's environment.
4. **Every module is understandable and testable on its own.**
5. **`intent_classification` returns only a label; it does not route.**
   Mapping a label to a retrieval method is the consuming project's
   (`chatbot`) decision.

For the full rationale, see `ARCHITECTURE.md`.

## Modules

```
src/medrag/api/retrieval/
  core/        shared interfaces + ingestion-contract schemas (small, stable)
  eval/        golden-set format + metrics + async runner (cross-cutting)
  modules/
    intent_classification/   query -> intent label   (usable)
    db_query/                query -> SQL -> rows    (usable)
    top_n/                   dense vector search     (usable)
    query_rewriting/         rewriting / expansion   (planned)
    raptor/                  hierarchical retrieval  (excluded -- see note)
    agentic/                 multi-step flows        (planned)
  tests/       each module has its own tests/<module>/
```

> RAPTOR is not used in production — `modules/raptor/` is kept as a
> skeleton and no new features are added to it.

### `medrag.api.retrieval.core`

The surface every module shares:

- `RetrievalResult` — a single scored result every backend returns the
  same way (`id`, `score`, `text`, `metadata`).
- `Retriever` — the single async method every backend implements:
  `async def retrieve(query, k) -> list[RetrievalResult]`. `run_sync` is
  a thin helper for synchronous callers.
- `IntentLabel` (9 classes) / `IntentResult`.
- Ingestion-contract schemas — pydantic models for the JSON an
  upstream ingestion/parsing project produces (chunks, product graph,
  document metadata). The contract is defined by *shape*, not by
  importing the producer — any project that emits these shapes can
  feed this layer. Models use `extra="ignore"` so upstream format
  drift does not break parsing.

### `intent_classification`

A prompt-based classifier on top of any `langchain-core` `BaseChatModel`.
Returns a single `IntentResult` and does nothing else — the
label → strategy mapping is the consumer's job.

```python
from medrag.api.retrieval.modules.intent_classification import (
    LLMIntentClassifier, build_default_chat_model,
)

clf = LLMIntentClassifier(build_default_chat_model())
result = await clf.classify("de1000 fiyati ne kadar")   # -> IntentResult
```

In tests, inject a fake `BaseChatModel` instead of the real factory — no
network, no model download. From synchronous code, use the
`classify_sync(clf, query)` helper (mirrors `run_sync` for retriever
callers).

### `top_n`

Dense vector search over an existing Qdrant collection. Two seams:

- `Embedder` (query → vector) — the shipped `OpenAICompatEmbedder` talks
  HTTP to any server exposing `POST {base}/embeddings`
  (ollama/openai/openrouter/local); the `EMBEDDING_*` env group is
  shared with the planned `raptor` module.
- `VectorStore` (vector → scored results) — the shipped
  `QdrantVectorStore` is read-only (search-only); this module never
  creates or writes to a collection.

```python
from medrag.api.retrieval.modules.top_n import build_default_retriever

retriever = build_default_retriever()   # wired from QDRANT_*/EMBEDDING_* env
results = await retriever.retrieve("de1000 operating temperature", k=5)
```

**The query must be embedded with the SAME model that wrote the vectors**
— a model mismatch produces meaningless similarity, and a dimension
mismatch makes Qdrant reject the search outright. In this project's
reference pipeline, `vectorize` writes the collection (it is not imported
here, only the format/name contract binds the two).

### `db_query`

Retrieval by generating and running SQL. The natural-language query is
turned into a SQL string by the `text2sql-engine-native` package (a
2-stage LLM pipeline: schema linking → SQL generation), that SQL is run
against a ready-made database, and the rows come back as
`RetrievalResult`s.

```python
from medrag.api.retrieval.modules.db_query import build_default_retriever

retriever = build_default_retriever(db_path="specs.db")
results = await retriever.retrieve("operating temperature of PN1309", k=5)
```

Two injected seams keep the module testable and dependency-isolated:

- `SqlGenerator` (NL → SQL) — the production implementation wraps
  `text2sql_native`'s engine (it talks to a remote OpenAI-compatible
  endpoint); unit tests inject a fake generator that returns canned SQL.
- `SqlExecutor` (SQL → rows) — the shipped `SQLiteExecutor` uses only
  the stdlib `sqlite3` and opens the database **read-only by default**;
  a write attempt in the generated SQL fails at the engine level.

Score defaults to `1.0` (a SQL result is an exact match set, not a
similarity ranking; ordering is whatever the query's `ORDER BY`
produces) and is overridable. The whole row sits in
`RetrievalResult.metadata`. This module connects to an existing
database — it does not build one (that is `facts`' job).

The full pipeline is not mandatory — every piece is importable on its
own:

```python
from medrag.api.retrieval.modules.db_query import SQLiteExecutor, rows_to_results

cols, rows = SQLiteExecutor("specs.db").execute("SELECT * FROM products LIMIT 5")
results = rows_to_results(cols, rows, score=0.9)   # -> list[RetrievalResult]
```

## Inference and GPU rule

This machine does NOT run LLM/VLM/embedding inference (repo-wide rule).
Model inference runs on a separate machine, behind an OpenAI-compatible
HTTP endpoint (or Ollama's native `/api/chat`/`/api/embeddings`); this
layer only makes requests to that endpoint — it never downloads a model
or runs one locally. The default test run (`pytest`) is fully mocked and
never hits a real endpoint — tests that talk to a live endpoint are
marked `integration` and skipped by default.

## Install

The root `pyproject.toml`'s `serve` optional-dependency group includes
this layer's production dependencies (`qdrant-client`, `langchain-core`,
`langchain-openai`, `text2sql-engine-native`, ...). From the repo root:

```bash
uv sync --extra serve       # or: pip install -e ".[serve]"
```

If you develop on this component only, also add the `dev` group
(`--extra dev`). The `retrieval/pyproject.toml` in this directory is
stale — do not install from it (see the note at the top).

## Configuration

No secrets or real values here, just pointers:

- `src/medrag/api/retrieval/.env.example` — which env variable belongs to
  which module, grouped (`[inference]` shared LLM/embedding endpoints,
  `[top_n]` Qdrant connection, `[db_query]` DB path + text2sql-engine
  settings, ...). Copy it to `.env` and fill in real values there.
- The Qdrant collection name **must match** `[qdrant].collection_name` in
  `vectorize`'s `config/default.toml` — the same name/model must be used
  in all three places (this layer, `chatbot`, the embedding model), or
  requests fail with a dimension error.

## Use / test

From the repo root:

```bash
python -m pytest src/medrag/api/retrieval/tests -q
```

The default run skips `integration`-marked tests (they require a real
endpoint — run them only on the GPU machine with `-m integration`).

## For more information

- `ARCHITECTURE.md` — full rationale for the inter-module boundaries
  (this directory).
- `src/medrag/api/retrieval/modules/<module>/README.md` — each module's
  own detailed note (e.g. `top_n/README.md`, `db_query/README.md`).

## License

MIT. See `LICENSE` (this directory).