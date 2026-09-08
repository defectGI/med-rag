# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[SemVer](https://semver.org/).

## [0.5.0] — 2026-07-23

Observability: surface what the multi-stage LLM pipeline already computes,
instead of silently discarding it on both success and failure.

### Added
- `core.IntentResult.raw_text` — set only when the classifier's label is a
  silent fallback (model output didn't match any known label and
  `out_of_scope` was substituted). `None` when the model genuinely returned
  a valid label.
- `intent_classification.parsing.parse_label_verbose(raw) -> (IntentLabel,
  bool)` — same mapping as `parse_label`, plus whether this was a fallback.
  `parse_label` itself is unchanged for existing callers.
- `db_query.generator.OnStage` type alias + `Text2SqlGenerator.generate(query,
  *, on_stage=None)` — purely observational callback invoked with
  linking/generation stage detail (tables, rationale, latency, tokens) on
  success, or the stage + raw model text on failure (via the underlying
  engine's error's `raw_text`, added upstream in `text2sql-engine`/
  `sqlretrieve`). Never changes what is returned or raised.
- `db_query.DbQueryRetriever.retrieve(..., on_stage=None)` — forwards to the
  generator if it supports `on_stage` (detected via `inspect.signature`, not
  a try/except-TypeError guess). Optional, additive; omitting it is
  byte-for-byte the prior behavior.

### Fixed
- `intent_classification.build_default_chat_model`'s thinking-suppression
  `extra_body` is now **opt-in** (`LLM_THINKING=off` explicitly) rather than
  a silent default — a default-injected `{"reasoning_effort": "none"}` a
  server rejects would previously hard-fail every classify call (no
  retry-without-body hook in langchain `ChatOpenAI`, unlike the raw-HTTP
  answering model in the consuming project).

## [0.4.1] — 2026-07-23

### Added
- `db_query.DbQueryRetriever.retrieve(..., on_sql=None)` — optional,
  purely-observational hook called with the final (cleaned + rewritten) SQL
  string right before execution.

## [0.3.0] — 2026-07-22

Architecture-review follow-ups: tighter contracts, leaner base, consistent sync.

### Added
- `db_query.SqlEngine` / `db_query.SqlResult` protocols — the engine injected
  into `Text2SqlGenerator` is now typed (previously untyped `Any`), and
  `build_default_engine` declares its return type. `text2sql-engine` stays
  optional and injectable; a fake `run(request) -> .sql` is a valid substitute.
- `intent_classification.classify_sync(classifier, query)` — a free helper that
  mirrors `retrieval.core.run_sync` for classifiers.

### Changed
- **`langchain-core` moved out of base `dependencies`** into the
  `intent_classification` extra — it is the only code that imports it. Installing
  just `[db_query]` no longer pulls LangChain (ARCHITECTURE.md #8, clarified).
- `RetrievalResult.metadata` typed as `dict[str, Any]` (was bare `dict`).

### Removed
- `LLMIntentClassifier.classify_sync` **method** — replaced by the free
  `classify_sync` helper, so the classifier's protocol surface stays just the
  async `classify`. (Was unused elsewhere; the public entry point is the helper.)

## [0.2.0] — 2026-07-22

Make `db_query` composable — no forced pipeline.

### Added
- `db_query.mapping` — standalone, importable row → `RetrievalResult` mapping
  (`rows_to_results`, `row_to_result`, `pick_row_id`, `render_row_text`,
  `DEFAULT_ID_CANDIDATES`). Anyone running SQL their own way can reuse the exact
  mapping without touching `DbQueryRetriever`.
- `DbQueryRetriever.generator` / `.executor` properties to reach the injected
  seams, and a `score` argument (default `1.0`) so callers that rank rows
  themselves aren't forced to a constant.

### Changed
- `DbQueryRetriever` now delegates row mapping to `db_query.mapping`; behavior
  is unchanged. `score` is overridable throughout (`rows_to_results`,
  `row_to_result`).

## [0.1.0] — 2026-07-22

First public release.

### Added
- **`retrieval.core`** — the small, stable shared surface every module depends
  on: `RetrievalResult`, the async `Retriever` protocol (+ `run_sync`),
  `IntentLabel` (9 classes) / `IntentResult`, and the ingestion-contract
  pydantic schemas (`ChunkFile`/`ChunkNode`, `ProductFile`/`ProductNode`,
  `DocumentFile`/`DocumentNode`) with file loaders.
- **`retrieval.eval`** — golden-set format, hand-written metrics
  (accuracy / macro-micro F1 / confusion, recall@k / MRR) and an async runner.
- **`intent_classification`** module (extra `intent_classification`) — an
  LLM-prompt classifier over any `langchain-core` `BaseChatModel`, returning a
  single `IntentResult`. Provider-agnostic; a factory wires a real
  OpenAI-compatible endpoint. Defensive label parsing, graceful
  logprobs→confidence.
- **`db_query`** module (extra `db_query`) — retrieval by generating SQL
  (via the `text2sql-engine` package) and running it against a ready-made
  database. Two injected seams: `SqlGenerator` (NL→SQL) and `SqlExecutor`
  (`SQLiteExecutor`, read-only). Rows map to `RetrievalResult` with a constant
  score of `1.0`.

### Notes
- Every module depends only on `retrieval.core`, never on another module.
- Model inference is expected behind a remote OpenAI-compatible endpoint; the
  library only sends requests and never runs local inference. The default test
  run is fully mocked and never touches a network endpoint.
