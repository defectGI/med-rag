# text2sql_native

> **INDEPENDENT FORK (originally 2026-07-30, fully decoupled 2026-08-06)** —
> lives in this repo at `packages/text2sql-native/`, wired into the root
> `pyproject.toml` as an editable path dependency
> (`[tool.uv.sources] text2sql-engine-native = { path = "packages/text2sql-native",
> editable = true }`, pulled in by the root `serve` extra). Distribution name
> `text2sql-engine-native`, import name `text2sql_native`, version `1.0.0`.
>
> This started as a fork of [defectGI/sqlretrieve](https://github.com/defectGI/sqlretrieve)
> (PyPI `text2sql-engine`, import name `text2sql`) adding a native `ollama`
> provider (see `text2sql_native/providers/ollama_provider.py`) + a `num_ctx`
> knob on `StageSettings`/`build_provider` — Ollama's `/v1` endpoint can't
> raise its context window per request (github.com/ollama/ollama/issues/5356);
> this provider talks to Ollama's native `/api/chat` instead, which can.
>
> **2026-08-06: fully renamed, not just version-pinned.** A version pin alone
> (`==0.2.0+ollama1`) was meant to stop `pip install -r requirements.txt`
> from silently falling back to the real, unmodified PyPI `text2sql-engine`
> — it happened anyway, for the fourth time, because the pin only makes the
> collision *improbable* (some install path can still name the old
> distribution) while both packages still shared the same import name
> (`text2sql`), so either one can silently shadow the other in one
> environment. Renaming both the distribution (`text2sql-engine-native`) and
> the import name (`text2sql_native`) makes the collision *impossible*:
> nothing on PyPI has ever published under either name, and this codebase —
> not the stale upstream repo — is now the only place this package lives.

A **schema-agnostic, config-driven Text-to-SQL library**. Give it a
natural-language request and metadata for any relational schema, and it returns
a SQL string.

It is a **library**, not an app. No UI, no web server, no CLI. It **never
connects to or runs against a database**. Generating the SQL is the last step.
Running it is the caller's job.

## Where this package sits in the pipeline

`text2sql_native` is an independent library meant to be imported on its
own; its only in-tree consumer inside medrag is
`src/medrag/api/retrieval/modules/db_query/`:

- `db_query/factory.py::build_default_engine()` lazy-imports
  `Text2SQL.from_config()` from this package and builds the engine (a
  clear `ImportError` is raised if the package is not installed — i.e.
  the root `serve` extra is missing).
- `db_query/generator.py::Text2SqlGenerator` consumes the generated SQL
  and `db_query/executor.py::SQLiteExecutor` runs it read-only.
- Together these are the SQL-generation step of the chatbot's
  `sql_topn` retrieval flow — the previous step is user-request /
  intent resolution, the next step is converting the rows returned by
  `SQLiteExecutor` into a response.
- **The input schema and runtime config do NOT live in this package**;
  they live wherever it is integrated: schema at
  `facts/db/schema.yaml`, behavioral `config.toml` + prompt templates
  under `chatbot/text2sql/`, provider/model selection (unprefixed
  `SCHEMA_PATH` / `LINKING_*` / `SQL_PROVIDER` and similar keys)
  documented in `src/medrag/api/.env.example`. The `.env` / `config.toml`
  walkthrough further down in this README describes the package's OWN
  (standalone) usage mode — meant for running it against the example
  schema and prompts under `examples/`.

## How it works — a 2-stage LLM pipeline

```
user request
      │
      ▼
┌─────────────────────────────┐   full schema metadata
│ Stage 1: Schema Linking LLM │◄──────────────────────
│  reduce schema → subset     │
└─────────────────────────────┘
      │  linked schema (tables, columns, joins, entities, filters, rationale)
      ▼
┌─────────────────────────────┐
│ Stage 2: SQL Generation LLM │
│  produce SQL in dialect     │
└─────────────────────────────┘
      │
      ▼
  Text2SQLResult(sql, linked_schema, metadata, explanation)
```

- **Stage 1 (linking)** takes the request and the *full* schema metadata. It
  reduces the schema to the relevant part and returns structured JSON. Output is
  forced via `response_format` (OpenAI) or tool-calling (Anthropic), with
  strict-JSON parsing and retry as a fallback.
- **Stage 2 (generation)** takes the request, that reduced subset, and Stage 1's
  reasoning (rationale, chosen joins, column reasons). It follows that logic and
  writes SQL in the target dialect. Two output modes (config `output_mode`):
  `structured` forces JSON with `sql` + `explanation`; `raw` asks for plain SQL
  text, for specialized text-to-SQL models.
- The two stages can use **different providers and models**. For example a cheap
  general model for linking, a stronger SQL model for generation.

## Installation

Not published to PyPI, and never will be.

**Inside the repo (normal path):** the root `pyproject.toml`'s `serve`
extra already pulls this package via the editable path in
`[tool.uv.sources]` — no manual install is needed:

```bash
uv sync --extra serve      # from the repo root
```

**When developing/testing this package on its own**, install editable
from this directory:

```bash
pip install -e .              # core library
pip install -e ".[all]"       # + openai and anthropic SDKs
pip install -e ".[dev]"       # + pytest
```

The provider SDKs (`openai`, `anthropic`) are imported lazily. So the library
imports, and the tests run, without them installed.

## Quick start

```python
from text2sql_native import Text2SQL

engine = Text2SQL.from_config()          # reads .env + config.toml
result = engine.run("get me the top 5 best-selling products last month")

print(result.sql)              # the generated SQL string
print(result.linked_schema)    # Stage 1 output (reduced schema)
print(result.metadata)         # models, tokens, latency, attempts
```

This package **does not ship its own `.env.example`** by design (in this
repo every `.env.example` belongs to a runtime component; this package
is a library only). There are two usage modes:

- **Real in-repo usage (the chatbot's `sql_topn` flow):** the
  configuration already lives in `src/medrag/api/.env.example` (unprefixed
  `SCHEMA_PATH`, `LINKING_*`, `SQL_*` keys) and
  `chatbot/text2sql/config.toml` — see "Where this package sits in the
  pipeline" above. You do not need to configure this package separately.
- **Standalone try-out (running this package on its own, against the
  `examples/` schema):** create your own `.env` in this directory by
  hand (with the keys listed in the table below) and point
  `SCHEMA_PATH` at `examples/schema.yaml`. There is no ready-made
  `config.toml` under `examples/` — write your own `config.toml` using
  the defaults from the `[general]` / `[linking]` / `[generation]` /
  `[retry]` tables below (or point `CONFIG_PATH` at an existing one,
  e.g. `../../chatbot/text2sql/config.toml`).

## Configuration — two separate layers

Config is split by concern. **Secrets, paths, and provider/model *selection* go
in `.env`. Everything tunable about *behavior* goes in `config.toml`.** One typed
layer (`text2sql_native.config.Settings`) loads both. It fails fast with a clear error
if a required `.env` variable is missing.

### `.env` — secrets, paths, provider & model selection

| Variable | Required | Description |
|----------|----------|-------------|
| `SCHEMA_PATH` | yes | Path to the schema file (`.json`, `.yaml`, `.yml`). |
| `PROMPT_DIR` | no | Directory with prompt templates (`linking.txt`, `generation.txt`). If unset, the templates **packaged inside the library** are used. So it works after pip install, from any working directory. |
| `CONFIG_PATH` | no | Override the `config.toml` location. Defaults to `./config.toml`. |
| `LINKING_PROVIDER` | yes | Provider for Stage 1. One of the registered names: `openai`, `anthropic`, `ollama` (native `/api/chat`, no SDK/API key required). |
| `LINKING_MODEL` | yes | Model id for Stage 1 (schema linking). |
| `LINKING_BASE_URL` | no | Endpoint override for Stage 1 (for OpenAI-compatible gateways). |
| `SQL_PROVIDER` | yes | Provider for Stage 2. |
| `SQL_MODEL` | yes | Model id for Stage 2 (SQL generation). |
| `SQL_BASE_URL` | no | Endpoint override for Stage 2. |
| `OPENAI_API_KEY` | if used | API key. Needed only if a stage uses provider `openai`. |
| `ANTHROPIC_API_KEY` | if used | API key. Needed only if a stage uses provider `anthropic`. |

Provider, model, and `base_url` are picked **per stage**, independently.

### `config.toml` — behavior

No magic numbers in code. Every tunable is here.

#### `[general]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sql_dialect` | string | `postgres` | Target dialect (e.g. `postgres`, `mysql`, `sqlite`). Passed to the prompts. |
| `log_level` | string | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |
| `strict_json` | bool | `true` | `true`: parse LLM JSON strictly. `false`: tolerate fences / extra prose. |

#### `[linking]` — Stage 1

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temperature` | float | `0.0` | Sampling temperature. |
| `top_p` | float | `1.0` | Nucleus-sampling cutoff. |
| `max_tokens` | int | `1024` | Max tokens for the linking response. |
| `prompt_template` | string | `linking.txt` | Template file, looked up in `PROMPT_DIR`. |
| `max_tables` | int | `8` | Max tables linking may return. |
| `timeout_seconds` | float | `60` | Per-request timeout. |
| `num_ctx` | int | unset | `ollama` provider only: forced context window via native `options.num_ctx` (see `providers/ollama_provider.py`). Other providers accept and ignore it. Unset means "don't force one", not "force 0". |

#### `[generation]` — Stage 2

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temperature` | float | `0.0` | Sampling temperature. |
| `top_p` | float | `1.0` | Nucleus-sampling cutoff. |
| `max_tokens` | int | `1024` | Max tokens for the SQL response. |
| `prompt_template` | string | `generation.txt` | Template file, looked up in `PROMPT_DIR`. In `raw` mode, defaults to `generation_raw.txt` when unset. |
| `timeout_seconds` | float | `60` | Per-request timeout. |
| `output_mode` | string | `structured` | `structured`: force JSON with `sql` + `explanation`, reliable parsing, best for general models. `raw`: ask for plain SQL text (no explanation), better for specialized text-to-SQL models trained to emit SQL only. |
| `num_ctx` | int | unset | Same as `[linking] num_ctx` above, for Stage 2. |

#### `[retry]` — shared by both stages

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_attempts` | int | `3` | Attempts per LLM call before failing. Retries on JSON-parse and transient provider errors. |
| `backoff_seconds` | float | `1.0` | First delay between retries. |
| `backoff_factor` | float | `2.0` | Delay is multiplied by this each retry. |

## Schema metadata format

The schema is **never hardcoded**. It is loaded from an external file through a
`SchemaProvider`. `examples/schema.yaml` (and the same `examples/schema.json`)
show the full format: a small e-commerce database with `customers`,
`categories`, `products`, `orders`, and `order_items`. Each table has a
description. Each column has a name, type, description, primary-key flag, and
sample values. Foreign keys are listed. Both JSON and YAML work, picked by
extension.

The example runs out of the box. The tests exercise the pipeline against it with
mocked LLM calls.

## Extending

Everything is swappable via **config + dependency injection**:

```python
from text2sql_native import Text2SQL, SchemaProvider, PromptTemplates
from text2sql_native.schema import Schema

# 1. Custom schema source, e.g. live DB introspection instead of a file.
class MyIntrospectionProvider(SchemaProvider):
    def load(self) -> Schema:
        ...   # build and return a Schema

# 2. Inject any part. Anything left None is built from config.
engine = Text2SQL.from_config(
    schema_provider=MyIntrospectionProvider(),
    prompts=PromptTemplates("/path/to/my/templates"),
    # linking_provider=..., generation_provider=...,
)
```

### Adding a new LLM provider

Subclass `LLMProvider`, implement `complete` (and optionally `complete_json` for
native structured output), and register it. One class, one decorator:

```python
from text2sql_native import LLMProvider, register_provider
from text2sql_native.types import LLMResponse

@register_provider("myprovider")
class MyProvider(LLMProvider):
    def complete(self, *, system, user, temperature, top_p, max_tokens) -> LLMResponse:
        ...
```

Then set `LINKING_PROVIDER=myprovider` (or `SQL_PROVIDER`) in `.env`.

### Prompts

Prompt templates are plain, editable text files in `PROMPT_DIR`. Each has a
`SYSTEM:` and a `USER:` section with `{placeholder}` substitution. Edit them
without touching code.

## Result object

`engine.run(...)` returns a `Text2SQLResult`:

| Field | Description |
|-------|-------------|
| `sql` | The generated SQL string. The deliverable. |
| `explanation` | Short explanation from Stage 2. |
| `linked_schema` | Stage 1 output: `tables`, `columns`, `joins`, `entities`, `filters`, `rationale`. |
| `metadata` | `RunMetadata`: per-stage provider/model, token usage, latency, attempts. Plus `total_usage` and `total_latency_seconds`. |
| `request` | The original request. |

## Error handling

Every error subclasses `Text2SQLError`:

- `ConfigError` — missing/invalid `.env` var or `config.toml`.
- `SchemaError` — schema file missing or invalid.
- `ProviderError` — provider build or API call failed.
- `StructuredOutputError` — output not parseable as JSON after all retries.
- `EmptyLinkingError` — Stage 1 matched no table/column.

Uses stdlib `logging` throughout, never `print`. Secrets are never logged.

## Try it end to end

With a real provider/model and (if needed) a key set in this package's own
`.env` (standalone mode — see Installation/Configuration above):

```bash
python examples/run_example.py "top 5 best-selling products last month"
```

Makes real LLM calls for both stages. GPU/LLM inference is NOT run on
this machine (repo-wide rule) — when `provider=ollama` is selected,
requests only go to an already-running Ollama HTTP endpoint over the
network (`http://localhost:11434` or `LINKING_BASE_URL`/`SQL_BASE_URL`),
no model is loaded or run in-process. Prints the linked schema, the
SQL, and the metadata. Never connects to a database.

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests mock all LLM calls (no network) and run on `examples/schema.yaml`.

## Project layout

```
text2sql_native/
  config.py              .env + config.toml loading, typed settings
  types.py               result dataclasses
  errors.py              exceptions
  logging_utils.py       logging setup
  schema/                SchemaProvider, file loader, models, serializer
  providers/             base + registry + openai + anthropic + ollama (native)
  prompts/               packaged default linking.txt / generation*.txt templates
  pipeline/              linking (Stage 1), generation (Stage 2), orchestrator
examples/
  schema.yaml / schema.json    sample metadata "spec"
  run_example.py               runnable end-to-end example (real LLM call)
tests/                   pipeline, schema, config, prompt, and ollama-provider tests
pyproject.toml           packaging + [tool.uv.sources] editable path (see root pyproject.toml)
CHANGELOG.md             version history
LICENSE                  MIT
```

Note: there is no `config.toml` / `.env` in this directory — this package
is a library and does not carry its own runtime config files (see
"Where this package sits in the pipeline" above). `examples/run_example.py`
needs its own `.env` for a standalone try-out, and a `config.toml` if
you want one.

## For more information

- `CHANGELOG.md` (this directory) — version history and the rationale
  for the `ollama` provider and the rename.

## License

MIT. See [LICENSE](LICENSE).
