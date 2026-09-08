# Changelog

Notable changes are listed here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-06 (independent fork, decoupled from upstream)

### Changed
- **Renamed distribution + import name** to stop colliding with the real
  PyPI `text2sql-engine` package (import name `text2sql`), which this fork
  diverged from. A version pin alone (`==0.2.0+ollama1`) made the collision
  merely improbable, not impossible — a plain `pip install text2sql-engine`
  (or any install step naming the old distribution) could still silently
  shadow this package, and it happened for the fourth time. Distribution is
  now `text2sql-engine-native`; import name is `text2sql_native`
  (`from text2sql_native import Text2SQL`). Nothing on PyPI has ever
  published under either name, so the collision is now structural, not just
  probabilistic.
- No functional/behavioral change from `0.2.0+ollama1` -- same providers,
  same `num_ctx` support, same config shape. Only names moved.

## [0.2.0+ollama1] - 2026-07-30 (local fork, not upstream)

### Added
- New `ollama` provider (`text2sql/providers/ollama_provider.py`): talks to
  Ollama's native `/api/chat` instead of the OpenAI-compatible `/v1`
  endpoint, because `/v1` cannot raise the context window per request
  (github.com/ollama/ollama/issues/5356) and silently truncates an
  over-budget prompt instead of rejecting it.
- `StageSettings.num_ctx` (`[linking]`/`[generation] num_ctx` in
  `config.toml`) — forwarded to `build_provider()`/`LLMProvider.__init__` as
  `num_ctx`; every other provider accepts and ignores it.

## [0.2.0] - 2026-07-22

### Changed
- Stage 2 now receives Stage 1's reasoning: the linking rationale, the chosen
  joins, and the per-column reasons are passed into the generation prompt. The
  SQL model follows the linking logic instead of guessing.
- Default generation prompts (`generation.txt`, `generation_raw.txt`) tell the
  model to respect the intended comparison semantics (numeric range / exact
  compare) and not fall back to substring/LIKE matching unless meant.
- New prompt placeholders: `{rationale}`, `{columns}`, `{joins}`. Custom
  templates keep working; they just do not have to use them.

## [0.1.1] - 2026-07-22

### Added
- `[generation] output_mode` config. `structured` (default) forces JSON with
  `sql` + `explanation`. `raw` asks for plain SQL text, for specialized
  text-to-SQL models. In raw mode the prompt defaults to `generation_raw.txt`
  unless a template is set.

## [0.1.0] - 2026-07-22

### Added
- First release.
- Two-stage Text-to-SQL pipeline. Stage 1 links the schema, Stage 2 generates
  the SQL.
- Provider-agnostic LLM layer with a registry/factory. Built-in OpenAI
  (OpenAI-compatible) and Anthropic providers. Adding a provider is one class.
- `SchemaProvider` abstraction. File loader supports JSON and YAML.
- Two config layers: `.env` (secrets, paths, provider/model selection) and
  `config.toml` (behavior).
- Forced Stage 1 output via `response_format` (OpenAI) or tool-calling
  (Anthropic). Strict-JSON parsing with retry/backoff as a fallback.
- Editable prompt templates. Packaged with the library, so they resolve from any
  working directory.
- Example e-commerce schema, a runnable example, and a test suite with mocked
  LLM calls (no network).
- Typed public API (`py.typed`), MIT license, CI on Python 3.11 / 3.12.
