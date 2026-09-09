# benchmark — parser-output quality scoring

Scores the parser's Markdown output against the **source PDF** using a
vision LLM as judge. Every document gets three sub-scores and a
weighted overall:

| dimension | what it measures |
|-----------|------------------|
| accuracy  | Is the content that IS in the Markdown correct against the PDF? |
| coverage  | What fraction of the PDF's in-scope content made it into the Markdown? |
| clarity   | Structure, table formatting, reading order, absence of leaked noise |

The PDF is treated as ground truth (rendered into page images); the
Markdown is the candidate. An optional **scope** file narrows what is
graded. The overall score is a weighted average of the sub-scores,
computed locally and deterministically (never asked of the model).

## Where it sits in the pipeline

`tools/benchmark` is an independent tool that **post-hoc verifies the
parser pipeline's output** -- not a part of the parse chain itself, and
no module imports it. Its input is the `pipeline/*/<doc>/` output
folders written by `src/medrag/pipeline/` (each containing `<name>.pdf`
+ `<name>.md`, and optionally `<name>.json` IR for chunking). Run it as
a standalone CLI: `python -m tools.benchmark`.

## Features

- Single-document **or batch** (folder-of-folders) scoring; ambiguous
  subfolders are skipped with a reason rather than guessed at.
- Provider-agnostic judge: reuses the parser's own `llm/` client --
  ollama, anthropic, openai-compatible, openrouter.
- Context-window safety net: pre-call rough estimate warning +
  post-call real `prompt_tokens` evidence of truncation.
- Page-aligned **chunking** for large documents (uses the parser's IR
  JSON), with page-weighted aggregation of per-chunk scores.
- Two-step **model preflight** (canary image read + thinking/reasoning
  behavior) -- catches a broken model setup before any document is
  graded.
- Optional **scope** file to define what's in/out of the score
  (`scope.example.md` is a template; `scope.md` is this repo's concrete
  example tailored to its parser).
- Grading tuning lives entirely in `config/default.toml` (no hardcoding
  in code), with a partial `--config` override supported.

## Install

`tools/` is an installed package (root `pyproject.toml`
`[tool.setuptools.packages.find] include = ["medrag*", "tools*"]`); there
is no separate `requirements.txt`. From the repo root:

```
uv sync --extra parse --extra dev
```

The `parse` extra pulls in the render (`pdfplumber`, `Pillow`) and LLM
client (`anthropic` included) dependencies -- this tool reuses
`src/medrag/pipeline/parser/llm/` and the PDF renderer **as-is** and adds
no new dependencies of its own. The `dev` extra is for tests (`pytest`).

## Usage

Run as a module from the repo root (bare-script execution is no longer
supported):

```
python -m tools.benchmark -f <folder> [-s scope.md]
```

`<folder>`:
- a folder holding one `.pdf` + one `.md` -> single document, or
- a folder of such folders -> **batch** (one score per subfolder +
  average). Matches the `pipeline/*/<doc>/` output layout directly.

Ambiguous subfolders (two PDFs, no `.md`, several `.md`s with no
matching PDF) are **skipped with a reason** and listed in the report --
never guessed at.

### Examples

```
# grade a single pipeline output folder end-to-end
python -m tools.benchmark -f pipeline/e2e_test_output/2026-07-08_13-15-22Z/01_DATASHEET_...

# batch, narrowed with scope.md, custom weights, JSON report
python -m tools.benchmark -f pipeline/e2e_test_output/2026-07-08_13-15-22Z \
  -s tools/benchmark/scope.example.md \
  --weights "accuracy=2,coverage=2,clarity=1" \
  -o report.json

# grading tuning from a TOML override (flags still beat it); model from .env
python -m tools.benchmark -f <folder> --config tools/benchmark/config.example.toml
```

## Model / provider

Model selection reuses `src/medrag/pipeline/parser/llm/`; every provider
that package supports works here too: **ollama, anthropic,
openai-compatible, openrouter**. The model MUST be multimodal (the PDF
is graded from page images). LLM/VLM inference is NOT run locally on
this machine -- if ollama is selected, requests go over HTTP to a
separate Ollama server (`VLM_BASE_URL`); the model is not loaded on
this machine.

The model connection lives in the ENVIRONMENT (not in any config file).
Resolution order, latest wins:

1. `.env` files, loaded LAYERED (highest precedence first): `--env-file`,
   then `tools/benchmark/.env`, then `./.env`, then
   `src/medrag/pipeline/parser/.env`
2. Flags: `--provider --model --base-url --api-key --thinking --num-ctx`

Flags map onto the `VLM_*` env vars (which fall back to `LLM_*`).
**`tools/benchmark/.env` is the default home for judge-specific config
and is auto-loaded -- no flag needed.** Because files are layered,
`tools/benchmark/.env` doesn't have to be complete: set only
`VLM_MODEL=...` and inherit provider/base_url from `parser/.env`. Fill
the whole `VLM_*` block and it's fully self-contained. This is how to
use a different judge model than the parser's without touching
`parser/.env`. Start by copying `tools/benchmark/.env.example` to
`tools/benchmark/.env`; `--model`/`--provider` per-run still wins.

## Config file (grading tuning only)

Grading tuning lives in `tools/benchmark/config/default.toml`
(committed, validated by `tools/benchmark/config.py`): `[render].dpi`,
`[grading].max_tokens/assume_context/max_pages`,
`[chunking].enabled/pages_per_window`, `[weights].accuracy/coverage/
clarity`, `[scope].exclude_types`. `--config <file.toml>` overrides a
subset (partial file, merged on top; see `config.example.toml`); a CLI
flag beats both. Precedence: `default.toml < --config toml < flag`.

Model/provider/api_key are NOT config-file keys -- they are environment
variables (`VLM_*`/`LLM_*`, see "Model / provider" above). Run inputs
(`folder`/`scope`/`out`) are CLI flags (`-f`/`-s`/`-o`), each with an
optional `.env` default the flag beats: `BENCHMARK_FOLDER`,
`BENCHMARK_SCOPE`, `BENCHMARK_OUT` -- the same way parser/pipeline
take their input paths from the env, so you can set a static folder in
`tools/benchmark/.env` and run with just `python -m tools.benchmark`.

## Other flags

- `--thinking {on,off}` explicitly sets `VLM_THINKING_ON` for the judge
  call (env var, not a config key). Default: whatever's in
  `parser/.env`, or off if unset. Leaving a hybrid-reasoning judge
  model ON risks burning its whole token budget on hidden reasoning
  and returning empty/unparseable content -- the `[2/2]` preflight
  check (below) catches this live before any document is graded,
  regardless of this setting.
- `--num-ctx N` (env: `VLM_NUM_CTX`; not a config key) forces the
  judge call's context window to N tokens, ollama only -- see below.
  Leave unset to take the safe default automatically.
- `--dpi N` page render resolution (config: `[render].dpi`, default 150).
- `--max-pages N` max pages sent to the model per document (config:
  `[grading].max_pages`, `0`=all). When a PDF is truncated the report
  and console say so (`(2/16p)`) and the judge is told to grade only
  the pages it can see -- coverage is never silently penalized.
- `--chunk` / `--no-chunk` (config: `[chunking].enabled`, default on)
  grade a document that doesn't fit one context window in page-aligned
  chunks rather than truncating -- see "Large documents" below.
  `--no-chunk` restores the old single-call behavior (large documents
  just warn).
- `--chunk-pages N` (config: `[chunking].pages_per_window`, `0`=auto)
  pages per chunk window. Auto = as many as fit the context budget; an
  explicit value caps the window height.
- `--max-tokens N` model response budget per document (config:
  `[grading].max_tokens`, default 2048). Limits the model's *output*,
  not input -- see the context-window note below.
- `--assume-context N` context window (tokens) used ONLY for the
  pre-call size warning when `num_ctx` is NOT enforced (config:
  `[grading].assume_context`, default 4096, Ollama's common default).
  When `num_ctx` is set/default, this is informational only -- the
  warning then compares against the real enforced number.
- `-o/--out PATH` JSON report path (default:
  `tools/benchmark/benchmark_reports/<timestamp>__<graded-folder-name>.json`
  -- never dropped into the graded folder; the name alone shows "when
  it ran" and "what it graded").

## Context window

`--max-tokens` limits the model's *output*; *input* (Markdown text plus
one image per page) is a separate problem. Ollama's `/v1`
(OpenAI-compatible) endpoint has no per-request way to raise the
context window ([ollama/ollama#5356](https://github.com/ollama/ollama/issues/5356))
-- an over-budget prompt is **truncated silently, from the start, with
no error** -- and the judge then scores pages / text it never saw.

**When the effective provider is `ollama` this is enforced by default,
not just warned about:** if `num_ctx` is left empty/`0`/`null`
everywhere (flag, config, `.env`), `src/medrag/pipeline/parser/llm/
__init__.py` applies a built-in default (`_DEFAULT_OLLAMA_NUM_CTX`,
currently 16384 tokens -- shared across the project, not benchmark-
specific) and routes the call through Ollama's own native `/api/chat`
instead of `/v1` -- the only endpoint that actually accepts
`options.num_ctx` per request. Every other provider is unaffected;
this only changes behavior for `ollama`. If your server has more VRAM
or your documents run longer than ~15-20 pages, override the number in
config via `"num_ctx"` or in `parser/.env` via `VLM_NUM_CTX` /
`LLM_NUM_CTX`.

Two more safety nets on top of the enforcement:

- **Before every call**: a rough (not exact) token estimate from page
  count + Markdown length; warns loudly if it approaches the enforced
  (or, if not enforced, default) window.
- **After every call**: if the server reported a `usage` /
  `prompt_eval_count` token count and that count is suspiciously close
  to a common context cap (2048 / 4096 / 8192 / ...), that's not an
  estimate but real evidence of truncation -- still warns and records
  it as `prompt_tokens` in the report.

If you still see the warning with `num_ctx` enforced: the estimate may
under-count (real tokenization varies per model) or the document truly
needs more headroom -- raise `num_ctx` further (limited by the server's
VRAM: a larger context window means a larger KV cache), or let chunking
take over instead of dropping pages via `--max-pages` (below).

## Large documents (chunking)

A document whose estimated prompt exceeds 80% of the context window is
graded in **page-aligned chunks** instead of one silently-truncated
call -- this is on by default (`--no-chunk` turns it off). Each chunk
is a contiguous page window (a few pages) sent with the Markdown slice
that belongs to exactly those pages, so the judge is never asked to
grade a slice against pages it cannot see (which would wrongly tank
coverage), nor shown pages the slice doesn't cover. Per-chunk scores
are then aggregated into one document score by a **page-weighted** mean
(a 10-page chunk counts ten times a 1-page one), so a chunked score
is directly comparable to a single-call one; the report carries the
per-chunk breakdown under `documents[].windows`.

Chunking requires the parser's **IR JSON** next to the `.md` (the
pipeline writes it there automatically -- `<name>.json`). That JSON
tags every block with its source page (`Span.page`), which is what lets
the Markdown be sliced to match each window -- the rendered `.md`
itself carries no page markers. A slice is re-rendered from the IR
through the parser's own Markdown renderer, so the chunks reproduce
the graded `.md` modulo small boundary effects (a list split at a
window edge, a table's trailing description). A document **without an
IR JSON** can't be aligned, so it falls back to a single call and the
same truncation warning.

Chunk height is auto by default (as many pages as fit the budget);
`--chunk-pages N` caps it. The console shows the split live
(`grading in 8 chunk(s) of <= 2 page(s)...`) with one line per window.

## Report

Console: one line per document (`accuracy coverage clarity overall`)
plus the batch average. JSON (`--out`): per-document sub-scores,
weights, page counts, `prompt_tokens` (if the server reported one),
per-dimension reasoning + concrete issues, a summary, an aggregate
block, and any skipped folders with reasons.

## Running tests

Offline -- hand-built folders + a fake VLM client; no model, no
network. From the repo root:

```
pytest tools/benchmark/tests
```