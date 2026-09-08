# chunker

A parser-independent chunking library that turns a parser's output (a
structural document IR) into structure-preserving chunks suitable for RAG.

The primary scenario is chunking the `ParsedDocument` IR JSON produced by
`medrag.pipeline.parser` (v7+), but the core engine knows nothing
parser-specific. Every parser-specific detail lives inside an **adapter** —
to use a different parser, write a new adapter; the engine and strategies
keep working unchanged.

## Where it sits in the pipeline

```
parser (IR JSON)  →  chunker (this component)  →  vectorize / facts / retrieval
```

- **Previous step:** `medrag.pipeline.parser` converts ACME document/product
  data into a structural IR (`ir_version` + `blocks`). The chunker reads
  this IR and never parses any document format itself.
- **This step:** Splits the IR into token-budgeted, structure-preserving
  chunks; resolves cross references; writes a single output root
  (`all_chunks.json` + `all_combined.json`).
- **Next steps:** `vectorize` (embedding + Qdrant), `facts` (provenanced
  knowledge layer), `retrieval` — all read this output as INPUT and do not
  import chunker code.

## Architecture

```
parser output (any format)
        │
        ▼
┌──────────────────┐   Parser-specific code lives ONLY here.
│ adapters/        │   The first_parse adapter is the reference implementation.
│  (IR → inner model) │  Each adapter converts to the engine's shared inner model.
└──────────────────┘
        │  shared inner document model (parser-agnostic)
        ▼
┌──────────────────┐   Splitting decisions live here: atomic blocks, token budget,
│ core/            │   flex allowance formula, hierarchical heading injection.
│  (chunking engine)│  Token counting goes through an abstract tokenizer interface.
└──────────────────┘
        │  chunks + base metadata
        ▼
┌──────────────────┐   Runs AFTER chunks are produced:
│ enrichment/      │   cross-reference → chunk id resolution.
└──────────────────┘
        │
        ▼
    chunk JSON output
```

- `adapters/` — parser-specific transformers (initial adapter: `first_parse` IR v7+)
- `core/` — parser-agnostic chunking engine and inner document model
- `tokenization/` — tokenizer abstraction (limit is in **tokens**, not characters)
- `enrichment/` — cross-reference resolution (RAPTOR was removed: the embedding-
  based summary tree added no value for a templated corpus with clean metadata
  hierarchy)
- `config/` — default config; includes flex allowance coefficients and every
  threshold that influences splitting
- `tests/` — test suite (offline; no LLM calls — this component runs entirely
  without LLMs)

(All sub-packages above live under `src/medrag/pipeline/chunker/`.)

## Features

- **Atomic blocks** — tables/lists/code are not split; if they do not fit
  they are split at their natural boundary (line/sentence/item) with overlap,
  without breaking the structure.
- **Token-based limit** — a pluggable tokenizer (default tiktoken
  `cl100k_base`, `fake` for tests).
- **Flex allowance** — a per-block-type formula (`config/default.toml`
  `[flex.ratios]`) lets the token limit be exceeded to avoid splitting a
  valuable structure; `hard_max_tokens` is still the absolute ceiling.
- **Hierarchical chunking** — section (heading) boundary strategy is
  configurable (`[packing] section_strategy`: `hard` | `merge`); each chunk
  carries a `heading_path` breadcrumb.
- **In-document cross-reference resolution** — when an `InlineRun.link`
  points at a heading in the same document, the target chunk id is written
  into metadata.
- **Image awareness** — `image_id`/`ocr_text`/`alt_text`/`description`/
  `visual_type` are carried in a structured `images` list; types whose
  content is excluded render as a placeholder (the entity is never hidden).
- **Solid provenance** — every `ChunkSet` is stamped with the source hash
  and version (for staleness detection); every node carries its own
  `content_sha256`.
- **Fully offline, LLM-free core engine.** RAPTOR (embedding-based summary
  tree) was removed entirely.

For detailed design decisions (flex formula, schema fields, output layout)
see the `src/medrag/pipeline/chunker/cli.py` docstring and the root
configuration documentation (`CONFIG.md`).

## Installation

In the repo root (this is not a sub-project; it's a package included in the
root `pyproject.toml`):

```
pip install -e .
```

The core chunking runs **offline and without LLMs** — no GPU or model server
is needed. The default tokenizer uses tiktoken (pinned in the root
`pyproject.toml` under the `serve` extra); if tiktoken is not installed,
`TOKENIZER=fake` (deterministic, not for production) lets it run offline.

## Configuration

- **`.env`**: see `src/medrag/pipeline/chunker/.env.example` (copy it to
  `src/medrag/pipeline/chunker/.env`). It contains NO secrets — only
  tokenizer selection and CLI input/output paths.
- **Splitting/threshold settings**: `src/medrag/pipeline/chunker/config/default.toml`
  (partially overridable via `CHUNKER_CONFIG`). No number that affects
  splitting is hardcoded in code; everything comes from here.

## Usage (CLI)

```
python -m medrag.pipeline.chunker
```

The interface is **environment variables** (the only exception is
`--limit N`, for trial/sampling — not a persistent setting, intentionally
a flag rather than env):

```
python -m medrag.pipeline.chunker --limit 5
```

| Variable | Meaning |
|---|---|
| `CHUNKER_INPUT_DIR` | required — root for IR JSON; recursively scanned; a `*.json` is treated as IR only when its top level has `ir_version` + `blocks` (side files are skipped with a log entry) |
| `CHUNKER_OUTPUT_DIR` | default `./storage` — output root: `all_chunks.json`, `all_combined.json`, `viz/`, `archive/` |
| `CHUNKER_CONFIG` | optional — partial override TOML applied on top of `config/default.toml` |
| `CHUNKER_VIZ` | `on` (default) \| `off` — for each document, produce an offline interactive HTML tree (`viz/{doc_id}.tree.html`) + `viz/index.html` |
| `CHUNKER_PROGRESS` | `on` (default) \| `off` — terminal progress bar (when tqdm is installed and stderr is a real terminal) |
| `TOKENIZER` | tiktoken encoding name (default `cl100k_base`) or `fake` |

Cross-reference resolution is always on (offline). Exit codes: `0` = all OK,
`1` = at least one document failed (run continues with the others),
`2` = configuration error (including no IR found).

## Output layout

`CHUNKER_OUTPUT_DIR` is a **root** (default `./storage`); for details and
the "when does it change" table see [`storage/README.md`](storage/README.md);
the layout itself is defined in `src/medrag/pipeline/chunker/layout.py`.

```
storage/
  all_chunks.json     documents: {doc_id -> ChunkSet}   → rewritten every run
  all_combined.json   wrapped form of the same dictionary
  archive/{generated_at}/   previous run output (not deleted)
  viz/                {doc_id}.tree.html + index.html   → derivative, overwritten each run
```

## Chunk schema (summary)

Single node type; since RAPTOR was removed, `core/` always produces only
leaves (`tree_level=0`). Every `ChunkSet` carries a `provenance` block
(`chunker_version`, `generated_at`, source IR identifier) — this is a
staleness gate that determines whether the run must regenerate. For the
full schema field list and decision rationale, see
`src/medrag/pipeline/chunker/core/chunk.py` and the root configuration
documentation (`CONFIG.md`).

## Running tests

```
pytest src/medrag/pipeline/chunker -q
```