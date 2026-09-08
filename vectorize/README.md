# vectorize

Independent component that reads the chunk JSONs produced by `chunker`,
vectorizes every chunk node with an OpenAI-compatible embedding provider
(via HTTP to Ollama — no LLM/GPU inference runs on this machine), and writes
to Qdrant.

## Overview

One of medrag's independent components (parser, pipeline,
chatbot-corpus, benchmark + chunker + retrieval) — none imports another
directly; see the root `README.md`.

Place in the pipeline: `chunker` (produces chunk JSON) → **vectorize**
(embeds and writes to Qdrant) → `chatbot`/`retrieval` (reads from Qdrant and
answers).

**The code lives under `src/medrag/pipeline/vectorize/`** — this folder
(`vectorize/`) only hosts config/README and the local `storage/`. The module
path is accordingly `medrag.pipeline.vectorize` — the commands below run from
the repo root with the `medrag` package installed.

## Features

- Discovers both chunker's legacy (`{doc_id}.chunks.json`) and new
  (`all_chunks.json`/`all_raptor.json`/`all_combined.json`) output layouts by
  content, without trusting name patterns (`discover.py`).
- One OpenAI-compatible embedding client (ollama/openai/openrouter/local) —
  same pattern as chunker's `openai_compat.py`
  (`embedder/openai_compat.py`).
- **Delete-then-reinsert** when writing to Qdrant: when a scope is
  re-embedded, its old points are deleted and the new set is written from
  scratch (`store/qdrant_store.py::replace_scope`).
- **Staleness gate**: a signature built from source hash + chunker version +
  RAPTOR model/prompt versions + sha256 of all node texts
  (`core.ChunkSet.signature`) is written to local `storage/state.json`; a
  scope with the same signature is skipped on the next run (bypass with
  `--force`).
- Configurable: whether RAPTOR summary nodes (`tree_level >= 1`) are
  embedded and whether the `heading_path` chain is prefixed to the embedded
  text (`config/default.toml`).
- On a collection size mismatch (`on_dim_mismatch`), the default behavior is
  to fail loudly, not silently overwrite.
- Read-only search helper (`query.py`) — completely separate from the write
  side, the concrete example of how a consuming project connects to Qdrant.
- Per-scope error isolation: a failed scope is logged and the run continues
  with the rest (see the `cli.py` docstring for the 0/1/2 exit-code
  contract).

## Installation

No separate `requirements.txt` — dependencies are grouped in the root
`pyproject.toml` (the `serve` extra carries everything vectorize needs,
including `qdrant-client`). From the repo root:

```bash
pip install -e ".[serve]"
```

The `.env` file is derived from
`src/medrag/pipeline/vectorize/.env.example` (copy and fill it in if missing;
the real `.env` is gitignored).

## Configuration

- Secret/connection values (`EMBEDDING_*`, `QDRANT_*`, input/output paths):
  see `src/medrag/pipeline/vectorize/.env.example`, and do NOT copy real values
  into it — write them in `.env`.
- Tuning (batch size, collection name, staleness on/off, RAPTOR node
  inclusion, `on_dim_mismatch`, ...):
  `src/medrag/pipeline/vectorize/config/default.toml`.
- **Critical warning**: `EMBEDDING_MODEL` in `.env`, `[qdrant].collection_name`
  in `default.toml`, and the chatbot-side `EMBEDDING_MODEL` must be the same
  in all three places — if they diverge, every query fails with a dimension
  error (details in `.env.example`).

## Usage

From the repo root (with the `medrag` package installed):

```bash
python -m medrag.pipeline.vectorize                # embed the whole corpus
python -m medrag.pipeline.vectorize --limit 5       # trial with the first 5 scopes in discovery order
python -m medrag.pipeline.vectorize --force         # ignore the staleness gate, re-embed everything
```

For detailed behavior (env vars, exit-code contract): the
`src/medrag/pipeline/vectorize/cli.py` docstring.

### Search (read side)

`cli.py` only WRITES — the concrete example of how a consuming project
("chatbot", `retrieval`, manual trials) connects to Qdrant is `query.py`:

```bash
python -m medrag.pipeline.vectorize.query "PN1015'in çalışma sıcaklığı nedir"
python -m medrag.pipeline.vectorize.query "..." --k 10 --json   # to pipe into another program
```

The question is embedded with the SAME `EMBEDDING_*` model as the write side
(a different model/size would mismatch the vectors in Qdrant). Results always
carry `score` + the full payload (`text`, `doc_id`, `node_id`,
`heading_path`, `page_start`/`page_end`, `keywords`,
`chunk_schema_version`, `chunker_version`, ...). `search()` is for
programmatic use — a chatbot layer can import `medrag.pipeline.vectorize.query`
and call it directly.

## Architecture

```
src/medrag/pipeline/vectorize/
  core.py             input contract: ChunkSet/ChunkNode (MIRRORED from chunker, not imported)
  discover.py         finds chunk scopes under VECTORIZE_INPUT_DIR
  config.py           config/default.toml -> VectorizeConfig
  embedder/           OpenAI-compatible embedding client (mirrored from chunker's openai_compat.py)
  store/              Qdrant client: ensure_collection + replace_scope (delete-then-reinsert)
  state.py            local staleness state (scope_id -> signature)
  layout.py           local output layout (storage/state.json)
  cli.py                end-to-end entrypoint (python -m medrag.pipeline.vectorize)
  query.py              read-only search helper (python -m medrag.pipeline.vectorize.query)
  config/default.toml    tuning (batch size, collection name, staleness on/off, ...)
  tests/                  pytest tests

vectorize/              this folder: config/README + local storage/
  storage/              local state.json (the real output is the Qdrant collection, see storage/README.md)
```

## Why Qdrant + delete-then-reinsert?

chunker's chunk `node_id`s (`{doc_id}::c{n}`) are SET-SCOPED — when a
document is re-chunked the numbering is reassigned from scratch, stability
across sets is not promised. Therefore, when a scope is re-embedded, ALL its
old points in Qdrant are deleted and the new set is written from scratch —
"upsert only what changed" would produce wrong matches.

## Staleness gate

Each scope's signature (source hash + chunker version + RAPTOR
model/prompt versions + sha256 of node texts) is written to local
`storage/state.json`; a scope with the same signature is skipped on the next
run (bypass with `--force`). A deliberately different decision than
chunker's "regenerate the whole corpus every run": embedding is a real
remote LLM/GPU call (HTTP to Ollama) and thus costly, unlike chunker's
local/free recomputation. Including `text_sha256` in the signature means an
in-place edit of chunk text (e.g. OCR cleanup) automatically triggers a
re-embed without needing `--force`.

## Tests

```bash
python -m pytest src/medrag/pipeline/vectorize/tests -q
```

## See also

- Root `README.md` — cross-component rules and contract decisions.
