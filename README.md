# med-rag

Ask questions about your medical documents and get answers with citations.

med-rag reads PDFs (digital and scanned/photo), DOCX, PPTX, XLSX, HTML and
Markdown, stores them as vectors in Qdrant, and answers questions that point to
the document, page and section they came from.

Models are not run here. med-rag talks to them over HTTP: local Ollama, any
OpenAI-compatible endpoint, OpenRouter, or Anthropic. It never downloads a model.

## What it does

- **Reads 6 formats** into one internal format (IR) plus clean Markdown.
- **Handles scanned PDFs.** It OCRs images and rebuilds tables from the page
  geometry; the model only judges the low-confidence parts.
- **Cites every claim.** Answers carry `doc_id + page + section`. If no source
  is found it says so; if two sources disagree you see both, with a warning.
- **Grows with your files.** Upload → parse → chunk → vectorize. Delete removes
  every trace. Re-uploading a changed file replaces the old version. Your notes
  join the corpus and are cited too.
- **Web UI.** React (Vite + Tailwind + shadcn/ui), light/dark theme, status
  badges, notes, and an evidence viewer.

## Architecture

```
 chatbot-corpus/                        src/medrag/pipeline/
 ──────────────                         ──────────────────
 BELGELER/ tree  ──► document_nodes.json (registry, catalog-less scan)
                                       │
                             ┌─────────▼─────────┐
                             │  cli: parse        │  parser: file → IR JSON + Markdown
                             │                    │  (digital / hybrid / scanned paths;
                             │                    │   image OCR, table structure + heading)
                             └─────────┬─────────┘
                             ┌─────────▼─────────┐
                             │  cli: chunk        │  chunker: IR → token-based,
                             │  all_chunks.json   │  structure-preserving (LLM-free)
                             └─────────┬─────────┘
                                       │
                             ┌─────────▼─────────┐
                             │  vectorize         │  chunk → embed → Qdrant upsert
                             └─────────┬─────────┘
                                       │  vector path
                             ┌─────────▼─────────┐
                             │  src/medrag/api/   │  intent → deterministic router →
                             │  chatbot (web UI)  │  flow → answer model (evidence panel,
                             │                    │  SSE trace)
                             └───────────────────┘
```

**Layer rule:** no component imports another. They meet only through the
database and the filesystem. `medrag.api` and `medrag.pipeline` stay separate,
and `medrag.core` sits at the bottom. `pyproject.toml` enforces this with
import-linter.

## Components

| Component | What it does | Code |
|---|---|---|
| **chatbot-corpus** | Scans `BELGELER/` and builds `document_nodes.json`. | [`chatbot-corpus/document_info/`](chatbot-corpus/document_info/README.md) |
| **parser** | 6 formats → IR + Markdown. Three PDF paths; OCR; tables. | [`src/medrag/pipeline/parser/`](src/medrag/pipeline/parser/README.md) |
| **pipeline/cli** | Runs stages and the nightly run (`medrag-nightly`). | [`src/medrag/pipeline/cli/`](src/medrag/pipeline/cli/README.md) |
| **chunker** | Splits IR into chunks with `doc`/`section`/`page` metadata. Offline. | [`src/medrag/pipeline/chunker/`](src/medrag/pipeline/chunker/README.md) |
| **vectorize** | Embeds chunks and writes them to Qdrant. | [`src/medrag/pipeline/vectorize/`](src/medrag/pipeline/vectorize/README.md) |
| **api (chatbot)** | Intent → router → flow → answer. Web UI + evidence panel. | [`src/medrag/api/`](src/medrag/api/) |
| **api/retrieval** | RAG retrieval: intent classification, top_n, query rewriting, raptor, db_query. | [`src/medrag/api/retrieval/`](src/medrag/api/retrieval/) · [`retrieval/ARCHITECTURE.md`](retrieval/ARCHITECTURE.md) |
| **facts (dormant)** | Extracts facts into `specs.db` + text2sql. Not used here. | [`src/medrag/pipeline/facts/`](src/medrag/pipeline/facts/) |
| **panel** | Read-only lineage/staleness panel. | [`src/medrag/api/panel/`](src/medrag/api/panel/README.md) |
| **core** | Shared config loader, sqlite helpers, LLM client. | [`src/medrag/core/`](src/medrag/core/) |
| **tools/benchmark** | Scores parser output against the source PDF with a VLM judge. | [`tools/benchmark/`](tools/benchmark/README.md) |
| **packages/text2sql-native** | Two-stage text2SQL engine used by facts (dormant). | [`packages/text2sql-native/`](packages/text2sql-native/README.md) |

The root folders `chatbot/`, `vectorize/`, `chunker/`, `facts/`, `pipeline/` and
`retrieval/` hold only data, config and docs. All code is in `src/medrag/`.

## Install

Python **≥ 3.11**.

```bash
uv sync --extra parse --extra serve --extra dev
# or
pip install -e ".[parse,serve,dev]"
```

## Quick start

### Docker (recommended)

```bash
cp .env.example .env          # set MEDRAG_SIFRE (password) + LLM/embedding values
docker compose up -d --build  # web + pipeline-worker + pipeline + qdrant
```

Open `http://localhost:8507`. Setup: [`DEPLOY.md`](DEPLOY.md) · usage:
[`USAGE.md`](USAGE.md).

### CLI (development)

```bash
# Parse one file, scanned PDFs included
python src/medrag/pipeline/parser/scripts/to_markdown.py document.pdf

# Stage by stage
python -m medrag.pipeline.cli.run_parse_pipeline     # registry → IR + Markdown
python -m medrag.pipeline.cli.run_chunk_pipeline     # IR → all_chunks.json
python -m medrag.pipeline.vectorize                  # chunk → embed → Qdrant

# Nightly chain
medrag-nightly                # scan → parse → chunk → ownership → facts → load → vectorize
medrag-nightly --from chunk   # start from a given stage

# Dev server
python -m medrag.api.webapp   # http://127.0.0.1:8507
```

Corpus layout: `BELGELER/` (PDFs) → `document_nodes.json` → `parsed/` →
`chunks/all_chunks.json` → Qdrant. Each `.env.example` template sets its own
paths (see [`CONFIG.md`](CONFIG.md)).

**Keep these in sync:** `EMBEDDING_MODEL` and the Qdrant collection name must be
the same in vectorize, retrieval and the chatbot.

## Tests

The suite is **offline** — no network, model or API key needed:

```bash
pytest src/ tools/ -q
python -m pytest src/medrag/tests/test_import_contracts.py   # layer boundaries
```

End-to-end test against a running stack:

```bash
python tools/e2e_smoke.py --base-url http://localhost:8507 --sample sample.pdf
```

## Status

Task groups in [`PLAN.md`](PLAN.md):

- **A — Document lifecycle** (upload/delete/change/status/queue/notes): done.
- **B — Frontend** (Vite + React + TS + Tailwind + shadcn/ui): done.
- **C — Evidence & prompts** (mandatory citation, conflict policy, not-found +
  clinician note, TR/EN language, router): done.
- **D — Design system** (Wada Sanzo ivory + sun, light/dark): done.
- **E — Deploy & ops** (compose + volumes + nightly backup/restore + logs + auth):
  done. See [`DEPLOY.md`](DEPLOY.md).
- **F — Quality**: the offline suite runs at its documented baseline; a few
  dormant-path tests (specs.db / text2sql / nightly-report) fail on a clean
  checkout — see [`PLAN.md`](PLAN.md). `ruff check src tools` and the
  layer-contract tests are green. The smoke test is ready.

## Deploy

Setup, backup/restore, observability and security live in [`DEPLOY.md`](DEPLOY.md).
Short version:

```bash
cp .env.example .env && docker compose up -d --build
```

Services: `web` (SPA + API, port 8507), `pipeline-worker` (job queue:
parse→chunk→vectorize), `pipeline` (idle exec target for the nightly run),
`qdrant`. The corpus sits in one host directory (`CORPUS_HOST_DIR` → `/corpus`).
`web` and `pipeline-worker` read and write it; the upload API is the writer.

## Origin

This started as a copy of an internal product-document assistant
(`doc-rag-pipeline`, package name `urun`), renamed for a different job: here
every clinical answer must cite the document, page and section it comes from.
The product catalog, WhatsApp bridge and structured-DB path were removed. What
is left is the document lifecycle, retrieval, and a chatbot that must cite.
That original project is not public.

## License

[MIT](LICENSE).
