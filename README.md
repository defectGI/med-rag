# med-rag

Medical document RAG assistant: it ingests clinical PDFs (digital **and** scanned/photo),
DOCX, PPTX, XLSX, HTML and Markdown, converts them into a chunked vector corpus in
Qdrant, and answers questions **with mandatory source citations**. In a medical setting
the provenance of every claim matters, so each answer links back to the exact document,
page and section it came from.

The model layer is provider-agnostic (local Ollama, any OpenAI-compatible endpoint,
OpenRouter or Anthropic). External calls are made over HTTP — no model is downloaded or
run on this machine.

## Highlights

- **6 input formats** → a common intermediate representation (IR) + clean Markdown.
- **Three PDF paths**: deterministic (digital), VLM-verified (hybrid) and
  render+VLM (**scanned/photo PDF**). Images are OCR'd, tables are reconstructed from
  the PDF's own vector geometry, and the model only arbitrates low-confidence regions.
- **Source-first answers**: every factual claim carries a `doc_id + page + section`
  citation (inline `[n]` badges + a numbered source list). Missing sources are reported
  honestly, and conflicting sources are shown side by side with a warning.
- **Self-growing corpus**: upload a file → event-driven pipeline
  (`parse → chunk → vectorize`) → searchable; delete removes every trace; re-uploading a
  changed file drops the old derivation and ingests the new one. Your own notes join the
  corpus and are cited like documents.
- **Web UX**: Vite + React + TypeScript + Tailwind + shadcn/ui, Wada Sanzo ivory + sun
  palette, light/dark theme, live status badges, note module and an evidence viewer.

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

**Layer rule:** no component imports another directly; the coupling is the database and
the filesystem. The independence of `medrag.api` ↔ `medrag.pipeline` and the bottom
placement of `medrag.core` are enforced by import-linter contracts in `pyproject.toml`.

## Components

| Component | What it does | Code |
|---|---|---|
| **chatbot-corpus** | Scans the `BELGELER/` tree and produces the `document_nodes.json` registry. Catalog-less mode: with no product whitelist every document is registered. | [`chatbot-corpus/document_info/`](chatbot-corpus/document_info/README.md) |
| **parser** | 6 formats → common IR + Markdown. Three PDF paths; image OCR; table reconstruction. | [`src/medrag/pipeline/parser/`](src/medrag/pipeline/parser/README.md) |
| **pipeline/cli** | Stage orchestration + nightly run (`medrag-nightly`). | [`src/medrag/pipeline/cli/`](src/medrag/pipeline/cli/README.md) |
| **chunker** | Token-based, structure-preserving chunks carrying `doc`/`section`/`page` metadata — the basis of evidence attribution. Fully offline. | [`src/medrag/pipeline/chunker/`](src/medrag/pipeline/chunker/README.md) |
| **vectorize** | Embedding → Qdrant upsert (delete-then-reinsert + staleness gate). | [`src/medrag/pipeline/vectorize/`](src/medrag/pipeline/vectorize/README.md) |
| **api (chatbot)** | Orchestration: intent → router → flow (`default_topn` main path) → answer model. Web UI + evidence panel. | [`src/medrag/api/`](src/medrag/api/) |
| **api/retrieval** | Independent RAG retrieval layer: intent classification, top_n (Qdrant), query rewriting, raptor, db_query. | [`src/medrag/api/retrieval/`](src/medrag/api/retrieval/) · [`retrieval/ARCHITECTURE.md`](retrieval/ARCHITECTURE.md) |
| **facts (dormant)** | Evidence-based spec extraction → `specs.db` + text2sql path. Present but unused in med-rag. | [`src/medrag/pipeline/facts/`](src/medrag/pipeline/facts/) |
| **panel** | Read-only lineage/staleness panel. | [`src/medrag/api/panel/`](src/medrag/api/panel/README.md) |
| **core** | Shared infrastructure: config loader, sqlite helpers, OpenAI-compatible LLM client. | [`src/medrag/core/`](src/medrag/core/) |
| **tools/benchmark** | Scores parser Markdown against the source PDF using a VLM judge. | [`tools/benchmark/`](tools/benchmark/README.md) |
| **packages/text2sql-native** | Two-stage text2SQL engine used by facts (dormant). | [`packages/text2sql-native/`](packages/text2sql-native/README.md) |

The root-level data directories (`chatbot/`, `vectorize/`, `chunker/`, `facts/`,
`pipeline/`, `retrieval/`) hold **only data, configuration and documentation**; all code
is under `src/medrag/`.

## Installation

Python **≥ 3.11**.

```bash
uv sync --extra parse --extra serve --extra dev
# or
pip install -e ".[parse,serve,dev]"
```

Model/embedding calls are made over HTTP (local Ollama or a hosted OpenAI-compatible
endpoint); no model is downloaded or executed on this machine.

## Quick start

### Docker (recommended — one command, full stack)

```bash
cp .env.example .env          # fill in: MEDRAG_SIFRE (password) + LLM/embedding values
docker compose up -d --build  # web + pipeline-worker + pipeline + qdrant
```

Browser: `http://localhost:8507`. Details: [`DEPLOY.md`](DEPLOY.md) ·
daily usage: [`USAGE.md`](USAGE.md).

### CLI (development / single stage)

```bash
# Parse a single file (scanned PDF included)
python src/medrag/pipeline/parser/scripts/to_markdown.py document.pdf

# Stage by stage
python -m medrag.pipeline.cli.run_parse_pipeline     # registry → IR + Markdown
python -m medrag.pipeline.cli.run_chunk_pipeline     # IR → all_chunks.json
python -m medrag.pipeline.vectorize                  # chunk → embed → Qdrant

# Nightly chain
medrag-nightly                # scan → parse → chunk → ownership → facts → load → vectorize
medrag-nightly --from chunk   # start from a given stage

# Chatbot (development server)
python -m medrag.api.webapp   # http://127.0.0.1:8507
```

Corpus layout: `BELGELER/` tree (PDFs) → `document_nodes.json` → `parsed/` →
`chunks/all_chunks.json` → Qdrant. All paths are configured from the component
`.env.example` templates (see [`CONFIG.md`](CONFIG.md)).

**Critical alignment:** `EMBEDDING_MODEL` and the Qdrant collection name must match
byte-for-byte across vectorize, retrieval and the chatbot.

## Tests

The whole test suite is **offline** — no network, model or API key required:

```bash
pytest src/ tools/ -q
python -m pytest src/medrag/tests/test_import_contracts.py   # layer boundaries
```

End-to-end acceptance against a running stack:

```bash
python tools/e2e_smoke.py --base-url http://localhost:8507 --sample sample.pdf
```

## Project status

The task groups in [`PLAN.md`](PLAN.md):

- **A — Document lifecycle** (upload/delete/change/status/job queue/notes): complete.
- **B — Frontend** (Vite + React + TS + Tailwind + shadcn/ui): complete.
- **C — Evidence & prompt** (mandatory citation, conflict policy, not-found + clinician
  note, TR/EN language policy, router simplification): complete.
- **D — Design system** (Wada Sanzo ivory + sun palette, light/dark theme): complete.
- **E — Deploy & operations** (compose + volume + nightly backup/restore + logs + auth
  hardening): complete; see [`DEPLOY.md`](DEPLOY.md).
- **F — Quality & acceptance**: the offline suite runs at its documented
  baseline (a small set of pre-existing failures on dormant specs.db / text2sql /
  nightly-report paths — see [`PLAN.md`](PLAN.md)); `ruff check src tools` and the
  layer-contract tests are green; the end-to-end smoke test `tools/e2e_smoke.py`
  is ready against a running stack.

## Deploy (Docker)

For setup, backup/restore drill, observability and security notes, see
[`DEPLOY.md`](DEPLOY.md). Short version:

```bash
cp .env.example .env && docker compose up -d --build
```

Services: `web` (SPA + API, port 8507), `pipeline-worker` (job queue:
parse→chunk→vectorize), `pipeline` (exec target for the nightly run/backup,
`sleep infinity`), `qdrant`. The corpus lives in a single host directory
(`CORPUS_HOST_DIR` → `/corpus`); `web` and `pipeline-worker` read-write the corpus (the
upload API is a writer).

## Origin

This repository was adapted from an internal product-document assistant
(`doc-rag-pipeline`, package name `urun`) by copy–strip–rename. The purpose differs: here
every answer to a clinical question — disease facts, drug doses — must cite the document,
page and section it is derived from. The product-catalog, WhatsApp bridge and
structured-database path were removed; what remains is the document lifecycle,
retrieval and a citation-forced chatbot. That original work is not public.

## License

[MIT](LICENSE).
