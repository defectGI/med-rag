# facts — the home of the fact (evidence-backed knowledge) layer

The component that holds ACME product/spec data as `facts/db/specs.db`
(SQLite) + `facts/db/schema.yaml`, which feeds the chatbot's text2sql
(SQL answer) path.

## Features / core capabilities

- **`specs.db`** (SQLite, committed) — canonical spec values for ACME
  products with evidence (`evidence[]`) and trust ranking
  (`document.trust_rank`). The same value arriving from a second source
  is corroboration, not conflict; a real conflict keeps the highest-
  trust source as `present` and marks the others `status='conflicting'`.
- **`schema.yaml` / `schema_model.yaml`** — schema source for the
  chatbot's text2sql phase; `schema_rag.sql`/`schema_pg.sql` are the
  DDL reference.
- **`vocabulary/SPEC_SCHEMA_CANONICAL.{json,md}`** — canonical `spec_key`
  dictionary from which `specs.db` is produced.
- **Cross-component boundary is locked by a test:** `chatbot` does NOT
  import this component's CODE; it only reads the produced files via
  PATH (see below).

## Where this component sits in the pipeline

Phase order: `parse → chunk → facts → vectorize`:

- **Previous step:** `chunker` — produces `chunker/storage/all_chunks.json`.
  The fact extraction flow reads these chunks as input.
- **This step:** writes spec values extracted from chunks into
  `facts/db/specs.db` (schema source: `facts/db/schema.yaml`).
- **Next step:** `chatbot` (text2sql / SQL answer path) reads `specs.db`
  and `schema.yaml`/`schema_model.yaml` via PATH, WITHOUT importing
  the code — the boundary is defined in
  `src/medrag/api/factory.py::_FACTS_DB_DIR` and
  `src/medrag/core/paths.py::FACTS_DB_DIR`, locked by a test
  (`src/medrag/api/tests/test_factory.py::test_chatbot_never_imports_facts_package`).
  `src/medrag/api/panel/ingest.py` also reads the same file with
  `mode=ro` for lineage/staleness tracking. The vector path fed by
  `vectorize` is a separate source, not dependent on `facts`.

## What's here now

- **`db/`** — everything the readers (`chatbot`, `src/medrag/api/panel`)
  consume: `specs.db` (SQLite, committed — `.gitignore` has the
  `!facts/db/specs.db` exception), `schema.yaml` + `schema_model.yaml`
  (the chatbot's text2sql-phase schema), `DATA_DICTIONARY.md`,
  `schema_rag.sql`/`schema_pg.sql` (DDL source — not code, reference),
  `specs.db.legacy_*` backups.
- **`ledger/`, `queue/`, `snapshots/`, `extraction_runs/`, `absent/`** —
  audit-trail data from past runs (not code). Their fate is a separate
  git-ballast cleanup decision, out of scope for this README.
- **`vocabulary/SPEC_SCHEMA_CANONICAL.{json,md}`** — canonical
  `spec_key` dictionary; the documentation record of which dictionary
  `specs.db` was produced with.

There is NO CODE here at the moment — only produced data + reference
documents. When the fact extraction code returns, this section and the
Install/Use sections below must be updated.

## Setup

This component has no dependency install of its own — the files under
`db/` come with the repo, no separate `pip`/`uv install` needed. The
reader side (`chatbot`) defines its own setup in its own README.

## Use

Until the code returns there is no CLI/script to run here. To reach
the deleted code's last state (reference/restore only):

```bash
git show facts-kodu-son-hali:facts/experiments/chunk_full_run/load_to_db.py
git checkout facts-kodu-son-hali -- facts/experiments/chunk_full_run/
```

This machine does NOT run LLM/VLM/GPU inference — when the fact
extraction flow returns (like parser/chunker) it will HTTP-call Ollama;
inference runs on a SEPARATE GPU machine / Ollama server, not here.

## Configuration

There is no component-specific `.env`/`config/default.toml` at the
moment (the old `facts/.env`, `facts/.env.example`, `facts/config/`
were deleted with the rest). The readers of `specs.db`/`schema.yaml`
(`chatbot`) keep their own settings in `src/medrag/api/.env`
(`CHATBOT_DB_QUERY_DB_PATH`, `SCHEMA_PATH`) — empty falls back to the
bundled files in this directory; see `.env.example` and
`src/medrag/api/factory.py`.

## Test

There is no component-specific test package at the moment (the old
`facts/tests/`, 27 files / 599 tests, was deleted with the rest).
The tests locking the existence of `specs.db`/`schema.yaml` and the
chatbot/facts code-import boundary live in
`src/medrag/api/tests/test_factory.py`:

```bash
pytest src/medrag/api/tests/test_factory.py -k facts
```

## Cross-references

- Why and how the old tree was deleted and what comes back ->
  `silinenler/`, `src/medrag/pipeline/facts/` README
- Contract decision (the fact layer's place in the repo, boundaries)
  -> `src/medrag/pipeline/facts/README.md`