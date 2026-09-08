# Lineage Panel

Observability layer for the pipeline (parser → chunker → facts → vectorize →
chatbot). **Read-only/observation** — it triggers, stops, or changes no stage;
it only reads existing artifacts and summarizes them into its own separate
observation database (`lineage.db`, not committed to git). For scope and design
rationale see `PANEL_ARCHITECTURE.md` (repo root).

## Why it exists

The repo kept no durable record of "when each stage ran, with what input, with
what output" (the designed `extraction_run` table had been removed). This panel
fills that gap OUTSIDE `specs.db`, in a separate SQLite file — it does not touch
the specs.db schema.

## Usage

```
python ingest.py    # reads document_nodes.json + chunker/storage/all_chunks.json
                    # + facts/db/specs.db + root storage/ mtime,
                    # APPENDS new rows to lineage.db (append-only)
python serve.py     # http://localhost:8010/ — localhost only, GET-only API
```

Data freshness is not the panel's own responsibility — the panel only shows what
the last `ingest.py` run saw. For periodic freshness, re-run `ingest.py` manually
or via a scheduled job (there is intentionally NO automatic scheduler in this repo
right now — this is the observation phase; automation is a later phase).

## Data model

`schema.sql` — three tables: `pipeline_runs` (append-only run-history, each
`ingest.py` run adds new rows), `lineage_edges` (general source→target edge
table: document→chunk_set, chunk_set→product, product→family,
chunk_set→spec_value_group, spec_value_group→conflict), `conflicts` (a snapshot
of the `status='conflicting'` rows in specs.db).

## Out of scope (deliberately)

- Triggering/stopping the pipeline (write operations) — a later phase.
- Querying Qdrant live for the vectorize stage (instead of adding a
  qdrant-client dependency, the file mtime of the root `storage/` is read — a
  crude but sufficient "last touched" signal, honoring the "no new dependency"
  constraint).
- Drawing thousands of `spec_value` rows as graph nodes — instead, nodes are
  aggregated at the document/chunk_set/product/family level with drill-down via
  the conflict panel.
