# PANEL_ARCHITECTURE.md — Pipeline Management/Lineage Panel

The app lives in `src/medrag/api/panel/`. This document records the design
decisions and their rationale; for usage see `src/medrag/api/panel/README.md`.

## Data model decision — why a fresh DB

The repo kept no durable run-history for any stage ("when did it run, with what
input, with what output"); the designed `extraction_run` table had been removed.
So the panel builds its data model from scratch.

Three tables — a minimal model — `pipeline_runs`, `lineage_edges`, `conflicts`,
defined in full in `src/medrag/api/panel/schema.sql`. Critical design decision:
these tables live **OUTSIDE specs.db**, in a separate SQLite file
(`lineage_panel_data/lineage.db`) — the panel touches no specs.db schema and
breaks no specs.db contract.

`pipeline_runs` is **append-only**: each `ingest.py` run adds new rows and never
overwrites. This yields the desired "run-history accumulates over time" behavior
— not a "latest state only" structure like the registry (`document_nodes.json`).

## Source data — what ingest.py reads

| Stage | Source read | Staleness rule |
|---|---|---|
| parse | `document_nodes.json` → `parse.status`, `parse.last_parsed`, `parse.parsed_from_hash` | (not implemented — the parser's own re-parse gate is already in the registry) |
| chunk | `document_nodes.json` → `chunk.chunked_from_hash`, `chunk.chunked_at` | `chunked_from_hash != scan.content_hash` → stale (document changed, chunk not regenerated) |
| facts | `facts/db/specs.db` → `spec_value.source_doc_id`/`extracted_at` (per-document `MAX`) | `MAX(extracted_at) < chunk.chunked_at` → stale (facts produced before the last chunking) |
| vectorize | mtime of the most recently modified file under root `storage/` | not implemented (per-document mapping would require a live Qdrant query — deliberately out of scope, noted in the README) |

The `conflicts` table appends a snapshot of the `spec_value WHERE
status='conflicting'` rows on every ingest (append, no dedup — conflicts from
past ingests also accumulate and are filtered by `ingested_at`).

`lineage_edges`: `document→chunk_set`, `chunk_set→product` (from specs.db
`document.product_codes`), `product→family` (specs.db `product.family`),
`chunk_set→spec_value_group` (which documents facts extraction ran for),
`spec_value_group→conflict`.

**Scale decision:** each of the ~23.704 `spec_value` rows is NOT a separate graph
node — `chunk_set→spec_value_group` is a node aggregated at the document level,
and real conflicts are seen via drill-down from the `conflicts` panel. A graph
with thousands of nodes would be neither renderable nor useful.

## Technical choice — fitting the existing stack

No new dependency. The panel deliberately follows a lightweight pattern — stdlib
`http.server`, localhost-only, vanilla JS/HTML/CSS static files — as
`src/medrag/api/panel/serve.py` +
`src/medrag/api/panel/static/{index.html,app.js,style.css}`. The chatbot's Flask
was avoided because the panel is not part of the chatbot's serving path; it is an
independent local tool.

Graph rendering: no external library, raw inline SVG (`app.js::drawGraph`) —
honoring the "no unnecessary new dependency" constraint.

## Observation boundary — a strictly enforced constraint

The `ROUTES` dict in `serve.py` holds **GET-only** endpoints: `/api/stages`,
`/api/conflicts`, `/api/graph`, `/api/cascade`. None of them calls any pipeline
script via subprocess/import. Even `/api/cascade` — "if I re-run this stage, what
goes stale" — only **returns information**, triggering nothing (an in-code comment
and the UI text both say so). A real "re-run" button is intentionally absent:
write operations belong to a later phase.

## UI — how the four requirements are met

1. **Stage timeline + stale/current** → `#stages` section, color-coded cards
   (`fresh`/`stale`/`notrun`).
2. **Conflict panel, filterable** → `#conflicts`, filter by `product_code`.
3. **Re-run warning (cascade, display only)** → `#cascade` section,
   `/api/cascade`.
4. **Dependency/lineage graph + navigation** → `#graph` section, search by
   node_type+node_id, showing the queried node's upstream and downstream edges.
   **Limitation:** node-to-node click-through navigation is NOT in this version —
   the user types a node_id and searches again. A fully interactive
   click-to-navigate UI did not fit the scope given the time/complexity
   trade-off; noted as a next improvement step.

## Validation

`ingest.py` was run against real repo data: 772 documents, 198 chunk sets,
~23.704 `spec_value` rows, 24 conflicts → 1551 run records, 1989 edges, 0 stale
stages (expected for the current consistent state). `serve.py`'s four API
endpoints were exercised with `curl`, and the static page returned 200. No visual
verification was done in a real browser session (browser automation was not used
here) — this remains an open risk.
