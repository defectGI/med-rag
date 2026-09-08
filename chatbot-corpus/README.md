# chatbot-corpus

Houses two independent deterministic pipelines that convert the ACME
product and document data into the two node graphs
(`product_nodes.json`, `document_nodes.json`) that are the foundation of
the chatbot / RAG corpus.

## Overview

This component sits at the very start of the pipeline — it has no
upstream dependencies; its inputs are the raw sources directly (the
Excel catalogue + the `BELGELER/` file tree). It makes NO model/LLM/VLM
calls; both pipelines are pure deterministic Excel-reading /
file-scanning + matching logic in Python.

```
ACME PRODUCTS & SOLUTIONS.xlsx ──export_products.py──► product_nodes.json
                                                              │
BELGELER/ (customer-facing files) ──classify_documents.py──► document_nodes.json
                                        (linked via owner_ids to product_nodes.json)
```

The two JSONs produced are the input of the `parser`/`pipeline` chain
(see the root `README.md`) and of the `facts/` layer — without these
nodes, there is no way to know which document belongs to which product.
The read-only front-end tree visualiser is no longer here: `viz/` and
`spec_schema/` have moved under `tools/viz/` and `tools/spec_schema/`;
anyone wanting to read these two JSONs for visualisation / atom
extraction should look there.

## Folders

- **`product_info/`** — converts `ACME PRODUCTS & SOLUTIONS.xlsx`
  (sheet `ACME PRODUCTS`) into a hierarchical `product_nodes.json`
  (`export_products.py`). Column mapping is done by header name, the
  hierarchy is built from the path in the `No` column, and `node_id` is
  stable independently of row order. For details and field definitions,
  see `product_info/README.md` + `product_node_schema.md`.
- **`document_info/`** — scans the customer-facing documents under
  `BELGELER/` (brochure, catalogue, datasheet, CE declaration, user
  manual, etc.) and writes `document_nodes.json` while linking them to
  the nodes in `product_nodes.json` (`classify_documents.py`). For
  details, see `document_info/README.md` + `document_node_schema.md`.
- **`BELGELER/`** — the source tree holding product images, technical
  files and customer-facing documents (not code; data).

## Features

- **Deterministic, no model calls** — both are fully rule/matching-based;
  network/LLM dependency is zero.
- **Re-runnable (rescan-merge)** — `classify_documents.py` does not
  rebuild the output file if it already exists; it merges into it: the
  `doc_id` and the editorial/pipeline state (`is_active`, `parse`,
  `chunk`) are preserved, and `scan_status` (NEW / UNCHANGED / MODIFIED /
  MOVED / DELETED-sticky) is derived by comparing against the previous
  scan. No separate sync script is needed.
- **Two-tier owner resolution** — a product code from the file path
  (`DE\d+` or wildcard); if none is found, the folder name is matched
  against family/subfamily/subfamily_2 text; if nothing matches, the
  file is not silently dropped — it lands in the "unresolved" list.
- **Category → subtree fan-out** — if a document matches a category node
  (family/subfamily/subfamily_2), `owner_ids` is deterministically
  expanded to that node plus every descendant in the tree (including
  intermediate categories and leaf products).
- **Scanning/classification tuning is NOT hard-coded** — which files to
  skip, `doc_type` keywords, the code pattern, extensions come from
  `document_info/config/default.toml`.
- **No silently-swallowed data errors** — `export_products.py` prints a
  separate `UYARI:` line for data errors such as colliding `node_id`.

## Setup

There is no `requirements.txt` of its own — dependencies (e.g.
`openpyxl`, `python-dotenv`, `pydantic`) are managed project-wide via
the root `pyproject.toml`/`uv.lock`; no separate optional group is
needed (they're in the base dependencies):

```
uv sync
```

The scripts can then be run via `uv run python ...` or by activating the
`.venv` that `uv sync` creates and using plain `python ...`; both use
the same dependency set.

## Configuration

No secrets / real paths are written here. Each of the two subfolders has
its own `.env` / `.env.example`; the settings are independent:

- `product_info/.env.example` — `EXCEL_PATH`, `SHEET_NAME`,
  `OUTPUT_PATH`.
- `document_info/.env.example` — `PRODUCT_INFO_DIR`, `BELGELER_DIR`,
  `OUTPUT_PATH`, optional `DOCUMENT_INFO_CONFIG` (partial TOML
  override).

Scanning/classification tuning knobs (exclude lists, `doc_type`
keywords, code pattern, extensions) are NOT in `.env`; they live in
`document_info/config/default.toml`.

## Usage

`document_info` depends on `product_nodes.json` produced by
`product_info` — run `product_info` first:

```
cd chatbot-corpus/product_info && python export_products.py
cd ../document_info && python classify_documents.py
```

Neither takes CLI arguments — all settings are read from `.env` /
`config/default.toml` (see "Configuration"). Both are re-runnable;
`classify_documents.py` does not reset the existing
`document_nodes.json`, it merges into it (see "Features" above).

## Tests

```
pytest chatbot-corpus/product_info/tests
pytest chatbot-corpus/document_info/tests
```

Both are offline (no Excel/network/model calls).

## For more information

- `product_info/product_node_schema.md`,
  `document_info/document_node_schema.md` — complete field definitions
  and rationale.
