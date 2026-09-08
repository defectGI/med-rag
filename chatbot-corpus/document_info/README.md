# classify_documents.py

Scans the customer-facing documents under `BELGELER` (brochure, catalogue,
datasheet, technical drawing, CE declaration, user manual, etc.),
classifies them according to the format in `document_node_schema.md`, and
writes `document_nodes.json`.

`owner_ids` is resolved in two tiers: first a code in the file path
(`DE\d+` or similar), validated against the whitelist in
`product_nodes.json`; if no code is found, the folder name is compared
against family/subfamily/subfamily_2 text. If neither matches, the file
lands in the "unresolved" list.

## Setup

```
pip install -r requirements.txt
```

## Settings

All paths are read from `.env` — no path is hard-coded.

1. Copy `.env.example` to `.env`.
2. Fill in the real paths for your machine:

```
# Root of the product_info folder - portable. The product_nodes.json path
# and BELGELER structure underneath are assumed fixed and not repeated here.
PRODUCT_INFO_DIR=C:\Users\...\chatbot-corpus\product_info

# Full path of the document_nodes.json file to produce
OUTPUT_PATH=C:\Users\...\chatbot-corpus\document_info\document_nodes.json
```

> Note: this script requires `product_info/export_products.py` to have
> been run first so that `product_nodes.json` exists.

## Running

```
python classify_documents.py
```

The script is re-runnable: if the output file already exists, it is not
rebuilt — it is MERGED into. Each record's identity (`doc_id`) and
editorial/pipeline state (`is_active`, `parse`, `chunk`) are preserved;
`scan_status` (NEW/UNCHANGED/MODIFIED/MOVED) is derived by matching
against the previous scan; records present in the previous scan but
missing from this one are not deleted, they stay sticky with
`scan_status=DELETED`. No separate sync script is needed.

## Output shape

```json
{
  "generated_at": "...",
  "summary": { "scanned_files": ..., "resolved": ..., "unresolved": ..., "unresolved_files": [...] },
  "documents": [ { ... }, ... ]
}
```

See `document_node_schema.md` for the field groups
(`identity/location/scan/type/curation/links/parse/chunk`) and their
rationale.
