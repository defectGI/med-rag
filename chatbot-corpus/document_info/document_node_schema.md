# Document Node — Metadata Schema (v2)

> **v2:** `doc_id`'s "assigned once" principle is now actually enforced
> in code (`classify_documents.py` merges on re-scan, preserving the
> identity + `parse`/`chunk` state; previously every scan regenerated
> all UUIDs, so the principle was described here but the code did not
> implement it). The unowned `vector` block was removed and replaced
> with the actually-written `chunk` block. All timestamps are
> offset-local ISO-8601; all hash fields are `sha256:<hex>`.

Scope: outward-facing documents that belong to products (brochure,
catalogue, datasheet, technical drawing, CAD/STP, CE declaration, user
manual, etc.). Component / BOM files (RoHS certificates, part images)
are deliberately OUT OF SCOPE — see DROP.

Reference: `<internal-share>/document_registry.json` (the manager's
working record system). This schema takes that one's pipeline/status
tracking, drops its redundant fields, and closes two critical
durability issues (trusting `mtime` + path-derived id).

---

## Design principles (why it is shaped this way)

1. **Identity ≠ location ≠ content.** `doc_id` is assigned once on
   first sight and carried through moves/edits. Location is tracked
   via `rel_path`, content via `content_hash`, separately. A path-
   derived sha1 changes `doc_id` when the file moves to another
   folder while the document tree is still under construction — every
   Qdrant vector tied to that document silently becomes orphaned. This
   schema prevents that structurally.
2. **Don't trust `mtime`.** The engine of change detection is
   `content_hash` (sha256). `mtime` is only a cheap pre-filter.
   Copying / `touch` / metadata edits lie.
3. **Change = one enum.** Instead of four separate booleans, use
   `scan_status`. Move detection comes for free (hash present, path
   missing → MOVED, same `doc_id`).
4. **Staleness self-validates.** `content_hash != parsed_from_hash` →
   parse stale. `content_hash != chunked_from_hash` → chunk stale.
   Crashes / half-runs / manual edits are all caught by this — better
   than a bare `was_modified` boolean. In addition to the hash, the
   **code version** is also a gate: `parse.parser_version !=
   PARSER_VERSION` (same bytes, old parser → re-parse), and on the
   chunk side the chunk file's own `provenance`.
5. **Keep derivable fields out.** `linked_doc_types` / `linked_scopes`
   are dropped: scope derives from the level of the node
   `owner_ids` points to; `doc_type` is already in the `doc_type`
   field. `exists` is dropped: it is derived from
   `scan_status == DELETED`.

---

## Fields (grouped)

The structure is not a flat single object but grouped into
`identity/location/scan/type/curation/links/parse/vector`. Document
counts at ACME are in the hundreds — there is no flat-filtering
advantage, but grouping gains readability: the "what's this document's
processing status?" query reduces to reading one group.

| Group | Field | Note |
|---|---|---|
| **identity** | `doc_id` | Assigned once; carried through moves/edits. |
| | `file_name` | Original name. |
| | `extension` | `.pdf` `.docx` `.xlsx` `.stp` `.png` etc. |
| **location** | `rel_path` | Canonical, mount-independent path. Matching is done against this. |
| | `file_path` | Derived: `mount_root` + `rel_path`. `mount_root` is kept in one place at registry level and is not repeated per record. |
| **scan** | `content_hash` | sha256(bytes). The engine of change detection. |
| | `size_bytes` | Cheap pre-filter before hashing (if `(size, mtime)` unchanged, skip hashing). |
| | `last_modified_time` | Same — pre-filter only, not a decision on its own. |
| | `scan_status` | `NEW` / `MODIFIED` / `MOVED` / `UNCHANGED` / `DELETED`. What happened in the last scan. DELETED is sticky (not deleted, retained). |
| **type** | `doc_type` | `BROCHURE` / `CATALOGUE` / `DATASHEET` / `CE_DECLARATION` / `TECHNICAL_DRAWING` / `USER_MANUAL` / `QUICK_START_GUIDE` / `STP` / `PRODUCT_IMAGE` |
| **curation** | `is_active` | Editorial decision (obsolete, don't show). Independent axis from `scan_status` — the file may exist but you can still say "don't show it". |
| **links** | `owner_ids` | Related node IDs — **the matched_node_id itself + all descendant nodes in the tree** (intermediate categories INCLUDED, leaf products INCLUDED). When a document resolves to a category (family/subfamily/subfamily_2), it is deterministically fan-out'd to that category itself + all its subfamily/subfamily_2 nodes + all leaf products: "a document belonging to a family also belongs to all its sub-families and all its products". Many-to-many. (Previously it was only at leaf-product level; category nodes were excluded.) |
| | `link_count` | Length of `owner_ids`. Derived but cached for fast filtering. |
| | `matched_node_id` | The single raw matched node (may be a category) — kept for traceability; now also one element of `owner_ids`. Equals the single element of `owner_ids` if no fan-out happened. Nullable (no match). |
| | `resolution_method` | `code_match` / `folder_name_match` (+ `_expanded` suffix when fan-out really expanded, kept consistent IN CODE — not delegated to the LLM) / `null` (unresolved). |
| **parse** | `parser` | Parser script used (`parser_datasheet.py`, etc.). |
| | `parser_version` | The `PARSER_VERSION` that produced this parse (single constant in `parser/parsers/base.py`). When parser logic changes, a re-parse is triggered: same bytes + old version → re-parse. |
| | `status` | `PENDING` / `SUCCESS` / `PARTIAL` / `FAILED` / `SKIPPED`. Overall summary, fast filter. Detail in `stages`. |
| | `parsed_from_hash` | The `content_hash` the parse ran against. Staleness check. |
| | `parsed_json_path` | Output JSON path. |
| | `last_parsed` | Last parse time. |
| | `error` | nullable; overall parse error. |
| | `stages` | Three-stage LLM pipeline record — see below. |
| | `models` | nullable; `{provider, model, num_ctx, thinking}` for each `LLM`/`VLM`/`VLM2` role that produced this parse — so later (without opening the IR JSON) you can see which model + which context window produced it. Different from the per-stage `model` string in `stages`: here the full config (context/thinking included) in one object; the same snapshot is also written to the IR's own `metadata.models` (`llm.describe_configured_models()`). |
| **chunk** | `status` | `PENDING` / `SUCCESS` / `FAILED`. Written by `run_chunk_pipeline.py`. |
| | `chunked_at` | nullable; chunk file's `provenance.generated_at`. |
| | `chunked_from_hash` | The `content_hash` the chunk set ran against (copied from the chunk file's `provenance.source.raw_sha256`, NOT re-derived — registry and file can never disagree). |
| | `chunks_path` | `{doc_id}.chunks.json` path. |
| | `chunker_version` | Chunker version that produced the chunks. |
| **facts** | `status` | `PENDING` / `SUCCESS` / `SKIPPED` / `FAILED`. Written by `run_facts_pipeline.py`. `SKIPPED`: out-of-scope doc_types (`PRODUCT_IMAGE`/`STP`) or a statement/observation mismatch (`ScopeMismatch`). |
| | `extracted_at` | nullable; last run time of pass 1 on this document. |
| | `extracted_from_hash` | The `content_hash` this run was bound to (staleness check, same pattern as `chunked_from_hash`). |
| | `extractor_version` | Pass 1 `Extractor.name` (e.g. `"fake"` / `"ollama:qwen2.5:32b"`). |
| | `prompt_version` | Pass 1 `Extractor.prompt_version`. |

There is NO single output-file-path field in the `facts` block like
the one in `chunk`/`vector`: the output of pass 1 is not a single file
but ledger rows split per product (`facts/ledger/{model}.jsonl`) —
"has this document been processed?" is answered by `status` /
`extracted_from_hash`, "which facts came out?" is answered by querying
the ledger itself (per model).

There is NO `vector` block: no code was filling it, and the empty
placeholder creates a "looks filled" risk. When the embedding stage
arrives in this repo it will add its own block in the same pattern
(status + `*_at` + `*_from_hash` + producer version).

---

## `parse.stages` — three-stage LLM pipeline

The parser uses an LLM, but a single model is not enough: three
separate models, three separate roles. They run sequentially and
gated. Each stage records its own `model` / `status` / `at` / `error`
so it's visible which stage ran which model and which one blew up.

Order: `ocr → ocr_check → table_desc`

| Stage | Role | Critical? |
|---|---|---|
| `ocr` | Extracts raw text from image/scanned regions. | **Yes** — if it fails there is no usable text. |
| `ocr_check` | Validates / corrects the OCR output (hallucination, error catch). | No. |
| `table_desc` | Writes the table description **and** validates it. | No. |

**Overall `parse.status` derivation rule:**
- All SUCCESS → `SUCCESS`
- `ocr` FAILED → `FAILED` (subsequent stages `SKIPPED`)
- Non-critical stage FAILED/SKIPPED but `ocr` succeeded → `PARTIAL`
- Never started → `PENDING`
- Deliberately not parsed (e.g. STP/CAD binary, nothing to extract) → `SKIPPED`

**Note:** There is no per-stage `from_hash`. All three stages run
against the same document `content_hash`; a single
`parse.parsed_from_hash` catches staleness. Because OCR is expensive,
an intermediate cache (store OCR output and only re-run the
downstream) is v2 work and is not added now.

---

## Example record (PN1127 datasheet)

```json
{
  "identity": {
    "doc_id": "d1f4a7b0-...",
    "file_name": "ACME_PN5029_Datasheet.pdf",
    "extension": ".pdf"
  },
  "location": {
    "rel_path": "AVIONICS INTERFACES/AVIOLINKS ETH-USB/ACME_PN5029_Datasheet.pdf",
    "file_path": "<mount_root>/BELGELER/AVIONICS INTERFACES/AVIOLINKS ETH-USB/ACME_PN5029_Datasheet.pdf"
  },
  "scan": {
    "content_hash": "sha256:9c2e...",
    "size_bytes": 482113,
    "last_modified_time": "2025-09-26T09:06:23+03:00",
    "scan_status": "UNCHANGED"
  },
  "doc_type": "DATASHEET",
  "is_active": true,
  "links": {
    "owner_ids": ["n_5d0906035152", "n_057b7cff529a", "n_e3d59d3b2bcb", "…12 leaf products…"],
    "link_count": 13,
    "matched_node_id": "n_5d0906035152",
    "resolution_method": "code_match_expanded"
  },
  "parse": {
    "parser": "parser_datasheet.py",
    "parser_version": "1.4.0",
    "status": "SUCCESS",
    "parsed_from_hash": "sha256:9c2e...",
    "parsed_json_path": ".../parsed_datasheet/ACME_PN5029_Datasheet.json",
    "last_parsed": "2026-01-19T22:02:53+03:00",
    "error": null,
    "stages": {
      "ocr":        { "model": "paddleocr-vl-0.9b", "status": "SUCCESS", "at": "2026-01-19T22:01:40+03:00", "error": null },
      "ocr_check":  { "model": "qwen3-32b-fp8",     "status": "SUCCESS", "at": "2026-01-19T22:02:10+03:00", "error": null },
      "table_desc": { "model": "qwen3-32b-fp8",     "status": "SUCCESS", "at": "2026-01-19T22:02:53+03:00", "error": null }
    },
    "models": {
      "LLM": { "provider": "ollama", "model": "qwen3-32b-fp8", "num_ctx": 16384, "thinking": false },
      "VLM": { "provider": "ollama", "model": "paddleocr-vl-0.9b", "num_ctx": 16384, "thinking": false }
    }
  },
  "chunk": {
    "status": "SUCCESS",
    "chunked_at": "2026-01-19T22:10:04+03:00",
    "chunked_from_hash": "sha256:9c2e...",
    "chunks_path": ".../chunks/d1f4a7b0-....chunks.json",
    "chunker_version": "1.0.0"
  }
}
```

*(Model names are placeholders — fill with your own stack.)*

---

## DROP — not included in this schema

- Component / part files (RoHS certificates, part images with `H...` /
  `DM...` / `DT...` codes) — belong to the BOM, not the product.
- Internal compliance-tracking spreadsheets like
  `Teknik Dosyalar.xlsx` — not a file, but process tracking.

## Dropped from the manager's schema

- `linked_doc_types` — `doc_type` repeated inside an array. Redundant.
- `linked_scopes` — scope derives from the node `owner_ids` points to.
  A separate field is unnecessary.
- `exists` (along with its four booleans) — reduced to the
  `scan_status` enum.

## Changed / added

- `doc_id`: path-derived sha1 → persistent, once-assigned identity.
- `content_hash`, `size_bytes`: added (change detection + pre-filter).
- `scan_status`: four booleans → single enum (+ free move-detection).
- `parse`: single `llm_model` → three-stage `stages` + `parsed_from_hash`
  + `parse_status` enum + `error`.
- `chunk` (v2): record of the chunk stage — `chunked_from_hash` +
  `status` enum + producer version.
- `is_active`: kept (editorial axis).

---

## Next step — not yet decided

1. **`owner_ids` resolution: DECIDED.** Two-tier resolution (tier 1
   code match, tier 2 folder name) → single `matched_node_id`; then
   deterministic tree fan-out fills `owner_ids` with `matched_node_id`
   itself + all descendant nodes (intermediate categories included,
   leaf products included). The LLM expansion step was retired
   (non-determinism removed).
2. **Future candidates (not now):** OCR intermediate cache (separate
   hash per stage), `chunk_count`, version chain
   (`supersedes`/`superseded_by`). For now `is_active` handles
   versioning — to avoid bloating the schema.
3. **Embedding / vector block:** added when that stage arrives in this
   repo (no unowned placeholders are kept).