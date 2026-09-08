# storage/

The parser's runtime data. Data folders, not code.

- `raw/` — raw input files. Not deleted once processing finishes; they're kept.
- `output/` — the resulting IR output (`ParsedDocument`), as JSON. Image/table enrichment
  results are also written back into this IR, so the final result lives here.
- `images/` — the image blob store. Addressed by sha256 (`image_id`), immutable and
  deduplicated (if the same image occurs again, only one copy is kept). Filenames
  are the BARE hex + the mime's extension, while the IR field says
  `sha256:<hex>` (`:` is not a legal Windows filename character) — see
  `sha256_id`/`hash_hex` in `parsers/base.py`.
- `labels/` — the visual-type classification cache (`images/visual_classify.py`).
  One JSON file per cache key, plus `unknown.jsonl` (an append-only log of every
  crop classified as `"unknown"`, for taxonomy review). The key is
  sha256(crop bytes + prompt version), so editing the classification prompt
   invalidates the cache instead of silently keeping old labels.
  Deliberately outside `output/`'s reach: it survives an IR/PARSER_VERSION-triggered
  re-parse, so a crop already labeled is never sent to the VLM twice.
  - `labels/pages/` — a second, separate cache namespace for
    `classify_page_regions` (one whole-page VLM_CLASSIFY call locating every
    visual region on the page at once, instead of one call per candidate
    crop). Key is sha256(page PNG render + `_PAGE_REGIONS_PROMPT_VERSION`),
    same versioning discipline as above but a different key space and a
    different payload shape (a list of `{visual_type, bbox, confidence}`
    regions, not one verdict) -- kept in its own subdir so the two caches
    can never collide on one file.

Note: there is no separate database; all records/state are kept in the IR JSON
(`labels/` is the one exception -- a small, explicitly out-of-band cache). The chunk
schema also does not belong to this store; chunking is a separate component's responsibility.
