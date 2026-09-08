# adapters

Parser-specific input transformers. An adapter's only job is to convert a
parser's output into `core/`'s shared inner document model. Every
parser-specific fact (schema, field names, version migration, format
quirks) stays HERE — it does not leak into the engine.

First adapter (reference implementation) — **`first_parse.py`**:

- **first_parse** — `ParsedDocument` IR v9; the source contract is
  `medrag.pipeline.parser.parsers.base` (a real package now; the earlier
  vendor copy was removed — staying in sync by hand was unsustainable).
  The version gate + `Cell.plain_text()` live in one place, the
  dependency lives only here. Field mapping:
  - `heading_path` → carried per block (None → empty list)
  - `description` → `Table.description`/`Image.description` (same shared
    field, IR v9 — see parser IR's `DescribableBlock`: a table is not
    a separate branch of this category, just the more deterministic
    member); `facts` → `Table.facts`; `excluded_at_parse` carried
    verbatim for both
  - `list_id` / `list_level` / `list_ordered` → contiguous groups
    become `ListBlock` (an interruption closes the list; the same id
    reopening produces `"L1#2"`); paragraph/heading open a new item,
    table/image/code attach to the previous item
  - `InlineRun.link` → `LinkRef` (consecutive same-target runs merge);
    text stays canonical, marks are NOT rendered
  - `Span.page` / `page_start` / `page_end` → normalized
    `page_start`/`page_end`
  - `Cell.plain_text()` → flat cell text (merge slot = empty string)
  - provenance labels → core enum (unknown label → `AdapterError`)
  - `header_rows`: IR field when present (forward-compatible; the parser
    side agreed to add it), otherwise `[table_split] default_header_rows`
    from config
  - `Heading.anchor_id`: IR field when present (HTML/markdown already
    populated; docx/pdf/pptx still None), otherwise None — resolution
    is `enrichment/cross_ref.py`'s job, no heuristic in the adapter
  - OCR text marked `ocr_meaningful is False` is NOT carried

Adding a new parser = adding a new adapter file under this folder. The
engine, strategies, and config are untouched.