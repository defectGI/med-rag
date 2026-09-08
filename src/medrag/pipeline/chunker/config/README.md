# config

Default configuration. Principle: no hardcoded thresholds in code — every
number that affects splitting comes from here and the user can override.

**Format decision: TOML + pydantic.** `default.toml` is read with stdlib
`tomllib` and validated by `ChunkerConfig` (a pydantic model) in
`chunker/config.py` (no new dependencies). There are NO code-side
defaults: every key must be present in `default.toml`; a missing or
unknown key makes loading fail loudly.

Override: a partial TOML containing only the keys you want to change is
deep-merged on top of the defaults via `load_config(override=...)`.

`default.toml` sections:

- `[limits]` — target chunk size and hard upper limit (tokens).
- `[flex]` / `[flex.ratios]` — **the flex allowance formula's coefficients**.
  The formula is `flex = target_tokens × ratio[kind]`,
  `effective_limit = min(target + flex, hard_max)`. The computation lives
  in `ChunkerConfig.effective_limit`; the engine only calls that.
- `[table_split]` — overlap row count (k), header-row repetition, fact
  injection; `default_header_rows` = the adapter's default when IR doesn't
  carry header info (the IR's `header_rows` field wins when present).
- `[heading_injection]` — how and how much of the heading chain is added
  to the text (structured `heading_path` metadata is always carried
  separately).
- `[images]` — `inject_ocr` (default **false**): image OCR text NEVER
  enters the chunk BODY; it's only carried in the structured channel
  (`ChunkNode.images[].ocr_text`). Tagging unverified OCR alone was tried
  and wasn't enough — even tagged, OCR doesn't separate from real
  paragraph/table text in the body and pushes retrieval toward wrong
  answers. `label_unverified_ocr`: when OCR enters the body
  (`inject_ocr = true`), tags OCR with `provenance == "unverified"`; no
  effect when `inject_ocr` is off.
- `[boilerplate]` — `drop_headings`: sections whose heading matches an
  entry here (Contents, Revision History…) are dropped entirely. Match
  uses casefold that respects Turkish İ/I (via `core/engine.py::_tr_casefold`
  — stdlib `casefold` mishandles Turkish dotted capital İ).
- `[enrichment]` — on/off flags (cross-reference resolution).

Secrets do NOT live here — they live in `.env`.