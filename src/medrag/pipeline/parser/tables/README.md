# tables/

Adds structure and a description to the table blocks (full JSON, including merges) produced by
the format parsers.

A table is not a category of its own here — it is the same kind of thing a chart or a block
diagram is (a "describable block"), only more deterministic (real cell data instead of pixels).
The actual description generation now lives in `describe/` (repo root of `parser/`), shared by
every describable type; this directory keeps only what's genuinely table-specific:

- **table strategy** (`describe/table.py`, not in this directory — see there for the FORMAT_SPEC,
  the LLM check/retry loop, and the digit-checked `facts` this strategy alone produces) fills
  the shared `description`/`facts`/`describe_status`/`describe_attempts` fields
  (`parsers/base.py`'s `DescribableBlock`) that `TableBlock` and `ImageBlock` both carry under
  the same names. Tuning: `[table]` for what's genuinely table-only (`max_rows`, `llm_check`,
  `facts`, `header_llm`, `check_retries`); `[describe]`/`[describe.context]` for what every
  describable type shares (concurrency, surrounding-context injection). `[table]`'s own knobs
  keep their historical env names (`TABLE_LLM_CHECK`, `TABLE_CHECK_RETRIES`, ...); `[describe]`'s
  are `DESCRIBE_*` (the old `TABLE_CONTEXT*`/`TABLE_CONCURRENCY` names were
  retired, not aliased — see `config.py`'s `_ENV_OVERRIDES`).

  LLM access is configured via env (the `llm/` layer): `LLM_PROVIDER` (`openai` |
  `anthropic` | `openrouter` | `ollama` | `local`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`,
  `LLM_THINKING_ON`. The last one matters here specifically: hybrid-reasoning models (Qwen3/3.5,
  gpt-oss, ...) default to thinking ON in both Ollama and OpenRouter, which can consume the
  entire `max_tokens` budget on hidden reasoning and leave the visible description empty —
  `LLM_THINKING_ON` defaults to off to avoid that; see the env docs in `llm/__init__.py` for
  details.

- `header_infer.py` — LLM *fallback* for `TableData.header_rows` (how many leading rows are
  header rows, IR v6). Parsers fill it deterministically where the source states it (html
  `<th>`/`<thead>`, markdown's pipe header row, docx `w:tblHeader`, pptx `firstRow`, the pdf
  band builder's recovered header); this pass only runs for tables still at `None`, asks the
  LLM a single constrained question ("how many leading rows are headers? answer one integer"),
  validates the answer against the table's shape (`0 <= n < n_rows`, capped at 3), and marks an
  accepted value with a `"header-llm"` entry in `table_flags` so consumers can tell inference
  from source fact. Any failure leaves `None` — never a wrong claim. `TABLE_HEADER_LLM=0`
  disables the pass (default on); shares `[describe.context]`/`[describe] concurrency` (env:
  `DESCRIBE_CONTEXT*`/`DESCRIBE_CONCURRENCY`) with the table describe strategy.

Note: a table block is atomic (not split) at the chunking stage — that rule belongs to the
chunker; this repo is only responsible for structuring and describing the table.

## `structure/` — optional model referee for the PDF table grid

Building a table's row/column grid (as opposed to describing it) is the PDF parser's job, not
this module's: `parsers/pdf_parser.py` locates table regions and rebuilds their grid
deterministically from the page's own geometry (`parsers/table_bands.py`) with no model
involved — see "Table structure" in `parsers/README.md` for the full picture. `tables/structure/`
only supplies the **optional referee** that gets consulted for a region the deterministic
builder itself flags as low-confidence:

- `TableStructureClient` (`base.py`) — the neutral protocol every referee adapter implements:
  image in, detected grid out. Never called unless `TABLE_STRUCT_PROVIDER` is set — with it
  unset (the default), every table is built by the deterministic path alone.
- `vlm_adapter.py` (`TABLE_STRUCT_PROVIDER=vlm`) — reuses the already-configured `VLM_*`/`LLM_*`
  vision model; no extra dependency. `TABLE_STRUCT_MAX_TOKENS` (default 8192) is this adapter's
  per-region token budget — raise it if a large table's JSON response comes back truncated. A
  single region whose response is unparseable is skipped on its own; it doesn't take down the
  other regions on the same page.
- `tableformer_adapter.py` (`TABLE_STRUCT_PROVIDER=tableformer`) — a local IBM TableFormer model
  (`docling-ibm-models`, optional dependency — see `requirements.txt`).
- `http_client.py` (`TABLE_STRUCT_PROVIDER=http`) — an external structure-recognition service;
  `TABLE_STRUCT_BASE_URL` required, `TABLE_STRUCT_API_KEY` optional.

Whichever adapter is configured, the model only ever decides the grid's SHAPE — every cell's
text is placed from the PDF's own digital characters afterwards, never from the model's own
reading, so a visual misread can never corrupt cell content (same "structure from the model,
text from the code" split the deterministic band builder itself follows).
