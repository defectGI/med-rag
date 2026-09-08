# parser

A modular parser that converts different file formats (docx, pptx, xlsx, html, pdf, markdown)
into a common intermediate representation (IR — `ParsedDocument`), runs images through OCR to
verify their meaningfulness, and structures and describes tables.

The IR is serialized as JSON (`.json` files under `storage/output/`). All enrichment results —
image OCR, table description, etc. — are written back directly into this IR JSON rather than
into a separate database. Raw image bytes are never embedded in the JSON; only an `image_id`
reference is carried.

Chunking, RAPTOR, and the chunk schema are out of scope for this repo; the parser only produces
the IR.

For what each format can currently do, see [`SCOPE.txt`](SCOPE.txt).

## Quick start — parse your own files with your own models

Everything below is driven by env vars (or a `.env` file at the repo root), so
you can point the parser at **any** folder, **any** files, and **any** model
(local Ollama, a hosted OpenAI-compatible API, OpenRouter, Anthropic) — with
reasoning/"thinking" **on or off** — without editing a single line of code.

**1. Install**

```
pip install -r requirements.txt
# optional extras: pytesseract (scanned-PDF text),
# docling-ibm-models (TableFormer) — see requirements.txt for when each is needed.
```

**2. Choose your models.** Copy `.env.example` to `.env` and set the two roles.
`LLM_*` is the text model (table descriptions, OCR check); `VLM_*` is the vision
model (image OCR, scanned/hybrid PDF pages). Every `VLM_*` var falls back to its
`LLM_*` counterpart, so if your LLM is already multimodal you can set only `LLM_*`.

Local Ollama example:
```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:32b        # any model you've pulled
LLM_THINKING_ON=0            # 0 = thinking OFF (default), 1 = ON

VLM_PROVIDER=ollama
VLM_MODEL=qwen2.5vl:7b
VLM_THINKING_ON=0
```
Hosted OpenAI-compatible / OpenRouter example:
```dotenv
LLM_PROVIDER=openrouter
LLM_MODEL=qwen/qwen-2.5-72b-instruct
LLM_API_KEY=sk-or-...
LLM_THINKING_ON=0
```

> **Thinking / reasoning toggle — read this.** Hybrid-reasoning models
> (Qwen3/3.5, Qwen3-VL, gpt-oss, …) default to thinking **ON**, and when left on
> they can spend the entire token budget on hidden reasoning and return **empty**
> content — which silently breaks OCR and table description. That's why
> `*_THINKING_ON` defaults to **off**. Set it to `1` per role only if you
> actually want reasoning. Note that some models/servers ignore the toggle: at
> least one build of `qwen3-vl:8b` on Ollama keeps thinking on no matter what —
> if OCR/table output comes back empty, switch to a non-reasoning model (e.g. the
> Qwen2.5 family) or verify the toggle works with a one-line smoke call first.

**3. (Ollama only) pull + warm the models** so the first file doesn't eat a
multi-GB download mid-run:
```
python ../pipeline/ensure_ollama_models.py
```

**4. Run.** One file → IR JSON **and** rendered Markdown side by side:
```
python scripts/to_markdown.py path/to/your_file.pdf
# writes storage/output/your_file.json + storage/output/your_file.md
```
Point the output somewhere else with `STORAGE_OUTPUT_DIR`:
```
# PowerShell
$env:STORAGE_OUTPUT_DIR="C:\out"; python scripts/to_markdown.py C:\docs\a.pdf
# bash
STORAGE_OUTPUT_DIR=./out python scripts/to_markdown.py docs/a.pdf
```
A whole folder (loop over it in your shell):
```powershell
# PowerShell — every PDF under a folder
Get-ChildItem C:\docs -Filter *.pdf | ForEach-Object {
    python scripts/to_markdown.py $_.FullName
}
```
```bash
# bash — every PDF under a folder
for f in docs/*.pdf; do python scripts/to_markdown.py "$f"; done
```
`scripts/to_markdown.py` enriches best-effort: if a model call fails the parse
and Markdown still complete, just without that enrichment. Supported inputs:
`.pdf .docx .pptx .xlsx .html .md` (see `SCOPE.txt`).

For a fixed, corpus-backed end-to-end regression run (10 datasheets + 10 mixed
documents, with model/dependency/thinking preflight checks), see
`../pipeline/README.md`.

## Testing

The whole suite is offline: no network calls, no LLM/VLM, no external services, no `.env`
required. Every test either builds its own tiny document byte-for-byte (PDF/DOCX/PPTX/XLSX/
HTML/Markdown — see e.g. `text_pdf()`/`grid_pdf()` in `tests/test_table_structure.py`, or
`tests/test_table_bands.py`'s content-stream builder for fill-rect/ruling geometry) or injects a
fake `LLMClient`/`VLMClient`/`TableStructureClient` through the parser's constructor instead of
calling a real model. That's deliberate: a fresh clone with no Ollama running and no API key set
can still run the full suite, and CI never depends on flaky network/model access.

```
pip install pytest
pytest                                    # everything
pytest tests/test_pdf_parser.py -q        # one file
pytest -k table_bands                     # by keyword
```

- `conftest.py` at the repo root puts the repo on `sys.path`, so `import parsers...` resolves
  under pytest without an editable install.
- One test is expected to `SKIP` (in `test_nested_tables.py`) — it documents a lossless
  round-trip-editing subsystem that doesn't exist yet; the skip reason explains what would need
  to land first.
- To fake model calls in your own test: pass `vlm=`, `vlm2=`, `detector=`, or `table_struct=` to
  `PdfParser(...)` (or the equivalent constructor arg on another parser) with an object exposing
  just the method the real client would (`complete_vision`, `detect`, ...) and a canned return
  value — see any existing `_Fake*`/`Fake*` class in `tests/` for the pattern.
- This is unit/regression coverage for parser *logic* (routing, grid geometry, verification,
  IR shape). It is not a substitute for `../pipeline/README.md`'s corpus-backed end-to-end run
  against real models, which is the only way to catch a real model's actual behavior (e.g. a
  provider silently changing its JSON formatting, or thinking mode swallowing the answer).

## Pipeline

1. `storage/raw/` — the input file is kept as-is.
2. `parsers/` — the parser appropriate to the file type converts it to the common
   `ParsedDocument` IR → `storage/output/`. For PDF specifically, this step already includes
   building every table's row/column grid — deterministically from the page's own geometry, no
   model required (`parsers/table_bands.py`); see `parsers/README.md`, "Table structure".
3. `images/` — runs the `<imageN>` markers placed during parsing through OCR, verifies their
   meaningfulness with an LLM, and writes the result back into the IR in place of the marker.
   The raw image is stored immutably and deduplicated by sha256 (`image_id`) in the
   `storage/images/` blob store; the record itself lives on the `ImageBlock` in the IR.
4. `describe/` — adds a natural-language `description` (and, for tables, digit-checked `facts`)
   to every "describable block" (table, chart, block_diagram, technical_drawing, flowchart —
   see `parsers/base.py`'s `DescribableBlock`): tables are described from their own cells with
   an optional LLM check, native charts/SmartArt are described deterministically from their own
   XML at parse time (no model call), everything else from a VLM reading its crop plus
   surrounding text. See `tables/README.md` for the `LLM_THINKING_ON` env flag —
   hybrid-reasoning models default to thinking on and can silently return empty descriptions if
   it's left on.

## Folders

- `parsers/` — format-specific parsers + the `BaseParser`/`ParsedDocument` contract;
  `table_bands.py` is the deterministic PDF table-grid builder (see `parsers/README.md`)
- `images/` — `image_handler` and `ocr_output_control`
- `describe/` — the shared describe pass (`core.py`) + its per-type strategies (`table.py`,
  `visual.py`) + surrounding-context gathering (`context.py`); see `describe/core.py`'s
  module docstring
- `tables/` — `header_infer.py` (LLM fallback for `header_rows`); `tables/structure/` is the
  OPTIONAL PDF table-grid model referee, consulted only for a region the deterministic builder
  itself flags low-confidence (`TABLE_STRUCT_PROVIDER=vlm|tableformer|http`, see
  `tables/README.md`)
- `tests/` — the test suite (see "Testing" above); mirrors the module it covers
  (`test_pdf_parser.py`, `test_table_bands.py`, `test_table_structure.py`, ...)
- `storage/` — raw data (`raw/`), resulting IR output (`output/`), image blob store (`images/`);
  these three paths are not hardcoded — they can be overridden via the `STORAGE_RAW_DIR` /
  `STORAGE_OUTPUT_DIR` / `STORAGE_IMAGES_DIR` env variables through `storage_paths.py`
  (see `.env.example`) — otherwise they fall back to the dev-time defaults here.
