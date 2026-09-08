# parsers/

A separate parser module per file format, each conforming to the common `BaseParser`
contract, converting the input file into the `ParsedDocument` IR.

- `base.py` — `BaseParser` (abstract interface) and `ParsedDocument` (IR) definitions. Every
  format parser implements this.
- `registry.py` — selects the right parser based on file extension/mimetype.
- `docx_parser.py`, `pptx_parser.py`, `xlsx_parser.py`, `html_parser.py`, `pdf_parser.py`,
  `markdown_parser.py` — format-specific implementations.

Notes:
- For what each format can currently do, see `SCOPE.txt` in the root directory.
- markitdown is not used: it doesn't preserve byte offset information from the original
  document, and the IR requires it.
- `pdf_parser.py` implements the "PDF pipeline" flow below. LLM access is model- and
  provider-agnostic: `llm.get_vlm_client()` (env: `VLM_*`, falls back to `LLM_*`; the second
  verifier model is `VLM2_*`). If the VLM isn't configured, the parser degrades gracefully:
  the hybrid page falls back to the code path, a scanned page stays a full-page `ImageBlock`
  (OCR is then done by the `images/` stage).
- Table blocks, including merges, are written to the IR here as structured JSON, with
  `provenance`/`table_confidence`/`table_flags` recording how the grid itself was built (see
  "Table structure" in the PDF pipeline below); generating the description is the `tables/`
  module's job.
- Wherever an image occurs, a marker like `<imageN>` is placed; filling in the marker is the
  `images/` module's job.
- Lists are not a separate container block: the IR is a flat block stream, and list membership
  is written onto ordinary blocks as metadata (`list_id` / `list_level` / `list_ordered`). This
  way, a table/image inside a list item is preserved as a real `TableBlock`/`ImageBlock` (not
  flattened into plain text); this also aligns with how docx (`w:numId`/`w:ilvl`) and pdf store
  lists. A "list" is reconstructed by grouping blocks that share the same `list_id`.

## PPTX pipeline

`pptx_parser.py` walks slides shape by shape (`_walk_shapes`), descending recursively into
group shapes (PowerPoint allows multiple shapes to be grouped; without recursion, text/table/
image inside a group would be lost entirely).

- **Heading**: a real TITLE/CENTER_TITLE/VERTICAL_TITLE placeholder is definitive and always
  wins (exact enum match — the old "TITLE" substring check was accidentally also catching
  SUBTITLE; this was fixed). If there is no placeholder (a slide designed to "look like a
  heading" using a free text-box), the same pattern as docx's pseudo-header logic runs: the
  slide's non-placeholder candidate paragraphs are scored with `PptxHeadingConfig` (bold/caps/
  title-case/centered/short/isolation/underlined/font-ratio weights + threshold), and the
  single highest-scoring candidate that clears the threshold is promoted to a HeadingBlock (one
  heading per slide). Plain-text clues are shared with docx via `heading_heuristics.py`; font/
  bold/underline/italic signals are read both from the run's own rPr and from the paragraph's
  default (a:pPr/a:defRPr).
- **Inline formatting**: `_walk_pptx_para` walks each `a:p` in order, converting a:r/a:fld into
  InlineRun (Mark: bold/italic/underline/strike/superscript/subscript), following OOXML's own
  resolution rule (if a run doesn't set a property in its own rPr, it falls back to the
  paragraph's a:pPr/a:defRPr — the same as docx's `_walk_para`). a:br (soft line break) is
  converted to an unmarked space; python-pptx returns it as a raw `\x0b` (vertical tab), which
  would otherwise leak a control character into the text, including into headings, if not
  cleaned up.
- **Lists**: not a separate container like in docx — written onto blocks as `list_id` (slide+
  shape based) / `list_level` (paragraph indent) / `list_ordered` (a:buAutoNum vs a:buChar)
  metadata.
- **Tables**: `TableBlock`, including merges (gridSpan/rowSpan, read via python-pptx's
  merge-cell API). Cells are still plain text (`text_cell`) — the docx parallel of `Cell.blocks`
  (in-cell runs/nested table/image) doesn't exist yet.
- **Images**: PICTURE shape -> `ImageBlock` (locator = media part path).
- **Embedded OLE objects** (e.g. an Excel table pasted into a slide): the raster preview that
  OOXML mandates (`mc:Fallback/p:oleObj/p:pic/blipFill/a:blip`, or `p:oleObj/p:pic` directly
  without AlternateContent) is extracted as an `ImageBlock`; `ole_format.prog_id` is written to
  `alt_text` (e.g. "Embedded object (Excel.Sheet.12)"). The preview usually comes in EMF
  (vector metafile, `image/x-emf`) format — not PNG/JPEG; the `images/` stage needs to
  rasterize it first, otherwise the OCR/vision model can't read it. If there's no preview
  (rare), no block is produced at all.
- **Speaker notes**: if `slide.notes_slide` exists, it's kept as a separate `ParagraphBlock`,
  distinguished from the body by `Span(part="ppt/notesSlides/notesSlideN.xml")`; consumers can
  distinguish body/notes by looking at span.part.
- **Locator**: shapes don't have a meaningful byte offset; `Span(part="ppt/slides/
  slideN.xml", page=N)` is used (Decision B best-effort, same as in the PDF pipeline).

### Known v1 limitations

- Table cells are plain text (`text_cell`); in-cell formatting/nested-table/image is not
  modeled.
- OLE previews can come as EMF; without rasterization they can't go through the OCR/vision
  stage.

## PDF pipeline

Unlike other formats, PDF does not follow a single lossless path; it's split into three paths
depending on the source, and content produced by the VLM is verified against an independent
source. `pdf_parser.py` implements this; triage is done per page (a single PDF can mix scanned
and digital pages), and the per-page decision is written to `metadata["pdf_pages"]`.
Provenance labels are carried in the IR on the `Block.provenance` / `Block.source_crop` fields
(see `base.py`).

### 0. Triage

Using pdfplumber, is there a text layer, and what's the coverage ratio?

- No text layer → **Scanned path**
- Text layer present + simple layout → **Code path**
- Text layer present + complex region (table/multi-column detected) → **Hybrid path**

### 1. Code path (born-digital, simple)

pdfplumber → IR blocks. Lossless, deterministic. Tagged `text-layer-verified`. Done.

### 2. Hybrid path (born-digital, complex)

1. Render the page
2. Feed to the VLM, adding the text layer to the prompt as grounding (document anchoring)
3. Fuzzy-match the output against the text layer
4. Match → `text-layer-verified`, no match → goes to Verification

### 3. Scanned path

1. Render
2. Text detector (bboxes)
3. VLM reading
4. Goes to Verification

### 4. Verification (for content without a text layer)

- Every VLM line must map to a detector bbox; one that doesn't map is suspected fabrication
- Suspicious/critical regions: crop-and-reread with a second model, fuzzy-match
- Domain regex (part number, unit, date)
- Agreement → `consensus-verified`, disagreement → **warning, no automatic resolution**

### 5. Output

IR blocks + a provenance tag on every block + a source crop reference (blob store hash) for
`unverified` ones. The citation pipeline reads its trust level from here.

**Summary:** Read with code if it's cheap, read with the VLM if you must, check everything the
VLM says against an independent source, and tag and keep a crop of whatever it couldn't verify.

### Configuration

Models (all optional; if none are set the parser runs with deterministic fallbacks):

- `VLM_*` — the primary vision model (`VLM_PROVIDER/MODEL/BASE_URL/API_KEY`; any variable
  that's missing falls back to its `LLM_*` counterpart).
- `VLM2_*` — the independent second model that verifies suspicious regions via
  crop-and-reread. Deliberately has no fallback: the verifier must be chosen independently of
  the primary model.
- Scanned-path text detector: used automatically if `pytesseract` is installed (language via
  `PDF_TESSERACT_LANG`); otherwise verification relies on VLM2.

Settings: `PDF_VLM=0` (disable all VLM use), `PDF_RENDER_DPI` (150),
`PDF_CROP_DIR` (storage/images), `PDF_VLM_MAX_TOKENS` (8192), `PDF_CONTAINMENT` (0.9),
`PDF_LINE_MATCH` (0.8). Table-structure specific settings are listed at the end of
"Table structure" below. Visual-region classification: `[visual]` in `config/default.toml`
(`classify`, `exclude_types`, `types`, `max_tokens`; `VISUAL_CLASSIFY`/`VISUAL_EXTRACT_CHARTS`/
`VISUAL_EXTRACT_SMARTART`/`VISUAL_MAX_TOKENS` env overrides) — see "Table structure" below and
`images/visual_classify.py`. Classification only LABELS a region; producing its natural-language
`description` (chart/SmartArt: deterministic, from the element's own XML —
`chart_extract.py`/`smartart_extract.py`; a block diagram/technical drawing/flowchart's crop: a
VLM reading it plus surrounding text) is a separate pass over the finished document — see
`describe/core.py`, not this module.

### Table structure: deterministic band builder + optional model referee

`find_tables()` (pdfplumber's line-geometry detector) is used for one thing only: **locating**
table regions on the page — its bboxes are reliable. It is never trusted to reconstruct the
GRID inside a region, because its "closed cell" requirement breaks on templates that draw a
table with no left border line and only per-row zebra-shading rectangles for structure: on an
unshaded row the leftmost cell has no closed bbox, so `find_tables()` drops that row's whole
first column (the label column vanishes on every other row); the same rect-vs-line coordinate
drift also invents extra hairline "phantom" columns from a shaded cell's own padding. Both were
the dominant real-world table defect in this codebase's corpus before the fix below.

**Classification gate (before any of the above):** `find_tables()` locates a region reliably,
but says nothing about what it actually IS — a block diagram or technical drawing whose frame/
connector lines happen to close into a grid passes the same geometry check a real table does
(e.g. a diagram's repeated short labels rendering as `| AAF | AAF | AAF |`, or a single diagram
caption becoming a one-cell "table"). So, with `[visual].classify` on and a VLM reachable, every
region is first classified (crop → VLM, cached by crop sha256 — see `images/visual_classify.py`)
before any of the band-builder/referee/lattice work below is spent on it: a region classified as
anything other than `"table"` is diverted straight to an `ImageBlock` (`visual_type` set to the
classified taxonomy label) and never reaches step 1. Fail-open when no VLM is reachable
(unset/disabled/down): every region is trusted as a table exactly as if classification didn't
exist — "never force unknown into a table" is about a genuine verdict of uncertainty, not about
the classifier being unavailable to ask. The VLM's own ungrounded `"table"` spec (§ hybrid path,
below) goes through the same gate via a text-only classification pass (no crop available there).

Every region that clears the gate then goes through `PdfParser._build_region_tables`, in this
order:

1. **Band builder** (`parsers/table_bands.py::build_band_table`, no model, no network) —
   the PRIMARY and default path. It reads the page's own vector geometry directly: column
   boundaries come from clustering vertical rule/rect-edge x-coordinates *across the whole
   region* (so a boundary recovered from just one shaded row's rect edge closes that column on
   *every* row — the fix for the dropped-column-0 defect above), with empty padding-strip
   "columns" between two real ones coalesced away (the phantom-column fix); when there are no
   vertical edges at all, columns fall back to persistent whitespace channels between words
   (lower confidence). Row boundaries come from horizontal rules / fill-band edges when present
   (correctly keeping a wrapped multi-line cell in one row), else from visual line midpoints. A
   header set in a shaded band directly above the ruled area (so `find_tables()`'s own bbox
   doesn't include it) is detected and pulled in as row 0. A colspan merge is only emitted with
   two independent pieces of geometric evidence at once — no rule crosses the row at that
   boundary, **and** the two sides' text actually runs together (contiguous, not just
   un-ruled) — so a header row's separately-labeled cells never wrongly fuse into one just
   because there's no rule above them. Every cell's text is always the region's own PDF words
   placed by this geometry — nothing is ever read from a bitmap or invented. Returns a
   `confidence` (0..1) and `flags` (e.g. `"header-recovered"`, `"cols-from-gaps"`,
   `"single-column"`, `"word-straddle"`) alongside the grid.
2. **Model referee** — only for regions whose band-builder confidence is below
   `PDF_TABLE_BANDS_MIN_CONFIDENCE` (default 0.7), and only if a table-structure model is
   configured (`TABLE_STRUCT_PROVIDER`, see `tables/structure/`). The model proposes a grid;
   cell text is still always placed from the PDF's own characters, never the model's reading
   (`_fill_cells_by_content`/`_fill_cells_by_containment`), and the proposal must pass an accept
   gate (fraction of the region's real content the grid accounted for, and no substantial
   invented number) or it's rejected outright. **A rejected/skipped region never falls back to
   raw `find_tables()` lattice geometry** — it keeps the band builder's own (already
   text-faithful) grid instead, so the old failure mode of a bad grid quietly degrading to a
   broken lattice table can't recur. `provenance` is `"table-bands"` or `"table-structure-model"`
   accordingly.
3. **Lattice, last resort** — only when the band builder found no placeable text in the region
   at all (a near-empty region, usually an over-detected diagram frame). Flagged
   `"lattice-fallback"` so it stays visible.

Every resulting table then passes a source-agnostic health check (`_apply_health`), regardless
of which of the three built it: it re-derives the region's real content words independently and
compares them against what actually landed in a cell (`"content-dropout"`, also folded into a
lower `confidence`), and separately flags the specific "first cell empty while the rest of the
row has content" shape (`"sparse-col0"`) — the exact signature of the old dropped-column-0
defect, worth catching wherever it recurs. Neither check ever changes cell TEXT, only
`TableBlock.table_confidence`/`table_flags` (both optional, IR v5+) — a low score or a flag is a
signal for a consumer or human reviewer, not a silent rewrite.

pdfplumber gives no rowspan/colspan directly; `Merge` records are derived from grid geometry
either by the band builder (above) or, for the lattice-fallback case, from `table.cells`'
covered-and-empty vs. genuinely-borderless cell distinction (`_table_to_data`). The hybrid (VLM)
path pairs the VLM's own "table" spec positionally with the page's already-built table (same
order, reading order) and uses that structured table outright instead of the VLM's flat JSON
grid — including its provenance/confidence/flags, which are left untouched rather than
overwritten by the hybrid path's usual whole-block text-containment check (a table's cells are
already grounded in the PDF's own text, so that coarser check would only be a downgrade). A
count mismatch falls back to the VLM's own ungrounded grid for the unpaired table(s) only.

A weaker VLM sometimes transcribes a table region as ordinary paragraphs and never emits a
"table" spec for it at all, so the positional pairing above never consumes that region's
already-built grid. `_merge_unpaired_tables` (run after verification, before figures) catches
exactly this: any geometric table the pairing left unconsumed is folded back into the block
stream — dropping the VLM paragraph/heading blocks whose text is almost entirely contained in
the table's own cells (`_UNPAIRED_TABLE_CONTAIN`, 0.8) and taking their reading-order slot, so
the clean grid replaces the duplicative prose in place rather than being lost. Its cells are
the PDF's own characters (never the model's reading), so this can't hallucinate content; a
recovered table carries a `"vlm-unpaired"` flag. If no block matches (the VLM didn't transcribe
the region at all) the table is appended so it can never silently vanish either way.

Table-structure settings (all optional): `PDF_TABLE_BANDS_MIN_CONFIDENCE` (0.7 — band-builder
confidence floor below which the model referee is consulted, if configured),
`PDF_TABLE_STRUCT_MIN_COVERAGE` (0.9 — the referee's own accept-gate floor), `TABLE_STRUCT_*`
(model referee provider/model/endpoint — see `tables/structure/`), `TABLE_STRUCT_MAX_TOKENS`
(8192 — the `provider=vlm` adapter's per-region token budget; raise it if a large table's JSON
comes back truncated).

### Heading level

In the code path, whether a line is a heading is still decided locally based on the median body
text size of its own column/page (`_is_heading`) — that hasn't changed. But which level (1-6)
it corresponds to is now determined by a **document-wide** font-size ranking: before actually
processing the pages, `parse()` scans all "code" and "hybrid" pages (including hybrid pages
that may fall back to the code path if left without a VLM) and collects all heading-candidate
sizes into a single set (`_page_heading_candidates`), sorts them, and passes this shared list to
every page (as the `heading_sizes` parameter to `_specs_from_lines`). This way, a section
heading on page 50 isn't incorrectly promoted to level 1 just because there's nothing bigger
than it on that page — if page 1 has a bigger heading, it stays level 2 (or lower), producing a
hierarchy that's consistent across the whole document. On hybrid pages where the VLM read
successfully, the heading level is still the VLM's own estimate (that's a separate mechanism);
this ranking only kicks in for hybrid pages that end up falling back to the code path due to a
missing or failed VLM.

`_page_heading_candidates` and the real per-page pass (`_specs_from_lines`, via `_parse_code`)
must always agree on which words/lines they even look at — if one excludes a word the other
still counts, the two can compute a different local median (`body`) for the same page, and a
line can end up heading-sized in the real pass without its size ever having been ranked, crashing
the `heading_sizes.index(...)` lookup. This is why table-header recovery and margin-noise
filtering (both below) are threaded into *both* passes identically, not just the real one.

### Heading merge & demotion

A heading-sized line rarely stands alone: a wrapped title spans 2-3 lines just like a paragraph
does, and so does a body sentence set in a heading-sized font by mistake — both look identical
line-by-line. `_specs_from_lines` buffers consecutive heading-sized lines (small gap, matching
font size) the same way it buffers paragraph continuations, and classifies the whole run only
once it's flushed:

- A run ending in sentence-terminal punctuation (`.`, `;`, `,`) is demoted to a plain paragraph,
  checked even for a single line — a misfont sentence's own lines often fail `_is_heading`'s
  word-count cap individually, leaving only its last, short fragment heading-sized and alone.
- A merged run over 14 words total is also demoted, but only once more than one line has actually
  merged (a single already-short line can't trip this on its own).

Figure/table captions (`heading_heuristics.CAPTION`, e.g. "Table 5:", "Figure 1:") are excluded
from heading candidacy outright, shared with the same regex docx/pptx already use.

### Table header recovery

A table's own header row is sometimes set in a shaded band with no ruling line directly above
it, so a naive grid built from ruling alone doesn't include it and the row's words would
otherwise survive the "words inside this table's bbox are removed from the text flow" filter and
surface as an ordinary (often heading-sized) stray line instead of table cells. Today this is
handled *inside* the band builder itself (`table_bands.py::_detect_header`, see "Table structure"
above) — it extends the table's own region to include the recovered header, so the header both
becomes row 0 of the grid and is excluded from the ordinary text flow as part of the same bbox.

`_recover_table_headers` — scan the lines directly above a table (nearest first, within a
generous vertical gap) for the first one whose whitespace-token count exactly matches the
table's column count — is the *older*, standalone version of this idea and is still present, but
in the live pipeline every table reaching it is already a band-built/referee `_PreparedTable`
(header already resolved one way or another), so it's a no-op there today. It's kept as a
defensive fallback for any caller that hands `_parse_code`/`_parse_hybrid` a raw, unprepared
pdfplumber table directly (bypassing `_triage`'s region-building step) — its exact-token-count
match guards against misreading an unrelated caption/row-group label as the header in that case.

### Running headers/footers and page numbers

Two independent, position-gated checks in `running_lines.py` drop page-margin noise before block
classification, both scoped to the top/bottom ~8% of the page (`in_margin_band`) so real content
elsewhere is never touched:

- **Running lines** (`detect_running_lines`): the exact same normalized text recurs, in the
  margin band, on at least 3 pages (or all of them, for a short document) — catches boilerplate
  that's identical everywhere (a URL, a "Caution" notice, ...).
- **Page numbers** (`looks_like_page_number`): the line's own shape ("4", "4/10", "Page 4", ...)
  marks it, since its *text* differs on every page and could never satisfy the repeat check above.

A one-off line that matches neither check is always left alone, even inside the margin band.

### Two-column gutter splitting

On a page with a detected gutter (two-column layout), a line is only split at the gutter
x-coordinate when there's a real gap there (`_split_gutter_groups`, gap ≥ `_COLUMN_GAP_MIN`,
comfortably between normal word spacing and `_find_gutter`'s own minimum qualifying band width).
A full-width line that merely crosses the gutter coordinate with no such gap — a wrapped sentence,
or a table's own header row, on an otherwise two-column page — stays whole in whichever column
its center falls into, instead of being cut into two disconnected fragments that land in the
wrong reading-order position relative to each other.

### Table of contents extraction

A heading whose text is exactly "Contents" or "Table of Contents" is followed, in many datasheets,
by a run of lines that are really TOC entries (a section title with its page number glued onto the
end — `"2.3. Relay Type 2"` as a numbered list item, or `"DESCRIPTION .......... 2"` as a
dot-leader paragraph with no list marker at all) rather than ordinary content; left alone, these
sit in `blocks` indistinguishable from a real section titled "Relay Type" elsewhere in the same
document. `_extract_toc` pulls any such run out of `blocks` into structured `metadata["toc"]`
entries (`{"title", "page", "level"}` — `level` from the entry's list nesting when it has one,
else 1) right after all pages are parsed, and renumbers block ids so they stay gapless. Detection
needs no page-position or font heuristic: a genuine TOC run is the only place consecutive lines'
trailing page numbers hold non-decreasing across the whole run; a run under 2 entries is left as
ordinary content (a single coincidental match is far more likely than a one-line TOC). The heading
itself ("Contents") is always kept — only the noisy entries under it are removed.

Known gap: a TOC laid out as two visually separate columns (titles in one, right-aligned page
numbers in the other, split by a real gutter — see "Two-column gutter splitting" above) reads as
two disconnected runs of blocks with no page number on the same line as its title, so this heuristic
doesn't fire and those entries are left as ordinary (numberless) heading-like paragraphs.

### Inline formatting (runs)

Bold/italic is always read from the word's own font name (e.g. "Arial-BoldMT") — the PDF
equivalent of reading a run's rPr in docx: deterministic, not a VLM guess. In the code path
this is direct (`_word_marks`/`_line_runs`). In the hybrid path, the page's own word+font
stream is extracted (`_word_marks_stream`, the same reading order as the code path's column/
gutter logic); if a block's text has already been verified against the text layer and is
`text-layer-verified`, that text's own words are aligned to this stream with a **forward-only,
never-rewinding** pointer (`_align_runs`), and matched words get their own font marks. Blocks
that couldn't be verified (`consensus-verified`/`unverified`) never go through alignment at all
— their text doesn't already match the text layer, so alignment wouldn't be reliable either.
On a scanned page (no text layer at all), `runs` is not filled in; there's no independent font
source to verify against, so this is out of scope.

### Known v1 limitations

- On both hybrid and scanned pages, figures the VLM calls out but that have no counterpart in
  pdfplumber's object model (e.g. a vector drawing, or a sub-figure embedded in a scanned
  background raster) remain an `unverified` `ImageBlock`; if the VLM supplied its own bbox
  guess, the audit crop is tightened to that guess and stored as `source_crop` (images/
  resolves the block from that crop, so it still gets an `image_id` and OCR), otherwise no crop
  is produced at all and the block stays permanently unresolved (a full-page dump would be
  misleading, so it's avoided).
- On a hybrid page, embedded images the VLM didn't count as a figure (i.e. the VLM missed them)
  are appended to the end of the page flow, without a position; ones the VLM did call a figure
  and that match a candidate from `_vlm_figure_regions` get both the reading-order position from
  the VLM and a real bbox -- from a single whole-page `classify_page_regions` call when
  `VLM_CLASSIFY` is reachable (a technical drawing's fragments already merged into one region
  there), or from a raw pdfplumber raster object as a fallback when it isn't.
- On a scanned page, `runs` (inline formatting) is not filled in (see above).
- `Span` has no byte offset, only a page number.
- Running-header/footer detection requires the exact same text to repeat on several pages; a
  footer that happens to sit slightly differently on one page (different margin-band offset,
  e.g. no page number printed there) isn't linked to its repeats elsewhere and can slip through.
- Table header recovery (`table_bands.py::_detect_header`) takes the nearest line above a
  table's region (within a generous vertical gap) that has 2+ words and fits the region's
  horizontal span — looser than the older column-count-matching approach it replaced (see
  "Table header recovery" above), and with no caption-line exclusion of its own; a two-or-more-
  word caption or row-group label placed directly above (rather than below) a table, within its
  horizontal span, could in principle be misread as the header instead. Not observed on the
  ACME corpus this was built against (captions there sit below the table), but not structurally
  ruled out for a differently-laid-out document.
- The band builder's column detection needs either real ruling/rect edges or a whitespace gap
  that holds consistently across (almost) all of a table's rows; a table with no ruling at all
  and columns that only align loosely row-to-row degrades to a single-column grid (flagged
  `"single-column"`, low `table_confidence`) rather than a guessed multi-column split.
- The band builder's colspan/merge detection requires geometric evidence (no crossing rule +
  contiguous text); a table region with no ruling at all therefore never emits a merge, even if
  a cell visually spans several columns — it stays several adjacent single-column cells instead.
- TOC extraction (see above) only recognizes a heading literally titled "Contents" or "Table of
  Contents", and only catches entries whose page number sits on the same line as its title —
  a two-column TOC layout (titles and page numbers split across the gutter) is not recognized.
