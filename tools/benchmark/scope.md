# Evaluation Scope

This file defines **what** the benchmark scores. It is passed verbatim into
every judge call; anything it places out of bounds must not raise OR lower
any score. The ground truth is the **source PDF**; the candidate is the
Markdown the parser produced. This scope is written against that PDF output.
(It is written in English on purpose, matching the rest of the judge prompt —
mixing languages inside one instruction risks the model applying scope rules
inconsistently.)

The goal is not a biased shield but an **honest boundary**: everything the
parser actually promises is in scope (so it is genuinely tested); everything
it does not promise, or that is still experimental, is out of scope (so its
absence is never counted as a defect). The scope decisions below are based on
`parser/SCOPE.txt` and the "Scope" section of the root `README.md`.

## In scope (score these)

- **Text fidelity.** Body text transcribed correctly against the PDF: numbers,
  units, technical values, part/model numbers, dates, and codes. Wrong
  numbers, garbled/mixed-up characters, and fabricated content (not present in
  the PDF) are all penalizable.
- **Heading hierarchy.** Headings appearing with the correct text and at the
  correct level (1-6). Level is ranked by font size document-wide; the same
  font size must map to the same level everywhere. Expected (not penalizable)
  behavior: a heading wrapped across lines merges into one heading; a
  sentence-like run set in heading-sized font (ending in punctuation, or very
  long) is demoted to a paragraph; "Table N:"/"Figure N:" captions are never
  classified as headings.
- **Lists.** Structure and content of bulleted and numbered lists.
- **Tables.** Two separate things are scored: (1) cell-content accuracy — cell
  text must always come from **the PDF's own characters**, never a model's
  invention; (2) row/column grid structure — merged cells, correct row/column
  counts, cells that aren't shifted. Also in scope: a table's own header row
  (a shaded band with no ruling line above it) must not have leaked out of the
  table as a stray heading/paragraph.
- **Cleanliness and reading order.** Page headers/footers, repeated
  boilerplate, and page numbers must not have leaked into the body; a
  reasonable reading order; two-column pages must not have their columns
  interleaved. Also in scope: a single-column Table-of-Contents entry must not
  have leaked into the body as plain content (it belongs in structured data
  instead). Two-column TOC layouts are explicitly excepted — see out of scope.
- **Visible text inside images (in-scope types only).** When an image of an
  in-scope type contains readable text (read via VLM-OCR) and it clears the
  confidence threshold, that **text** must be captured reasonably. Text only —
  not the image itself (see below). Images classified into the excluded types
  are out entirely — see "Excluded visual types" below.
- **Visual descriptions.** For describable visuals — tables (an LLM-written
  summary/facts alongside the cells), charts, block diagrams, technical
  drawings, and flowcharts — the parser emits a natural-language description.
  Where such a description is present, it is graded for **faithfulness to the
  source visual in the PDF**: a description that misstates what the figure
  actually shows is penalizable (accuracy), and a describable visual left with
  no description where the figure is clearly locatable counts against coverage.
  (A table's cell grid and cell text are graded separately and stay in scope —
  see "Tables". The unlocatable/unmatched-figure exception below still applies:
  a figure the parser could not place is not expected to carry a description.)
- **Inline formatting (digital pages only).** Bold/italic emphasis is expected
  to carry through on digital (text-layer) PDF pages. It is not read on
  scanned pages — its absence there is not penalizable.

## Out of scope (must NOT affect the score)

- **The original embedded image itself.** The parser does not export the
  original embedded image stream for PDF figures; it only renders the page
  and crops it to a bounding box. The absence of the original image/resolution
  in the Markdown, or a loose/approximate crop, must not lower the score.
  (This render-crop limitation is specific to PDF figures only.)
- **Excluded visual types (`exclude_types`).** Visuals the parser classifies as
  **product_photo**, **decorative**, or **unknown** are out of scope entirely —
  neither their presence, their OCR text, nor any description affects the score.
  (`exclude_types = [product_photo, decorative, unknown]`; kept in
  `benchmark/config/default.toml [scope]` too.) The parser promises no
  meaningful content for a decorative flourish, a product photo, or an
  unclassifiable region, so its absence is never a defect. (Charts, block
  diagrams, drawings, and flowcharts are NOT excluded — their descriptions ARE
  graded, see "Visual descriptions" above.)
- **Unlocatable / unmatched figures.** When the VLM cannot produce a bbox
  guess for a figure, no crop is produced at all and the figure may simply be
  absent from the Markdown; this is not a penalty. (Only the figure's text, if
  any, is scored — not its presence.)
- **A visual too small to describe.** A crop whose shorter edge falls below
  the parser's objective size gate (`[describe] min_visual_edge_px`, default
  48px) gets no description at all — deliberately, since asking a model to
  describe pixels it cannot actually resolve only produces a confident-sounding
  fabrication. This missing description is not a coverage defect. (Its OCR
  text, if any, is still scored normally.)
- **A softened description on a low-confidence classification.** When a
  visual's type classification falls below `[describe] min_visual_confidence`,
  the description is still produced, but the prompt presents the type as an
  unconfirmed guess rather than asserted fact and explicitly permits "I can't
  tell." A resulting hedged or partial description is not a defect — it is
  graded for faithfulness like any other (see "Visual descriptions" above),
  not penalized for being cautious instead of thorough.
- **Remote (http/https) images.** Never downloaded; their absence is out of
  scope.
- **Low-confidence OCR text.** Image text that falls below the confidence
  threshold and is deliberately not kept is not counted as missing.
- **The experimental table-structure path.** `TABLE_STRUCT_PROVIDER=tableformer`
  is experimental and currently out of scope; differences specific to that
  path do not factor into grading. (The deterministic grid and the VLM-referee
  path ARE in scope — see "Tables" above.)
- **Two-column TOC layouts.** A Table-of-Contents whose titles and page numbers
  are split across the column gutter into separate lines is not yet recognized,
  so its entries may leak into the body as ordinary content. This is a known,
  un-promised limitation; such leakage must not lower the score. (Single-column
  TOC extraction IS in scope — see "Cleanliness and reading order" above.)
- **Chunking / RAPTOR / chunk schema.** Outside this repo's scope entirely;
  the parser only produces IR/Markdown. This covers every RAPTOR tree scope —
  per-document, corpus-level (`_corpus.chunks.json`), and profile-level
  (`_profile.*.chunks.json`) summary trees are all built by the `chunker`
  sub-project and are never scored here.
