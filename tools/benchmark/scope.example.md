# Evaluation Scope (example)

This file defines **what** the benchmark scores. It is passed verbatim into
every judge call; anything it places out of bounds must not raise OR lower
any score. The text here is calibrated to what the parser actually
promises (see the "Scope" section of the root `README.md`) -- so the
parser is never penalized for things it never claimed to deliver. Edit
it for your needs.

## In scope (score these)

- **Text fidelity.** Body text transcribed correctly against the PDF: numbers,
  units, technical values, code/part numbers.
- **Heading hierarchy.** Headings appear at the correct level and with the
  correct text.
- **Lists.** Structure and content of bulleted and numbered lists.
- **Tables.** Cell-content accuracy (cell text must always come from the
  PDF's own characters) and row/column grid structure.
- **Cleanliness.** Page headers/footers, page numbers, and repeated
  boilerplate must not have leaked into the body; sensible reading order.
- **Text inside images (in-scope types only).** When an in-scope visual
  contains readable text (read via VLM-OCR), that text must be captured
  reasonably. Text only -- see below. Images classified into the excluded
  types are entirely out of scope (see below).
- **Visual descriptions.** For describable visuals (tables: an LLM-written
  summary/facts alongside the cells; chart, block_diagram,
  technical_drawing, flowchart), the parser emits a natural-language
  description. When such a description is present, it is graded for
  **faithfulness to the source visual in the PDF**: a description that
  misstates what the figure actually shows is penalizable (accuracy);
  a describable visual left with no description where the figure is
  clearly locatable counts against coverage. (A table's cell grid and
  cell text are graded separately and stay in scope -- see "Tables".
  The unlocatable/unmatched-figure exception below still applies.)

## Out of scope (must NOT affect the score)

- **The original embedded image itself.** The parser does not export the
  embedded image stream for PDF figures; it only renders the page and
  crops it to a bounding box. The absence of the original image /
  resolution in the Markdown, or a loose/approximate crop, must not
  lower the score.
- **Excluded visual types (`exclude_types`).** Visuals the parser
  classifies as **product_photo**, **decorative**, or **unknown** are
  out of scope entirely -- neither their presence, their OCR text, nor
  any description affects the score. (`exclude_types = [product_photo,
  decorative, unknown]`; also kept in `benchmark/config/default.toml
  [scope]`.) The parser promises no meaningful content for a decorative
  flourish, a product photo, or an unclassifiable region, so its absence
  is never a defect. (Charts, diagrams, drawings, flowcharts are NOT
  excluded -- their descriptions ARE graded, see above.)
- **Unlocatable / unmatched figures.** When the VLM cannot produce a
  bbox guess for a figure, no crop is produced at all and the figure
  may simply be absent from the Markdown; this is not a penalty.
- **A visual too small to describe.** A crop whose shorter edge falls
  below the parser's objective size gate (`[describe]
  min_visual_edge_px`) gets no description at all. Low-confidence
  classifications (`[describe] min_visual_confidence` or below) get a
  softened description -- also not a defect.
- **Remote (http/https) images.** Never downloaded; their absence is
  out of scope.
- **Low-confidence OCR.** Image text that falls below the confidence
  threshold and is deliberately not kept is not counted as missing.
- **The experimental table-structure path.** `TableFormer`-based
  structure recognition is experimental; differences specific to that
  path do not factor into grading.
- **Chunking / RAPTOR / chunk schema.** Outside this repo's scope
  entirely; the parser only produces IR/Markdown. This covers every
  RAPTOR tree scope -- per-document, corpus-level
  (`_corpus.chunks.json`), and profile-level (`_profile.*.chunks.json`)
  summary trees -- all built by the `chunker` sub-project, none scored
  here.