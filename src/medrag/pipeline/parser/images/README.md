# images/

Processes the image markers (`<imageN>`, each an `ImageBlock`) placed into the IR during
parsing.

- `image_handler.py` — locates the raw image based on each `ImageBlock`'s `locator` (docx/pptx/
  xlsx: zip media part; html/markdown: data-uri or local file; pdf: the page is rendered and
  cropped from the bbox), classifies what it actually is (table/chart/block_diagram/
  technical_drawing/flowchart/product_photo/decorative/unknown — `visual_classify.py`; skipped
  if the block arrives already classified, e.g. one of `pdf_parser.py`'s reclassified table
  candidates) unless a type in `[visual].exclude_types` stops here (no OCR/description
  generated, only existence+type+blob kept — see `ImageBlock.excluded_at_parse`), runs it
  through OCR with a VLM, and verifies the result with
  `ocr_output_control.py`; if meaningful **and** confident enough (see below), writes the
  corrected text into `ImageBlock.ocr_text`
  — i.e. into the marker's *own slot* in the IR, not into the surrounding paragraph/cell text
  (that text's `Span` must stay byte-exact to the source). Regardless of whether it's meaningful
  or not, stores the image immutably and deduplicated by sha256 (`image_id`) in the
  `storage/images/` blob store. The image's record (image_id, locator, ocr_text,
  ocr_meaningful, ocr_confidence, mime, width/height) is kept on the `ImageBlock` in the IR; `doc_id` and
  `access_level` come from the document. There is no separate database. Images inside table
  cells (including nested tables at any depth) are processed too. A remote (http/https) `src`
  is left unresolved — the pipeline does not make network requests. A pdf `ImageBlock` whose
  locator has no bbox (an unpaired/"unverified" figure — see parsers/pdf_parser.py) falls back
  to `ImageBlock.source_crop`: the parser already rendered and stored an audit crop for it in
  this same blob store, so it resolves from that file instead of staying permanently
  unprocessed.
- `ocr_output_control.py` — asks an LLM whether the OCR output is meaningful, rates its
  `confidence` (0.0–1.0) that the transcription is accurate, and fixes spelling mistakes and
  format corruption.

Note: this module classifies and OCRs the image; it never writes a natural-language
*description* of a non-photo visual (a chart, a block diagram, ...) — that is `describe/`'s job,
a separate pass over the finished document (see `parser/describe/README` equivalent: the module
docstring of `describe/core.py`). `image_handler.py` does honor exclusion for the type it OWNS
(OCR), but excluding a describable type (`[visual].exclude_types`) additionally stops
`describe/` from ever being asked about it.

Rule: the raw image bytes are never embedded in the IR — only the `image_id` reference is
carried. If the OCR is not meaningful, `ocr_text` stays empty (only `ocr_meaningful=False` is
set) but the blob + the record in the IR are kept — resolving the marker at search/display time
(substituting it with text, or dropping it) is the read-time consumer's (webapp/chunker) job.

Confidence gate: image OCR is inherently unverified — a VLM reads pixels with no text layer to
align against (unlike a born-digital PDF page). So even a "meaningful" transcription is only
kept as `ocr_text` when its `ocr_confidence` clears `IMAGE_OCR_CONFIDENCE_THRESHOLD` (default
`0.7`, never hardcoded — set the env var to tune it). A lower-confidence reading is withheld
exactly like a not-meaningful one (`ocr_meaningful=False`, blob + record kept), but its score
is still recorded in `ocr_confidence`. This is **fail-closed**: a reading the reviewer gives no
confidence for at all is withheld too — an unquantified image transcription is treated as no
more trustworthy than a low-scored one. Because the text is never fully trustworthy, the Markdown renderer labels it
`*OCR (unverified image text, may be unreliable — confidence 0.NN): …*` and the JSON carries
`ocr_confidence`, so both outputs make the OCR origin and its unreliability explicit.
