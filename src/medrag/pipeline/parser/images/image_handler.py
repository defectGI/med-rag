"""Fetches the raw bytes an ImageBlock's locator points at, OCRs them via a
VLM, validates/cleans the OCR text via ocr_output_control, and stores every
image -- regardless of OCR outcome -- in the sha256-keyed blob store.

Pipeline per image:
    1. fetch raw bytes from the source document (format-specific: a zip
       media part for docx/pptx/xlsx, a data: URI or local file for
       html/markdown -- a remote http(s) src is left unresolved, see below
       -- or a rendered-and-cropped page region for pdf),
    2. store the bytes in storage/images/ by sha256 (dedup, immutable) and
       fill image_id/mime/width/height,
    3. classify what the image actually is (table/chart/block_diagram/
       technical_drawing/flowchart/product_photo/decorative/unknown -- see
       visual_classify.py), UNLESS it arrives already classified (e.g. a
       pdf_parser.py table-candidate diverted by its own classification
       step); a type in `[visual].exclude_types` stops here -- no OCR/
       description content generated for it, only existence+type+crop,
    4. ask a VLM to transcribe any text the image contains,
    5. hand the raw transcription to ocr_output_control, which judges
       whether it's meaningful and returns a cleaned version; ocr_text and
       ocr_meaningful are filled from that verdict.

Steps 4 and 5 use DIFFERENT models (VLM vs LLM); on a single-GPU Ollama that
can't hold both, running them back-to-back per image swaps models constantly.
`handle_images(..., raw_sink=...)` therefore runs steps 1-4 only, parking raw
transcriptions in the sink, and `apply_ocr_checks(doc, raw_ocr)` runs step 5
later -- so a pipeline can batch all VLM work, then all LLM work.

Design decisions
-----------------
* The corrected text becomes the value of `ImageBlock.ocr_text` -- the
  image's own slot in the IR -- not a splice into a sibling paragraph's
  text. Every block's `Span` stays byte-exact to the source (see
  parsers/base.py); resolving a `<imageN>` marker to its image's text for
  display/search is a read-time concern for consumers (webapp/chunker), not
  something this stage does in place.
* A remote (http/https) `src` in html/markdown is left unresolved: no
  network fetch happens from this pipeline. image_id/ocr fields simply stay
  None for that ImageBlock.
* OCR uses a VLM first: one `complete_vision` call per image, consistent
  with how parsers/pdf_parser.py already reads scanned pages. With
  IMAGE_OCR_FALLBACK=tesseract set, a failing/unconfigured VLM falls back
  to local tesseract OCR -- a VLM-only design would mean a systemic VLM
  outage disables the "fallback" OCR stage along with the parser's own page
  reads. Fallback text still passes the same LLM
  meaningfulness check before landing in ocr_text.
* Storing the blob never depends on OCR/VLM availability -- an unconfigured
  or failing VLM only leaves ocr_text/ocr_meaningful unset.
* An unpaired PDF figure (`provenance="unverified"`, no `bbox` in its
  locator -- see parsers/pdf_parser.py) is not a dead end: the parser already
  rendered an audit crop for it into this same blob store and recorded its
  sha256 as `ImageBlock.source_crop`. When the normal locator-based fetch
  can't resolve such a block, this stage falls back to that existing crop
  instead of leaving it unprocessed -- the bytes are already sitting there,
  they were just never wired up to `image_id`/OCR.

Environment (all optional):
    IMAGE_OCR_MAX_TOKENS  VLM transcription token budget (default 1024)
    IMAGE_OCR_FALLBACK    "tesseract" -> local OCR when the VLM is
                          unavailable or a transcription call fails
                          (uses PDF_TESSERACT_LANG); unset -> no fallback
    IMAGE_OCR_CONFIDENCE_THRESHOLD
                          minimum reviewer confidence (0.0-1.0, default 0.7) for
                          an image's transcription to be kept as `ocr_text`; a
                          lower-confidence reading is withheld (blob/record kept)
                          since image OCR has no text layer to verify against.
                          Fail-closed: a reading with no confidence at all is
                          withheld too.
    PDF_RENDER_DPI        page render resolution for pdf crops (default 150;
                           shared with parsers/pdf_parser.py for consistency)
    IMAGE_CONCURRENCY     images fetched/OCR'd in parallel (default 4) -- each
                          is a blocking HTTP call, so threads overlap network
                          wait time across images
"""

from __future__ import annotations

import base64
import io
import mimetypes
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from PIL import Image

from medrag.pipeline.parser import progress
from medrag.pipeline.parser.config import get_config
from medrag.pipeline.parser.llm import (
    LLMClient,
    LLMError,
    VLMClient,
    get_client,
    get_vlm_classify_client,
    get_vlm_client,
)
from medrag.pipeline.parser.parsers.base import (
    ImageBlock,
    ParsedDocument,
    TableBlock,
    hash_hex,
    sha256_id,
)
from medrag.pipeline.parser.storage_paths import images_dir, tesseract_lang_override

from .ocr_output_control import check_ocr_text
from .visual_classify import classify_crop, log_unknown

_OCR_SYSTEM = (
    "Transcribe any text visible in this image exactly as it appears, "
    "preserving line breaks. If the image contains no legible text (e.g. a "
    "photo, icon, logo or decorative graphic), reply with exactly: NO_TEXT"
)

Fetcher = Callable[[Path, dict[str, Any]], "tuple[bytes, str | None] | None"]


def _apply_verdict(block: ImageBlock, llm: LLMClient, raw_text: str) -> None:
    """Run the reviewer over one raw transcription and record the verdict on
    the block, gating `ocr_text` on the confidence threshold.

    Image OCR has no text layer to verify against, so a low-confidence reading
    is treated as untrustworthy and its text withheld (blob + record are still
    kept -- same contract as a "not meaningful" verdict). The threshold is read
    from IMAGE_OCR_CONFIDENCE_THRESHOLD (default 0.7), so it is never hardcoded.

    Fail-closed: a reviewer that reports no confidence at all (None) is treated
    as not trustworthy and its text withheld too -- an unquantified image
    transcription doesn't clear the bar any more than a low-scored one does."""
    meaningful, cleaned, confidence = check_ocr_text(llm, raw_text)
    block.ocr_confidence = confidence
    threshold = get_config().image_ocr.confidence_threshold
    confident_enough = confidence is not None and confidence >= threshold
    accept = meaningful and confident_enough
    block.ocr_meaningful = accept
    block.ocr_text = cleaned if accept else None


def _collect_images(doc: ParsedDocument) -> list[ImageBlock]:
    """Every ImageBlock in reading order, including ones nested inside table
    cells at any depth -- `ParsedDocument.images()` only looks at top-level
    blocks, which misses cell images (a real, tested feature) entirely."""
    found: list[ImageBlock] = []

    def walk_block(b) -> None:
        if isinstance(b, ImageBlock):
            found.append(b)
        elif isinstance(b, TableBlock):
            for row in b.table.cells:
                for cell in row:
                    if cell is None:
                        continue
                    for cb in cell.blocks:
                        walk_block(cb)

    for b in doc.blocks:
        walk_block(b)
    return found


def _ext_for(mime: str | None) -> str:
    return mimetypes.guess_extension(mime or "") or ".bin"


def _store_blob(data: bytes, mime: str | None) -> str:
    """Write into the shared immutable blob store, deduped by sha256; return
    the image_id (`sha256:<hex>`; the filename keeps the bare hex)."""
    hash_id = sha256_id(data)
    root = images_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{hash_hex(hash_id)}{_ext_for(mime)}"
    if not path.exists():
        path.write_bytes(data)
    return hash_id


# -- fetchers, one per locator shape -----------------------------------------


def _fetch_zip_part(raw_path: Path, locator: dict[str, Any]):
    """docx/pptx/xlsx: locator = {"part": "word/media/image1.png"}."""
    part = locator.get("part")
    if not part:
        return None
    try:
        with zipfile.ZipFile(raw_path) as zf:
            data = zf.read(part.lstrip("/"))
    except (KeyError, OSError):
        return None
    return data, mimetypes.guess_type(part)[0]


def _fetch_src(raw_path: Path, locator: dict[str, Any]):
    """html/markdown: locator = {"src": ...}.

    A data: URI is decoded inline; a local path is resolved relative to the
    source document. A remote http(s) URL is left unresolved by design (no
    network fetch from the parsing pipeline)."""
    src = locator.get("src")
    if not src:
        return None
    if src.startswith("data:"):
        header, sep, encoded = src.partition(",")
        if not sep or ";base64" not in header:
            return None
        mime = header[len("data:"):].split(";")[0] or None
        try:
            return base64.b64decode(encoded), mime
        except Exception:  # noqa: BLE001 -- broken base64 data: URI -> None; the function's "an unresolvable source returns None" contract (same path as a remote URL / missing src)
            return None
    if urlsplit(src).scheme in ("http", "https"):
        return None
    candidate = Path(src)
    if not candidate.is_absolute():
        candidate = raw_path.parent / candidate
    if not candidate.is_file():
        return None
    return candidate.read_bytes(), mimetypes.guess_type(str(candidate))[0]


def _fetch_pdf_region(raw_path: Path, locator: dict[str, Any], cache: dict):
    """pdf: locator = {"page": N, "bbox": [x0, top, x1, bottom], ...} in
    pdfplumber's native point units. Re-renders that page and crops -- there
    is no embedded-object extraction, so this mirrors the audit-crop path
    parsers/pdf_parser.py already uses for unverified figures.

    A "vlm_figure" locator with no bbox (the VLM called out a figure that
    neither pdfplumber's object model nor its own bbox guess can back)
    cannot be resolved to any bytes at all; that ImageBlock stays
    unresolved, same as parsers/pdf_parser.py's own unverified-figure path.
    """
    page_no = locator.get("page")
    bbox = locator.get("bbox")
    if page_no is None or bbox is None:
        return None
    dpi = get_config().pdf.render_dpi
    key = (str(raw_path), page_no, dpi)
    png = cache.get(key)
    if png is None:
        import pdfplumber

        from medrag.pipeline.parser.parsers.pdf_lock import PDFIUM_LOCK

        # Holds the lock across both the cache check and the render so two
        # threads racing on the same page don't double-render, and so this
        # never overlaps another thread's pdfplumber.open() elsewhere
        # (PDFium itself isn't safe to touch from multiple threads at once).
        with PDFIUM_LOCK:
            png = cache.get(key)
            if png is None:
                try:
                    with pdfplumber.open(raw_path) as pdf:
                        png = pdf.pages[page_no - 1].to_image(resolution=dpi).original
                except Exception:  # noqa: BLE001 -- pdfplumber/PDFium render error -> no crop is produced (None), the parse chain does not fall
                    return None
                cache[key] = png
    scale = dpi / 72.0
    x0, top, x1, bottom = bbox
    box = tuple(round(v) for v in (x0 * scale, top * scale, x1 * scale, bottom * scale))
    buf = io.BytesIO()
    png.crop(box).save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _fetcher_for(fmt: str) -> Fetcher | None:
    if fmt in ("docx", "pptx", "xlsx"):
        return _fetch_zip_part
    if fmt in ("html", "markdown"):
        return _fetch_src
    if fmt == "pdf":
        cache: dict[Any, Any] = {}
        return lambda raw_path, locator: _fetch_pdf_region(raw_path, locator, cache)
    return None


def _ocr_via_vlm(vlm: VLMClient, data: bytes, mime: str) -> str:
    text = vlm.complete_vision(
        system=_OCR_SYSTEM, user="Transcribe this image.",
        images=[(mime, data)], max_tokens=get_config().image_ocr.max_tokens,
    ).strip()
    return "" if text == "NO_TEXT" else text


def _ocr_via_tesseract(data: bytes) -> str | None:
    """Opt-in (IMAGE_OCR_FALLBACK=tesseract) local OCR used when the VLM is
    unavailable or its call fails. Without this, a systemic VLM outage takes
    the whole OCR stage down with it -- the "fallback" would be the same
    model. None = fallback disabled/failed/read nothing; the
    transcription still goes through the usual LLM meaningfulness check, so
    tesseract noise doesn't land in ocr_text unvetted."""
    if get_config().image_ocr.fallback.strip().lower() != "tesseract":
        return None
    try:
        import pytesseract  # deferred: optional dependency

        text = pytesseract.image_to_string(
            Image.open(io.BytesIO(data)),
            lang=tesseract_lang_override())
    except Exception:  # noqa: BLE001 -- optional pytesseract: an import/OCR error equals the docstring's "None = fallback disabled/failed" state
        return None
    return text.strip() or None


def _fetch_source_crop(source_crop: str) -> tuple[bytes, str | None] | None:
    """Load the audit crop pdf_parser.py already rendered and stored for an
    unpaired ("unverified") figure -- same blob store, sha256-named, always a
    PNG (see `_store_crop`/`_figure_crop` in parsers/pdf_parser.py)."""
    path = images_dir() / f"{hash_hex(source_crop)}.png"
    if not path.is_file():
        return None
    return path.read_bytes(), "image/png"


def _process_image(block: ImageBlock, *, fetch: Fetcher, raw_path: Path,
                    vlm: VLMClient | None, vlm_classify: VLMClient | None,
                    llm: LLMClient | None,
                    doc_id: str, fmt: str,
                    raw_sink: dict[str, str] | None = None,
                    classify_only: bool = False) -> None:
    """Resolve a single ImageBlock, mutating it in place. Independent of every
    other image -- safe to run concurrently, one block per thread.

    With `raw_sink` set, the LLM meaningfulness check is DEFERRED: the raw VLM
    transcription is parked in `raw_sink[block.marker]` and no LLM is touched
    (`apply_ocr_checks` runs the check later). This lets a model-affinity
    pipeline do all VLM work first, then all LLM work, instead of swapping
    models in and out of VRAM per image.

    With `classify_only` set, OCR itself is ALSO deferred: this call stops
    right after classification (and the exclude-type check), never touching
    `vlm` at all. This is what lets a genuinely different VLM_CLASSIFY model
    (see images/visual_classify.py) get warmed and used for every image in
    one pass, before the primary VLM is ever loaded -- otherwise classify and
    OCR calls alternate per image (and across images, since they run
    concurrently), forcing Ollama to reload a different model constantly.
    Since `classify_crop` caches its verdict on disk by crop sha256 (see
    visual_classify.py), a later non-`classify_only` call for the same image
    finds `block.visual_type` already set here and skips classification
    outright -- no double model call, whether that second call happens in
    the same process or a later one.

    Format-agnostic classification: this one call site handles every format's
    images (`_fetcher_for` already dispatched docx/pptx/xlsx/html/markdown/pdf
    to the right byte fetcher above), so stamping `visual_type` here is the
    single place the whole repo classifies a "plain" image block, regardless
    of source format -- pdf_parser.py's own reclassified table candidates
    arrive already stamped (see its `visual_type is not None` check below) and
    are never reclassified a second time."""
    fetched = fetch(raw_path, block.locator)
    if fetched is None and block.source_crop:
        fetched = _fetch_source_crop(block.source_crop)
    if fetched is None:
        return
    data, mime = fetched
    mime = mime or block.mime
    block.image_id = _store_blob(data, mime)
    block.mime = mime
    try:
        with Image.open(io.BytesIO(data)) as img:
            block.width, block.height = img.size
    except Exception:  # noqa: BLE001, S110 -- size metadata only; if PIL can't open it, width/height stay None -- the visual blob was already stored
        pass

    if block.visual_type is None:
        cfg_v = get_config().visual
        verdict = classify_crop(data, mime or "image/png", vlm_classify, cfg_v)
        block.visual_type = verdict.visual_type
        block.visual_type_source = verdict.source
        block.visual_type_confidence = verdict.confidence
        if verdict.visual_type == "unknown" and verdict.source != "unavailable":
            log_unknown(doc_id, block.id, block.image_id, {"fmt": fmt})

    cfg_v = get_config().visual
    if cfg_v.classify and block.visual_type in cfg_v.exclude_types:
        # Excluded type: existence + type + a stored crop (image_id, just set
        # above) are enough for a chunk-time placeholder -- no OCR/description
        # content is generated for it.
        block.excluded_at_parse = True
        return

    if classify_only:
        return

    if not get_config().image_ocr.enabled:
        # OCR is unconditionally off (`[image_ocr] enabled = false`). Kept
        # SEPARATE from the `exclude_types` path: that one says "this TYPE is
        # left out" and marks `excluded_at_parse` (stopping describe too); here
        # only the OCR step is skipped, the block itself and its type stay normal.
        return

    raw_text: str | None
    if vlm is None:
        raw_text = _ocr_via_tesseract(data)  # None unless fallback opted in
        if raw_text is None:
            return
    else:
        try:
            raw_text = _ocr_via_vlm(vlm, data, mime or "image/png")
        except LLMError:
            raw_text = _ocr_via_tesseract(data)
            if raw_text is None:
                return
    if not raw_text:
        block.ocr_meaningful = False
        return

    if raw_sink is not None:
        raw_sink[block.marker] = raw_text
        return
    if llm is None:
        return
    _apply_verdict(block, llm, raw_text)


def handle_images(doc: ParsedDocument, vlm: VLMClient | None = None,
                   llm: LLMClient | None = None, *,
                   raw_sink: dict[str, str] | None = None,
                   classify_only: bool = False) -> ParsedDocument:
    """Fill every unresolved ImageBlock's image_id/mime/width/height, and --
    when a VLM is available and the image resolves -- its ocr_text/
    ocr_meaningful.

    Mutates `doc` in place and returns it. Idempotent: an ImageBlock that
    already has an `image_id` is left untouched, so re-running the stage (or
    running it after a partial failure) neither re-fetches nor re-OCRs it.
    No-op when the document has no unresolved images, so no client/network
    is needed for image-free documents.

    With `raw_sink` given, this is the VLM-ONLY half of the stage: raw OCR
    transcriptions land in `raw_sink` keyed by `ImageBlock.marker` and NO LLM
    is built or called -- the caller persists the sink and later runs
    `apply_ocr_checks(doc, raw_sink)` to finish. Model-affinity split: all
    VLM work can then run before any LLM work, so a single-GPU Ollama isn't
    forced to reload a large model per image.

    With `classify_only=True`, this is the CLASSIFY-ONLY third of the split:
    every image is fetched and classified (see `_process_image`'s
    `classify_only`) but neither the primary VLM (OCR) nor the LLM is built or
    called -- `vlm`/`llm` are ignored entirely in this mode. VLM_CLASSIFY can
    now be configured as a genuinely different model from the primary VLM
    (images/visual_classify.py), so running it interleaved with OCR (the old
    per-image order) made Ollama swap models on every image instead of once
    per run. Calling `handle_images(doc, classify_only=True)` for the whole
    corpus before any OCR/LLM phase warms `classify_crop`'s on-disk cache
    (keyed by crop sha256), so the later OCR-only call below finds every
    `visual_type` already set and never touches VLM_CLASSIFY again.
    `raw_sink` and `classify_only` are mutually exclusive in practice (the
    caller picks one deferred mode per call); `classify_only` wins if both
    are somehow set, since neither the OCR nor the LLM step would run anyway.

    Images are resolved concurrently (see IMAGE_CONCURRENCY): fetch+OCR+check
    per image is a couple of blocking HTTP calls, and images don't depend on
    each other, so threads overlap the network wait instead of paying for it
    image-by-image.
    """
    images = [im for im in _collect_images(doc) if im.image_id is None]
    if not images:
        return doc

    fetch = _fetcher_for(doc.fmt)
    if fetch is None:
        return doc
    raw_path = Path(doc.source_path)

    if vlm is None:
        try:
            vlm = get_vlm_client()
        except LLMError:
            vlm = None
    # `vlm` is only resolved here as a fallback CANDIDATE for
    # get_vlm_classify_client (used when VLM_CLASSIFY_* isn't set) --
    # constructing a client makes no network call by itself. In classify_only
    # mode `_process_image` never calls `vlm` for OCR (it returns right after
    # classification), so building it costs nothing but keeps the same
    # fallback-to-primary-VLM behavior classify has always had.
    vlm_classify = get_vlm_classify_client(vlm)
    if llm is None and raw_sink is None and not classify_only:  # deferred modes never touch the LLM
        try:
            llm = get_client()
        except LLMError:
            llm = None
    if classify_only:
        vlm = None  # _process_image must not call OCR in this mode

    max_workers = max(1, get_config().image_ocr.concurrency)
    label = "image classify" if classify_only else "image classify/OCR"
    with ThreadPoolExecutor(max_workers=max_workers) as pool, \
            progress.bar(len(images), f"{raw_path.name}: {label}",
                         unit="img") as pbar:
        futures = [
            pool.submit(_process_image, block, fetch=fetch, raw_path=raw_path,
                        vlm=vlm, vlm_classify=vlm_classify, llm=llm,
                        doc_id=doc.doc_id, fmt=doc.fmt, raw_sink=raw_sink,
                        classify_only=classify_only)
            for block in images
        ]
        for f in as_completed(futures):
            f.result()
            pbar.update()

    return doc


def apply_ocr_checks(doc: ParsedDocument, raw_ocr: dict[str, str],
                     llm: LLMClient | None = None) -> ParsedDocument:
    """The deferred LLM half of the OCR pipeline: judge + clean the raw VLM
    transcriptions a `handle_images(..., raw_sink=...)` call parked earlier
    (possibly in a previous process -- `raw_ocr` round-trips through JSON).

    Keys are `ImageBlock.marker` (`<imageN>` -- unique per document, stable
    across IR serialization). Mutates `doc` in place and returns it.
    Idempotent: a block whose `ocr_meaningful` is already set is skipped, so
    re-running after a partial failure only pays for what's still pending.
    """
    by_marker = {b.marker: b for b in _collect_images(doc)}
    pending = [(by_marker[m], text) for m, text in raw_ocr.items()
               if text and m in by_marker and by_marker[m].ocr_meaningful is None]
    if not pending:
        return doc

    if llm is None:
        try:
            llm = get_client()
        except LLMError:
            return doc

    def _check_one(block: ImageBlock, raw_text: str) -> None:
        _apply_verdict(block, llm, raw_text)

    max_workers = max(1, get_config().image_ocr.concurrency)
    with ThreadPoolExecutor(max_workers=max_workers) as pool, \
            progress.bar(len(pending), f"{Path(doc.source_path).name}: OCR check",
                         unit="img") as pbar:
        futures = [pool.submit(_check_one, block, text)
                   for block, text in pending]
        for f in as_completed(futures):
            f.result()
            pbar.update()

    return doc
