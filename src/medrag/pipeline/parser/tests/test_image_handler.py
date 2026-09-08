"""images/image_handler.py tests: locator -> bytes fetchers, blob store
dedup, nested table-cell collection, and the meaningful/meaningless OCR
paths. No network: VLM/LLM are fakes injected as arguments; a remote
http(s) src is asserted to stay unresolved rather than fetched.
"""

from __future__ import annotations

import base64
import zipfile

import pytest

from medrag.pipeline.parser.images.image_handler import (
    _collect_images,
    _fetch_pdf_region,
    _fetch_source_crop,
    _fetch_src,
    _fetch_zip_part,
    _store_blob,
    apply_ocr_checks,
    handle_images,
)
from medrag.pipeline.parser.parsers.base import (
    Cell,
    ImageBlock,
    ParagraphBlock,
    ParsedDocument,
    TableBlock,
    TableData,
    hash_hex,
)

_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


@pytest.fixture(autouse=True)
def _isolate_labels_dir(tmp_path, monkeypatch):
    """Every test here now potentially exercises the visual-type classify
    step (image_handler.py calls it for every image, see A.5) -- isolate its
    disk cache so no test run ever writes into the real repo's storage/labels/."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls.append({"system": system, "n_images": len(images)})
        return self.text


class FakeLLM:
    def __init__(self, verdict_json: str) -> None:
        self.verdict_json = verdict_json
        self.calls = 0

    def complete(self, *, system, user, max_tokens=1024):
        self.calls += 1
        return self.verdict_json


# ---------------------------------------------------------------------------
# Minimal one-page PDF builder (same low-level shape as test_pdf_parser.py)
# ---------------------------------------------------------------------------


def _build_pdf(objs: list[bytes]) -> bytes:
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF").encode()
    return out


def _blank_pdf(width: int = 200, height: int = 100) -> bytes:
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        f"/Contents 4 0 R >>".encode(),
        (b"<< /Length 0 >>\nstream\n\nendstream"),
    ])


# ---------------------------------------------------------------------------
# Blob store
# ---------------------------------------------------------------------------


def test_store_blob_dedups_by_sha256(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    sha1 = _store_blob(_PNG_1PX, "image/png")
    sha2 = _store_blob(_PNG_1PX, "image/png")
    assert sha1 == sha2
    # The field value is `sha256:<hex>`, the filename the bare hex.
    assert sha1.startswith("sha256:")
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].name == f"{hash_hex(sha1)}.png"


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


def test_fetch_zip_part_reads_media_from_zip(tmp_path):
    zpath = tmp_path / "doc.docx"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("word/media/image1.png", _PNG_1PX)

    result = _fetch_zip_part(zpath, {"part": "word/media/image1.png"})
    assert result is not None
    data, mime = result
    assert data == _PNG_1PX
    assert mime == "image/png"


def test_fetch_zip_part_missing_member_returns_none(tmp_path):
    zpath = tmp_path / "doc.docx"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("word/media/other.png", _PNG_1PX)
    assert _fetch_zip_part(zpath, {"part": "word/media/image1.png"}) is None


def test_fetch_src_data_uri(tmp_path):
    encoded = base64.b64encode(_PNG_1PX).decode()
    locator = {"src": f"data:image/png;base64,{encoded}"}
    data, mime = _fetch_src(tmp_path / "doc.md", locator)
    assert data == _PNG_1PX
    assert mime == "image/png"


def test_fetch_src_local_file_relative_to_source(tmp_path):
    (tmp_path / "img.png").write_bytes(_PNG_1PX)
    data, mime = _fetch_src(tmp_path / "doc.md", {"src": "img.png"})
    assert data == _PNG_1PX
    assert mime == "image/png"


def test_fetch_src_remote_url_left_unresolved(tmp_path):
    assert _fetch_src(tmp_path / "doc.md",
                       {"src": "https://example.com/x.png"}) is None


def test_fetch_pdf_region_crops_bbox_scaled_by_dpi(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_RENDER_DPI", "72")  # 1 point == 1 pixel
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_blank_pdf(width=200, height=100))

    data, mime = _fetch_pdf_region(
        pdf_path, {"page": 1, "bbox": [10, 20, 60, 70]}, {})
    assert mime == "image/png"
    import io

    from PIL import Image
    img = Image.open(io.BytesIO(data))
    assert img.size == (50, 50)


def test_fetch_source_crop_reads_existing_blob(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    sha = _store_blob(_PNG_1PX, "image/png")
    data, mime = _fetch_source_crop(sha)
    assert data == _PNG_1PX
    assert mime == "image/png"


def test_fetch_source_crop_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    assert _fetch_source_crop("does-not-exist") is None


# ---------------------------------------------------------------------------
# Recursive collection (including table-cell images)
# ---------------------------------------------------------------------------


def test_collect_images_includes_nested_table_cell_images():
    body_img = ImageBlock(id="b0", image_index=1, locator={"src": "a.png"})
    cell_img = ImageBlock(id="b1", image_index=2, locator={"src": "b.png"})
    table = TableBlock(
        id="t0",
        table=TableData(n_rows=1, n_cols=1,
                         cells=[[Cell(blocks=[cell_img])]]))
    doc = ParsedDocument(doc_id="d", source_path="x.md", fmt="markdown",
                         blocks=[body_img, table])
    found = _collect_images(doc)
    assert found == [body_img, cell_img]


# ---------------------------------------------------------------------------
# handle_images: end-to-end with fakes
# ---------------------------------------------------------------------------


def _doc_with_data_uri_image() -> ParsedDocument:
    encoded = base64.b64encode(_PNG_1PX).decode()
    img = ImageBlock(id="b0", image_index=1,
                      locator={"src": f"data:image/png;base64,{encoded}"})
    return ParsedDocument(doc_id="d", source_path="unused.md", fmt="markdown",
                         blocks=[img])


class DispatchVLM:
    """Answers the classify call and the OCR call differently, dispatching
    on the system prompt -- image_handler.py's own classify prompt always
    mentions "classify" (see visual_classify.py), the OCR prompt never does."""

    def __init__(self, classify_json: str, ocr_text: str) -> None:
        self.classify_json = classify_json
        self.ocr_text = ocr_text
        self.calls: list[str] = []  # one entry per call: "classify" | "ocr"

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        if "classify" in system.lower():
            self.calls.append("classify")
            return self.classify_json
        self.calls.append("ocr")
        return self.ocr_text


def test_handle_images_classifies_and_stores_visual_type(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "product_photo", "confidence": 0.85}', "some text")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "some text", "confidence": 0.9}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.visual_type == "product_photo"
    assert block.visual_type_source == "vlm"
    assert block.excluded_at_parse is False
    # not excluded (default exclude_types=[]) -> OCR still ran normally
    assert block.ocr_text == "some text"
    assert vlm.calls == ["classify", "ocr"]


def test_handle_images_excluded_type_skips_ocr(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    ov = tmp_path / "visual_override.toml"
    ov.write_text('[visual]\nexclude_types = ["decorative"]\n', encoding="utf-8")
    monkeypatch.setenv("PARSER_CONFIG", str(ov))

    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "decorative", "confidence": 0.9}', "should not run")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "should not run"}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.visual_type == "decorative"
    assert block.excluded_at_parse is True
    assert block.ocr_text is None
    assert block.ocr_meaningful is None  # never touched -- OCR was skipped
    assert vlm.calls == ["classify"]     # no OCR call at all
    assert llm.calls == 0
    # the blob is still stored (mechanical, needed for a chunk-time
    # placeholder's path) even though no content was generated for it.
    assert block.image_id is not None


def test_image_ocr_disabled_skips_ocr_but_keeps_classification(tmp_path, monkeypatch):
    """`IMAGE_OCR_ENABLED=0` skips the OCR step UNCONDITIONALLY. Two differences
    from the `exclude_types` path, both deliberate:
      1. it does NOT require `[visual] classify` to be on,
      2. it does NOT mark `excluded_at_parse` -- the block was not left out, only
         its OCR was skipped in this run.
    Classification is a separate switch, so it keeps working."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_OCR_ENABLED", "0")

    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "product_photo", "confidence": 0.85}', "should not run")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "should not run"}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.visual_type == "product_photo"   # classification is a separate switch
    assert block.excluded_at_parse is False       # NOT left out
    assert block.ocr_text is None
    assert block.ocr_meaningful is None
    assert vlm.calls == ["classify"]              # NO OCR call at all
    assert llm.calls == 0
    assert block.image_id is not None             # the blob is still stored


def test_image_ocr_disabled_works_with_classify_off(tmp_path, monkeypatch):
    """What `exclude_types` cannot do: stop OCR while `classify` is OFF. The
    only path it offered depended on the `cfg_v.classify and ...` condition, so
    turning classification off brought OCR back -- the user's request to "turn
    all four off at once" could not be met with a single switch that way."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_OCR_ENABLED", "0")
    monkeypatch.setenv("VISUAL_CLASSIFY", "0")

    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "product_photo", "confidence": 0.85}', "should not run")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "should not run"}')

    handle_images(doc, vlm=vlm, llm=llm)

    assert doc.blocks[0].ocr_text is None
    assert vlm.calls == []   # neither classification nor OCR -- no VLM call
    assert llm.calls == 0


def test_handle_images_prestamped_visual_type_not_reclassified(tmp_path, monkeypatch):
    """A block that arrives already classified (e.g. one of pdf_parser.py's
    A.4.1-reclassified table candidates) is never sent through classify_crop
    a second time -- single source of truth for the label."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    encoded = base64.b64encode(_PNG_1PX).decode()
    img = ImageBlock(id="b0", image_index=1,
                     locator={"src": f"data:image/png;base64,{encoded}"},
                     visual_type="block_diagram", visual_type_source="vlm")
    doc = ParsedDocument(doc_id="d", source_path="unused.md", fmt="markdown",
                        blocks=[img])
    vlm = DispatchVLM('{"type": "product_photo", "confidence": 0.9}', "some text")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "some text"}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.visual_type == "block_diagram"  # unchanged, not overwritten
    assert vlm.calls == ["ocr"]  # classify never called again


def test_handle_images_meaningful_ocr_fills_ocr_text(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("Reveneu Q3: 42")
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "Revenue Q3: 42", "confidence": 0.9}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.image_id == _store_blob(_PNG_1PX, "image/png")
    assert block.mime == "image/png"
    assert block.width == 1 and block.height == 1
    assert block.ocr_meaningful is True
    assert block.ocr_text == "Revenue Q3: 42"
    assert len(vlm.calls) == 2  # classify (A.5) + OCR transcription


def test_handle_images_no_text_skips_llm_check(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("NO_TEXT")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "should not be used"}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.image_id is not None  # blob stored regardless of OCR outcome
    assert block.ocr_meaningful is False
    assert block.ocr_text is None
    assert llm.calls == 0


def test_handle_images_meaningless_ocr_keeps_blob_drops_text(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("l0go inc.")
    llm = FakeLLM('{"meaningful": false, "cleaned_text": ""}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.image_id is not None
    assert block.ocr_meaningful is False
    assert block.ocr_text is None


def test_handle_images_stores_confidence_and_accepts_above_threshold(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_OCR_CONFIDENCE_THRESHOLD", "0.7")
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("Revenue Q3: 42")
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "Revenue Q3: 42", "confidence": 0.9}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.ocr_meaningful is True
    assert block.ocr_text == "Revenue Q3: 42"
    assert block.ocr_confidence == 0.9


def test_handle_images_withholds_text_below_confidence_threshold(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_OCR_CONFIDENCE_THRESHOLD", "0.7")
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("blurry photo text")
    # meaningful, but the reviewer isn't confident it read the pixels right
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "blurry photo text", "confidence": 0.4}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.image_id is not None            # blob kept regardless
    assert block.ocr_meaningful is False         # gated out by low confidence
    assert block.ocr_text is None                # text withheld
    assert block.ocr_confidence == 0.4           # but the score is recorded


def test_handle_images_withholds_text_when_no_confidence_reported(
        tmp_path, monkeypatch):
    # Fail-closed: a meaningful verdict with no confidence field at all is
    # still withheld -- an unquantified image transcription isn't trusted.
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "unquantified"}')

    handle_images(doc, vlm=FakeVLM("unquantified"), llm=llm)

    block = doc.blocks[0]
    assert block.image_id is not None            # blob kept regardless
    assert block.ocr_meaningful is False         # withheld, fail-closed
    assert block.ocr_text is None
    assert block.ocr_confidence is None


def test_confidence_threshold_is_configurable_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_OCR_CONFIDENCE_THRESHOLD", "0.3")  # lenient
    doc = _doc_with_data_uri_image()
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "borderline", "confidence": 0.4}')

    handle_images(doc, vlm=FakeVLM("borderline"), llm=llm)

    block = doc.blocks[0]
    assert block.ocr_meaningful is True          # 0.4 >= 0.3 now clears the bar
    assert block.ocr_text == "borderline"


def test_handle_images_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("some text")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "some text"}')

    handle_images(doc, vlm=vlm, llm=llm)
    handle_images(doc, vlm=vlm, llm=llm)  # already resolved -> no-op

    assert len(vlm.calls) == 2  # classify (A.5) + OCR, only from the FIRST call


def test_handle_images_falls_back_to_source_crop_for_unpaired_pdf_figure(
        tmp_path, monkeypatch):
    """An unverified figure whose locator has no bbox (pdf_parser.py's
    unpaired-figure case) can't be fetched by _fetch_pdf_region -- but the
    parser already rendered and stored an audit crop for it (source_crop),
    in this same blob store. This should resolve from that crop instead of
    being left as a permanently empty ImageBlock."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    crop_sha = _store_blob(_PNG_1PX, "image/png")

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_blank_pdf())
    img = ImageBlock(id="b0", image_index=1,
                      locator={"page": 1, "region": "vlm_figure"},
                      provenance="unverified", source_crop=crop_sha)
    doc = ParsedDocument(doc_id="d", source_path=str(pdf_path), fmt="pdf",
                         blocks=[img])
    vlm = FakeVLM("Pin 1: VCC")
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "Pin 1: VCC", "confidence": 0.9}')

    handle_images(doc, vlm=vlm, llm=llm)

    block = doc.blocks[0]
    assert block.image_id == crop_sha
    assert block.ocr_meaningful is True
    assert block.ocr_text == "Pin 1: VCC"


def test_handle_images_no_source_crop_leaves_unpaired_figure_unresolved(
        tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(_blank_pdf())
    img = ImageBlock(id="b0", image_index=1,
                      locator={"page": 1, "region": "vlm_figure"},
                      provenance="unverified")
    doc = ParsedDocument(doc_id="d", source_path=str(pdf_path), fmt="pdf",
                         blocks=[img])

    handle_images(doc, vlm=FakeVLM("text"), llm=FakeLLM("{}"))

    assert doc.blocks[0].image_id is None


def test_handle_images_noop_without_images():
    doc = ParsedDocument(doc_id="d", source_path="x.md", fmt="markdown",
                         blocks=[ParagraphBlock(id="b0", text="hello")])
    # would raise if it tried to build a real VLM/LLM client
    handle_images(doc)


def test_handle_images_roundtrip_json(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    handle_images(doc, vlm=FakeVLM("NO_TEXT"))
    restored = ParsedDocument.from_json(doc.to_json())
    assert restored.to_dict() == doc.to_dict()


# ---------------------------------------------------------------------------
# classify_only (three-phase model-affinity split: VLM_CLASSIFY pre-warm
# pass, run before the VLM-OCR / LLM-check phases below)
# ---------------------------------------------------------------------------


def test_classify_only_stops_before_ocr_and_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "product_photo", "confidence": 0.85}', "should not run")
    llm = FakeLLM('{"meaningful": true, "cleaned_text": "should not run"}')

    handle_images(doc, vlm=vlm, llm=llm, classify_only=True)

    block = doc.blocks[0]
    assert block.visual_type == "product_photo"
    assert block.image_id is not None            # fetch still happens
    assert vlm.calls == ["classify"]              # OCR never called
    assert llm.calls == 0                         # LLM never touched
    assert block.ocr_text is None
    assert block.ocr_meaningful is None


def test_classify_only_respects_exclude_types(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    ov = tmp_path / "visual_override.toml"
    ov.write_text('[visual]\nexclude_types = ["decorative"]\n', encoding="utf-8")
    monkeypatch.setenv("PARSER_CONFIG", str(ov))
    doc = _doc_with_data_uri_image()
    vlm = DispatchVLM('{"type": "decorative", "confidence": 0.9}', "should not run")

    handle_images(doc, vlm=vlm, classify_only=True)

    block = doc.blocks[0]
    assert block.excluded_at_parse is True
    assert vlm.calls == ["classify"]


def test_classify_only_then_ocr_pass_never_reclassifies(tmp_path, monkeypatch):
    """The exact sequence run_parse_pipeline.py's phase 0 -> phase 1 uses: a
    classify_only pass over one document's `parse()` output, then a LATER,
    SEPARATE `parse()` + raw_sink (OCR-only) pass over the same source file
    (phase 1 always re-parses from scratch -- see phase1_one). The two
    passes share nothing but the on-disk classify_crop cache (keyed by crop
    sha256, same STORAGE_LABELS_DIR here) -- that cache is what must save the
    second pass from calling VLM_CLASSIFY again."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc_phase0 = _doc_with_data_uri_image()
    vlm_phase0 = DispatchVLM('{"type": "product_photo", "confidence": 0.85}', "unused")
    handle_images(doc_phase0, vlm=vlm_phase0, classify_only=True)
    assert vlm_phase0.calls == ["classify"]

    # A brand-new ParsedDocument/ImageBlock (image_id=None, visual_type=None
    # again) with byte-identical image content -> same crop sha256 -> same
    # cache entry phase 0 just wrote.
    doc_phase1 = _doc_with_data_uri_image()
    vlm_phase1 = DispatchVLM('{"type": "product_photo", "confidence": 0.85}',
                            "Reveneu Q3: 42")
    sink: dict[str, str] = {}
    handle_images(doc_phase1, vlm=vlm_phase1, raw_sink=sink)

    block = doc_phase1.blocks[0]
    assert vlm_phase1.calls == ["ocr"]              # cache hit -> no "classify" call
    assert sink == {block.marker: "Reveneu Q3: 42"}
    assert block.visual_type == "product_photo"     # from the cache, not this VLM


def test_classify_only_builds_no_llm_client(tmp_path, monkeypatch):
    """Would raise if classify_only tried to build a real LLM client (none
    configured here) -- classify_only must never touch the LLM at all, same
    guarantee raw_sink already gives for OCR's deferred verdict check."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    handle_images(doc, vlm=FakeVLM('{"type": "product_photo", "confidence": 0.9}'),
                 classify_only=True)
    assert doc.blocks[0].visual_type == "product_photo"


# ---------------------------------------------------------------------------
# deferred OCR check (model-affinity split: VLM half now, LLM half later)
# ---------------------------------------------------------------------------


def test_raw_sink_defers_llm_check(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    vlm = FakeVLM("Reveneu Q3: 42")
    sink: dict[str, str] = {}

    # would raise if it tried to build a real LLM client (none configured):
    # deferred mode must never touch the LLM side at all
    handle_images(doc, vlm=vlm, raw_sink=sink)

    block = doc.blocks[0]
    assert block.image_id is not None            # VLM half fully done
    assert sink == {block.marker: "Reveneu Q3: 42"}
    assert block.ocr_meaningful is None          # verdict still pending


def test_raw_sink_empty_ocr_settled_without_llm(tmp_path, monkeypatch):
    # NO_TEXT needs no verdict -- settled in the VLM half, nothing parked.
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    sink: dict[str, str] = {}
    handle_images(doc, vlm=FakeVLM("NO_TEXT"), raw_sink=sink)
    assert sink == {}
    assert doc.blocks[0].ocr_meaningful is False


def test_apply_ocr_checks_finishes_after_json_roundtrip(tmp_path, monkeypatch):
    """The two halves run in different processes in the phased pipeline: the
    doc and the sink both round-trip through JSON between them."""
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path))
    doc = _doc_with_data_uri_image()
    sink: dict[str, str] = {}
    handle_images(doc, vlm=FakeVLM("Reveneu Q3: 42"), raw_sink=sink)

    restored = ParsedDocument.from_json(doc.to_json())
    llm = FakeLLM(
        '{"meaningful": true, "cleaned_text": "Revenue Q3: 42", "confidence": 0.9}')
    apply_ocr_checks(restored, dict(sink), llm=llm)

    block = restored.blocks[0]
    assert block.ocr_meaningful is True
    assert block.ocr_text == "Revenue Q3: 42"
    assert llm.calls == 1


def test_apply_ocr_checks_is_idempotent_and_skips_unknown_markers():
    img = ImageBlock(id="b0", image_index=1, locator={}, image_id="sha",
                     ocr_meaningful=True, ocr_text="done")
    doc = ParsedDocument(doc_id="d", source_path="x.md", fmt="markdown",
                         blocks=[img])
    llm = FakeLLM('{"meaningful": false, "cleaned_text": ""}')
    # already-verdicted marker + a marker no block has: neither reaches the LLM
    apply_ocr_checks(doc, {img.marker: "done", "<image9>": "ghost"}, llm=llm)
    assert llm.calls == 0
    assert img.ocr_text == "done"
