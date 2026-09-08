"""pdf parser tests: triage routing, code path, hybrid/scanned verification.

No network and no real models: PDFs are tiny hand-built byte streams, the VLM
and the text detector are fakes injected through PdfParser's constructor.
"""

from __future__ import annotations

import io
import json

import pytest

from medrag.pipeline.parser.parsers.base import (
    HeadingBlock,
    ImageBlock,
    ParagraphBlock,
    Span,
    TableBlock,
    hash_hex,
)
from medrag.pipeline.parser.parsers.pdf_parser import (
    PDFIUM_LOCK,
    PROV_CONSENSUS,
    PROV_TABLE_BANDS,
    PROV_TEXT_LAYER,
    PROV_UNVERIFIED,
    DetectedLine,
    PdfParser,
    _containment,
    _Counters,
    _criticals,
    _find_gutter,
    _is_hard_number,
    _merge_unpaired_tables,
    _norm,
    _parse_vlm_blocks,
    _PreparedTable,
    _recover_dropped_headings,
    _tok_list,
)


@pytest.fixture(autouse=True)
def _isolate_labels_dir(tmp_path, monkeypatch):
    """A page with real ruled-table geometry and an injected (non-None) VLM
    now also runs the visual-type classify step (A.4.1) -- isolate its disk
    cache for every test in this file so none of them writes into the real
    repo's storage/labels/, regardless of whether that specific test means
    to exercise classification at all."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))


# ---------------------------------------------------------------------------
# Minimal PDF builders (no third-party writer; enough for pdfplumber/pdfium)
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


def _stream(head: bytes, data: bytes) -> bytes:
    return head + f" /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def text_pdf(lines: list[tuple[float, float, float, str]]) -> bytes:
    """One-page born-digital PDF. lines: (x, top_from_page_top, size, text)."""
    content = b""
    for x, top, size, text in lines:
        y = 792 - top - size  # PDF origin is bottom-left
        esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content += f"BT /F1 {size} Tf {x} {y} Td ({esc}) Tj ET\n".encode("latin-1")
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", content),
    ])


def scanned_pdf() -> bytes:
    """One-page PDF whose only content is a page-filling raster image."""
    w = h = 50
    pixels = bytes([180, 180, 180]) * (w * h)
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        _stream(b"<< /Type /XObject /Subtype /Image /Width 50 /Height 50 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8", pixels),
        _stream(b"<<", b"q 612 0 0 792 0 0 cm /Im1 Do Q"),
    ])


def scanned_pdf_with_subfigure() -> bytes:
    """Same page-filling scan, plus one distinct, non-page-spanning raster
    (e.g. a photo printed on the scanned page) drawn on top of it."""
    bg = bytes([180, 180, 180]) * (50 * 50)
    fg = bytes([90, 60, 30]) * (10 * 10)
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im1 4 0 R /Im2 5 0 R >> >> "
            b"/Contents 6 0 R >>"
        ),
        _stream(b"<< /Type /XObject /Subtype /Image /Width 50 /Height 50 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8", bg),
        _stream(b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8", fg),
        _stream(b"<<", b"q 612 0 0 792 0 0 cm /Im1 Do Q\n"
                        b"q 100 0 0 80 250 350 cm /Im2 Do Q"),
    ])


def vector_ink_pdf(n_rects: int = 60) -> bytes:
    """One-page PDF with no text layer and no rasters, only vector paths --
    the shape of a page whose text was flattened to glyph outlines (SLSC
    catalogue) or of a technical drawing's linework."""
    content = b""
    for i in range(n_rects):
        x = 40 + (i % 10) * 50
        y = 700 - (i // 10) * 30
        content += f"{x} {y} 8 12 re f\n".encode()
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R >>"
        ),
        _stream(b"<<", content),
    ])


def tiled_images_pdf() -> bytes:
    """One-page PDF with no text layer and several small rasters, none of
    which alone covers 20% of the page but which together do -- the SLSC
    p.1 layout that the old max-based img_frac misrouted to "empty"."""
    w = h = 10
    pixels = bytes([120, 120, 120]) * (w * h)
    img = _stream(b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 "
                  b"/ColorSpace /DeviceRGB /BitsPerComponent 8", pixels)
    # four tiles of ~7.5% each => max 0.075 < 0.2, sum 0.30 >= 0.2
    content = b""
    for i, (x, y) in enumerate([(30, 500), (330, 500), (30, 200), (330, 200)]):
        content += f"q 250 0 0 145 {x} {y} cm /Im1 Do Q\n".encode()
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        img,
        _stream(b"<<", content),
    ])


def mixed_font_pdf(segments: list[tuple[str, float, float, str]]) -> bytes:
    """One-page PDF where each segment is (font, x, top, text), each drawn in
    its own absolute BT/ET block so multiple fonts can land on one visual
    line -- used to exercise font-based bold/italic run detection."""
    size = 11.0
    content = b""
    for font, x, top, text in segments:
        y = 792 - top - size
        esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content += f"BT /{font} {size} Tf {x} {y} Td ({esc}) Tj ET\n".encode("latin-1")
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R /F2 6 0 R /F3 7 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>",
    ])


TWO_COL_LEFT = [f"left column line {i} alpha beta gamma" for i in range(1, 9)]
TWO_COL_RIGHT = [f"right column line {i} delta epsilon zeta" for i in range(1, 9)]


def two_column_pdf() -> bytes:
    lines = []
    for i, text in enumerate(TWO_COL_LEFT):
        lines.append((50.0, 100.0 + 20 * i, 10.0, text))
    for i, text in enumerate(TWO_COL_RIGHT):
        lines.append((340.0, 100.0 + 20 * i, 10.0, text))
    return lines and text_pdf(lines)


def two_column_with_image_pdf() -> bytes:
    """Same two-column layout, plus one real embedded raster near the top of
    the page (well above the text so it never overlaps a column)."""
    content = b""
    for x, top, size, text in (
        [(50.0, 100.0 + 20 * i, 10.0, t) for i, t in enumerate(TWO_COL_LEFT)]
        + [(340.0, 100.0 + 20 * i, 10.0, t) for i, t in enumerate(TWO_COL_RIGHT)]
    ):
        y = 792 - top - size
        esc = text_escape(text)
        content += f"BT /F1 {size} Tf {x} {y} Td ({esc}) Tj ET\n".encode("latin-1")
    content += b"q 100 0 0 30 250 750 cm /Im1 Do Q\n"
    pixels = bytes([200, 150, 100]) * (10 * 10)
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> /XObject << /Im1 5 0 R >> >> "
            b"/Contents 6 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8", pixels),
        _stream(b"<<", content),
    ])


def two_page_pdf_mixed_heading_sizes() -> bytes:
    """Page 1 has a size-18 heading (the document's biggest) plus size-11
    body lines; page 2 has only a size-14 heading plus size-11 body lines --
    big enough to be the biggest thing on its own page, but smaller than page
    1's, so document-wide heading-level ranking must place it at level 2,
    not level 1 (which page-local ranking would give it)."""
    def page_content(heading_size: float, heading_text: str) -> bytes:
        content = b""
        y = 792 - 80 - heading_size
        content += (f"BT /F1 {heading_size} Tf 72 {y} Td "
                    f"({text_escape(heading_text)}) Tj ET\n").encode("latin-1")
        for i, body in enumerate(["body line one here",
                                  "body line two here",
                                  "body line three here"]):
            top = 140.0 + 20 * i
            y = 792 - top - 11
            content += (f"BT /F1 11 Tf 72 {y} Td "
                        f"({text_escape(body)}) Tj ET\n").encode("latin-1")
        return content

    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 6 0 R >> >> /Contents 5 0 R >>"
        ),
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>"
        ),
        _stream(b"<<", page_content(18.0, "Big Heading")),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", page_content(14.0, "Small Heading")),
    ])


def two_column_pdf_with_bold_word() -> bytes:
    """Same two-column layout as two_column_pdf(), but "delta" in the first
    right-column line is rendered in Helvetica-Bold -- exercises hybrid-path
    run backfill against the page's own text layer. Kept clear of the left
    column's max extent so gutter detection is unaffected."""
    content = b""
    size = 10.0
    for i, text in enumerate(TWO_COL_LEFT):
        top = 100.0 + 20 * i
        y = 792 - top - size
        esc = text_escape(text)
        content += f"BT /F1 {size} Tf 50 {y} Td ({esc}) Tj ET\n".encode("latin-1")
    y0 = 792 - 100.0 - size
    for font, x, text in (
        ("F1", 340.0, "right column line 1"),
        ("F2", 460.0, "delta"),
        ("F1", 500.0, "epsilon zeta"),
    ):
        esc = text_escape(text)
        content += f"BT /{font} {size} Tf {x} {y0} Td ({esc}) Tj ET\n".encode("latin-1")
    for i, text in enumerate(TWO_COL_RIGHT[1:], start=1):
        top = 100.0 + 20 * i
        y = 792 - top - size
        esc = text_escape(text)
        content += f"BT /F1 {size} Tf 340 {y} Td ({esc}) Tj ET\n".encode("latin-1")
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R /F2 6 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", content),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ])


def text_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeVLM:
    """Returns a canned payload; records how it was called."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls.append({"system": system, "user": user,
                           "n_images": len(images), "images": images})
        return self.payload


class FakeLLM:
    """Returns a canned text payload; records call count."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, *, system, user, max_tokens=1024):
        self.calls += 1
        return self.payload


class FakeDetector:
    def __init__(self, lines: list[DetectedLine]) -> None:
        self.lines = lines

    def detect(self, png: bytes) -> list[DetectedLine]:
        return self.lines


def offline_parser(**kw) -> PdfParser:
    """PdfParser with every model/detector explicitly disabled unless given."""
    kw.setdefault("vlm", None)
    kw.setdefault("vlm2", None)
    kw.setdefault("detector", None)
    return PdfParser(**kw)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_containment_and_criticals():
    hay = _tok_list("Part 99-1234 ships on 2026-07-04 for 500 USD.")
    from collections import Counter
    assert _containment(_tok_list("ships on 2026-07-04"), Counter(hay)) == 1.0
    assert _containment(_tok_list("totally absent words"), Counter(hay)) == 0.0
    crits = _criticals("Part 99-1234 for 500 USD")
    assert "99-1234" in crits and "500" in crits and "part" not in crits


def test_norm_folds_glyph_variants():
    # Dash variants (en/em/horizontal-bar/minus) all fold to ASCII hyphen; the
    # ASCII hyphen itself is untouched so real values keep their punctuation.
    assert _norm("5 V ―") == "5 v -"      # horizontal bar (PN1015 tables)
    assert _norm("a – b") == "a - b"       # en dash
    assert _norm("99-1234") == "99-1234"        # ASCII hyphen preserved
    # Super/subscript digits fold to ASCII (footnote marker "¹" -> "1",
    # so a VLM reading it as "1" matches the PDF's superscript).
    assert _norm("1 A ¹") == "1 a 1"
    assert _norm("H₂O") == "h2o"
    # Non-breaking space normalizes like a normal space.
    assert _norm("5 V") == "5 v"


def test_is_hard_number():
    # Substantial numeric tokens (>=2 chars with a digit) are hard rejects when
    # invented; a lone digit or folded footnote marker is not.
    assert _is_hard_number("6V")
    assert _is_hard_number("264VAC")
    assert _is_hard_number("2026")
    assert not _is_hard_number("1")        # lone digit / footnote marker
    assert not _is_hard_number("A")        # no digit
    assert not _is_hard_number("-")        # punctuation only


def test_tok_list_drops_pure_symbol_tokens():
    # Two independent transcriptions of the same formula, differing only in
    # operator glyphs (→ vs ->, − vs -): symbol-only tokens must not count as
    # content, or short formula lines spuriously fail containment even though
    # every letter/digit agrees (see corpus_pdf/LecNotes.pdf page 13).
    from collections import Counter
    a = _tok_list("w = (24 − 11) / 5 = 2.6, always rounding up → w = 3")
    b = _tok_list("w = (24 - 11) / 5 = 2.6, always rounding up -> w = 3")
    assert "−" not in a and "→" not in a and "=" not in a and "/" not in a
    assert _containment(a, Counter(b)) == 1.0


def test_find_gutter_two_columns():
    words = []
    for row in range(20):
        for x in (40, 80, 120, 160):
            words.append({"x0": x, "x1": x + 30, "top": row * 12,
                          "bottom": row * 12 + 10})
        for x in (340, 380, 420, 460):
            words.append({"x0": x, "x1": x + 30, "top": row * 12,
                          "bottom": row * 12 + 10})
    gx = _find_gutter(words, 612.0)
    assert gx is not None and 200 < gx < 340


def test_find_gutter_single_column():
    words = [{"x0": 50 + (i % 8) * 60, "x1": 50 + (i % 8) * 60 + 50,
              "top": (i // 8) * 12, "bottom": (i // 8) * 12 + 10}
             for i in range(80)]
    assert _find_gutter(words, 612.0) is None


def test_parse_vlm_blocks_fenced_and_invalid():
    payload = [{"type": "heading", "level": 2, "text": "Title"},
               {"type": "paragraph", "text": "Body"},
               {"type": "table", "rows": [["a", "b"], ["c", None]]},
               {"type": "nonsense", "text": "x"}]
    raw = "```json\n" + json.dumps(payload) + "\n```"
    specs = _parse_vlm_blocks(raw)
    assert [s["type"] for s in specs] == ["heading", "paragraph", "table"]
    assert specs[2]["rows"][1] == ["c", ""]
    assert _parse_vlm_blocks("no json here") is None
    assert _parse_vlm_blocks("") is None


# ---------------------------------------------------------------------------
# Code path
# ---------------------------------------------------------------------------


def test_code_path_heading_paragraph_list(tmp_path):
    pdf = text_pdf([
        (72, 80, 18, "Document Title"),
        (72, 130, 11, "This is the first body paragraph of the document."),
        (72, 145, 11, "It continues on a second wrapped line here."),
        (72, 180, 11, "- first bullet item"),
        (72, 195, 11, "- second bullet item"),
    ])
    path = tmp_path / "simple.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "doc1")

    assert doc.fmt == "pdf" and doc.page_count == 1
    assert doc.metadata["pdf_pages"][0]["route"] == "code"

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert len(headings) == 1
    assert headings[0].text == "Document Title" and headings[0].level == 1
    assert headings[0].provenance == PROV_TEXT_LAYER
    assert headings[0].span.page == 1

    items = [b for b in doc.blocks if b.list_id is not None]
    assert len(items) == 2
    assert items[0].text == "first bullet item"
    assert items[0].list_id == items[1].list_id
    assert items[0].list_ordered is False

    paras = [b for b in doc.blocks
             if isinstance(b, ParagraphBlock) and b.list_id is None]
    assert len(paras) == 1  # the two wrapped lines merged into one paragraph
    assert "second wrapped line" in paras[0].text

    # ids are sequential in reading order
    assert [b.id for b in doc.blocks] == [f"b{i}" for i in range(len(doc.blocks))]


def test_code_path_bold_italic_runs_from_font_name(tmp_path):
    # F1=Helvetica (plain), F2=Helvetica-Bold, F3=Helvetica-Oblique, all on
    # one visual line -- deterministic from the PDF's own font names, same
    # idea as reading a docx run's rPr.
    pdf = mixed_font_pdf([
        ("F1", 72, 100, "plain"),
        ("F2", 140, 100, "boldword"),
        ("F3", 230, 100, "italicword"),
    ])
    path = tmp_path / "mixed.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "doc_runs")
    assert doc.metadata["pdf_pages"][0]["route"] == "code"

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert len(paras) == 1
    para = paras[0]
    assert para.text == "plain boldword italicword"
    assert "".join(r.text for r in para.runs) == para.text

    from medrag.pipeline.parser.parsers.base import Mark
    marks_by_word = {r.text.strip(): set(r.marks) for r in para.runs}
    assert marks_by_word["plain"] == set()
    assert marks_by_word["boldword"] == {Mark.BOLD}
    assert marks_by_word["italicword"] == {Mark.ITALIC}


def test_code_path_heading_level_ranked_document_wide(tmp_path):
    # Page 2's only heading (size 14) is locally the biggest thing on its own
    # page -- page-local ranking would call it level 1 -- but page 1 has a
    # bigger heading (size 18), so document-wide ranking must give page 2's
    # heading level 2.
    path = tmp_path / "twopage.pdf"
    path.write_bytes(two_page_pdf_mixed_heading_sizes())

    doc = offline_parser().parse(path, "doc_headings")
    assert doc.page_count == 2
    assert doc.metadata["pdf_pages"][0]["route"] == "code"
    assert doc.metadata["pdf_pages"][1]["route"] == "code"

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert len(headings) == 2
    big = next(h for h in headings if h.span.page == 1)
    small = next(h for h in headings if h.span.page == 2)
    assert big.text == "Big Heading" and big.level == 1
    assert small.text == "Small Heading" and small.level == 2


def test_heading_reconcile_on_by_default_produces_document_wide_ranking(tmp_path):
    # pdf_parser.py used to read PDF_HEADING_RECONCILE via a bare
    # os.environ.get() that bypassed parser/config.py's TOML+env system
    # entirely -- a PARSER_CONFIG override could never reach this check.
    # Locks the default (TOML `pdf.heading_reconcile = true`, no env set):
    # document-wide ranking runs and is surfaced in metadata.
    path = tmp_path / "twopage.pdf"
    path.write_bytes(two_page_pdf_mixed_heading_sizes())

    doc = offline_parser().parse(path, "doc_headings_on")

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    small = next(h for h in headings if h.span.page == 2)
    assert small.level == 2  # document-wide: page 1's bigger heading wins level 1
    assert "heading_reconcile" in doc.metadata


def test_heading_reconcile_off_skips_document_wide_ranking(tmp_path, monkeypatch):
    # Same fixture as the "on" test above, only PDF_HEADING_RECONCILE=0 --
    # must now read through get_config().pdf.heading_reconcile (env override
    # registry), not the old bare os.environ check. No heading_reconcile
    # stats are recorded, proving reconcile_headings() itself never ran
    # (the direct signal pdf_parser.py's own gate writes to metadata).
    monkeypatch.setenv("PDF_HEADING_RECONCILE", "0")
    path = tmp_path / "twopage.pdf"
    path.write_bytes(two_page_pdf_mixed_heading_sizes())

    doc = offline_parser().parse(path, "doc_headings_off")

    assert "heading_reconcile" not in doc.metadata


def test_code_path_numbered_subheading_at_bodylike_size_becomes_heading(tmp_path):
    # A deep numbered sub-heading is often set only a touch larger than body
    # text (here 10pt over a 9pt body, ratio 1.111) -- well under the 1.15x
    # margin an unnumbered heading needs, so on looks alone it would fall
    # through to _BULLET_RE's list-item branch and lose the very number that
    # marks it as a heading (see ACME_PN5057_User_Manual.pdf p.20). A bare
    # number with no trailing "." or ")" at the very same size bump (a
    # spec-badge caption, "5 V / 20 W power output") must NOT get the same
    # promotion -- it has no real section-numbering shape.
    pdf = text_pdf([
        (72, 60, 9, "This is regular body text for the description section here."),
        (72, 90, 9, "Another line of ordinary body prose sits here for padding."),
        (72, 120, 10, "5.4.2.1. Controlling Cells"),
        (72, 150, 9, "Body text describing controlling cells follows this heading."),
        (72, 180, 10, "5 V / 20 W power output"),
        (72, 210, 9, "Final body paragraph text closing out the page for the test."),
    ])
    path = tmp_path / "bodyheading.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "doc_bodyheading")

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["5.4.2.1. Controlling Cells"]

    badge = next(b for b in doc.blocks if "5 V" in (b.text or ""))
    assert isinstance(badge, ParagraphBlock)


def test_code_path_toc_numbered_variant_extracted_to_metadata(tmp_path):
    # "1.1. Key Features" is indented (x=90) relative to its top-level
    # siblings (x=72) -- same shape as heading_path nesting, mirrored into
    # the extracted entry's "level".
    pdf = text_pdf([
        (72, 60, 16, "Contents"),
        (72, 100, 11, "1. Description 1"),
        (90, 115, 11, "1.1. Key Features 1"),
        (72, 130, 11, "2. Hardware Overview 2"),
        (72, 160, 16, "1. Description"),
        (72, 200, 11, "Real body text for the description section."),
    ])
    path = tmp_path / "toc_numbered.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "doc_toc1")

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["Contents", "1. Description"]

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [p.text for p in paras] == ["Real body text for the description section."]
    assert all(b.list_id is None for b in doc.blocks)

    assert doc.metadata["toc"] == [
        {"title": "Description", "page": 1, "level": 1},
        {"title": "Key Features", "page": 1, "level": 2},
        {"title": "Hardware Overview", "page": 2, "level": 1},
    ]

    # ids stay sequential/gapless after the drop
    assert [b.id for b in doc.blocks] == [f"b{i}" for i in range(len(doc.blocks))]


def test_code_path_toc_dot_leader_variant_extracted_to_metadata(tmp_path):
    # No numbering marker at all -- title and page number are only separated
    # by a run of dot-leader characters, so these lines never touch
    # `_BULLET_RE`/list_id, only the ordinary paragraph branch.
    pdf = text_pdf([
        (72, 60, 16, "Contents"),
        (72, 100, 11, "DESCRIPTION " + "." * 60 + " 1"),
        (72, 125, 11, "HARDWARE OVERVIEW " + "." * 55 + " 2"),
        (72, 160, 16, "Description"),
        (72, 200, 11, "Real body text goes here for real."),
    ])
    path = tmp_path / "toc_dotleader.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "doc_toc2")

    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["Contents", "Description"]

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [p.text for p in paras] == ["Real body text goes here for real."]

    assert doc.metadata["toc"] == [
        {"title": "DESCRIPTION", "page": 1, "level": 1},
        {"title": "HARDWARE OVERVIEW", "page": 2, "level": 1},
    ]


def test_extract_toc_steps_over_stray_lines_inside_run():
    # A revision note pasted between TOC entries doesn't end the run: it is
    # stepped over (and kept as an ordinary block) while the entries around
    # it -- pages still non-decreasing -- all land in metadata.
    from medrag.pipeline.parser.parsers.pdf_parser import _extract_toc
    blocks = [
        HeadingBlock(id="h", text="Contents", level=1),
        ParagraphBlock(id="e1", text="Description 2"),
        ParagraphBlock(id="e2", text="Hardware Overview 3"),
        ParagraphBlock(id="s1", text="Revision No. 0 View Revision History"),
        ParagraphBlock(id="e3", text="Signal Connections 5"),
        ParagraphBlock(id="e4", text="Safety Guidelines 14"),
        ParagraphBlock(id="body", text="Real body text after the contents."),
    ]
    out, toc = _extract_toc(blocks)
    assert [e["title"] for e in toc] == [
        "Description", "Hardware Overview", "Signal Connections",
        "Safety Guidelines"]
    assert [b.id for b in out] == ["h", "s1", "body"]


def test_extract_toc_recognizes_localized_heading():
    from medrag.pipeline.parser.parsers.pdf_parser import _extract_toc
    blocks = [
        HeadingBlock(id="h", text="İçindekiler", level=1),
        ParagraphBlock(id="e1", text="Tanım 2"),
        ParagraphBlock(id="e2", text="Donanım 3"),
    ]
    _, toc = _extract_toc(blocks)
    assert [e["title"] for e in toc] == ["Tanım", "Donanım"]


def test_extract_toc_recovers_split_heading_and_bare_page_number():
    # A hybrid-route VLM reading a ruled (no dot-leader) printed TOC reads
    # title and page number as two separate blocks instead of one glued
    # paragraph -- title survives as a real numbered HeadingBlock (the VLM
    # keeps the number in .text), the page number lands in the very next
    # ParagraphBlock, alone. Mixed here with one glued-shape entry (e3) to
    # confirm a single contiguous run accepts both shapes, matching a TOC
    # that spans a code-route page then hybrid-route pages.
    from medrag.pipeline.parser.parsers.pdf_parser import _extract_toc
    blocks = [
        HeadingBlock(id="h", text="Contents", level=1),
        HeadingBlock(id="e1", text="5.4.2.5. Temperature Sensors", level=4),
        ParagraphBlock(id="n1", text="16"),
        HeadingBlock(id="e2", text="5.4.2.6. Fans", level=4),
        ParagraphBlock(id="n2", text="17"),
        # already number-stripped, as _BULLET_RE would leave it upstream:
        ParagraphBlock(id="e3", text="Configuration Settings 18"),
        ParagraphBlock(id="body", text="Real body text after the contents."),
    ]
    out, toc = _extract_toc(blocks)
    assert toc == [
        {"title": "Temperature Sensors", "page": 16, "level": 4},
        {"title": "Fans", "page": 17, "level": 4},
        {"title": "Configuration Settings", "page": 18, "level": 1},
    ]
    assert [b.id for b in out] == ["h", "body"]


def test_extract_toc_split_shape_rejects_unrelated_heading_and_number():
    # A genuine body heading followed by an unrelated one-line paragraph that
    # happens to be a bare number must NOT be swept into a fake TOC run --
    # the >=2-entries safety net still applies uniformly across both shapes.
    from medrag.pipeline.parser.parsers.pdf_parser import _extract_toc
    blocks = [
        HeadingBlock(id="h", text="Contents", level=1),
        HeadingBlock(id="e1", text="3. Signal Connections", level=1),
        ParagraphBlock(id="n1", text="7"),
        ParagraphBlock(id="body", text="Ordinary paragraph text follows."),
    ]
    out, toc = _extract_toc(blocks)
    assert toc == []
    assert [b.id for b in out] == ["h", "e1", "n1", "body"]


def test_multicolumn_without_vlm_falls_back_to_code(tmp_path):
    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())

    doc = offline_parser().parse(path, "doc2")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "hybrid" and meta["used"] == "code"
    texts = [b.text for b in doc.blocks if isinstance(b, ParagraphBlock)]
    joined = " ".join(texts)
    # left column content must come before right column content
    assert joined.index("left column line 1") < joined.index("right column line 1")
    assert all(b.provenance == PROV_TEXT_LAYER for b in doc.blocks)


def _ruled_grid_pdf() -> bytes:
    """One-page PDF with a real ruled 2-row x 1-col table (find_tables()
    detects it by line geometry) whose only cell text is a repeated short
    label -- the "AAF" false-positive shape: geometrically a table, but a
    classifier should read it as something else (a diagram, in this test)."""
    content = (
        b"1 w\n"
        b"60 702 m 300 702 l S\n"
        b"60 662 m 300 662 l S\n"
        b"60 622 m 300 622 l S\n"
        b"60 702 m 60 622 l S\n"
        b"300 702 m 300 622 l S\n"
        b"BT /F1 12 Tf 72 680 Td (AAF) Tj ET\n"
        b"BT /F1 12 Tf 72 640 Td (AAF) Tj ET\n"
    )
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", content),
    ])


def test_reclassified_table_candidate_becomes_image_not_table(tmp_path, monkeypatch):
    """End-to-end regression test for the "AAF" bug: a geometrically-detected
    grid whose cells repeat the same short label is classified as a
    block diagram and ends up as an ImageBlock in the final document --
    never a TableBlock -- and its own words ("AAF") do not leak back into
    the document as a stray ParagraphBlock (see pdf_parser.py's
    `_table_boxes`)."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(_ruled_grid_pdf())

    vlm = FakeVLM(json.dumps({"type": "block_diagram", "confidence": 0.9}))
    doc = offline_parser(vlm=vlm).parse(path, "doc_reclassified")

    assert not any(isinstance(b, TableBlock) for b in doc.blocks)
    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert images[0].visual_type == "block_diagram"
    assert images[0].visual_type_source == "vlm"
    assert images[0].source_crop

    paragraphs = [b.text for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert not any("AAF" in text for text in paragraphs)


# ---------------------------------------------------------------------------
# Hybrid path
# ---------------------------------------------------------------------------


def _hybrid_payload(extra: list[dict] | None = None) -> str:
    blocks = [{"type": "paragraph", "text": t} for t in TWO_COL_LEFT]
    blocks += [{"type": "paragraph", "text": t} for t in TWO_COL_RIGHT]
    return json.dumps(blocks + (extra or []))


def test_hybrid_verified_against_text_layer(tmp_path, monkeypatch):
    # Not testing figure-region classification here -- disabled so the fake
    # VLM (reused as VLM_CLASSIFY, since none is configured) isn't also hit
    # by _vlm_figure_regions' page-region call before the primary read.
    monkeypatch.setenv("VISUAL_CLASSIFY", "0")
    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())

    vlm = FakeVLM(_hybrid_payload())
    doc = offline_parser(vlm=vlm).parse(path, "doc3")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "hybrid" and meta["used"] == "hybrid"
    assert len(vlm.calls) == 1 and vlm.calls[0]["n_images"] == 1
    assert "TEXT LAYER:" in vlm.calls[0]["user"]  # grounding was passed

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert len(paras) == 16
    assert all(b.provenance == PROV_TEXT_LAYER for b in paras)
    assert all(b.source_crop is None for b in paras)


def test_hybrid_hallucination_is_unverified_with_crop(tmp_path, monkeypatch):
    crop_dir = tmp_path / "crops"
    monkeypatch.setenv("PDF_CROP_DIR", str(crop_dir))
    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())

    fake = {"type": "paragraph", "text": "Part 77-9999 costs 500 USD"}
    vlm = FakeVLM(_hybrid_payload([fake]))
    doc = offline_parser(vlm=vlm).parse(path, "doc4")

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    bad = [b for b in paras if "77-9999" in b.text]
    assert len(bad) == 1
    assert bad[0].provenance == PROV_UNVERIFIED
    assert bad[0].source_crop is not None
    assert (crop_dir / f"{hash_hex(bad[0].source_crop)}.png").exists()
    good = [b for b in paras if "77-9999" not in b.text]
    assert all(b.provenance == PROV_TEXT_LAYER for b in good)


def test_hybrid_bold_run_backfilled_from_text_layer(tmp_path):
    # "delta" is Helvetica-Bold in the PDF's own text layer; the VLM's
    # transcription carries no font info at all, but its wording matches the
    # text layer verbatim (PROV_TEXT_LAYER), so runs are backfilled from the
    # same deterministic font-name signal the code path reads directly.
    path = tmp_path / "twocol_bold.pdf"
    path.write_bytes(two_column_pdf_with_bold_word())

    vlm = FakeVLM(_hybrid_payload())
    doc = offline_parser(vlm=vlm).parse(path, "doc_hybrid_runs")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "hybrid" and meta["used"] == "hybrid"

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    target = next(b for b in paras if b.text == TWO_COL_RIGHT[0])
    assert target.provenance == PROV_TEXT_LAYER
    assert "".join(r.text for r in target.runs) == target.text

    from medrag.pipeline.parser.parsers.base import Mark
    bold_runs = [r for r in target.runs if Mark.BOLD in r.marks]
    assert len(bold_runs) == 1 and bold_runs[0].text.strip() == "delta"

    # Untouched paragraphs (including ones after "delta" in reading order)
    # still align cleanly -- the shared pointer never rewound past them.
    other = next(b for b in paras if b.text == TWO_COL_RIGHT[1])
    assert other.runs == []


def test_hybrid_figure_unpaired_is_unverified_no_crop(tmp_path, monkeypatch):
    # No embedded raster in this PDF at all: a VLM-claimed figure has nothing
    # to pair with geometrically, so it must still surface as an ImageBlock,
    # flagged unverified (never silently dropped). With no bbox guess either,
    # a full-page dump wouldn't be a useful crop, so none is stored.
    crop_dir = tmp_path / "crops"
    monkeypatch.setenv("PDF_CROP_DIR", str(crop_dir))
    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())

    payload = json.dumps([{"type": "figure"}] +
                          [{"type": "paragraph", "text": t} for t in TWO_COL_LEFT] +
                          [{"type": "paragraph", "text": t} for t in TWO_COL_RIGHT])
    vlm = FakeVLM(payload)
    doc = offline_parser(vlm=vlm).parse(path, "doc10")

    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert "bbox" not in images[0].locator
    assert images[0].locator["region"] == "vlm_figure"
    assert images[0].provenance == PROV_UNVERIFIED
    assert images[0].source_crop is None
    # the figure spec was first in the VLM's array -> first block in the doc
    assert doc.blocks[0] is images[0]


def test_hybrid_figure_paired_with_geometry_keeps_position(tmp_path):
    # A real embedded raster this time: the VLM's single figure spec should
    # pair positionally with it, giving a real bbox instead of a bare guess.
    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())

    payload = json.dumps([{"type": "figure"}] +
                          [{"type": "paragraph", "text": t} for t in TWO_COL_LEFT] +
                          [{"type": "paragraph", "text": t} for t in TWO_COL_RIGHT])
    vlm = FakeVLM(payload)
    doc = offline_parser(vlm=vlm).parse(path, "doc11")

    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert "bbox" in images[0].locator
    assert images[0].provenance is None  # deterministic, same as the code path
    assert images[0].source_crop is None
    assert doc.blocks[0] is images[0]


def test_hybrid_image_not_mentioned_by_vlm_still_appended(tmp_path):
    # A real embedded raster the VLM's payload never calls out as a figure
    # (it under-counted): the image must still be emitted, just without a
    # reading-order position -- never silently lost.
    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())

    vlm = FakeVLM(_hybrid_payload())  # no "figure" spec at all
    doc = offline_parser(vlm=vlm).parse(path, "doc12")

    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert "bbox" in images[0].locator
    assert images[0].provenance is None
    # appended after the VLM-driven text flow, so it lands last
    assert doc.blocks[-1] is images[0]


def test_hybrid_primary_read_masks_figure_regions(tmp_path, monkeypatch):
    # The image sent to the model for the primary read must NOT be the raw
    # page render when the page has a figure -- `_mask_figures` blanks it to
    # reduce fine-detail load (see its docstring), while the figure itself
    # still gets paired geometrically (unaffected by what the model saw,
    # since pairing is purely positional -- see
    # test_hybrid_figure_paired_with_geometry_keeps_position).
    # Classify disabled: this test targets _page_images' geometric fragment
    # (see _vlm_figure_regions), not the VLM-grounded region path, and would
    # otherwise also record the fake VLM's page-region classify call below.
    monkeypatch.setenv("VISUAL_CLASSIFY", "0")
    import pdfplumber

    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())
    with PDFIUM_LOCK, pdfplumber.open(path) as pdf:
        raw_render = pdf.pages[0].to_image(resolution=150).original
        buf = io.BytesIO()
        raw_render.save(buf, format="PNG")
        raw_png = buf.getvalue()

    vlm = FakeVLM(_hybrid_payload([{"type": "figure"}]))
    offline_parser(vlm=vlm).parse(path, "doc_mask")

    assert len(vlm.calls) == 1
    sent_png = vlm.calls[0]["images"][0][1]
    assert sent_png != raw_png


class _SequencedVLM:
    """Like FakeVLM, but returns a different canned payload per call, in
    call order -- needed where one page triggers two distinct VLM calls
    with different expected reply shapes (the page-region classify call,
    then the primary hybrid read)."""

    def __init__(self, payloads: list[str]) -> None:
        self.payloads = payloads
        self.calls: list[dict] = []

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls.append({"system": system, "user": user, "n_images": len(images)})
        return self.payloads[len(self.calls) - 1]


def test_hybrid_figure_uses_vlm_grounded_region_over_geometric_fragment(tmp_path):
    """When a page-region grounding call is available (here, the same fake
    VLM reused as VLM_CLASSIFY, since no VLM_CLASSIFY_* is configured in
    tests -- see llm.get_vlm_classify_client), its whole-figure bbox and
    type replace `_page_images`' raw pdfplumber raster object for the
    figure's ImageBlock -- this is the fix for one diagram fragmenting into
    many small ImageBlocks: _vlm_figure_regions' merged region is used in
    place of the geometric fragment list entirely."""
    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())

    page_regions = json.dumps({"regions": [
        {"type": "product_photo", "bbox": [0.3, 0.02, 0.7, 0.08], "confidence": 0.85},
    ]})
    vlm = _SequencedVLM([page_regions, _hybrid_payload([{"type": "figure"}])])
    doc = offline_parser(vlm=vlm).parse(path, "doc_regions")

    # 1 page-region call + 1 primary read -- the page-region call happens
    # during _triage (this page has no table candidates, so _triage's own
    # warm-up branch calls classify_page_regions there; see its docstring),
    # so _parse_hybrid's later _vlm_figure_regions call is a cache hit, not
    # a second real call.
    assert len(vlm.calls) == 2
    figures = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(figures) == 1
    fig = figures[0]
    assert fig.visual_type == "product_photo"
    assert fig.visual_type_source == "cache"


def test_triage_warms_page_regions_for_figure_only_hybrid_page(tmp_path, monkeypatch):
    """The gap `classify_only`'s docstring calls out: a page routed "hybrid"
    purely by its two-column gutter (no geometric table candidate, so
    `_build_region_tables` never runs) still must not leave
    `classify_page_regions`'s cache cold for `_parse_hybrid` to hit live --
    `_triage` has its own warm-up branch for exactly this case. Simulates
    phase 0 (`classify_only`) pre-warming the cache, then a later `parse()`
    call (phase 1, a fresh VLM instance -- phase 0 and phase 1 are genuinely
    separate processes/documents in the pipeline) must get a cache hit, not
    a fresh VLM call, for the figure's classification."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())

    page_regions = json.dumps({"regions": [
        {"type": "product_photo", "bbox": [0.3, 0.02, 0.7, 0.08], "confidence": 0.85},
    ]})
    warm_vlm = FakeVLM(page_regions)
    offline_parser(vlm=warm_vlm).classify_only(path, "doc_warm")
    assert len(warm_vlm.calls) == 1  # only the page-region warm-up call

    # Phase 1: a differently-behaved VLM -- if the cache weren't warm, this
    # would be called for page-regions too and its reply (not shaped like
    # {"regions": [...]}) would parse to zero regions.
    vlm = FakeVLM(_hybrid_payload([{"type": "figure"}]))
    doc = offline_parser(vlm=vlm).parse(path, "doc_warm")

    assert len(vlm.calls) == 1  # only the primary read; classify was a cache hit
    figures = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(figures) == 1
    assert figures[0].visual_type == "product_photo"
    assert figures[0].visual_type_source == "cache"


def test_hybrid_figure_falls_back_to_page_images_when_no_regions(tmp_path):
    """The page-region call ran (classify is on by default) but returned no
    usable regions (e.g. a reply the model didn't ground) -- the figure
    must still be recovered via `_page_images`' geometric fragment, not
    silently dropped."""
    path = tmp_path / "twocol_img.pdf"
    path.write_bytes(two_column_with_image_pdf())

    vlm = _SequencedVLM(["not json at all", _hybrid_payload([{"type": "figure"}])])
    doc = offline_parser(vlm=vlm).parse(path, "doc_fallback")

    assert len(vlm.calls) == 2
    figures = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(figures) == 1
    assert figures[0].visual_type is None  # no classify verdict to stamp it with

    figures = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(figures) == 1 and "bbox" in figures[0].locator


def test_hybrid_primary_read_unchanged_when_page_has_no_figure(tmp_path):
    # No image on the page -- `_mask_figures` is a no-op (returns None
    # immediately, without even re-rendering), so the model sees the exact
    # same render the rest of the pipeline (crops, second reader) would.
    import pdfplumber

    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())
    with PDFIUM_LOCK, pdfplumber.open(path) as pdf:
        raw_render = pdf.pages[0].to_image(resolution=150).original
        buf = io.BytesIO()
        raw_render.save(buf, format="PNG")
        raw_png = buf.getvalue()

    vlm = FakeVLM(_hybrid_payload())
    offline_parser(vlm=vlm).parse(path, "doc_nomask")

    assert vlm.calls[0]["images"][0][1] == raw_png


def _drop_page(tmp_path):
    """A single-column page whose text layer holds two numbered headings
    (16pt/14pt over 10pt body), rendered to a real pdfplumber page for the
    recovery tests below."""
    import pdfplumber

    pdf = text_pdf([
        (72, 80, 16, "2. Hardware Overview"),
        (72, 120, 10, "Body paragraph about the hardware goes here for length."),
        (72, 160, 14, "2.1. Circuitry"),
        (72, 200, 10, "Circuit description body text follows the section head."),
    ])
    path = tmp_path / "drop.pdf"
    path.write_bytes(pdf)
    ctx = pdfplumber.open(path)
    return ctx


DROP_DECLARED = frozenset({"hardware overview", "circuitry"})


def test_recover_dropped_heading_from_text_layer(tmp_path):
    # The VLM produced everything on the page EXCEPT "2.1. Circuitry" (dropped
    # while transcribing, say, an intervening diagram). It's still verbatim in
    # the text layer AND declared by the outline, so recovery reinstates it --
    # text-layer-verified, and slotted into reading order between the overview
    # body and the circuit body by its own text-layer position.
    with _drop_page(tmp_path) as pdf:
        page = pdf.pages[0]
        prep = {"words": page.extract_words(extra_attrs=["size", "fontname"]),
                "tables": []}
        blocks = [
            HeadingBlock(id="b0", span=Span(page=1), text="2. Hardware Overview"),
            ParagraphBlock(id="b1", span=Span(page=1),
                           text="Body paragraph about the hardware goes here for length."),
            ParagraphBlock(id="b2", span=Span(page=1),
                           text="Circuit description body text follows the section head."),
        ]
        n = _recover_dropped_headings(blocks, page, prep, 1, frozenset(),
                                      _Counters(), DROP_DECLARED)

    assert n == 1
    recovered = [b for b in blocks
                 if isinstance(b, HeadingBlock) and "Circuitry" in b.text]
    assert len(recovered) == 1
    assert recovered[0].text == "2.1. Circuitry"
    assert recovered[0].provenance == PROV_TEXT_LAYER
    idx = blocks.index(recovered[0])
    assert blocks[idx - 1].text.startswith("Body paragraph")
    assert blocks[idx + 1].text.startswith("Circuit description")


def test_recover_skips_heading_the_vlm_already_captured(tmp_path):
    # Both headings present in blocks -> nothing to recover, no duplication.
    with _drop_page(tmp_path) as pdf:
        page = pdf.pages[0]
        prep = {"words": page.extract_words(extra_attrs=["size", "fontname"]),
                "tables": []}
        blocks = [
            HeadingBlock(id="b0", span=Span(page=1), text="2. Hardware Overview"),
            HeadingBlock(id="b1", span=Span(page=1), text="2.1. Circuitry"),
        ]
        n = _recover_dropped_headings(blocks, page, prep, 1, frozenset(),
                                      _Counters(), DROP_DECLARED)
    assert n == 0


def test_recover_does_not_duplicate_a_number_stripped_heading(tmp_path):
    # The VLM kept the heading but dropped its number ("Circuitry" for
    # "2.1. Circuitry") -- number-insensitive matching must treat it as
    # already present, so no second, numbered copy is injected.
    with _drop_page(tmp_path) as pdf:
        page = pdf.pages[0]
        prep = {"words": page.extract_words(extra_attrs=["size", "fontname"]),
                "tables": []}
        blocks = [
            HeadingBlock(id="b0", span=Span(page=1), text="2. Hardware Overview"),
            HeadingBlock(id="b1", span=Span(page=1), text="Circuitry"),
        ]
        n = _recover_dropped_headings(blocks, page, prep, 1, frozenset(),
                                      _Counters(), DROP_DECLARED)
    assert n == 0


def test_recover_skips_heading_shaped_line_not_declared(tmp_path):
    # A heading-shaped, heading-sized line the outline never declares (a
    # vector-diagram label like "Main Controller", or a dimension callout) is
    # NOT recovered -- the declared-set gate is what keeps big-font figure
    # text out of the hierarchy.
    with _drop_page(tmp_path) as pdf:
        page = pdf.pages[0]
        prep = {"words": page.extract_words(extra_attrs=["size", "fontname"]),
                "tables": []}
        blocks = [
            HeadingBlock(id="b0", span=Span(page=1), text="2. Hardware Overview"),
            ParagraphBlock(id="b1", span=Span(page=1),
                           text="Body paragraph about the hardware goes here for length."),
            ParagraphBlock(id="b2", span=Span(page=1),
                           text="Circuit description body text follows the section head."),
        ]
        # "circuitry" absent from declared -> even though it's a real dropped
        # heading in the text layer, without the document declaring it we
        # don't invent it.
        n = _recover_dropped_headings(blocks, page, prep, 1, frozenset(),
                                      _Counters(), frozenset({"hardware overview"}))
    assert n == 0


def test_hybrid_bad_vlm_json_falls_back_to_code(tmp_path, monkeypatch):
    # Classify disabled: this test counts primary-read retries, and would
    # otherwise also count the fake VLM's page-region classify call.
    monkeypatch.setenv("VISUAL_CLASSIFY", "0")
    path = tmp_path / "twocol.pdf"
    path.write_bytes(two_column_pdf())

    vlm = FakeVLM("I refuse to answer in JSON.")
    doc = offline_parser(vlm=vlm).parse(path, "doc5")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["used"] == "code" and "fallback" in meta["note"]
    assert len(vlm.calls) == 2  # one retry with the stricter reminder
    assert any(isinstance(b, ParagraphBlock) for b in doc.blocks)
    assert all(b.provenance == PROV_TEXT_LAYER for b in doc.blocks)


# ---------------------------------------------------------------------------
# Scanned path
# ---------------------------------------------------------------------------


def test_scanned_routes_and_consensus(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    vlm = FakeVLM(json.dumps([
        {"type": "paragraph", "text": "Hello scanned world"}]))
    det = FakeDetector([DetectedLine("Hello scanned world", (10, 10, 500, 40))])
    doc = offline_parser(vlm=vlm, detector=det).parse(path, "doc6")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned" and meta["used"] == "scanned"
    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert len(paras) == 1
    assert paras[0].text == "Hello scanned world"
    assert paras[0].provenance == PROV_CONSENSUS


def test_scanned_detector_disagreement_unverified(tmp_path, monkeypatch):
    monkeypatch.setenv("PDF_CROP_DIR", str(tmp_path / "crops"))
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    vlm = FakeVLM(json.dumps([
        {"type": "paragraph", "text": "Invoice total 500 USD"}]))
    det = FakeDetector([DetectedLine("Invoice total 600 USD", (10, 10, 500, 40))])
    doc = offline_parser(vlm=vlm, detector=det).parse(path, "doc7")

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert len(paras) == 1
    # 500 vs 600: critical tokens disagree, no second model -> stays unverified
    assert paras[0].provenance == PROV_UNVERIFIED
    assert paras[0].source_crop is not None


def test_scanned_second_vlm_rescues_disputed_line(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    vlm = FakeVLM(json.dumps([
        {"type": "paragraph", "text": "Invoice total 500 USD"}]))
    det = FakeDetector([DetectedLine("Invoice total 600 USD", (10, 10, 500, 40))])
    vlm2 = FakeVLM("Invoice total 500 USD")  # independent reread agrees w/ VLM1
    doc = offline_parser(vlm=vlm, detector=det, vlm2=vlm2).parse(path, "doc8")

    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert paras[0].provenance == PROV_CONSENSUS
    assert len(vlm2.calls) == 1  # the crop-and-reread call


def test_scanned_figure_over_background_scan_is_unverified_no_crop(tmp_path, monkeypatch):
    # scanned_pdf()'s only raster IS the page-spanning scan itself -- it must
    # be excluded from geometric pairing, so a VLM-claimed figure here has
    # nothing to pair with and surfaces as unverified. With no bbox guess
    # either, a full-page dump wouldn't be a useful crop, so none is stored.
    crop_dir = tmp_path / "crops"
    monkeypatch.setenv("PDF_CROP_DIR", str(crop_dir))
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    vlm = FakeVLM(json.dumps([{"type": "figure"}]))
    doc = offline_parser(vlm=vlm).parse(path, "doc9b")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["used"] == "scanned"
    assert len(doc.blocks) == 1
    img = doc.blocks[0]
    assert isinstance(img, ImageBlock)
    assert "bbox" not in img.locator
    assert img.locator["region"] == "vlm_figure"
    assert img.provenance == PROV_UNVERIFIED
    assert img.source_crop is None


def test_scanned_figure_with_vlm_bbox_gets_tight_crop(tmp_path, monkeypatch):
    # Same page-spanning-scan situation as the test above, but this time the
    # VLM volunteers its own (ungrounded) bbox for the sub-figure. It still
    # can't be geometrically confirmed, so it stays unverified -- but since
    # there's a bbox guess to work with, a tight audit crop is stored (unlike
    # the no-bbox case, which stores none rather than a full-page dump).
    crop_dir = tmp_path / "crops"
    monkeypatch.setenv("PDF_CROP_DIR", str(crop_dir))
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    vlm = FakeVLM(json.dumps(
        [{"type": "figure", "bbox": [0.1, 0.1, 0.4, 0.3]}]))
    doc = offline_parser(vlm=vlm).parse(path, "doc9d")

    img = doc.blocks[0]
    assert isinstance(img, ImageBlock)
    assert img.provenance == PROV_UNVERIFIED
    assert "vlm_bbox" not in img.locator  # consumed, not left dangling
    assert img.source_crop is not None
    assert (crop_dir / f"{hash_hex(img.source_crop)}.png").exists()


def test_scanned_figure_paired_with_real_subfigure_keeps_bbox(tmp_path):
    # A distinct, non-page-spanning embedded raster alongside the scan
    # background: the VLM's figure spec should pair with it (not the
    # background) and get a real geometric bbox, deterministic provenance.
    path = tmp_path / "scan_sub.pdf"
    path.write_bytes(scanned_pdf_with_subfigure())

    vlm = FakeVLM(json.dumps([{"type": "figure"},
                              {"type": "paragraph", "text": "Caption text"}]))
    det = FakeDetector([DetectedLine("Caption text", (10, 400, 300, 430))])
    doc = offline_parser(vlm=vlm, detector=det).parse(path, "doc9c")

    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert "bbox" in images[0].locator
    assert images[0].provenance is None
    assert images[0].source_crop is None


def test_scanned_without_vlm_keeps_page_as_image(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(scanned_pdf())

    doc = offline_parser().parse(path, "doc9")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned" and meta["used"] == "image-fallback"
    assert len(doc.blocks) == 1
    img = doc.blocks[0]
    assert isinstance(img, ImageBlock)
    assert img.locator["region"] == "full_page"
    assert img.locator["page"] == 1


def test_vector_ink_page_routes_scanned_not_empty(tmp_path):
    # Flattened-to-outlines page (no chars, no rasters, many vector paths)
    # must not vanish through the "empty" route: it routes "scanned" and,
    # without a VLM, survives as a full-page image block.
    path = tmp_path / "vec.pdf"
    path.write_bytes(vector_ink_pdf())

    doc = offline_parser().parse(path, "vec1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned" and meta["used"] == "image-fallback"
    assert meta["vector_objects"] >= 40
    assert len(doc.blocks) == 1
    assert isinstance(doc.blocks[0], ImageBlock)
    assert doc.blocks[0].locator["region"] == "full_page"


def test_sparse_vector_page_still_empty(tmp_path):
    # A couple of decorative rules on an otherwise blank page stay "empty",
    # and the route now says why.
    path = tmp_path / "blank.pdf"
    path.write_bytes(vector_ink_pdf(n_rects=3))

    doc = offline_parser().parse(path, "vec2")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "empty"
    assert "note" in meta
    assert doc.blocks == []


def test_tiled_images_page_routes_scanned_not_empty(tmp_path):
    # Total raster coverage counts, not the largest single image: four small
    # tiles summing past the threshold route "scanned" even though each one
    # alone is far below it.
    path = tmp_path / "tiles.pdf"
    path.write_bytes(tiled_images_pdf())

    doc = offline_parser().parse(path, "tiles1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned" and meta["used"] == "image-fallback"
    assert len(doc.blocks) == 1
    assert isinstance(doc.blocks[0], ImageBlock)


# ---------------------------------------------------------------------------
# IR round-trip with provenance
# ---------------------------------------------------------------------------


def test_provenance_serializes_and_roundtrips(tmp_path):
    from medrag.pipeline.parser.parsers.base import ParsedDocument

    pdf = text_pdf([(72, 80, 12, "hello world roundtrip")])
    path = tmp_path / "rt.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "rt")
    again = ParsedDocument.from_json(doc.to_json())
    assert again.blocks[0].provenance == PROV_TEXT_LAYER
    assert again.blocks[0].source_crop is None
    d = doc.blocks[0].to_dict()
    assert d["provenance"] == PROV_TEXT_LAYER and "source_crop" not in d


def test_registry_selects_pdf(tmp_path):
    from medrag.pipeline.parser.parsers.registry import parser_for

    assert isinstance(parser_for("x.pdf"), PdfParser)


# ---------------------------------------------------------------------------
# Unpaired-table recovery: a geometric table the VLM transcribed as paragraphs
# (never tagged "table") must still land in the IR as a real table, in place,
# without duplicating the paragraph prose that carried the same text.
# ---------------------------------------------------------------------------


def _spec_table_from(bbox):
    from medrag.pipeline.parser.parsers.base import TableData, text_cell

    data = TableData(n_rows=2, n_cols=4, cells=[
        [text_cell("Specification"), text_cell("Min"),
         text_cell("Typical"), text_cell("Max")],
        [text_cell("Voltage"), text_cell("1"), text_cell("2"), text_cell("3")],
    ])
    return _PreparedTable(bbox=bbox, data=data, provenance=PROV_TABLE_BANDS,
                          confidence=0.9, flags=["header-recovered"])


def test_blocks_from_vlm_leaves_unpaired_tables_for_the_caller():
    # Two geometric tables on the page, but the VLM emitted only ONE "table"
    # spec: the first is paired (grid replaces the model rows), the second is
    # left in the passed list for the caller to recover.
    ct = _Counters()
    t0 = _spec_table_from((0, 0, 100, 50))
    t1 = _spec_table_from((0, 60, 100, 110))
    geom = [t0, t1]
    specs = [{"type": "table", "rows": [["ignored"]]}]
    blocks = offline_parser()._blocks_from_vlm(specs, 1, ct, geom_tables=geom)

    assert len(blocks) == 1 and isinstance(blocks[0], TableBlock)
    # Paired table carries the band builder's provenance, not the VLM's rows.
    assert blocks[0].provenance == PROV_TABLE_BANDS
    assert blocks[0].table.n_cols == 4
    # The unpaired second table is still in the list, unconsumed.
    assert geom == [t1]


# ---------------------------------------------------------------------------
# Ungrounded VLM-table gate: a "table" spec with NO geometric region to pair
# with (geom_tables empty) is the weakest-evidence table candidate in the
# pipeline -- classified by cell text alone (no crop available) before it's
# trusted as a real table.
# ---------------------------------------------------------------------------


def test_ungrounded_vlm_table_reclassified_as_image(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    ct = _Counters()
    llm = FakeLLM(json.dumps({"type": "block_diagram", "confidence": 0.8}))
    specs = [{"type": "table", "rows": [["AAF", "AAF"], ["", ""]]}]
    blocks = offline_parser(text_llm=llm)._blocks_from_vlm(
        specs, 1, ct, geom_tables=[])

    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].visual_type == "block_diagram"
    assert blocks[0].visual_type_source == "text-llm"
    assert llm.calls == 1


def test_ungrounded_vlm_table_kept_when_classified_as_table():
    ct = _Counters()
    llm = FakeLLM(json.dumps({"type": "table", "confidence": 0.95}))
    specs = [{"type": "table", "rows": [["Spec", "Value"], ["Voltage", "5V"]]}]
    blocks = offline_parser(text_llm=llm)._blocks_from_vlm(
        specs, 1, ct, geom_tables=[])

    assert len(blocks) == 1
    assert isinstance(blocks[0], TableBlock)
    assert blocks[0].table.n_rows == 2


def test_ungrounded_vlm_table_fails_open_without_text_llm():
    # No text LLM reachable -> keep today's behavior (accept the VLM's own
    # rows as a table) rather than newly regressing this already-verified
    # path just because no one could be asked.
    ct = _Counters()
    specs = [{"type": "table", "rows": [["Spec", "Value"]]}]
    blocks = offline_parser()._blocks_from_vlm(specs, 1, ct, geom_tables=[])

    assert len(blocks) == 1
    assert isinstance(blocks[0], TableBlock)


def test_merge_unpaired_tables_replaces_paragraph_prose_in_place():
    ct = _Counters()
    pt = _spec_table_from((50, 100, 500, 200))
    blocks = [
        HeadingBlock(id="h0", text="Electrical Characteristics", level=2),
        ParagraphBlock(id="p0", text="Specification Min Typical Max"),
        ParagraphBlock(id="p1", text="Voltage 1 2 3"),
        ParagraphBlock(id="p2", text="The following section describes device "
                                     "behaviour over the full temperature range"),
    ]
    out = _merge_unpaired_tables(blocks, [pt], 1, ct)

    # p0 and p1 (the table, transcribed as prose) are gone; the table takes p0's
    # slot; the heading before and the real prose after are untouched.
    kinds = [type(b).__name__ for b in out]
    assert kinds == ["HeadingBlock", "TableBlock", "ParagraphBlock"]
    assert out[2].text.startswith("The following section")
    tb = out[1]
    assert tb.provenance == PROV_TABLE_BANDS and tb.table_confidence == 0.9
    assert "vlm-unpaired" in tb.table_flags and "header-recovered" in tb.table_flags
    # No hallucination: every cell's text is the table's own, unchanged.
    assert tb.table.cells[1][0].plain_text() == "Voltage"


def test_merge_unpaired_tables_appends_when_no_block_matches():
    ct = _Counters()
    pt = _spec_table_from((50, 100, 500, 200))
    blocks = [ParagraphBlock(id="p0",
                             text="An unrelated paragraph mentioning nothing "
                                  "from the grid at all whatsoever")]
    out = _merge_unpaired_tables(blocks, [pt], 1, ct)

    # Nothing matched -> the table is appended (never silently dropped), and the
    # unrelated prose is left in place.
    assert [type(b).__name__ for b in out] == ["ParagraphBlock", "TableBlock"]
    assert "vlm-unpaired" in out[1].table_flags


def test_merge_unpaired_tables_keeps_prose_that_only_shares_a_few_terms():
    # A real sentence that happens to name one column header must NOT be eaten:
    # its containment in the table's tokens stays well below the threshold.
    ct = _Counters()
    pt = _spec_table_from((50, 100, 500, 200))
    blocks = [ParagraphBlock(
        id="p0", text="The maximum voltage rating depends on ambient "
                      "conditions and must be derated accordingly")]
    out = _merge_unpaired_tables(blocks, [pt], 1, ct)

    assert [type(b).__name__ for b in out] == ["ParagraphBlock", "TableBlock"]
    assert out[0].text.startswith("The maximum voltage")


# ---------------------------------------------------------------------------
# Reading-robustness fixes: word joining, label split, icon glyphs, triage
# holes, tesseract fallback, partial-read completeness, full-page render,
# broken cmap
# ---------------------------------------------------------------------------

from medrag.pipeline.parser.parsers.base import InlineRun, Mark
from medrag.pipeline.parser.parsers.pdf_parser import (
    _cluster_lines,
    _det_paragraphs,
    _line_runs,
    _scan_broken_cmap,
    _split_label_group,
    _strip_symbol_glyphs,
)
from medrag.pipeline.parser.parsers.table_bands import join_words, word_sep


def _wd(text, x0, x1, top=100.0, bottom=110.0, size=10.0, font="Helvetica"):
    return {"text": text, "x0": x0, "x1": x1, "top": top, "bottom": bottom,
            "size": size, "fontname": font, "upright": True}


def test_word_sep_tight_gap_joins_without_space():
    # extract_words splits 'support@example.com' at the regular->bold font
    # change; the halves are near-touching and must re-join with no space.
    a = _wd("support@", 50, 90)
    b = _wd("example.com", 90.5, 150, font="Helvetica-Bold")
    assert word_sep(a, b) == ""
    assert join_words([a, b]) == "support@example.com"
    runs = _line_runs([a, b])
    assert "".join(r.text for r in runs) == "support@example.com"
    assert runs[-1].marks == (Mark.BOLD,)


def test_word_sep_normal_gap_keeps_space():
    a = _wd("hello", 50, 75)
    b = _wd("world", 78.5, 100)  # 3.5pt gap at 10pt: a real word space
    assert word_sep(a, b) == " "
    assert join_words([a, b]) == "hello world"


def test_word_sep_cross_line_always_space():
    a = _wd("end", 300, 320, top=100, bottom=110)
    b = _wd("start", 50, 80, top=115, bottom=125)
    assert word_sep(a, b) == " "


def test_cluster_lines_merges_font_split_word():
    words = [_wd("support@", 50, 90),
             _wd("example.com", 90.4, 150, font="Helvetica-Bold")]
    lines = _cluster_lines(words)
    assert len(lines) == 1
    assert lines[0].text == "support@example.com"
    assert "".join(r.text for r in lines[0].runs) == lines[0].text


def test_label_group_splits_callout_label_from_body():
    # "Caution" printed at the same top as a body line, far to its left and
    # in a different style (the PN1225 p.4 geometry: 7pt Black label, 9pt
    # Regular body, ~12pt gap) becomes its own line -- and is marked `label`
    # so the specs builder never re-merges it into the sentence.
    body_words = [_wd(t, 112 + i * 40, 140 + i * 40, size=9.0)
                  for i, t in enumerate(["in", "the", "event", "of", "damage"])]
    words = [_wd("Caution", 50, 99, size=7.0,
                 font="ABCDEF+RedHatDisplay-Black")] + body_words
    lines = _cluster_lines(words)
    assert [l.text for l in lines] == ["Caution", "in the event of damage"]
    assert lines[0].label and not lines[1].label


def test_label_group_keeps_same_style_line_whole():
    # Same size across a wide gap (a spaced key/value line, justified text):
    # no size discontinuity, stays one line.
    words = [_wd("Weight", 50, 90),
             _wd("1.2", 250, 265), _wd("kg", 268, 280)]
    assert _split_label_group(words) == [words]
    assert len(_cluster_lines(words)) == 1


def test_label_group_face_only_change_not_split():
    # A bold word after a wide gap on an otherwise same-size line is
    # ordinary emphasis, not a callout label seam.
    words = [_wd("plain", 50, 77),
             _wd("boldword", 140, 195, font="Helvetica-Bold")]
    assert _split_label_group(words) == [words]


def test_label_group_keeps_wide_two_fragment_line_whole():
    # Both sides of the widest gap carry >2 words: a spec/table-ish line,
    # not a label seam -- stays one line even across a style change.
    left = [_wd(t, 50 + i * 40, 80 + i * 40)
            for i, t in enumerate(["alpha", "beta", "gamma"])]
    right = [_wd(t, 350 + i * 40, 380 + i * 40, font="Helvetica-Bold")
             for i, t in enumerate(["delta", "epsilon", "zeta"])]
    assert _split_label_group(left + right) == [left + right]
    assert len(_cluster_lines(left + right)) == 1


def test_label_becomes_standalone_block_in_code_path(tmp_path):
    # End to end: the callout label (smaller size, wide gap to the body at
    # the same top) must surface as its own block, with the body sentence
    # intact beside it.
    pdf = text_pdf([
        (72, 100, 10, "A perfectly ordinary paragraph line first."),
        (72, 131, 7, "Caution"),
        (160, 130, 10, "the product shall be returned for repair."),
        (72, 160, 10, "Another ordinary paragraph line after."),
    ])
    path = tmp_path / "callout.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "callout1")

    texts = [b.text for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert "Caution" in texts
    assert any(t.startswith("the product shall be returned") for t in texts)
    assert not any("Caution the" in t for t in texts)


def test_trailing_icon_font_glyph_dropped():
    # The Glyphter checkmark that decodes as a stray trailing "A" is removed;
    # a real ampere value in the body font is not.
    words = [_wd("copper", 50, 90),
             _wd("A", 95, 100, font="ABCDEF+Glyphter")]
    lines = _cluster_lines(words)
    assert lines[0].text == "copper"

    amps = [_wd("16", 50, 60), _wd("A", 63, 68)]
    assert _cluster_lines(amps)[0].text == "16 A"


def test_lone_icon_glyph_line_dropped():
    assert _cluster_lines([_wd("A", 50, 55, font="Glyphter")]) == []
    assert _strip_symbol_glyphs([_wd("A", 50, 55, font="Glyphter")]) == []


def test_det_paragraphs_groups_by_vertical_gap():
    lines = [DetectedLine("first line", (10, 10, 200, 22)),
             DetectedLine("second line", (10, 26, 200, 38)),
             DetectedLine("new paragraph", (10, 90, 200, 102))]
    assert _det_paragraphs(lines) == ["first line second line",
                                      "new paragraph"]


def test_scan_broken_cmap_flags_and_repairs():
    blocks = [ParagraphBlock(id="p0", span=Span(page=1),
                             text="SoŌware seƩings maƩer for soŌer craŌs",
                             runs=[InlineRun("SoŌware seƩings maƩer "
                                             "for soŌer craŌs")])]
    stats = _scan_broken_cmap(blocks, min_hits=5)
    assert stats and stats["junk_chars"] == 5
    assert blocks[0].text == "Software settings matter for softer crafts"
    assert "".join(r.text for r in blocks[0].runs) == blocks[0].text
    assert stats["junk_remaining"] == 0


def test_scan_broken_cmap_leaves_healthy_document_alone():
    blocks = [ParagraphBlock(id="p0", span=Span(page=1),
                             text="Ōsaka is a city in Japan")]
    assert _scan_broken_cmap(blocks, min_hits=5) is None
    assert blocks[0].text == "Ōsaka is a city in Japan"


def _pdf_with_font_and_content(content: bytes) -> bytes:
    """One-page PDF with an F1 Helvetica resource and arbitrary content."""
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        _stream(b"<<", content),
    ])


def vector_ink_with_text_scrap_pdf(n_rects: int = 60) -> bytes:
    """A flattened page (heavy vector ink) that ALSO carries a few real
    text-layer characters (a page number) -- the one-character-wide re-opening
    of the SLSC hole."""
    content = b""
    for i in range(n_rects):
        x = 40 + (i % 10) * 50
        y = 700 - (i // 10) * 30
        content += f"{x} {y} 8 12 re f\n".encode()
    content += b"BT /F1 9 Tf 300 30 Td (3) Tj ET\n"
    return _pdf_with_font_and_content(content)


def small_raster_pdf() -> bytes:
    """No text layer, no vector ink, one small raster (~6% of the page) --
    used to be routed 'empty', silently dropping the raster."""
    pixels = bytes([120, 120, 120]) * (10 * 10)
    return _build_pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        _stream(b"<< /Type /XObject /Subtype /Image /Width 10 /Height 10 "
                b"/ColorSpace /DeviceRGB /BitsPerComponent 8", pixels),
        _stream(b"<<", b"q 175 0 0 175 100 300 cm /Im1 Do Q"),
    ])


def rotated_text_pdf() -> bytes:
    """Normal body lines plus one 90-degree-rotated boilerplate string (the
    technical-drawing copyright strip)."""
    content = b""
    for i, text in enumerate(["normal body line one goes here",
                              "normal body line two goes here",
                              "normal body line three goes here"]):
        y = 792 - (100.0 + 20 * i) - 10
        content += f"BT /F1 10 Tf 72 {y} Td ({text}) Tj ET\n".encode("latin-1")
    content += (b"BT /F1 8 Tf 0 1 -1 0 30 300 Tm "
                b"(CONFIDENTIALROTATED do not reproduce) Tj ET\n")
    return _pdf_with_font_and_content(content)


def vector_heavy_text_pdf(n_rects: int = 1100) -> bytes:
    """A drawing-like page: real text lines over massive vector linework."""
    content = b""
    for i, text in enumerate(["Box Type Bus Coupler title block",
                              "part number and revision data here"]):
        y = 792 - (60.0 + 20 * i) - 10
        content += f"BT /F1 10 Tf 72 {y} Td ({text}) Tj ET\n".encode("latin-1")
    for i in range(n_rects):
        x = 30 + (i * 37) % 540
        y = 120 + (i * 53) % 560
        content += f"{x} {y} 4 3 re f\n".encode()
    return _pdf_with_font_and_content(content)


def test_flattened_page_with_text_scrap_routes_scanned(tmp_path):
    path = tmp_path / "scrap.pdf"
    path.write_bytes(vector_ink_with_text_scrap_pdf())

    doc = offline_parser().parse(path, "scrap1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned"
    assert meta["chars"] > 0
    # without a VLM the page still survives as a full-page image
    assert any(isinstance(b, ImageBlock)
               and b.locator.get("region") == "full_page" for b in doc.blocks)


def test_small_raster_page_routes_scanned_not_empty(tmp_path):
    path = tmp_path / "smallraster.pdf"
    path.write_bytes(small_raster_pdf())

    doc = offline_parser().parse(path, "sr1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] == "scanned"
    assert 0.01 <= meta["image_coverage"] < 0.2
    assert len(doc.blocks) >= 1


def test_truly_blank_page_still_empty(tmp_path):
    path = tmp_path / "blank2.pdf"
    path.write_bytes(_pdf_with_font_and_content(b""))

    doc = offline_parser().parse(path, "blank2")

    assert doc.metadata["pdf_pages"][0]["route"] == "empty"
    assert doc.blocks == []


def test_rotated_boilerplate_dropped_from_flow(tmp_path):
    path = tmp_path / "rot.pdf"
    path.write_bytes(rotated_text_pdf())

    doc = offline_parser().parse(path, "rot1")

    all_text = " ".join(getattr(b, "text", "") for b in doc.blocks)
    assert "CONFIDENTIALROTATED" not in all_text
    assert "normal body line one goes here" in all_text
    assert doc.metadata["pdf_pages"][0].get("rotated_words_dropped", 0) >= 1


def test_vector_heavy_text_page_keeps_full_page_render(tmp_path):
    path = tmp_path / "drawing.pdf"
    path.write_bytes(vector_heavy_text_pdf())

    doc = offline_parser().parse(path, "drw1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["route"] in ("code", "hybrid")
    assert meta.get("full_page_render") is True
    renders = [b for b in doc.blocks if isinstance(b, ImageBlock)
               and b.locator.get("region") == "full_page_render"]
    assert len(renders) == 1
    assert renders[0].locator["bbox"] == [0, 0, 612.0, 792.0]
    # the text still parsed normally alongside
    all_text = " ".join(getattr(b, "text", "") for b in doc.blocks)
    assert "title block" in all_text


def test_scanned_vlm_fail_emits_tesseract_fallback_text(tmp_path):
    # VLM answers garbage (bad-json) -> the detector reading becomes
    # unverified paragraphs and the full-page image is still kept.
    path = tmp_path / "scanfb.pdf"
    path.write_bytes(scanned_pdf())
    det = FakeDetector([
        DetectedLine("recovered first line", (10, 10, 300, 22)),
        DetectedLine("recovered second line", (10, 26, 300, 38)),
        DetectedLine("separate paragraph here", (10, 200, 300, 212)),
    ])
    parser = PdfParser(vlm=FakeVLM("not json at all"), vlm2=None,
                       detector=det, table_struct=None)

    doc = parser.parse(path, "fb1")

    meta = doc.metadata["pdf_pages"][0]
    assert meta["used"] == "image-fallback"
    assert "tesseract-fallback" in meta["note"]
    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [p.text for p in paras] == [
        "recovered first line recovered second line",
        "separate paragraph here"]
    assert all(p.provenance == PROV_UNVERIFIED for p in paras)
    assert any(isinstance(b, ImageBlock)
               and b.locator.get("region") == "full_page" for b in doc.blocks)


def test_scanned_partial_vlm_read_keeps_full_page_image(tmp_path, monkeypatch):
    # The VLM reads one short paragraph off a page the detector says is
    # dense: the shortfall is noted and the full page is kept as an image.
    monkeypatch.setenv("PDF_CROP_DIR", str(tmp_path / "crops"))
    path = tmp_path / "partial.pdf"
    path.write_bytes(scanned_pdf())
    det_lines = [DetectedLine(f"unique{i} tokens{i} here{i}",
                              (10, 10 + 14 * i, 300, 22 + 14 * i))
                 for i in range(12)]
    parser = PdfParser(
        vlm=FakeVLM('[{"type": "paragraph", "text": "unique0 tokens0 here0"}]'),
        vlm2=None, detector=FakeDetector(det_lines), table_struct=None)

    doc = parser.parse(path, "part1")

    meta = doc.metadata["pdf_pages"][0]
    assert "partial-vlm-read" in meta.get("note", "")
    assert any(isinstance(b, ImageBlock)
               and b.locator.get("region") == "full_page" for b in doc.blocks)


def test_toc_page_meta_notes_extracted_entries(tmp_path):
    pdf = text_pdf([
        (72, 60, 16, "Contents"),
        (72, 100, 11, "1. Description 1"),
        (90, 115, 11, "1.1. Key Features 1"),
        (72, 130, 11, "2. Hardware Overview 2"),
        (72, 160, 16, "1. Description"),
        (72, 200, 11, "Real body text for the description section."),
    ])
    path = tmp_path / "tocnote.pdf"
    path.write_bytes(pdf)

    doc = offline_parser().parse(path, "tocnote1")

    assert doc.metadata["toc"]
    assert doc.metadata["pdf_pages"][0]["toc_extracted"] == 3
