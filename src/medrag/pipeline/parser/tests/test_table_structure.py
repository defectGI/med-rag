"""tables/structure/ tests: the neutral protocol, the HTTP adapter's JSON
contract, cell-text placement by character containment, and pdf_parser.py's
wiring of a configured structure model into the code path.

No network and no real TableFormer: a FakeTableStructureClient is injected
through PdfParser's constructor (same pattern test_pdf_parser.py uses for
VLM/detector fakes), and the HTTP adapter is tested by monkeypatching
urllib.request.urlopen.
"""

from __future__ import annotations

import json
from io import BytesIO

import pdfplumber
import pytest

from medrag.pipeline.parser.parsers.base import TableBlock, hash_hex
from medrag.pipeline.parser.parsers.pdf_parser import (
    PROV_TABLE_BANDS,
    PROV_TABLE_STRUCTURE,
    PdfParser,
    _apply_health,
    _as_table_data,
    _best_overlap_index,
    _fill_cells_by_containment,
    _is_degenerate_table,
    _PreparedTable,
    _sparse_first_column,
)
from medrag.pipeline.parser.tables.structure import (
    DetectedCell,
    DetectedTable,
    TextCellHint,
)
from medrag.pipeline.parser.tables.structure.http_client import HttpTableStructureClient


@pytest.fixture(autouse=True)
def _isolate_labels_dir(tmp_path, monkeypatch):
    """`_build_region_tables` now also runs the visual-type classify step
    (A.4.1) for any test with a real geometric table and a non-None VLM --
    isolate its disk cache so no test here writes into the real repo's
    storage/labels/, regardless of whether that specific test means to
    exercise classification at all."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))


# ---------------------------------------------------------------------------
# Minimal one-page born-digital PDF builder (same low-level shape as
# test_pdf_parser.py/test_image_handler.py -- no third-party writer needed)
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


def grid_pdf() -> bytes:
    """One-page PDF with a real ruled 2-row x 1-col table (so find_tables()
    detects it by line geometry) and one word per row. Used to exercise the
    low-coverage fallback: a structure model whose cell boxes miss the text
    should degrade to this line-detected table, not emit empty cells."""
    # PDF origin is bottom-left. Table x:60..300; rows top-from-top 90..130
    # and 130..170 -> horizontal rules at pdf-y 702/662/622, verticals x 60/300.
    content = (
        b"1 w\n"
        b"60 702 m 300 702 l S\n"
        b"60 662 m 300 662 l S\n"
        b"60 622 m 300 622 l S\n"
        b"60 702 m 60 622 l S\n"
        b"300 702 m 300 622 l S\n"
        b"BT /F1 12 Tf 72 680 Td (Alpha) Tj ET\n"
        b"BT /F1 12 Tf 72 640 Td (Bravo) Tj ET\n"
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


def offline_parser(**kw) -> PdfParser:
    kw.setdefault("vlm", None)
    kw.setdefault("vlm2", None)
    kw.setdefault("detector", None)
    kw.setdefault("table_struct", None)
    return PdfParser(**kw)


# ---------------------------------------------------------------------------
# _fill_cells_by_containment -- text placed from real PDF chars by center
# containment, using real pdfplumber chars (so extract_text has full dicts).
# to_pt=1.0 keeps the detected-cell bboxes in the same point space as chars.
# ---------------------------------------------------------------------------


def _chars_of(tmp_path, lines):
    path = tmp_path / "doc.pdf"
    path.write_bytes(text_pdf(lines))
    with pdfplumber.open(path) as pdf_doc:
        return list(pdf_doc.pages[0].chars)


def test_fill_by_containment_places_text_despite_loose_cell_boxes(tmp_path):
    # "Alpha" ~ top 100-112, "Bravo" ~ top 140-152. The detected cell boxes
    # below are deliberately LOOSE (not tight around the glyphs) -- an exact
    # clip would be fragile, but center-containment still routes each word to
    # its row. This is the regression guard for the "structure correct, cells
    # empty" bug.
    chars = _chars_of(tmp_path, [(72, 100, 12, "Alpha"), (72, 140, 12, "Bravo")])
    dt = DetectedTable(
        bbox=(60, 95, 300, 160),
        cells=[
            DetectedCell(row=0, col=0, bbox=(60, 90, 300, 125)),
            DetectedCell(row=1, col=0, bbox=(60, 125, 300, 160)),
        ],
    )
    data, coverage = _fill_cells_by_containment(chars, dt, to_pt=1.0)

    assert (data.n_rows, data.n_cols) == (2, 1)
    assert data.cells[0][0].plain_text() == "Alpha"
    assert data.cells[1][0].plain_text() == "Bravo"
    assert coverage == pytest.approx(1.0)


def test_fill_by_containment_reports_low_coverage_when_text_unplaced(tmp_path):
    # Region spans both words but the single detected cell only covers the
    # first -> half the region's characters land nowhere, so coverage drops
    # (the signal _structure_model_tables uses to fall back to geometry).
    chars = _chars_of(tmp_path, [(72, 100, 12, "Alpha"), (72, 140, 12, "Bravo")])
    dt = DetectedTable(
        bbox=(60, 95, 300, 160),
        cells=[DetectedCell(row=0, col=0, bbox=(60, 90, 300, 125))],
    )
    data, coverage = _fill_cells_by_containment(chars, dt, to_pt=1.0)

    assert data.cells[0][0].plain_text() == "Alpha"
    assert coverage == pytest.approx(0.5)


def test_fill_by_containment_never_invents_text(tmp_path):
    # A cell that contains no character stays empty -- text is only ever real
    # PDF characters, never fabricated.
    chars = _chars_of(tmp_path, [(72, 100, 12, "Only")])
    dt = DetectedTable(
        bbox=(60, 95, 300, 200),
        cells=[
            DetectedCell(row=0, col=0, bbox=(60, 95, 300, 125)),
            DetectedCell(row=1, col=0, bbox=(60, 160, 300, 200)),  # empty band
        ],
    )
    data, _ = _fill_cells_by_containment(chars, dt, to_pt=1.0)
    assert data.cells[0][0].plain_text() == "Only"
    assert data.cells[1][0].plain_text() == ""


# ---------------------------------------------------------------------------
# _fill_cells_by_content -- text placed by matching the model's CLAIMED cell
# content against the region's real words; the claim itself is never emitted.
# to_pt=1.0 keeps the detected bboxes in the same point space as the words.
# ---------------------------------------------------------------------------


def _word(x0, top, text):
    return {"x0": x0, "x1": x0 + 8.0 * len(text), "top": top,
            "bottom": top + 10.0, "text": text}


def test_fill_by_content_places_real_words_not_the_claim():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    words = [_word(60, 100, "Voltage"), _word(200, 100, "5V"),
             _word(60, 140, "Weight"), _word(200, 140, "500g")]
    dt = DetectedTable(bbox=(50, 90, 300, 160), cells=[
        DetectedCell(row=0, col=0, text="VOLTAGE"),  # misread case
        DetectedCell(row=0, col=1, text="5V"),
        DetectedCell(row=1, col=0, text="Weight"),
        DetectedCell(row=1, col=1, text="500g"),
    ])
    data, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert (data.n_rows, data.n_cols) == (2, 2)
    # Output text is the PDF's own word, not the model's uppercased claim.
    assert data.cells[0][0].plain_text() == "Voltage"
    assert data.cells[0][1].plain_text() == "5V"
    assert data.cells[1][1].plain_text() == "500g"
    assert placed == pytest.approx(1.0)
    assert invented == 0


def test_fill_by_content_repeated_tokens_resolve_in_reading_order():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    words = [_word(60, 100, "1"), _word(200, 100, "GND"),
             _word(60, 140, "2"), _word(200, 140, "GND")]
    dt = DetectedTable(bbox=(50, 90, 300, 160), cells=[
        DetectedCell(row=0, col=0, text="1"),
        DetectedCell(row=0, col=1, text="GND"),
        DetectedCell(row=1, col=0, text="2"),
        DetectedCell(row=1, col=1, text="GND"),
    ])
    data, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert data.cells[0][1].plain_text() == "GND"
    assert data.cells[1][1].plain_text() == "GND"
    assert placed == pytest.approx(1.0)
    assert invented == 0


def test_fill_by_content_detects_data_loss():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    # The model's matrix simply omitted "500g" -- a grid that loses real text
    # must be measurable so the caller can fall back to line geometry.
    words = [_word(60, 100, "Weight"), _word(200, 100, "500g")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="Weight"),
        DetectedCell(row=0, col=1, text=""),
    ])
    _, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert placed == pytest.approx(0.5)
    assert invented == 0


def test_fill_by_content_flags_hallucinated_numbers():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    words = [_word(60, 100, "Voltage"), _word(200, 100, "5V")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="Voltage"),
        DetectedCell(row=0, col=1, text="6V"),  # number not on the page
    ])
    data, _, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert invented == 1
    # And the fabricated value never reached the output.
    assert data.cells[0][1].plain_text() == ""


def test_fill_by_content_lone_digit_claim_is_not_invented():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    # A single-character digit claim that matches nothing (a stray footnote
    # marker misread, "¹" -> "1") must NOT count as invented -- too weak to
    # reject a whole grid on. Only substantial (>=2-char) numbers are hard.
    words = [_word(60, 100, "Voltage")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="Voltage"),
        DetectedCell(row=0, col=1, text="1"),  # lone digit not on the page
    ])
    _, _, invented = _fill_cells_by_content(words, dt, to_pt=1.0)
    assert invented == 0


def test_fill_by_content_matches_dash_glyph_variant():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    # PDF prints "not applicable" as a horizontal bar (U+2015); the model wrote
    # an ASCII hyphen. Glyph folding makes them match, so the dash cell places
    # and the grid isn't rejected -- and the emitted text is the PDF's own char.
    words = [_word(60, 100, "Input"), _word(200, 100, "―")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="Input"),
        DetectedCell(row=0, col=1, text="-"),
    ])
    data, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert placed == pytest.approx(1.0)   # the alnum content word placed
    assert invented == 0
    assert data.cells[0][1].plain_text() == "―"   # real PDF glyph, not the model's


def test_fill_by_content_matches_superscript_footnote():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    # PDF has a superscript footnote marker "¹" (U+00B9); the model read it as
    # ASCII "1". Folding lets it match; the real superscript reaches the cell.
    words = [_word(60, 100, "5"), _word(75, 100, "A"), _word(90, 100, "¹")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="5 A 1"),
    ])
    data, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert placed == pytest.approx(1.0)
    assert invented == 0
    assert "¹" in data.cells[0][0].plain_text()


# ---------------------------------------------------------------------------
# Source-agnostic health gate (_apply_health) -- flags a lossy/odd grid
# whatever built it, without ever changing cell text.
# ---------------------------------------------------------------------------


def test_apply_health_flags_content_dropout():
    from medrag.pipeline.parser.parsers.base import TableData, text_cell

    # region holds "Alpha" and "Beta"; the grid dropped "Beta".
    words = [_word(60, 100, "Alpha"), _word(200, 100, "Beta")]
    data = TableData(n_rows=1, n_cols=2,
                     cells=[[text_cell("Alpha"), text_cell("")]])
    pt = _PreparedTable(bbox=(50, 90, 300, 120), data=data,
                        provenance=PROV_TABLE_BANDS, confidence=1.0)
    _apply_health(pt, words)

    assert "content-dropout" in pt.flags
    assert pt.confidence < 1.0     # folded the ~0.5 dropout into confidence


def test_apply_health_flags_sparse_col0():
    from medrag.pipeline.parser.parsers.base import TableData, text_cell

    # header + two body rows whose first cell is empty while the rest isn't --
    # the lattice column-0 dropout signature.
    data = TableData(n_rows=3, n_cols=2, cells=[
        [text_cell("H1"), text_cell("H2")],
        [text_cell(""), text_cell("x")],
        [text_cell(""), text_cell("y")],
    ])
    pt = _PreparedTable(bbox=(0, 0, 100, 100), data=data,
                        provenance=PROV_TABLE_BANDS, confidence=1.0)
    _apply_health(pt, words=[])   # no region words -> dropout 0, isolates the flag

    assert "sparse-col0" in pt.flags


def test_sparse_first_column_ignores_header_and_dash_rows():
    from medrag.pipeline.parser.parsers.base import TableData, text_cell

    # header row (skipped) + rows whose first cell is present -> not sparse.
    ok = TableData(n_rows=3, n_cols=2, cells=[
        [text_cell("Spec"), text_cell("Val")],
        [text_cell("A"), text_cell("1")],
        [text_cell("B"), text_cell("2")],
    ])
    assert not _sparse_first_column(ok)


def test_fill_by_content_matches_across_tokenization_drift():
    from medrag.pipeline.parser.parsers.pdf_parser import _fill_cells_by_content

    # The text layer split what the model read as one token ("5VDC,1A" vs
    # "5 V DC, 1 A") -- the concat-run fallback still places every word.
    words = [_word(60, 100, "5"), _word(75, 100, "V"), _word(90, 100, "DC,"),
             _word(120, 100, "1"), _word(135, 100, "A")]
    dt = DetectedTable(bbox=(50, 90, 300, 120), cells=[
        DetectedCell(row=0, col=0, text="5VDC,1A"),
    ])
    data, placed, invented = _fill_cells_by_content(words, dt, to_pt=1.0)

    assert data.cells[0][0].plain_text() == "5 V DC, 1 A"
    assert placed == pytest.approx(1.0)
    assert invented == 0


def test_best_overlap_index_picks_max_overlap():
    a = _StubGeomTable((0, 0, 100, 100))
    b = _StubGeomTable((90, 90, 300, 300))
    assert _best_overlap_index((80, 80, 200, 200), [a, b]) == 1
    assert _best_overlap_index((0, 0, 50, 50), [a, b]) == 0
    assert _best_overlap_index((500, 500, 600, 600), [a, b]) is None


# ---------------------------------------------------------------------------
# _as_table_data / _PreparedTable
# ---------------------------------------------------------------------------


def test_as_table_data_passes_prepared_table_through():
    from medrag.pipeline.parser.parsers.base import Cell, ParagraphBlock, TableData

    data = TableData(n_rows=1, n_cols=1,
                     cells=[[Cell(blocks=[ParagraphBlock(id="p0", text="x")])]])
    prepared = _PreparedTable(bbox=(0, 0, 10, 10), data=data)
    assert _as_table_data(prepared) is data


# ---------------------------------------------------------------------------
# Structure-model refinement: _structure_model_tables refines pdfplumber's
# found REGIONS (geom_tables) rather than discovering tables itself -- so
# these call it directly with a stub geom_tables list (a bare object with
# just `.bbox`, all _structure_model_tables ever reads off it) instead of
# needing a synthetic PDF with real ruling lines for find_tables() to find.
# ---------------------------------------------------------------------------


class _StubGeomTable:
    """Minimal stand-in for a pdfplumber Table -- only `.bbox` is read."""

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        self.bbox = bbox


class FakeTableStructureClient:
    """Returns pre-baked geometry; deliberately claims implausible cell text
    is irrelevant -- the point of the design is that pdf_parser.py ignores
    whatever "text" the model might imply and pulls the real digital text
    itself, so this fake never even offers a text field."""

    def __init__(self, tables: list[DetectedTable]) -> None:
        self._tables = tables
        self.calls: list[tuple[list, list[TextCellHint]]] = []

    def detect(self, image: bytes, *, table_bboxes=(), text_cells=()) -> list[DetectedTable]:
        self.calls.append((list(table_bboxes), list(text_cells)))
        return self._tables


def test_structure_model_table_uses_digital_text(tmp_path):
    # Two cells stacked vertically, matching two text_pdf lines exactly.
    pdf = text_pdf([
        (72, 100, 12, "Spec"),
        (72, 120, 12, "42"),
    ])
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf)

    dpi = 150
    to_px = dpi / 72.0

    def px(bbox_pt):
        return tuple(v * to_px for v in bbox_pt)

    region_bbox_pt = (60, 95, 300, 140)
    detected = DetectedTable(
        bbox=px(region_bbox_pt),
        cells=[
            DetectedCell(row=0, col=0, bbox=px((60, 95, 300, 118))),
            DetectedCell(row=1, col=0, bbox=px((60, 115, 300, 140))),
        ],
    )
    fake = FakeTableStructureClient([detected])
    parser = offline_parser(table_struct=fake)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        png = parser._render(page)
        # _StubGeomTable.bbox mirrors a real pdfplumber Table.bbox: PDF POINT
        # space. _structure_model_tables converts it to pixels itself before
        # calling detect(), same as it does for a real Table's .bbox.
        geom_tables = [_StubGeomTable(region_bbox_pt)]
        result = parser._structure_model_tables(page, png, geom_tables)

    assert result is not None
    assert len(result) == 1
    table = result[0]
    assert table.data.n_rows == 2 and table.data.n_cols == 1
    assert table.data.cells[0][0].plain_text() == "Spec"
    assert table.data.cells[1][0].plain_text() == "42"

    # the fake was handed the region bbox converted to pixel space, and real
    # digital text cells to match against -- not just a bare image
    assert fake.calls, "detect() was never called"
    called_bboxes, called_hints = fake.calls[0]
    assert called_bboxes == [px(region_bbox_pt)]
    hint_texts = {h.text for h in called_hints}
    assert "Spec" in hint_texts and "42" in hint_texts


def test_structure_model_rowspan_becomes_merge(tmp_path):
    pdf = text_pdf([(72, 100, 12, "Merged")])
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf)

    dpi = 150
    to_px = dpi / 72.0

    def px(bbox_pt):
        return tuple(v * to_px for v in bbox_pt)

    region_bbox_pt = (60, 95, 300, 140)
    detected = DetectedTable(
        bbox=px(region_bbox_pt),
        cells=[
            DetectedCell(row=0, col=0, rowspan=2, colspan=1, bbox=px(region_bbox_pt)),
        ],
    )
    fake = FakeTableStructureClient([detected])
    parser = offline_parser(table_struct=fake)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        png = parser._render(page)
        geom_tables = [_StubGeomTable(region_bbox_pt)]  # PDF point space
        result = parser._structure_model_tables(page, png, geom_tables)

    table = result[0]
    assert table.data.n_rows == 2
    assert len(table.data.merges) == 1
    m = table.data.merges[0]
    assert (m.row, m.col, m.rowspan, m.colspan) == (0, 0, 2, 1)
    assert table.data.cells[1][0] is None  # covered by the merge


def test_structure_model_low_coverage_rejects_region_as_none(tmp_path):
    # The model returns a plausible 2x1 grid but its cell boxes sit far from
    # the real text (coverage ~0). The referee must REJECT it -- signalled as
    # None in the aligned result -- rather than emit an empty grid or fall back
    # to raw lattice here. The caller (_build_region_tables) then uses its own
    # deterministic band grid for the region; a rejected referee never emits.
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    dpi = 150
    to_px = dpi / 72.0

    def px(bbox_pt):
        return tuple(v * to_px for v in bbox_pt)

    region_pt = (60, 88, 300, 172)  # overlaps the real table region
    detected = DetectedTable(
        bbox=px(region_pt),
        cells=[
            DetectedCell(row=0, col=0, bbox=px((400, 400, 450, 450))),  # off the text
            DetectedCell(row=1, col=0, bbox=px((400, 460, 450, 500))),
        ],
    )
    fake = FakeTableStructureClient([detected])
    parser = offline_parser(table_struct=fake)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        png = parser._render(page)
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        assert geom_tables, "grid_pdf() should yield a line-detected table"
        result = parser._structure_model_tables(page, png, geom_tables)

    # aligned to the one input region, and that slot is None (rejected)
    assert result == [None]


def test_build_region_tables_uses_bands_when_referee_rejects(tmp_path):
    # End to end for the fallback the test above sets up: the referee rejects
    # the badly-grounded grid (None), so _build_region_tables keeps the
    # deterministic band-built grid, which recovers the real cell text.
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    dpi = 150
    to_px = dpi / 72.0
    detected = DetectedTable(
        bbox=tuple(v * to_px for v in (60, 88, 300, 172)),
        cells=[DetectedCell(row=0, col=0, bbox=tuple(v * to_px for v in (400, 400, 450, 450)))],
    )
    parser = offline_parser(table_struct=FakeTableStructureClient([detected]))

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, _diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert len(tables) == 1
    texts = [c.plain_text() for row in tables[0].data.cells for c in row if c]
    assert "Alpha" in texts and "Bravo" in texts  # bands recovered the text
    assert tables[0].provenance == "table-bands"


class _FakeClassifyVLM:
    """Returns a canned classification verdict; records call count."""

    def __init__(self, visual_type: str, confidence: float = 0.9) -> None:
        self.visual_type = visual_type
        self.confidence = confidence
        self.calls = 0

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls += 1
        return json.dumps({"type": self.visual_type, "confidence": self.confidence})


class _FakePageRegionVLM:
    """Returns a canned classify_page_regions-shaped reply (one region,
    confidently covering grid_pdf()'s table bbox); records call count."""

    def __init__(self, visual_type: str, confidence: float = 0.9) -> None:
        self.visual_type = visual_type
        self.confidence = confidence
        self.calls = 0

    def complete_vision(self, *, system, user, images, max_tokens=2048):
        self.calls += 1
        return json.dumps({"regions": [
            {"type": self.visual_type, "bbox": [0.0, 0.0, 0.6, 0.3],
             "confidence": self.confidence},
        ]})


def test_build_region_tables_page_region_match_skips_crop_classify(tmp_path, monkeypatch):
    """A candidate confidently covered by a whole-page region call (see
    _best_region_match) is confirmed a table without a second, per-candidate
    classify_crop call -- the cost reduction _build_region_tables' page-
    region-first docstring describes."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    vlm = _FakePageRegionVLM("table")
    parser = offline_parser(vlm=vlm)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert len(tables) == 1
    assert diverted == []
    assert vlm.calls == 1  # only the page-region call -- no crop fallback needed


def test_build_region_tables_page_region_match_diverts_without_crop_classify(
        tmp_path, monkeypatch):
    """A candidate confidently covered by a whole-page region typed as
    something other than "table" is diverted directly from that match --
    still without a second classify_crop call."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    vlm = _FakePageRegionVLM("block_diagram")
    parser = offline_parser(vlm=vlm)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert tables == []
    assert vlm.calls == 1
    assert len(diverted) == 1
    entry = diverted[0]
    assert entry["visual_type"] == "block_diagram"
    assert entry["visual_type_source"] == "vlm"
    assert entry["crop_sha"]


def test_build_region_tables_diverts_non_table_classification(tmp_path, monkeypatch):
    """A geometrically-detected region the classifier says is NOT a table
    (the "AAF"/"Digital Input" false-positive class) is diverted out of
    `tables` entirely -- no band/health work wasted on it -- and returned in
    `diverted` with its type and a stored crop, ready to become an
    ImageBlock instead."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    vlm = _FakeClassifyVLM("block_diagram")
    parser = offline_parser(vlm=vlm)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert tables == []
    # 2 calls: the whole-page classify_page_regions call first (this fake's
    # reply has no "regions" key, so it parses to zero regions -- no
    # confident match for the candidate), then the per-candidate
    # classify_crop fallback that actually produces the verdict below (see
    # _build_region_tables' page-region-first, crop-fallback docstring).
    assert vlm.calls == 2
    assert len(diverted) == 1
    entry = diverted[0]
    assert entry["visual_type"] == "block_diagram"
    assert entry["visual_type_source"] == "vlm"
    assert entry["crop_sha"]
    assert (tmp_path / "images" / f"{hash_hex(entry['crop_sha'])}.png").is_file()


def test_build_region_tables_keeps_table_when_classified_as_table(tmp_path, monkeypatch):
    """Regression guard: a region the classifier confirms IS a table behaves
    exactly as before classification existed."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("STORAGE_IMAGES_DIR", str(tmp_path / "images"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    vlm = _FakeClassifyVLM("table")
    parser = offline_parser(vlm=vlm)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert len(tables) == 1
    assert diverted == []
    texts = [c.plain_text() for row in tables[0].data.cells for c in row if c]
    assert "Alpha" in texts and "Bravo" in texts


def test_build_region_tables_fails_open_without_vlm(tmp_path, monkeypatch):
    """No VLM configured/reachable -> classification is skipped entirely
    (fail-open, trust the geometry) rather than diverting every real table
    to "unknown" just because no one could be asked."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    parser = offline_parser()  # vlm=None

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert len(tables) == 1
    assert diverted == []


def test_build_region_tables_classify_disabled_via_config(tmp_path, monkeypatch):
    """`[visual].classify=False` also skips classification even with a VLM
    available -- the config kill switch."""
    monkeypatch.setenv("STORAGE_LABELS_DIR", str(tmp_path / "labels"))
    monkeypatch.setenv("VISUAL_CLASSIFY", "0")
    path = tmp_path / "grid.pdf"
    path.write_bytes(grid_pdf())

    vlm = _FakeClassifyVLM("block_diagram")
    parser = offline_parser(vlm=vlm)

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        words = page.extract_words(extra_attrs=["size", "fontname"])
        geom_tables = [t for t in page.find_tables() if not _is_degenerate_table(t)]
        tables, diverted, _png = parser._build_region_tables(
            page, geom_tables, words, 1, "doc")

    assert len(tables) == 1
    assert diverted == []
    assert vlm.calls == 0


def test_structure_model_unconfigured_returns_none(tmp_path):
    pdf = text_pdf([(72, 100, 12, "No structure model here")])
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf)

    parser = offline_parser()  # table_struct=None by default
    assert parser._table_structure() is None

    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        png = parser._render(page)
        result = parser._structure_model_tables(
            page, png, [_StubGeomTable((0, 0, 10, 10))])
    assert result is None


def test_structure_model_error_falls_back_gracefully(tmp_path):
    from medrag.pipeline.parser.tables.structure import TableStructureError

    class FailingClient:
        def detect(self, image, *, table_bboxes=(), text_cells=()):
            raise TableStructureError("model unavailable")

    pdf = text_pdf([(72, 100, 12, "Plain text, no table")])
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf)

    parser = offline_parser(table_struct=FailingClient())
    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        png = parser._render(page)
        result = parser._structure_model_tables(
            page, png, [_StubGeomTable((0, 0, 10, 10))])
    assert result is None  # caller falls back to geom_tables unrefined

    # And the full parse() must still succeed end to end despite the failure.
    doc = parser.parse(path, "doc1")
    assert doc.page_count == 1


def test_parse_code_tags_prepared_table_with_structure_provenance(tmp_path):
    from medrag.pipeline.parser.parsers.base import Cell, ParagraphBlock, TableData
    from medrag.pipeline.parser.parsers.pdf_parser import _Counters

    pdf = text_pdf([(72, 100, 12, "unrelated paragraph")])
    path = tmp_path / "doc.pdf"
    path.write_bytes(pdf)

    data = TableData(n_rows=1, n_cols=1,
                     cells=[[Cell(blocks=[ParagraphBlock(id="p0", text="x")])]])
    prepared = _PreparedTable(bbox=(0.0, 0.0, 10.0, 10.0), data=data)

    parser = offline_parser()
    with pdfplumber.open(path) as pdf_doc:
        page = pdf_doc.pages[0]
        blocks = parser._parse_code(
            page, pageno=1,
            prep={"tables": [prepared], "gutter": None, "words": []},
            ct=_Counters(),
        )

    tables = [b for b in blocks if isinstance(b, TableBlock)]
    assert len(tables) == 1
    assert tables[0].provenance == PROV_TABLE_STRUCTURE
    assert tables[0].table is data


# ---------------------------------------------------------------------------
# HttpTableStructureClient
# ---------------------------------------------------------------------------


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_http_client_parses_tables_and_sends_text_cells(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeHTTPResponse({
            "tables": [{
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "cells": [
                    {"row": 0, "col": 0, "bbox": [1.0, 2.0, 2.0, 3.0]},
                    {"row": 0, "col": 1, "rowspan": 1, "colspan": 2,
                     "bbox": [2.0, 2.0, 3.0, 3.0]},
                ],
            }],
        })

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = HttpTableStructureClient(base_url="http://example.test/detect")
    result = client.detect(b"fake-png-bytes",
                           table_bboxes=[(5.0, 6.0, 7.0, 8.0)],
                           text_cells=[TextCellHint(text="hi", bbox=(0, 0, 1, 1))])

    assert len(result) == 1
    assert result[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert len(result[0].cells) == 2
    assert result[0].cells[1].colspan == 2

    assert captured["body"]["text_cells"] == [{"text": "hi", "bbox": [0, 0, 1, 1]}]
    assert captured["body"]["table_bboxes"] == [[5.0, 6.0, 7.0, 8.0]]


def test_http_client_bad_shape_raises_table_structure_error(monkeypatch):
    from medrag.pipeline.parser.tables.structure import TableStructureError

    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse({"unexpected": "shape"})

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    client = HttpTableStructureClient(base_url="http://example.test/detect")
    with pytest.raises(TableStructureError):
        client.detect(b"fake-png-bytes")


# ---------------------------------------------------------------------------
# VLMStructureAdapter -- reuses an existing VLMClient, no extra dependency
# ---------------------------------------------------------------------------


class _FakeVLM:
    """Stands in for llm.VLMClient -- records what it was asked and returns
    a canned JSON response per call (one per table_bbox, in order)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete_vision(self, *, system, user, images, max_tokens=4096):
        self.calls.append({"system": system, "user": user,
                           "n_images": len(images), "max_tokens": max_tokens})
        return self._responses[len(self.calls) - 1]


def _make_page_png(width=200, height=100) -> bytes:
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (width, height), color=(255, 255, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_vlm_adapter_returns_claimed_cell_text():
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    response = json.dumps([
        {"row": 0, "col": 0, "text": "Specification"},
        {"row": 0, "col": 1, "text": "Min"},
    ])
    fake = _FakeVLM([response])
    adapter = VLMStructureAdapter(vlm_client=fake)

    png = _make_page_png(200, 100)
    result = adapter.detect(png, table_bboxes=[(50.0, 20.0, 150.0, 80.0)])

    assert len(result) == 1
    table = result[0]
    assert table.bbox == (50.0, 20.0, 150.0, 80.0)
    assert len(table.cells) == 2

    left, right = table.cells
    assert (left.row, left.col) == (0, 0)
    assert left.text == "Specification"
    assert (right.row, right.col) == (0, 1)
    assert right.text == "Min"

    assert len(fake.calls) == 1
    assert fake.calls[0]["n_images"] == 1


def test_vlm_adapter_rowspan_colspan_and_fenced_json():
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    response = "```json\n" + json.dumps([
        {"row": 0, "col": 0, "rowspan": 2, "colspan": 1, "text": "merged"},
    ]) + "\n```"
    fake = _FakeVLM([response])
    adapter = VLMStructureAdapter(vlm_client=fake)

    png = _make_page_png(200, 100)
    result = adapter.detect(png, table_bboxes=[(0.0, 0.0, 100.0, 100.0)])

    cell = result[0].cells[0]
    assert (cell.rowspan, cell.colspan) == (2, 1)
    assert cell.text == "merged"


def test_vlm_adapter_no_table_bboxes_returns_empty_without_calling_vlm():
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    fake = _FakeVLM([])
    adapter = VLMStructureAdapter(vlm_client=fake)
    result = adapter.detect(_make_page_png(), table_bboxes=[])

    assert result == []
    assert fake.calls == []


def test_vlm_adapter_bad_json_skips_only_that_region():
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    # One region's response is unparseable, another's is fine. The bad region
    # is isolated (skipped) so the good one still refines -- the caller has a
    # per-region deterministic fallback for whatever comes back missing, so a
    # single truncated response must not discard every table on the page.
    good = json.dumps([{"row": 0, "col": 0, "text": "OK"}])
    fake = _FakeVLM(["not json at all", good])
    adapter = VLMStructureAdapter(vlm_client=fake)

    result = adapter.detect(
        _make_page_png(),
        table_bboxes=[(0.0, 0.0, 50.0, 50.0), (0.0, 50.0, 50.0, 100.0)])

    assert len(result) == 1
    assert result[0].bbox == (0.0, 50.0, 50.0, 100.0)
    assert result[0].cells[0].text == "OK"


def test_vlm_adapter_transport_error_raises():
    from medrag.pipeline.parser.tables.structure import TableStructureError
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    class _DeadVLM:
        def complete_vision(self, **kw):
            raise RuntimeError("connection refused")

    adapter = VLMStructureAdapter(vlm_client=_DeadVLM())

    # A failed CALL (model/transport down) affects every region, so it still
    # raises -- the caller stops trying this page rather than looping N dead calls.
    with pytest.raises(TableStructureError):
        adapter.detect(_make_page_png(), table_bboxes=[(0.0, 0.0, 50.0, 50.0)])


def test_vlm_adapter_valid_empty_array_skips_table_without_raising():
    from medrag.pipeline.parser.tables.structure.vlm_adapter import VLMStructureAdapter

    fake = _FakeVLM(["[]"])
    adapter = VLMStructureAdapter(vlm_client=fake)
    result = adapter.detect(_make_page_png(), table_bboxes=[(0.0, 0.0, 50.0, 50.0)])

    assert result == []  # a genuinely empty (but valid) response is not an error
