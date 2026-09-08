"""Tests for the deterministic band table-grid builder (parsers/table_bands.py).

No model of any kind is involved -- the builder reads a page's own vector
geometry. Synthetic one-page PDFs are drawn at the content-stream level (fill
rectangles, ruling segments, text) so each geometry pattern from the ACME
corpus can be reproduced exactly and asserted on:

  * a borderless-left / zebra-shaded table whose unshaded rows lose their
    column-0 cell under pdfplumber's lattice (the dominant real-world defect),
  * a header set in a band above the ruled area,
  * per-cell padding rectangles that explode the column count,
  * a wrapped multi-line cell that naive line-clustering would split,
  * a lineless whitespace-aligned table (lower confidence, no invented merges).
"""

from __future__ import annotations

import pdfplumber

from medrag.pipeline.parser.parsers.table_bands import build_band_table

# ---------------------------------------------------------------------------
# Content-stream PDF builder (draws fills, ruling segments and text directly)
# ---------------------------------------------------------------------------


def _build_pdf(content: bytes) -> bytes:
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<<" + f" /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
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


def _content(*, fills=(), vlines=(), hlines=(), texts=()) -> bytes:
    """fills: (x0, top, x1, bottom). vlines: (x, top, bottom).
    hlines: (y, x0, x1). texts: (x, top, size, text). Top-based coords
    (like pdfplumber); converted to PDF bottom-left here."""
    ops: list[str] = []
    for x0, top, x1, bottom in fills:
        ops.append(f"0.86 0.86 0.86 rg {x0:.2f} {792 - bottom:.2f} "
                   f"{x1 - x0:.2f} {bottom - top:.2f} re f")
    ops.append("0 0 0 RG 1 w")
    for x, t0, t1 in vlines:
        ops.append(f"{x:.2f} {792 - t0:.2f} m {x:.2f} {792 - t1:.2f} l S")
    for y, x0, x1 in hlines:
        ops.append(f"{x0:.2f} {792 - y:.2f} m {x1:.2f} {792 - y:.2f} l S")
    ops.append("0 0 0 rg")
    for x, top, size, text in texts:
        esc = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        ops.append(f"BT /F1 {size} Tf {x:.2f} {792 - top - size:.2f} Td ({esc}) Tj ET")
    return ("\n".join(ops)).encode("latin-1")


def _page(tmp_path, content: bytes):
    path = tmp_path / "t.pdf"
    path.write_bytes(_build_pdf(content))
    pdf = pdfplumber.open(path)
    return pdf, pdf.pages[0]


def _rows_text(data):
    return [[(c.plain_text() if c is not None else None) for c in row]
            for row in data.cells]


def _all_text(data):
    return {c.plain_text() for row in data.cells for c in row
            if c is not None and c.plain_text()}


# ---------------------------------------------------------------------------
# The central case: zebra shading + no left rule -> lattice drops column 0
# on the unshaded rows; the band builder recovers it from the region-global
# column boundaries.
# ---------------------------------------------------------------------------


def _zebra_content():
    # region x 60..360, 3 columns (bounds 60/160/260/360), 3 data rows in
    # bands 100-120 / 120-140 / 140-160, plus a header line above at top 84.
    row_tops = [100, 120, 140, 160]
    hlines = [(y, 60, 360) for y in row_tops]           # per-full-width rules
    vlines = [(x, 100, 160) for x in (160, 260, 360)]   # NO left rule at x=60
    # zebra fills on rows 0 and 2 -> they alone draw the x=60 left edge
    fills = [(60, 100, 360, 120), (60, 140, 360, 160)]
    texts = [
        (64, 84, 10, "Name"), (180, 84, 10, "Min"), (300, 84, 10, "Max"),
        (64, 103, 10, "Alpha"), (180, 103, 10, "1"), (300, 103, 10, "9"),
        (64, 123, 10, "Bravo"), (180, 123, 10, "2"), (300, 123, 10, "8"),
        (64, 143, 10, "Gamma"), (180, 143, 10, "3"), (300, 143, 10, "7"),
    ]
    return _content(fills=fills, vlines=vlines, hlines=hlines, texts=texts)


def test_bands_recovers_column0_on_all_rows(tmp_path):
    pdf, page = _page(tmp_path, _zebra_content())
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (60, 100, 360, 160), words)
    finally:
        pdf.close()

    assert bt is not None
    col0 = [row[0] for row in _rows_text(bt.data)]
    # every data label present -- including "Bravo" on the UNSHADED row that
    # lattice would drop for want of a left border line
    assert "Alpha" in col0 and "Bravo" in col0 and "Gamma" in col0


def test_bands_recovers_header_band(tmp_path):
    pdf, page = _page(tmp_path, _zebra_content())
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (60, 100, 360, 160), words)
    finally:
        pdf.close()

    assert "header-recovered" in bt.flags
    header = _rows_text(bt.data)[0]
    # the header pulled in from above the region, split into its own cells
    assert header == ["Name", "Min", "Max"]


def test_bands_never_invents_text(tmp_path):
    pdf, page = _page(tmp_path, _zebra_content())
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        source = {w["text"] for w in words}
        bt = build_band_table(page, (60, 100, 360, 160), words)
    finally:
        pdf.close()

    # every emitted token comes from a real source word; nothing fabricated
    emitted = set()
    for cell in _all_text(bt.data):
        emitted.update(cell.split())
    assert emitted <= source


def test_bands_high_confidence_for_clean_ruled_table(tmp_path):
    pdf, page = _page(tmp_path, _zebra_content())
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (60, 100, 360, 160), words)
    finally:
        pdf.close()
    assert bt.confidence >= 0.9
    assert "word-straddle" not in bt.flags


# ---------------------------------------------------------------------------
# Per-cell padding rectangles invent narrow empty columns -> coalesced away.
# ---------------------------------------------------------------------------


def test_bands_coalesces_phantom_columns(tmp_path):
    # Each of 3 logical columns is drawn as its own shaded rect with a small
    # padding gap between them, so the vertical edges land at 6 distinct x's
    # (two per column). Naively that is 5 columns; coalescing the empty padding
    # strips must bring it back to 3.
    fills = []
    texts = []
    # columns occupy 60-150, 160-250, 260-350 (10pt padding gaps at 150-160,
    # 250-260). Draw each cell rect per row -> left+right edges per column.
    col_ranges = [(60, 150), (160, 250), (260, 350)]
    labels = [["A", "B", "C"], ["D", "E", "F"]]
    for r, top in enumerate((100, 120)):
        bottom = top + 20
        for (cx0, cx1) in col_ranges:
            fills.append((cx0, top, cx1, bottom))
        for (cx0, _cx1), txt in zip(col_ranges, labels[r]):
            texts.append((cx0 + 4, top + 5, 10, txt))
    pdf, page = _page(tmp_path, _content(fills=fills, texts=texts))
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (60, 100, 350, 140), words)
    finally:
        pdf.close()

    assert bt.data.n_cols == 3, _rows_text(bt.data)
    assert _rows_text(bt.data)[0] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# A wrapped multi-line cell must stay one row when ruling delimits rows.
# ---------------------------------------------------------------------------


def test_bands_groups_wrapped_cell_into_one_row(tmp_path):
    # 2 columns (x bounds 60/200/360), 2 logical rows delimited by full-width
    # rules at y 100/140/180.
    # Row 1's second cell wraps onto two visual lines (top 144 and 158); the
    # rule-defined band must keep them in a single row, not split into two.
    hlines = [(y, 60, 360) for y in (100, 140, 180)]
    vlines = [(x, 100, 180) for x in (200, 360)]
    fills = [(60, 100, 360, 140)]  # row 0 shaded -> left edge at 60
    texts = [
        (64, 116, 10, "Key"), (204, 116, 10, "Simple"),
        (64, 150, 10, "Note"),
        (204, 144, 10, "wrapped line one"),
        (204, 158, 10, "and line two"),
    ]
    pdf, page = _page(tmp_path, _content(fills=fills, vlines=vlines,
                                         hlines=hlines, texts=texts))
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (60, 100, 360, 180), words)
    finally:
        pdf.close()

    assert bt.data.n_rows == 2, _rows_text(bt.data)
    assert _rows_text(bt.data)[1][0] == "Note"
    assert bt.data.cells[1][1].plain_text() == "wrapped line one and line two"


# ---------------------------------------------------------------------------
# A lineless whitespace-aligned table: columns come from gaps, confidence is
# lower, and no spurious merges are invented (no ruling => no merge evidence).
# ---------------------------------------------------------------------------


def test_bands_lineless_uses_gaps_and_lowers_confidence(tmp_path):
    # No rules, no fills -- three columns separated only by wide whitespace,
    # consistent across every row.
    texts = []
    rows = [["Length", "10", "mm"], ["Width", "20", "mm"], ["Mass", "300", "g"]]
    for r, top in enumerate((100, 116, 132)):
        for (cx, txt) in zip((60, 200, 320), rows[r]):
            texts.append((cx, top, 10, txt))
    pdf, page = _page(tmp_path, _content(texts=texts))
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        bt = build_band_table(page, (55, 98, 360, 146), words)
    finally:
        pdf.close()

    assert bt.data.n_cols == 3, _rows_text(bt.data)
    assert "cols-from-gaps" in bt.flags
    assert bt.confidence < 0.9              # not a fully-pinned grid
    assert not bt.data.merges               # no ruling -> never invents a span
    assert _rows_text(bt.data)[0] == ["Length", "10", "mm"]


def test_bands_returns_none_for_empty_region(tmp_path):
    pdf, page = _page(tmp_path, _content(texts=[(64, 100, 10, "Alpha")]))
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        # a region far from the only word holds nothing placeable
        bt = build_band_table(page, (400, 400, 500, 460), words)
    finally:
        pdf.close()
    assert bt is None
