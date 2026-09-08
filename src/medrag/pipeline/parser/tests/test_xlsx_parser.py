"""xlsx_parser.py: embedded pictures (xl/drawings) extraction and the
numeric (not lexicographic) sheetN.xml ordering fix.
"""

from __future__ import annotations

import openpyxl
from openpyxl.drawing.image import Image as XLImage
from PIL import Image as PILImage

from medrag.pipeline.parser.parsers.base import HeadingBlock, ImageBlock, TableBlock
from medrag.pipeline.parser.parsers.xlsx_parser import XlsxParser


def _png(tmp_path, name="pic.png"):
    p = tmp_path / name
    PILImage.new("RGB", (4, 4), color="red").save(p)
    return str(p)


def test_basic_sheet_and_merges(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "Header"
    ws["A2"] = "Value"
    ws.merge_cells("A1:B1")
    xlsx = tmp_path / "basic.xlsx"
    wb.save(xlsx)

    doc = XlsxParser().parse(xlsx, "doc")
    heading = next(b for b in doc.blocks if isinstance(b, HeadingBlock))
    table = next(b for b in doc.blocks if isinstance(b, TableBlock))
    assert heading.text == "Data"
    assert table.table.cells[0][0].plain_text() == "Header"
    assert len(table.table.merges) == 1


def test_embedded_image_extracted(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws["A1"] = "x"
    ws.add_image(XLImage(_png(tmp_path)), "C2")
    xlsx = tmp_path / "img.xlsx"
    wb.save(xlsx)

    doc = XlsxParser().parse(xlsx, "doc")
    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    assert images[0].locator["part"] == "xl/media/image1.png"
    assert images[0].locator["anchor_row"] == 1   # C2 -> 0-based row 1
    assert images[0].locator["anchor_col"] == 2   # C2 -> 0-based col 2
    assert images[0].mime == "image/png"


def test_image_only_sheet_not_dropped(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "x"
    pic_only = wb.create_sheet("PicOnly")
    pic_only.add_image(XLImage(_png(tmp_path)), "A1")
    xlsx = tmp_path / "piconly.xlsx"
    wb.save(xlsx)

    doc = XlsxParser().parse(xlsx, "doc")
    headings = [b.text for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert "PicOnly" in headings
    # the block right after the PicOnly heading is its image, not a TableBlock
    idx = doc.blocks.index(next(b for b in doc.blocks
                                if isinstance(b, HeadingBlock) and b.text == "PicOnly"))
    assert isinstance(doc.blocks[idx + 1], ImageBlock)


def test_native_chart_becomes_image_with_deterministic_description(tmp_path):
    """A `graphicFrame` anchor whose graphicData names the chart schema
    used to be silently skipped entirely (module docstring's own documented
    gap, `if pic is None: continue`) -- now becomes an ImageBlock tagged
    "chart" with a description read straight from the chart's own data.

    openpyxl writes a chart's cat/val as a live formula reference with no
    cached points (`<c:f>...</c:f>`, no `c:numCache`/`c:strCache`) -- valid
    OOXML, but nothing for chart_extract.py to read without Excel itself
    recalculating first. A real Excel-authored file always ships the cache
    (so viewers that can't recalculate still render something), so the
    chart XML is replaced here with an equivalent, fully-cached one -- same
    schema, same relationship wiring, just populated -- to test what a real
    corpus file actually looks like.
    """
    import zipfile

    from openpyxl.chart import BarChart, Reference

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    for row in (["Q1", 10], ["Q2", 15], ["Q3", 12]):
        ws.append(row)
    chart = BarChart()
    values = Reference(ws, min_col=2, min_row=1, max_row=3)
    cats = Reference(ws, min_col=1, min_row=1, max_row=3)
    chart.add_data(values, titles_from_data=False)
    chart.set_categories(cats)
    ws.add_chart(chart, "D2")
    xlsx = tmp_path / "chart.xlsx"
    wb.save(xlsx)

    cached_chart_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
        b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        b'<c:chart><c:plotArea><c:barChart><c:ser><c:idx val="0"/>'
        b'<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt>'
        b'</c:strCache></c:strRef></c:tx>'
        b'<c:cat><c:strRef><c:strCache>'
        b'<c:pt idx="0"><c:v>Q1</c:v></c:pt><c:pt idx="1"><c:v>Q2</c:v></c:pt>'
        b'<c:pt idx="2"><c:v>Q3</c:v></c:pt></c:strCache></c:strRef></c:cat>'
        b'<c:val><c:numRef><c:numCache>'
        b'<c:pt idx="0"><c:v>10</c:v></c:pt><c:pt idx="1"><c:v>15</c:v></c:pt>'
        b'<c:pt idx="2"><c:v>12</c:v></c:pt></c:numCache></c:numRef></c:val>'
        b'</c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>')

    # Rewrite the zip, swapping chart1.xml for the cached equivalent above --
    # everything else (drawing, rels, sheet data) is untouched.
    with zipfile.ZipFile(xlsx) as zf:
        entries = {n: zf.read(n) for n in zf.namelist()}
    entries["xl/charts/chart1.xml"] = cached_chart_xml
    with zipfile.ZipFile(xlsx, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            zf.writestr(name, content)

    doc = XlsxParser().parse(xlsx, "doc")
    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    img = images[0]
    assert img.visual_type == "chart"
    assert img.visual_type_source == "structural"
    assert img.description == (
        "Bar chart. Series 'Revenue': Q1=10.0, Q2=15.0, Q3=12.0.")
    assert img.locator.get("part") is None  # no media-part locator for a chart
    # the sheet's real table is untouched by the chart
    table = next(b for b in doc.blocks if isinstance(b, TableBlock))
    assert table.table.cells[0][0].plain_text() == "Q1"


def test_many_sheets_numeric_order(tmp_path):
    # 11 sheets so "sheet10.xml"/"sheet11.xml" would sort before "sheet2.xml"
    # under a naive lexicographic string sort.
    wb = openpyxl.Workbook()
    wb.active.title = "S1"
    wb.active["A1"] = "v1"
    for i in range(2, 12):
        ws = wb.create_sheet(f"S{i}")
        ws["A1"] = f"v{i}"
    xlsx = tmp_path / "many.xlsx"
    wb.save(xlsx)

    doc = XlsxParser().parse(xlsx, "doc")
    tables = [b for b in doc.blocks if isinstance(b, TableBlock)]
    assert len(tables) == 11
    for i, t in enumerate(tables, start=1):
        assert t.table.cells[0][0].plain_text() == f"v{i}"
        assert t.span.part == f"xl/worksheets/sheet{i}.xml"
