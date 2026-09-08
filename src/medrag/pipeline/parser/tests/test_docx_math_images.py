"""DOCX extraction fixes for known gaps:

1. Office Math (OMML m:t) text — was dropped because only w:t was read.
3. Legacy/OLE images (VML v:imagedata r:id) — were ignored (only a:blip handled).
4. Images inside table cells — were dropped (cell reader produced text only).

Items 1 and 4 also cover the body-vs-cell parity: the same `_walk_para` powers
both, so math/images work wherever they appear.
"""

from __future__ import annotations

import struct
import zlib
from io import BytesIO
from typing import ClassVar

import docx
from docx.oxml import parse_xml

from medrag.pipeline.parser.parsers.base import (
    ImageBlock,
    ParsedDocument,
)
from medrag.pipeline.parser.parsers.docx_parser import (
    DocxParser,
    _ImageEmit,
    _walk_para,
)


def _text(runs):
    return "".join(r.text for r in runs)


# --- helpers ----------------------------------------------------------------

_W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
_A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
_V = 'xmlns:v="urn:schemas-microsoft-com:vml"'
_R = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
_MC = 'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006"'

_OMATH = f'<m:oMath {_M}><m:r><m:t>∇φ</m:t></m:r></m:oMath>'  # ∇φ


def _png_1x1() -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 truecolor
    idat = zlib.compress(b"\x00\xff\x00\x00")            # one red pixel
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


class _FakePart:
    def __init__(self, name: str = "", ct: str = "", blob: bytes = b"") -> None:
        self.partname, self.content_type, self.blob = name, ct, blob


_CHART_XML = (
    b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    b'<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
    b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    b'<c:chart><c:plotArea><c:barChart><c:ser><c:idx val="0"/>'
    b'<c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt>'
    b'</c:strCache></c:strRef></c:tx>'
    b'<c:cat><c:strRef><c:strCache><c:pt idx="0"><c:v>Q1</c:v></c:pt>'
    b'<c:pt idx="1"><c:v>Q2</c:v></c:pt></c:strCache></c:strRef></c:cat>'
    b'<c:val><c:numRef><c:numCache><c:pt idx="0"><c:v>10</c:v></c:pt>'
    b'<c:pt idx="1"><c:v>20</c:v></c:pt></c:numCache></c:numRef></c:val>'
    b'</c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>'
)


_SMARTART_XML = (
    b'<dgm:dataModel '
    b'xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" '
    b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    b'<dgm:ptLst>'
    b'<dgm:pt modelId="0" type="doc"/>'
    b'<dgm:pt modelId="1"><dgm:t><a:p><a:r><a:t>Input</a:t></a:r></a:p></dgm:t></dgm:pt>'
    b'<dgm:pt modelId="2"><dgm:t><a:p><a:r><a:t>Output</a:t></a:r></a:p></dgm:t></dgm:pt>'
    b'</dgm:ptLst></dgm:dataModel>'
)


class _FakeDoc:
    """Stand-in for a python-docx Document exposing only related_parts."""
    class _P:
        related_parts: ClassVar[dict] = {
            "rId5": _FakePart("/word/media/imageA.png", "image/png"),
            "rId7": _FakePart("/word/media/imageB.png", "image/png"),
            "rIdChart1": _FakePart(blob=_CHART_XML),
            "rId1": _FakePart(blob=_SMARTART_XML),
        }
    part = _P()


# --- 1. Office Math ---------------------------------------------------------


def test_math_in_body_paragraph_is_captured():
    p = parse_xml(f'<w:p {_W} {_M}><w:r><w:t>x=</w:t></w:r>{_OMATH}</w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert _text(runs) == "x=∇φ"   # "x=∇φ", math no longer dropped
    assert images == []


def test_math_in_table_cell_is_captured(tmp_path):
    d = docx.Document()
    cell = d.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].add_run("x=")
    cell.paragraphs[0]._p.append(parse_xml(_OMATH))
    p = tmp_path / "math.docx"
    d.save(str(p))

    doc = DocxParser().parse(p, "math")
    txt = doc.tables()[0].table.cells[0][0].plain_text()
    assert "∇φ" in txt              # ∇φ present
    assert txt == "x=∇φ"


# --- 3. Legacy / OLE (VML) images -------------------------------------------


def test_vml_imagedata_becomes_an_image():
    p = parse_xml(
        f'<w:p {_W} {_V} {_R}><w:r><w:pict>'
        f'<v:shape><v:imagedata r:id="rId7"/></v:shape>'
        f'</w:pict></w:r></w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert _text(runs) == "<image1>"
    assert images == [_ImageEmit(image_index=1, part="word/media/imageB.png",
                                 mime="image/png")]


def test_alternate_content_image_is_not_double_counted():
    """mc:AlternateContent ships a modern (Choice/a:blip) and legacy
    (Fallback/v:imagedata) rendering of ONE image; count it once."""
    p = parse_xml(
        f'<w:p {_W} {_MC} {_A} {_V} {_R}><mc:AlternateContent>'
        f'<mc:Choice Requires="wps"><w:r><w:drawing>'
        f'<a:blip r:embed="rId5"/></w:drawing></w:r></mc:Choice>'
        f'<mc:Fallback><w:r><w:pict><v:shape>'
        f'<v:imagedata r:id="rId7"/></v:shape></w:pict></w:r></mc:Fallback>'
        f'</mc:AlternateContent></w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert images == [_ImageEmit(image_index=1, part="word/media/imageA.png",
                                 mime="image/png")]  # Choice only
    assert _text(runs).count("<image") == 1


# --- native chart / SmartArt -------------------------------------------------
# Both used to fall through every check in `visit()` (no a:blip, no w:t) and
# vanish with zero trace -- not even as a paragraph, unlike a plain unhandled
# tag. A picture's own a:graphicData (uri = the picture namespace, not chart/
# diagram) must keep falling through to find its nested a:blip -- verified
# below alongside the two new cases.

_CHART_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"
_DIAGRAM_NS = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
_PICTURE_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"


def test_docx_native_chart_becomes_image_with_deterministic_description():
    p = parse_xml(
        f'<w:p {_W} {_A} {_R}><w:r><w:drawing>'
        f'<a:graphic><a:graphicData uri="{_CHART_NS}">'
        f'<c:chart xmlns:c="{_CHART_NS}" r:id="rIdChart1"/>'
        f'</a:graphicData></a:graphic></w:drawing></w:r></w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert _text(runs) == "<image1>"
    assert len(images) == 1
    ref = images[0]
    assert ref.visual_type == "chart"
    assert ref.visual_type_source == "structural"
    assert ref.description == (
        "Bar chart. Series 'Revenue': Q1=10.0, Q2=20.0.")
    assert ref.part is None  # no media-part locator for a native chart


def test_docx_smartart_becomes_block_diagram_with_deterministic_description():
    p = parse_xml(
        f'<w:p {_W} {_A} {_R}><w:r><w:drawing>'
        f'<a:graphic><a:graphicData uri="{_DIAGRAM_NS}">'
        f'<dgm:relIds xmlns:dgm="{_DIAGRAM_NS}" '
        f'r:dm="rId1" r:lo="rId2" r:qs="rId3" r:cs="rId4"/>'
        f'</a:graphicData></a:graphic></w:drawing></w:r></w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert _text(runs) == "<image1>"
    assert len(images) == 1
    ref = images[0]
    assert ref.visual_type == "block_diagram"
    assert ref.visual_type_source == "heuristic"
    assert ref.description == "Diagram with 2 elements: Input; Output."
    assert ref.description_source == "structural"
    assert ref.part is None  # still no media-part locator: no pixels exist


def test_docx_smartart_extraction_disabled_by_config(monkeypatch):
    monkeypatch.setenv("VISUAL_EXTRACT_SMARTART", "0")
    p = parse_xml(
        f'<w:p {_W} {_A} {_R}><w:r><w:drawing>'
        f'<a:graphic><a:graphicData uri="{_DIAGRAM_NS}">'
        f'<dgm:relIds xmlns:dgm="{_DIAGRAM_NS}" '
        f'r:dm="rId1" r:lo="rId2" r:qs="rId3" r:cs="rId4"/>'
        f'</a:graphicData></a:graphic></w:drawing></w:r></w:p>')
    _runs, images = _walk_para(p, _FakeDoc(), [0])
    ref = images[0]
    assert ref.visual_type == "block_diagram"  # type still tagged
    assert ref.description is None             # but no extraction attempted


def test_docx_picture_wrapped_in_graphic_data_still_found():
    """A real picture's a:blip sits inside its OWN a:graphicData (uri = the
    picture namespace) -- the new chart/diagram uri check must fall through
    to the generic recursion for this uri, exactly like it always has,
    rather than swallowing the picture silently."""
    p = parse_xml(
        f'<w:p {_W} {_A} {_R}><w:r><w:drawing>'
        f'<a:graphic><a:graphicData uri="{_PICTURE_NS}">'
        f'<pic:pic xmlns:pic="{_PICTURE_NS}"><pic:blipFill>'
        f'<a:blip r:embed="rId5"/></pic:blipFill></pic:pic>'
        f'</a:graphicData></a:graphic></w:drawing></w:r></w:p>')
    runs, images = _walk_para(p, _FakeDoc(), [0])
    assert _text(runs) == "<image1>"
    assert images == [_ImageEmit(image_index=1, part="word/media/imageA.png",
                                 mime="image/png")]


def test_docx_chart_with_unresolvable_relationship_still_tags_type():
    """The chart part can't be resolved (missing/broken relationship) --
    still tagged "chart" (certain from the XML), just no description."""
    p = parse_xml(
        f'<w:p {_W} {_A} {_R}><w:r><w:drawing>'
        f'<a:graphic><a:graphicData uri="{_CHART_NS}">'
        f'<c:chart xmlns:c="{_CHART_NS}" r:id="rIdMissing"/>'
        f'</a:graphicData></a:graphic></w:drawing></w:r></w:p>')
    _runs, images = _walk_para(p, _FakeDoc(), [0])
    assert len(images) == 1
    assert images[0].visual_type == "chart"
    assert images[0].description is None


# --- 4. Images inside table cells -------------------------------------------


def test_image_in_table_cell_emits_imageblock(tmp_path):
    d = docx.Document()
    cell = d.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].add_run().add_picture(BytesIO(_png_1x1()))
    p = tmp_path / "cellimg.docx"
    d.save(str(p))

    doc = DocxParser().parse(p, "cellimg")
    # image did not leak to document body...
    assert doc.images() == [] or all(
        b.locator for b in doc.images())
    blocks = doc.tables()[0].table.cells[0][0].blocks
    imgs = [b for b in blocks if isinstance(b, ImageBlock)]
    assert len(imgs) == 1
    assert imgs[0].locator["part"].startswith("word/media/")
    # the marker shows in the flat text view exactly once
    assert doc.tables()[0].table.cells[0][0].plain_text() == imgs[0].marker


def test_body_and_cell_images_share_one_counter(tmp_path):
    """Image indices are globally sequential in reading order (body then cell)."""
    d = docx.Document()
    d.add_paragraph().add_run().add_picture(BytesIO(_png_1x1()))  # body image -> 1
    cell = d.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].add_run().add_picture(BytesIO(_png_1x1()))  # cell image -> 2
    p = tmp_path / "mixed.docx"
    d.save(str(p))

    doc = DocxParser().parse(p, "mixed")
    body_imgs = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    cell_imgs = [b for row in doc.tables()[0].table.cells for c in row if c
                 for b in c.blocks if isinstance(b, ImageBlock)]
    assert [b.image_index for b in body_imgs] == [1]
    assert [b.image_index for b in cell_imgs] == [2]


def test_roundtrip_with_cell_image(tmp_path):
    d = docx.Document()
    cell = d.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].add_run().add_picture(BytesIO(_png_1x1()))
    p = tmp_path / "rt.docx"
    d.save(str(p))

    doc = DocxParser().parse(p, "rt")
    restored = ParsedDocument.from_json(doc.to_json())
    assert restored.to_dict() == doc.to_dict()
