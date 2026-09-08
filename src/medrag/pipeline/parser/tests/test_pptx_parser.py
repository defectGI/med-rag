"""PPTX extraction fixes:

1. Group shapes (p:grpSp) were not walked -> nested text/pictures/tables were
   silently dropped.
2. Embedded OLE objects (e.g. an embedded Excel sheet) were not handled at all
   -> the OOXML-mandated raster fallback preview (p:pic/blipFill/a:blip) is now
   captured as an ImageBlock.
3. Speaker notes were never read -> now emitted as a ParagraphBlock spanned to
   ppt/notesSlides/notesSlideN.xml.
4. `_is_title` matched "TITLE" as a substring, so SUBTITLE placeholders (whose
   enum name contains "TITLE") were wrongly promoted to HeadingBlocks.
5. Slides that fake a title via a free-standing textbox (no TITLE placeholder
   at all) never produced a HeadingBlock -> scored heuristic promotion, mirroring
   docx's formatting-faked-heading detection (see PptxHeadingConfig).
6. Inline formatting (bold/italic/underline/strike/super/subscript) was never
   read -> `runs` was always empty for pptx blocks. Now read per-run from a:rPr,
   falling back to the paragraph's a:pPr/a:defRPr for whichever attribute a run
   doesn't set itself (both patterns occur in real decks).
"""

from __future__ import annotations

import io
import struct
import zlib

from pptx import Presentation
from pptx.enum.shapes import PROG_ID
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from medrag.pipeline.parser.parsers.base import (
    HeadingBlock,
    ImageBlock,
    Mark,
    ParagraphBlock,
    ParsedDocument,
)
from medrag.pipeline.parser.parsers.pptx_parser import PptxParser


def _png_1x1() -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return (struct.pack(">I", len(data)) + body
                + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\xff\x00\x00")
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def test_text_inside_group_shape_is_captured(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(2), Inches(1))
    tb.text_frame.text = "Grouped Label"
    slide.shapes.add_group_shape([tb])
    p = tmp_path / "grouped.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "grouped")
    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [b.text for b in paras] == ["Grouped Label"]


def test_speaker_notes_become_a_paragraph_block(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.notes_slide.notes_text_frame.text = "Talking points for slide one."
    p = tmp_path / "notes.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "notes")
    notes = [b for b in doc.blocks
             if isinstance(b, ParagraphBlock) and "notesSlide" in (b.span.part or "")]
    assert len(notes) == 1
    assert notes[0].text == "Talking points for slide one."
    assert notes[0].span.part == "ppt/notesSlides/notesSlide1.xml"


def test_slide_without_notes_emits_no_notes_block(tmp_path):
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    p = tmp_path / "nonotes.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "nonotes")
    assert not any("notesSlide" in (b.span.part or "") for b in doc.blocks)


def test_ole_object_fallback_preview_becomes_an_image(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_ole_object(
        io.BytesIO(b"fake xlsx bytes" * 10), PROG_ID.XLSX,
        Inches(1), Inches(3),
        icon_file=io.BytesIO(_png_1x1()), icon_width=Inches(1), icon_height=Inches(1))
    p = tmp_path / "ole.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "ole")
    imgs = doc.images()
    assert len(imgs) == 1
    assert imgs[0].locator["part"].startswith("ppt/media/")
    assert "Excel.Sheet.12" in imgs[0].alt_text


def _bold_run(shape, text, *, size=None, bold=None, center=False):
    p = shape.text_frame.paragraphs[0]
    run = p.add_run()
    run.text = text
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if center:
        p.alignment = PP_ALIGN.CENTER
    return p


def test_subtitle_placeholder_is_not_promoted_to_heading(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[0])  # Title Slide: title + subtitle
    slide.placeholders[0].text_frame.text = "Chapter 4"
    slide.placeholders[1].text_frame.text = "Selected Questions"
    p = tmp_path / "subtitle.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "subtitle")
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["Chapter 4"]


def test_freeform_title_textbox_is_promoted_to_heading(tmp_path):
    """No TITLE placeholder at all (deck built from plain textboxes, like
    BlueYonder_Presentation.pptx in the real corpus) -> the bold/centered/short
    textbox should still be recognized as the slide's title."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    title_box = slide.shapes.add_textbox(Inches(1), Inches(0.3), Inches(6), Inches(1))
    _bold_run(title_box, "Quarterly Sales Review", size=32, bold=True, center=True)

    body_box = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(2))
    _bold_run(body_box, "This section walks through the quarterly numbers in detail.")

    p = tmp_path / "freeform.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "freeform")
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [h.text for h in headings] == ["Quarterly Sales Review"]
    assert [pb.text for pb in paras] == [
        "This section walks through the quarterly numbers in detail."]


def test_caption_like_bold_line_is_not_promoted(tmp_path):
    """Bold + centered + short would otherwise score above threshold, but a
    figure/table caption is gated out explicitly (never a heading)."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1))
    _bold_run(box, "Table 1: Revenue by Quarter", size=24, bold=True, center=True)
    p = tmp_path / "caption.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "caption")
    assert not any(isinstance(b, HeadingBlock) for b in doc.blocks)


def test_only_the_best_candidate_is_promoted_per_slide(tmp_path):
    """Two textboxes both clear the threshold; only the single best-scoring one
    (a slide has exactly one title) becomes a HeadingBlock."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    strong = slide.shapes.add_textbox(Inches(1), Inches(0.3), Inches(6), Inches(1))
    _bold_run(strong, "Executive Summary", size=32, bold=True, center=True)

    weak = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(6), Inches(1))
    _bold_run(weak, "Overview", size=20, bold=True, center=True)

    p = tmp_path / "twocandidates.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "twocandidates")
    headings = [b for b in doc.blocks if isinstance(b, HeadingBlock)]
    assert [h.text for h in headings] == ["Executive Summary"]
    paras = [b for b in doc.blocks if isinstance(b, ParagraphBlock)]
    assert [pb.text for pb in paras] == ["Overview"]


def test_mid_paragraph_run_marks_are_captured(tmp_path):
    """A paragraph mixing a plain run and a bold+italic run keeps one plain
    `text` view but records the per-run marks in `runs`."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    para = box.text_frame.paragraphs[0]
    r1 = para.add_run()
    r1.text = "Please note: "
    r2 = para.add_run()
    r2.text = "handle with care"
    r2.font.bold = True
    r2.font.italic = True
    p = tmp_path / "runs.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "runs")
    pb = next(b for b in doc.blocks if isinstance(b, ParagraphBlock))
    assert pb.text == "Please note: handle with care"
    assert [(r.text, r.marks) for r in pb.runs] == [
        ("Please note: ", ()),
        ("handle with care", (Mark.BOLD, Mark.ITALIC)),
    ]


def test_paragraph_defrpr_fallback_marks_an_unmarked_run(tmp_path):
    """A run with no rPr of its own inherits bold from the paragraph's
    a:pPr/a:defRPr (python-pptx's `paragraph.font`) — the same fallback the
    title heuristic already relies on. Text is a full sentence so the heading
    heuristic doesn't also promote it (this test is only about `runs`)."""
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    para = box.text_frame.paragraphs[0]
    para.font.bold = True
    run = para.add_run()
    run.text = "This entire sentence inherits bold from the paragraph default."
    p = tmp_path / "defrpr.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "defrpr")
    b = next(b for b in doc.blocks if b.text.startswith("This entire sentence"))
    assert [(r.text, r.marks) for r in b.runs] == [(b.text, (Mark.BOLD,))]


def test_soft_line_break_becomes_an_unmarked_space(tmp_path):
    """a:br (Shift+Enter inside one paragraph) used to leak into the IR as a
    raw \\x0b control character; it's now a plain space in both `text` and `runs`."""
    from pptx.oxml.ns import qn as _qn
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    para = box.text_frame.paragraphs[0]
    r1 = para.add_run()
    r1.text = "Line one"
    para._p.append(para._p.makeelement(_qn("a:br"), {}))
    r2 = para.add_run()
    r2.text = "Line two"
    p = tmp_path / "linebreak.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "linebreak")
    pb = next(b for b in doc.blocks if isinstance(b, ParagraphBlock))
    assert pb.text == "Line one Line two"
    assert "\x0b" not in pb.text


def test_pptx_roundtrip_with_runs(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(4), Inches(1))
    run = box.text_frame.paragraphs[0].add_run()
    run.text = "bold text"
    run.font.bold = True
    p = tmp_path / "rt.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "rt")
    restored = ParsedDocument.from_json(doc.to_json())
    assert restored.to_dict() == doc.to_dict()


# ---------------------------------------------------------------------------
# Native chart / SmartArt -- both used to be silently dropped entirely
# (shape_type is CHART/None, has_table and has_text_frame both False, so no
# existing branch caught them at all -- not even a "not tabular" fallback,
# they just vanished with zero trace).
# ---------------------------------------------------------------------------


def test_native_chart_becomes_image_with_deterministic_description(tmp_path):
    from pptx.chart.data import CategoryChartData
    from pptx.enum.chart import XL_CHART_TYPE

    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    chart_data = CategoryChartData()
    chart_data.categories = ["Q1", "Q2", "Q3"]
    chart_data.add_series("Revenue", (10, 15, 12))
    slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1),
                           Inches(5), Inches(4), chart_data)
    p = tmp_path / "chart.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "chart1")
    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    img = images[0]
    assert img.visual_type == "chart"
    assert img.visual_type_source == "structural"
    assert img.description == (
        "Bar chart. Series 'Revenue': Q1=10.0, Q2=15.0, Q3=12.0.")
    # no TableBlock, no ParagraphBlock -- the chart is never mistaken for
    # tabular or plain text content.
    assert not any(hasattr(b, "table") for b in doc.blocks)


_SMARTART_DATA_XML = (
    b'<dgm:dataModel '
    b'xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" '
    b'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
    b'<dgm:ptLst>'
    b'<dgm:pt modelId="0" type="doc"/>'
    b'<dgm:pt modelId="1"><dgm:t><a:p><a:r><a:t>Input</a:t></a:r></a:p></dgm:t></dgm:pt>'
    b'<dgm:pt modelId="2"><dgm:t><a:p><a:r><a:t>Output</a:t></a:r></a:p></dgm:t></dgm:pt>'
    b'</dgm:ptLst></dgm:dataModel>'
)

_DIAGRAM_DATA_RT = ("http://schemas.openxmlformats.org/officeDocument/2006/"
                    "relationships/diagramData")


def _add_smartart_placeholder(slide, *, with_data_part: bool = True) -> None:
    """python-pptx has no API to create real SmartArt; a graphicFrame whose
    a:graphicData carries the diagram namespace is the exact shape SmartArt
    produces (verified against python-pptx's own shape_type/has_chart/
    has_table/has_text_frame behavior for this element -- all fall through
    to "nothing", the bug this test guards against).

    `with_data_part` wires a real diagramData relationship + part (python-pptx
    assigns the r:dm relationship id, so the graphicFrame XML reads it back
    rather than hardcoding one)."""
    from pptx.opc.package import Part
    from pptx.opc.packuri import PackURI
    from pptx.oxml import parse_xml

    rid = "rId1"
    if with_data_part:
        part = Part(PackURI("/ppt/diagrams/data1.xml"), "application/xml",
                    slide.part.package, blob=_SMARTART_DATA_XML)
        rid = slide.part.relate_to(part, _DIAGRAM_DATA_RT)

    gf_xml = (
        '<p:graphicFrame '
        'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<p:nvGraphicFramePr><p:cNvPr id="5" name="Diagram 4"/>'
        '<p:cNvGraphicFramePr/><p:nvPr/></p:nvGraphicFramePr>'
        '<p:xfrm><a:off x="0" y="0"/><a:ext cx="100" cy="100"/></p:xfrm>'
        '<a:graphic><a:graphicData '
        'uri="http://schemas.openxmlformats.org/drawingml/2006/diagram">'
        '<dgm:relIds '
        'xmlns:dgm="http://schemas.openxmlformats.org/drawingml/2006/diagram" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        f'r:dm="{rid}" r:lo="rId2" r:qs="rId3" r:cs="rId4"/>'
        '</a:graphicData></a:graphic></p:graphicFrame>')
    slide.shapes._spTree.append(parse_xml(gf_xml))


def test_smartart_becomes_block_diagram_with_deterministic_description(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_smartart_placeholder(slide)
    p = tmp_path / "smartart.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "smartart1")
    images = [b for b in doc.blocks if isinstance(b, ImageBlock)]
    assert len(images) == 1
    img = images[0]
    assert img.visual_type == "block_diagram"
    assert img.visual_type_source == "heuristic"
    assert img.description == "Diagram with 2 elements: Input; Output."
    assert img.description_source == "structural"
    assert img.alt_text == "Diagram 4"


def test_smartart_with_unresolvable_relationship_still_tags_type(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_smartart_placeholder(slide, with_data_part=False)  # rId1 dangles
    p = tmp_path / "smartart_broken.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "smartart2")
    img = next(b for b in doc.blocks if isinstance(b, ImageBlock))
    assert img.visual_type == "block_diagram"
    assert img.description is None


def test_smartart_extraction_disabled_by_config(tmp_path, monkeypatch):
    monkeypatch.setenv("VISUAL_EXTRACT_SMARTART", "0")
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_smartart_placeholder(slide)
    p = tmp_path / "smartart_disabled.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "smartart3")
    img = next(b for b in doc.blocks if isinstance(b, ImageBlock))
    assert img.visual_type == "block_diagram"  # type still tagged
    assert img.description is None             # but no extraction attempted


def test_picture_alt_text_reads_real_description_not_shape_name(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    pic = slide.shapes.add_picture(io.BytesIO(_png_1x1()), Inches(1), Inches(1))
    from pptx.oxml.ns import qn as _qn
    cNvPr = pic._element.find(".//" + _qn("p:cNvPr"))
    cNvPr.set("descr", "A photo of the enclosure")
    p = tmp_path / "alt.pptx"
    prs.save(str(p))

    doc = PptxParser().parse(p, "alt1")
    img = next(b for b in doc.blocks if isinstance(b, ImageBlock))
    assert img.alt_text == "A photo of the enclosure"
