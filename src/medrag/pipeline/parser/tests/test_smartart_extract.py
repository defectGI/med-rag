"""parsers/smartart_extract.py: SmartArt node text read from its own
data-model XML -- deterministic, no VLM (it has no pixels to show one).

The XML here follows the DrawingML diagram schema: a `dgm:dataModel` whose
`dgm:ptLst` holds `dgm:pt` nodes, each with a `dgm:t` text body of `a:p`
paragraphs of `a:r`/`a:t` runs. Only real data nodes carry content -- "doc",
"pres", "parTrans" and "sibTrans" points are structure/layout, not text.
"""

from __future__ import annotations

from medrag.pipeline.parser.parsers.smartart_extract import (
    describe_smartart,
    extract_smartart,
)

_DGM = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _pt(model_id: str, text: str | None = None, ptype: str | None = None,
        paragraphs: list[str] | None = None) -> str:
    attrs = f'modelId="{model_id}"'
    if ptype:
        attrs += f' type="{ptype}"'
    if paragraphs is None:
        paragraphs = [text] if text is not None else []
    if not paragraphs:
        return f"<dgm:pt {attrs}/>"
    body = "".join(
        f"<a:p><a:r><a:t>{p}</a:t></a:r></a:p>" for p in paragraphs)
    return f"<dgm:pt {attrs}><dgm:t>{body}</dgm:t></dgm:pt>"


def _model(*pts: str) -> bytes:
    return (
        f'<dgm:dataModel xmlns:dgm="{_DGM}" xmlns:a="{_A}">'
        f'<dgm:ptLst>{"".join(pts)}</dgm:ptLst>'
        f'</dgm:dataModel>'
    ).encode()


# --- extract_smartart ---------------------------------------------------------


def test_node_texts_in_document_order():
    xml = _model(_pt("0", ptype="doc"),
                 _pt("1", "Input"),
                 _pt("2", "Controller"),
                 _pt("3", "Output"))
    assert extract_smartart(xml) == ["Input", "Controller", "Output"]


def test_presentation_and_transition_nodes_are_skipped():
    """Only real data nodes are content: "pres" is layout, parTrans/sibTrans
    are the connectors. Including them would inject layout noise."""
    xml = _model(_pt("0", ptype="doc"),
                 _pt("1", "Real node"),
                 _pt("2", "Layout only", ptype="pres"),
                 _pt("3", "", ptype="parTrans"),
                 _pt("4", "", ptype="sibTrans"))
    assert extract_smartart(xml) == ["Real node"]


def test_node_type_node_is_content():
    xml = _model(_pt("1", "Named node", ptype="node"))
    assert extract_smartart(xml) == ["Named node"]


def test_multi_run_node_text_is_joined():
    """A node's label is a full text body -- "Power Supply" can arrive as
    several runs (a formatting change mid-label splits them)."""
    xml = (f'<dgm:dataModel xmlns:dgm="{_DGM}" xmlns:a="{_A}"><dgm:ptLst>'
           f'<dgm:pt modelId="1"><dgm:t><a:p>'
           f'<a:r><a:t>Power </a:t></a:r><a:r><a:t>Supply</a:t></a:r>'
           f'</a:p></dgm:t></dgm:pt>'
           f'</dgm:ptLst></dgm:dataModel>').encode()
    assert extract_smartart(xml) == ["Power Supply"]


def test_multi_paragraph_node_text_is_joined_with_space():
    xml = _model(_pt("1", paragraphs=["Line one", "Line two"]))
    assert extract_smartart(xml) == ["Line one Line two"]


def test_empty_nodes_are_dropped():
    xml = _model(_pt("1", "Real"), _pt("2", "   "), _pt("3"))
    assert extract_smartart(xml) == ["Real"]


def test_diagram_with_no_text_is_none():
    """A purely decorative SmartArt -- None so the caller keeps the bare type
    label instead of emitting an empty description."""
    assert extract_smartart(_model(_pt("0", ptype="doc"), _pt("1"))) is None


def test_malformed_xml_is_none():
    assert extract_smartart(b"<not-xml") is None


def test_xml_without_a_point_list_is_none():
    assert extract_smartart(b'<foo xmlns="urn:x"><bar/></foo>') is None


# --- describe_smartart --------------------------------------------------------


def test_description_names_the_nodes():
    out = describe_smartart(["Input", "Controller", "Output"])
    assert out == "Diagram with 3 elements: Input; Controller; Output."


def test_description_singular_for_one_node():
    assert describe_smartart(["Solo"]) == "Diagram with 1 element: Solo."


def test_description_claims_no_structure():
    """The connective structure (arrows, nesting) is exactly what ISN'T read
    here, so the wording must not imply an order or relationship."""
    out = describe_smartart(["A", "B"])
    for overclaim in ("flows", "then", "connects", "leads to"):
        assert overclaim not in out
