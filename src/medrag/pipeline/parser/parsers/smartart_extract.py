"""Deterministic, VLM-free extraction of SmartArt (DrawingML diagram) node text
from its own data-model XML -- the sibling of `chart_extract.py`, and for the
same reason: the content is already written down in the source, so reading it
beats asking a model to guess at pixels.

SmartArt is the awkward case in the taxonomy. It is unambiguously a diagram
(the `dgm` element says so, no classification needed), it usually carries real
information (a product's block structure, a process's stages) -- and it has NO
pixels anywhere: Word/PowerPoint render it at display time, so there is no
image part to crop and hand to a VLM. Rendering one needs infrastructure this
parser doesn't have (LibreOffice headless / Windows COM -- a separate piece of
work). Until then a SmartArt block was a type label and nothing else.

But its node texts sit in plain sight in the diagram's data-model part
(`diagrams/data1.xml`), which is exactly the situation `chart_extract.py`
already handles for charts. So the same answer applies: read the source's own
data, produce a description from it, never a model call, never a hallucination.
This does NOT make the deferred pixel-render work unnecessary -- geometry,
arrow directions and grouping are still lost -- it just stops the text from
being lost too.

`extract_smartart` returns None for XML it doesn't recognize or that has no
node text, rather than guessing (chart_extract.py's stance).

Node selection: the data model lists every `dgm:pt`, but most are not content.
A `dgm:pt` with no `type` (or type "node") is a real data node; "doc" is the
diagram root, "pres" nodes carry layout/presentation only, and "parTrans"/
"sibTrans" are the connectors between nodes. Including those would put layout
noise and empty strings into the description, so only real nodes are read.
"""

from __future__ import annotations

from lxml import etree

_DGM_NS = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

# `dgm:pt/@type` values that are NOT content (see module docstring).
_NON_CONTENT_TYPES = {"doc", "pres", "parTrans", "sibTrans"}


def _dgm(tag: str) -> str:
    return f"{{{_DGM_NS}}}{tag}"


def _a(tag: str) -> str:
    return f"{{{_A_NS}}}{tag}"


def _node_text(pt) -> str | None:
    """Text of one diagram node: every `a:t` run under its `dgm:t`, joined.

    A node's text is a full DrawingML text body (paragraphs of runs), so a
    node reading "Power Supply" may well be several runs; joining the runs of
    one paragraph without a separator and paragraphs with a space reproduces
    what the node visually reads as.
    """
    body = pt.find(_dgm("t"))
    if body is None:
        return None
    paragraphs: list[str] = []
    for para in body.iter(_a("p")):
        runs = "".join(t.text or "" for t in para.iter(_a("t")))
        if runs.strip():
            paragraphs.append(runs.strip())
    return " ".join(paragraphs) or None


def extract_smartart(xml_bytes: bytes) -> list[str] | None:
    """Node texts of a SmartArt `dataN.xml` part, in document order.

    Returns None when the XML is malformed, isn't a diagram data model, or
    holds no node text at all (a purely decorative SmartArt) -- callers then
    keep the plain type label rather than emitting an empty description.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None

    pt_list = root.find(f".//{_dgm('ptLst')}")
    if pt_list is None:
        return None

    texts: list[str] = []
    for pt in pt_list.findall(_dgm("pt")):
        if pt.get("type") in _NON_CONTENT_TYPES:
            continue
        text = _node_text(pt)
        if text:
            texts.append(text)
    return texts or None


def describe_smartart(texts: list[str]) -> str:
    """Deterministic, template-based summary of a SmartArt's node texts -- no
    model call, so it can never claim something the diagram doesn't say.

    Deliberately modest about what it knows: it says the diagram CONTAINS these
    labels, not how they relate, because the connective structure is precisely
    what isn't being read here (see module docstring).

    E.g. "Diagram with 3 elements: Input; Controller; Output."
    """
    joined = "; ".join(texts)
    count = len(texts)
    noun = "element" if count == 1 else "elements"
    return f"Diagram with {count} {noun}: {joined}."
