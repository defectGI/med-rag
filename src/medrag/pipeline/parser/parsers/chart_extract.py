"""Deterministic, VLM-free extraction of native chart data from raw OOXML
chart XML -- the DrawingML chart schema (`c:chart`) is byte-for-byte the
same regardless of whether it's embedded in a docx, pptx, or xlsx part, so
one extractor serves all three (see docx_parser.py/pptx_parser.py/
xlsx_parser.py, each resolving their own chart-part relationship and
handing the raw bytes here).

Reads chart type + series (name, category labels, values) straight from the
XML tree -- no model call, no pixels involved. This is *more* trustworthy
than a VLM guessing at chart pixels would be, since it comes from the
chart's own underlying data. `extract_chart` returns None for chart types
outside the curated set below (or malformed/unrecognized XML) rather than
guessing -- an unsupported chart is silently skipped, never a crash and
never a wrong description; widening the curated set is a separate,
incremental change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lxml import etree

_C_NS = "http://schemas.openxmlformats.org/drawingml/2006/chart"


def _c(tag: str) -> str:
    return f"{{{_C_NS}}}{tag}"


# Chart-type element -> human-readable label. Only common, unambiguous plot
# types; radar/bubble/surface/stock/3-D variants and combo charts are out of
# scope for now -- extract_chart returns None rather than mislabeling one of
# these as something it isn't.
_CHART_TYPE_LABELS = {
    "barChart": "bar", "bar3DChart": "bar",
    "lineChart": "line", "line3DChart": "line",
    "pieChart": "pie", "pie3DChart": "pie", "doughnutChart": "doughnut",
    "scatterChart": "scatter", "areaChart": "area", "area3DChart": "area",
}


@dataclass
class ChartSeries:
    name: str | None
    categories: list[str]
    values: list[float | str | None]


@dataclass
class ChartData:
    chart_type: str  # human-readable label, e.g. "bar" -- see _CHART_TYPE_LABELS
    series: list[ChartSeries] = field(default_factory=list)


def _text_of(el) -> str | None:
    """First `c:v` text found under `el` (e.g. a series' `c:tx` name, which
    is either a literal `c:v` or a `c:strRef/c:strCache/c:pt/c:v`)."""
    if el is None:
        return None
    v = el.find(f".//{_c('v')}")
    if v is None or not v.text:
        return None
    return v.text.strip() or None


def _numeric(text: str | None) -> float | str | None:
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _points(ref_el) -> list[tuple[int, str | None]]:
    """(idx, text) for every `c:pt` under a numRef/strRef/numLit/strLit, in
    document order. `idx` is explicit in the schema (points can be sparse),
    so callers place by idx rather than by position."""
    if ref_el is None:
        return []
    out: list[tuple[int, str | None]] = []
    for pt in ref_el.iter(_c("pt")):
        idx_raw = pt.get("idx")
        try:
            idx = int(idx_raw) if idx_raw is not None else len(out)
        except ValueError:
            idx = len(out)
        v = pt.find(_c("v"))
        out.append((idx, v.text if v is not None and v.text else None))
    return out


def _dense(points: list[tuple[int, str | None]]) -> list[str | None]:
    """Points placed at their `idx`, gaps filled with None (a sparse point
    list -- e.g. a category with no value -- must not shift later points)."""
    if not points:
        return []
    size = max(idx for idx, _ in points) + 1
    out: list[str | None] = [None] * size
    for idx, text in points:
        if 0 <= idx < size:
            out[idx] = text
    return out


def _series_from(ser_el) -> ChartSeries:
    name = _text_of(ser_el.find(_c("tx")))
    categories = [t or "" for t in _dense(_points(ser_el.find(_c("cat"))))]
    values = [_numeric(t) for t in _dense(_points(ser_el.find(_c("val"))))]
    return ChartSeries(name=name, categories=categories, values=values)


def extract_chart(xml_bytes: bytes) -> ChartData | None:
    """Parse a `chartN.xml` part's raw bytes into structured chart data.

    Returns None when the XML is malformed, has no plot area, or its plot
    type isn't in the curated set this extractor understands yet."""
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None

    plot_area = root.find(f".//{_c('plotArea')}")
    if plot_area is None:
        return None

    for child in plot_area:
        label = _CHART_TYPE_LABELS.get(etree.QName(child.tag).localname)
        if label is None:
            continue
        series = [_series_from(ser) for ser in child.findall(_c("ser"))]
        if not series:
            return None
        return ChartData(chart_type=label, series=series)
    return None


def describe_chart(data: ChartData) -> str:
    """Deterministic, template-based natural-language summary of a chart's
    own data -- no model call, so it can never hallucinate a value the
    chart doesn't actually contain.

    E.g. "Bar chart. Series 'Revenue': Q1=10.0, Q2=15.0, Q3=12.0, Q4=18.0."
    """
    parts = [f"{data.chart_type.capitalize()} chart."]
    for ser in data.series:
        name = ser.name or "(unnamed series)"
        if ser.categories and len(ser.categories) == len(ser.values):
            pairs = ", ".join(
                f"{cat}={val}" if val is not None else f"{cat}=?"
                for cat, val in zip(ser.categories, ser.values))
        else:
            pairs = ", ".join(
                str(v) if v is not None else "?" for v in ser.values)
        parts.append(f"Series '{name}': {pairs}.")
    return " ".join(parts)
