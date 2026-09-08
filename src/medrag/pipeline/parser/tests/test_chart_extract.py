"""parsers/chart_extract.py tests: deterministic OOXML chart XML -> chart
data + natural-language description. No model, no network -- pure XML
parsing, so these are plain unit tests against inline chart XML fixtures
(the same `c:chart` schema docx/pptx/xlsx all embed byte-for-byte identically).
"""

from __future__ import annotations

from medrag.pipeline.parser.parsers.chart_extract import describe_chart, extract_chart

_NS = (
    'xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
)


def _bar_chart_xml() -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <c:chartSpace {_NS}>
      <c:chart>
        <c:plotArea>
          <c:barChart>
            <c:ser>
              <c:idx val="0"/>
              <c:tx><c:strRef><c:strCache><c:pt idx="0"><c:v>Revenue</c:v></c:pt>
                </c:strCache></c:strRef></c:tx>
              <c:cat><c:strRef><c:strCache>
                <c:pt idx="0"><c:v>Q1</c:v></c:pt>
                <c:pt idx="1"><c:v>Q2</c:v></c:pt>
                <c:pt idx="2"><c:v>Q3</c:v></c:pt>
              </c:strCache></c:strRef></c:cat>
              <c:val><c:numRef><c:numCache>
                <c:pt idx="0"><c:v>10</c:v></c:pt>
                <c:pt idx="1"><c:v>15</c:v></c:pt>
                <c:pt idx="2"><c:v>12</c:v></c:pt>
              </c:numCache></c:numRef></c:val>
            </c:ser>
          </c:barChart>
        </c:plotArea>
      </c:chart>
    </c:chartSpace>""".encode()


def _radar_chart_xml() -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <c:chartSpace {_NS}>
      <c:chart><c:plotArea><c:radarChart>
        <c:ser><c:idx val="0"/></c:ser>
      </c:radarChart></c:plotArea></c:chart>
    </c:chartSpace>""".encode()


def _empty_pie_chart_xml() -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <c:chartSpace {_NS}>
      <c:chart><c:plotArea><c:pieChart/></c:plotArea></c:chart>
    </c:chartSpace>""".encode()


def _sparse_points_xml() -> bytes:
    # idx 1 is missing entirely -- must not shift idx 2's value into slot 1.
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <c:chartSpace {_NS}>
      <c:chart><c:plotArea><c:lineChart>
        <c:ser>
          <c:cat><c:strRef><c:strCache>
            <c:pt idx="0"><c:v>A</c:v></c:pt>
            <c:pt idx="2"><c:v>C</c:v></c:pt>
          </c:strCache></c:strRef></c:cat>
          <c:val><c:numRef><c:numCache>
            <c:pt idx="0"><c:v>1</c:v></c:pt>
            <c:pt idx="2"><c:v>3</c:v></c:pt>
          </c:numCache></c:numRef></c:val>
        </c:ser>
      </c:lineChart></c:plotArea></c:chart>
    </c:chartSpace>""".encode()


def test_extract_bar_chart():
    data = extract_chart(_bar_chart_xml())
    assert data is not None
    assert data.chart_type == "bar"
    assert len(data.series) == 1
    ser = data.series[0]
    assert ser.name == "Revenue"
    assert ser.categories == ["Q1", "Q2", "Q3"]
    assert ser.values == [10.0, 15.0, 12.0]


def test_describe_chart_bar():
    data = extract_chart(_bar_chart_xml())
    text = describe_chart(data)
    assert text == "Bar chart. Series 'Revenue': Q1=10.0, Q2=15.0, Q3=12.0."


def test_unsupported_chart_type_returns_none():
    assert extract_chart(_radar_chart_xml()) is None


def test_chart_with_no_series_returns_none():
    assert extract_chart(_empty_pie_chart_xml()) is None


def test_malformed_xml_returns_none():
    assert extract_chart(b"not xml at all") is None


def test_no_plot_area_returns_none():
    xml = f'<?xml version="1.0"?><c:chartSpace {_NS}><c:chart/></c:chartSpace>'
    assert extract_chart(xml.encode("utf-8")) is None


def test_sparse_points_keep_their_index():
    data = extract_chart(_sparse_points_xml())
    assert data is not None
    ser = data.series[0]
    assert ser.categories == ["A", "", "C"]
    assert ser.values == [1.0, None, 3.0]


def test_describe_chart_no_series_name_uses_placeholder():
    xml = f"""<?xml version="1.0"?>
    <c:chartSpace {_NS}>
      <c:chart><c:plotArea><c:pieChart>
        <c:ser>
          <c:val><c:numRef><c:numCache>
            <c:pt idx="0"><c:v>5</c:v></c:pt>
          </c:numCache></c:numRef></c:val>
        </c:ser>
      </c:pieChart></c:plotArea></c:chart>
    </c:chartSpace>""".encode()
    data = extract_chart(xml)
    text = describe_chart(data)
    assert "(unnamed series)" in text
    assert "5.0" in text
