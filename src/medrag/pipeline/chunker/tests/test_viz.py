"""`chunker/viz.py` offline tests.

Visualization must produce a single, self-contained HTML (no CDN/script/
font/network); document content must not leak as markup. Tests work with
hand-built small `ChunkSet`s — no network, no browser.
"""

from medrag.pipeline.chunker.core.chunk import ChunkNode, ChunkSet
from medrag.pipeline.chunker.viz import (
    IndexEntry,
    render_document_html,
    render_index_html,
    write_document_html,
    write_index_html,
)


def _leaf(doc_id, n, text, **ekstra):
    return ChunkNode(node_id=f"{doc_id}::c{n}", doc_id=doc_id, tree_level=0,
                     text=text, **ekstra)


def _agac():
    """One root summary + two leaf children; the leaves are enriched."""
    doc = "d"
    l1 = _leaf(doc, 0, "Birinci parça metni.", summary="birinci",
               keywords=["a", "b"], heading_path=["Bölüm", "Alt"],
               parent_id=f"{doc}::s1.0", token_count=42)
    l2 = _leaf(doc, 1, "İkinci parça metni.", summary="ikinci",
               parent_id=f"{doc}::s1.0", token_count=17)
    kok = ChunkNode(node_id=f"{doc}::s1.0", doc_id=doc, tree_level=1,
                    text="Birleşik özet.", summary="Birleşik özet.",
                    keywords=["ozet"], heading_path=["Bölüm"],
                    child_ids=[l1.node_id, l2.node_id])
    return ChunkSet(doc_id=doc, nodes=[l1, l2, kok])


# -- self-contained -------------------------------------------------------------


def test_html_dis_kaynak_icermez():
    html = render_document_html(_agac())
    # No external origin (CDN/font/script/img) references.
    for kotu in ("http://", "https://", "src=\"//", "//cdn", "@import"):
        assert kotu not in html
    assert "<!doctype html>" in html.lower()


def test_veri_gomulu_ve_dugumler_var():
    html = render_document_html(_agac())
    assert 'id="data"' in html
    for node_id in ("d::c0", "d::c1", "d::s1.0"):
        assert node_id in html
    # Summary and keywords also carried in the data.
    assert "Birleşik özet." in html and "ozet" in html


# -- content must NOT leak as HTML markup ------------------------------------------


def test_icerik_html_olarak_sizmaz():
    # `<script>` / `</script>` inside content must not close the embedded
    # JSON early; `<` inside JSON is escaped, raw `</script>` outside the
    # template must not appear.
    cs = ChunkSet(doc_id="x", nodes=[
        _leaf("x", 0, "<script>alert(1)</script> & <b>bold</b>")])
    html = render_document_html(cs)
    assert "<script>alert(1)" not in html  # must be escaped
    assert "\\u003cscript\\u003e" in html
    # The page's real <script> tags are only the template's two (data + code).
    assert html.count("<script") == 2


def test_bos_set():
    html = render_document_html(ChunkSet(doc_id="bos", nodes=[]))
    assert "<!doctype html>" in html.lower()
    assert '"nodes": []' in html or '"nodes":[]' in html


def test_coklu_kok():
    # Two parentless root nodes (RAPTOR's natural stopping can leave multiple roots).
    cs = ChunkSet(doc_id="m", nodes=[
        _leaf("m", 0, "yalnız leaf bir"), _leaf("m", 1, "yalnız leaf iki")])
    html = render_document_html(cs)
    assert "m::c0" in html and "m::c1" in html


# -- index ------------------------------------------------------------------------


def test_indeks_dokumanlari_baglar():
    entries = [IndexEntry("d", "d.tree.html", 2, 3),
               IndexEntry("e", "e.tree.html", 5, 7)]
    html = render_index_html(entries)
    assert "d.tree.html" in html and "e.tree.html" in html
    assert "http://" not in html and "https://" not in html


def test_bos_indeks():
    html = render_index_html([])
    assert "<!doctype html>" in html.lower()
    assert '"nodes"' not in html  # index uses a different payload


# -- writing to disk --------------------------------------------------------------


def test_dosya_yazimi(tmp_path):
    cs = _agac()
    html_path = write_document_html(cs, tmp_path)
    assert html_path.name == "d.tree.html"
    assert html_path.read_text(encoding="utf-8").startswith("<!doctype html>")

    idx = write_index_html([IndexEntry("d", "d.tree.html", 2, 3)], tmp_path)
    assert idx.name == "index.html"
    assert "d.tree.html" in idx.read_text(encoding="utf-8")