"""Cross-reference resolution (`chunker.enrichment.cross_ref`) tests.

Offline, fake tokenizer; `anchor_id` is given to `Heading` directly by
hand (the adapter doesn't produce it yet — see `Heading.anchor_id`
docstring).
"""

import logging

from medrag.pipeline.chunker.adapters.first_parse import adapt
from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import Document, Heading, LinkRef, Paragraph
from medrag.pipeline.chunker.core.engine import chunk_document
from medrag.pipeline.chunker.enrichment.cross_ref import resolve_cross_refs
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer
from medrag.pipeline.parser.parsers.markdown_parser import MarkdownParser

TOK = FakeTokenizer()


def belge(bloklar, **degisiklik) -> Document:
    alanlar = {"doc_id": "d1", "source_path": "raw/kilavuz.pdf", "fmt": "pdf",
               "blocks": bloklar}
    alanlar.update(degisiklik)
    return Document(**alanlar)


def kirintisiz(cfg):
    """`min_chunk_tokens=0` copy: a two-separate-chunk setup must not hit
    crumb merging (same helper as in test_engine.py)."""
    return cfg.model_copy(update={
        "packing": cfg.packing.model_copy(update={"min_chunk_tokens": 0})})


def cfg_yap(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return load_config(dosya)


def test_anchor_id_ile_ayni_chunka_cozulur(tmp_path):
    # section_strategy="hard" pinned by hand: the test's goal is to verify
    # resolution to a DIFFERENT chunk (default "merge" would merge the two
    # sections into one chunk and trivialize the test).
    cfg = cfg_yap(tmp_path,
                  "[packing]\nsection_strategy = \"hard\"\nmin_chunk_tokens = 0\n")
    doc = belge([
        Heading(id="h1", text="Giriş", level=1, anchor_id="giris"),
        Paragraph(id="p1", text="bkz kurulum", heading_path=["Giriş"], links=[
            LinkRef(text="kurulum", target="#kurulum"),
        ]),
        Heading(id="h2", text="Kurulum", level=1, anchor_id="kurulum"),
        Paragraph(id="p2", text="adım 1", heading_path=["Kurulum"]),
    ])
    cs = resolve_cross_refs(chunk_document(doc, cfg, TOK), doc)
    kaynak = cs.nodes[0]
    hedef = cs.nodes[1]
    assert kaynak.heading_path == ["Giriş"]
    assert hedef.heading_path == ["Kurulum"]
    assert len(kaynak.cross_refs) == 1
    assert kaynak.cross_refs[0].target_chunk_id == hedef.node_id


def test_anchor_id_yoksa_cozulmez_ve_loglanir(caplog):
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),  # no anchor_id (current real state)
        Paragraph(id="p1", text="bkz kurulum", heading_path=["Giriş"], links=[
            LinkRef(text="kurulum", target="#kurulum"),
        ]),
        Heading(id="h2", text="Kurulum", level=1),
        Paragraph(id="p2", text="adım 1", heading_path=["Kurulum"]),
    ])
    with caplog.at_level(logging.WARNING):
        cs = resolve_cross_refs(chunk_document(doc, load_config(), TOK), doc)
    assert cs.nodes[0].cross_refs[0].target_chunk_id is None
    assert "Unresolvable" in caplog.text


def test_cakisan_anchor_id_belirsiz_sayilir_ve_loglanir(caplog):
    doc = belge([
        Heading(id="h1", text="Giriş", level=1, anchor_id="dup"),
        Paragraph(id="p1", text="bkz", heading_path=["Giriş"], links=[
            LinkRef(text="x", target="#dup"),
        ]),
        Heading(id="h2", text="Ek", level=1, anchor_id="dup"),
        Paragraph(id="p2", text="içerik", heading_path=["Ek"]),
    ])
    with caplog.at_level(logging.WARNING):
        cs = resolve_cross_refs(chunk_document(doc, load_config(), TOK), doc)
    assert cs.nodes[0].cross_refs[0].target_chunk_id is None
    assert "Unresolvable" in caplog.text


def test_zaten_cozulmus_ref_dokunulmadan_kalir():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1, anchor_id="giris"),
        Paragraph(id="p1", text="bkz kurulum", heading_path=["Giriş"], links=[
            LinkRef(text="kurulum", target="#kurulum"),
        ]),
        Heading(id="h2", text="Kurulum", level=1, anchor_id="kurulum"),
        Paragraph(id="p2", text="adım 1", heading_path=["Kurulum"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    once = resolve_cross_refs(cs, doc)
    iki_kez = resolve_cross_refs(once, doc)
    assert iki_kez.nodes[0].cross_refs[0].target_chunk_id == \
        once.nodes[0].cross_refs[0].target_chunk_id


def test_uctan_uca_gercek_markdown_parser_ile_gle_m_slug_cozulur(tmp_path):
    """Not a hand-built `anchor_id` but the real GFM slug from a real
    MarkdownParser — end-to-end: file -> parser -> adapter -> engine ->
    resolve_cross_refs (the actual integration path that closes the gap)."""
    p = tmp_path / "doc.md"
    p.write_text(
        "## Giriş\n\n[kuruluma bak](#installation-guide)\n\n"
        "## Installation Guide\n\nadım 1\n", encoding="utf-8")
    parsed = MarkdownParser().parse(p, "d1")
    # section_strategy="hard": test goal is resolution to a DIFFERENT
    # chunk (default "merge" would merge both sections into one).
    cfg = cfg_yap(tmp_path, "[packing]\nsection_strategy = \"hard\"\n")
    doc = adapt(parsed, config=cfg)
    cs = resolve_cross_refs(chunk_document(doc, cfg, TOK), doc)

    # MarkdownParser doesn't fill heading_path (known, out-of-scope
    # boundary) — sections are distinguished by their text.
    giris = next(n for n in cs.leaves() if "Giriş" in n.text)
    hedef = next(n for n in cs.leaves() if "Installation Guide" in n.text)
    assert giris.cross_refs[0].target_chunk_id == hedef.node_id


def test_dis_link_hala_cross_ref_uretmez():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1, anchor_id="giris"),
        Paragraph(id="p1", text="bkz", heading_path=["Giriş"], links=[
            LinkRef(text="dış", target="https://example.com"),
        ]),
    ])
    cs = resolve_cross_refs(chunk_document(doc, load_config(), TOK), doc)
    assert cs.nodes[0].cross_refs == []