"""first_parse adapter tests (offline: IR objects built by hand; no file/
network/LLM; config loaded from real `default.toml`, numbers not asserted).
"""

from __future__ import annotations

import pytest

import medrag.pipeline.parser.parsers.base as ir
from medrag.pipeline.chunker.adapters.first_parse import AdapterError, adapt, adapt_json
from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import (
    Code,
    Heading,
    Image,
    ListBlock,
    Paragraph,
    Provenance,
    Table,
)
from medrag.pipeline.chunker.core.engine import chunk_document
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer

CFG = load_config()


def doc(blocks, **kwargs) -> ir.ParsedDocument:
    fields = {"doc_id": "d1", "source_path": "raw/a.docx", "fmt": "docx"}
    fields.update(kwargs)
    return ir.ParsedDocument(**fields, blocks=blocks)


# ---------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------


def test_dokuman_kimligi_ve_metadata_tasinir():
    parsed = doc([], raw_sha256="abc", parser_version="p1",
                 page_count=3, metadata={"toc": [{"title": "Giriş"}]})
    out = adapt(parsed, config=CFG)
    assert (out.doc_id, out.source_path, out.fmt) == ("d1", "raw/a.docx", "docx")
    assert out.page_count == 3
    assert out.metadata["toc"] == [{"title": "Giriş"}]
    assert out.metadata["raw_sha256"] == "abc"
    assert out.metadata["parser_version"] == "p1"
    assert out.metadata["ir_version"] == ir.IR_VERSION


def test_parsed_metadata_anahtari_passthrough_alanina_ezilmez():
    parsed = doc([], raw_sha256="gercek", metadata={"raw_sha256": "kendi"})
    assert adapt(parsed, config=CFG).metadata["raw_sha256"] == "kendi"


def test_adapt_json_versiyon_kapisindan_gecer():
    text = doc([ir.ParagraphBlock(id="b0", text="merhaba")]).to_json()
    out = adapt_json(text, config=CFG)
    assert isinstance(out.blocks[0], Paragraph)
    assert out.blocks[0].text == "merhaba"


def test_adapt_json_gelecek_versiyonu_reddeder():
    with pytest.raises(ir.IRParseError):
        adapt_json(f'{{"ir_version": {ir.IR_VERSION + 1}, "doc_id": "d", "source_path": "s", '
                   '"fmt": "docx", "blocks": []}',
                   config=CFG)


# ---------------------------------------------------------------------------
# Common fields: page, heading_path, provenance, link
# ---------------------------------------------------------------------------


def test_tek_sayfa_araliga_normalize_edilir():
    parsed = doc([ir.ParagraphBlock(id="b0", text="x", span=ir.Span(page=4))])
    block = adapt(parsed, config=CFG).blocks[0]
    assert (block.page_start, block.page_end) == (4, 4)


def test_sayfa_araligi_ve_eksik_uc():
    blocks = [
        ir.ParagraphBlock(id="b0", text="x",
                          span=ir.Span(page_start=2, page_end=5)),
        ir.ParagraphBlock(id="b1", text="y", span=ir.Span(page_start=3)),
        ir.ParagraphBlock(id="b2", text="z"),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert (out[0].page_start, out[0].page_end) == (2, 5)
    assert (out[1].page_start, out[1].page_end) == (3, 3)
    assert (out[2].page_start, out[2].page_end) == (None, None)


def test_heading_path_none_bos_listeye_duser():
    blocks = [
        ir.ParagraphBlock(id="b0", text="x"),
        ir.ParagraphBlock(id="b1", text="y", heading_path=["Kök", "Alt"]),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert out[0].heading_path == []
    assert out[1].heading_path == ["Kök", "Alt"]


def test_provenance_etiketleri_maplenir():
    blocks = [
        ir.ParagraphBlock(id="b0", text="a", provenance="text-layer-verified"),
        ir.ParagraphBlock(id="b1", text="b", provenance="consensus-verified"),
        ir.ParagraphBlock(id="b2", text="c", provenance="unverified"),
        ir.ParagraphBlock(id="b3", text="d"),
        # v7 PDF table paths: bands deterministic → VERIFIED;
        # structure-model has visual-model grid inference → CONSENSUS.
        ir.ParagraphBlock(id="b4", text="e", provenance="table-bands"),
        ir.ParagraphBlock(id="b5", text="f", provenance="table-structure-model"),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert [b.provenance for b in out] == [
        Provenance.VERIFIED, Provenance.CONSENSUS, Provenance.UNVERIFIED, None,
        Provenance.VERIFIED, Provenance.CONSENSUS]


def test_bilinmeyen_provenance_sinirda_patlar():
    parsed = doc([ir.ParagraphBlock(id="b7", text="x", provenance="yepyeni")])
    with pytest.raises(AdapterError, match="b7"):
        adapt(parsed, config=CFG)


def test_linkler_runlardan_toplanir_ardisik_ayni_hedef_birlesir():
    runs = [
        ir.InlineRun(text="bkz. "),
        ir.InlineRun(text="bölüm ", link="#sec2"),
        ir.InlineRun(text="2", marks=(ir.Mark.BOLD,), link="#sec2"),
        ir.InlineRun(text=" ve "),
        ir.InlineRun(text="ek", link="https://x.example"),
        ir.InlineRun(text="   ", link="#bos"),  # empty-text link dropped
    ]
    parsed = doc([ir.ParagraphBlock(id="b0", text="...", runs=runs)])
    links = adapt(parsed, config=CFG).blocks[0].links
    assert [(l.text, l.target) for l in links] == [
        ("bölüm 2", "#sec2"), ("ek", "https://x.example")]


# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------


def test_baslik_paragraf_kod_maplenir():
    blocks = [
        ir.HeadingBlock(id="b0", text="Başlık", level=2),
        ir.ParagraphBlock(id="b1", text="Gövde"),
        ir.CodeBlock(id="b2", text="print('x')", language="python"),
        ir.CodeBlock(id="b3", text="raw"),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert isinstance(out[0], Heading) and out[0].level == 2
    assert isinstance(out[1], Paragraph)
    assert isinstance(out[2], Code) and out[2].language == "python"
    assert isinstance(out[3], Code) and out[3].language is None


def test_baslik_anchor_id_tasinir():
    blocks = [
        ir.HeadingBlock(id="b0", text="Kurulum", level=1, anchor_id="kurulum"),
        ir.HeadingBlock(id="b1", text="Diğer", level=1),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert out[0].anchor_id == "kurulum"
    assert out[1].anchor_id is None


def test_bilinmeyen_blok_tipi_sinirda_patlar():
    class YeniBlok(ir.Block):
        pass

    with pytest.raises(AdapterError, match="b9"):
        adapt(doc([YeniBlok(id="b9")]), config=CFG)


def test_gorsel_alanlari_ve_ocr_guven_filtresi():
    blocks = [
        ir.ImageBlock(id="b0", image_index=1, image_id="sha", alt_text="şema",
                      ocr_text="devre şeması", ocr_meaningful=True),
        ir.ImageBlock(id="b1", image_index=2, ocr_text="gürültü",
                      ocr_meaningful=False),
        ir.ImageBlock(id="b2", image_index=3, ocr_text="henüz"),  # None = pass
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert (out[0].image_id, out[0].ocr_text, out[0].alt_text) == (
        "sha", "devre şeması", "şema")
    assert out[1].ocr_text is None
    assert out[2].ocr_text == "henüz"


def test_gorsel_siniflandirma_alanlari_tasinir():
    # Visual classification plan: visual_type/visual_type_source/
    # excluded_at_parse/description carried verbatim from IR.
    blocks = [
        ir.ImageBlock(id="b0", image_index=1, image_id="sha1",
                      visual_type="chart", visual_type_source="structural",
                      description="Bar chart. Series 'X': A=1, B=2.",
                      description_source="structural"),
        ir.ImageBlock(id="b1", image_index=2, source_crop="cropsha",
                      visual_type="block_diagram", visual_type_source="heuristic",
                      excluded_at_parse=True),
        ir.ImageBlock(id="b2", image_index=3),  # not classified at all
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert (out[0].visual_type, out[0].description) == (
        "chart", "Bar chart. Series 'X': A=1, B=2.")
    assert out[0].excluded_at_parse is False
    assert (out[1].visual_type, out[1].source_crop, out[1].excluded_at_parse) == (
        "block_diagram", "cropsha", True)
    assert out[2].visual_type is None
    assert out[2].excluded_at_parse is False


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def simple_table(block_id="t0", rows=(("A", "B"), ("1", "2")), **kwargs):
    cells = [[ir.text_cell(v) for v in row] for row in rows]
    data = ir.TableData(n_rows=len(cells), n_cols=len(cells[0]) if cells else 0,
                        cells=cells)
    return ir.TableBlock(id=block_id, table=data, **kwargs)


def test_tablo_source_crop_tasinir():
    data = ir.TableData(n_rows=1, n_cols=1, cells=[[ir.text_cell("x")]])
    block = ir.TableBlock(id="t0", table=data, source_crop="cropsha",
                          provenance="unverified")
    out = adapt(doc([block]), config=CFG).blocks[0]
    assert out.source_crop == "cropsha"


def test_tablo_duzlesir_merge_slotu_bos_string_olur():
    cells = [[ir.text_cell("Başlık"), None],
             [ir.text_cell("a"), ir.text_cell("b")]]
    data = ir.TableData(n_rows=2, n_cols=2, cells=cells,
                        merges=[ir.Merge(row=0, col=0, colspan=2)])
    out = adapt(doc([ir.TableBlock(id="t0", table=data)]), config=CFG).blocks[0]
    assert out.cells == [["Başlık", ""], ["a", "b"]]


def test_ic_ice_tablolu_hucre_plain_text_ile_duzlesir():
    inner = ir.TableData(n_rows=1, n_cols=2,
                         cells=[[ir.text_cell("x"), ir.text_cell("y")]])
    nested_cell = ir.Cell(blocks=[ir.TableBlock(id="t1", table=inner)])
    data = ir.TableData(n_rows=1, n_cols=1, cells=[[nested_cell]])
    out = adapt(doc([ir.TableBlock(id="t0", table=data)]), config=CFG).blocks[0]
    # Flattening has one authority — Cell.plain_text(): the output matches it exactly.
    assert out.cells == [[nested_cell.plain_text()]]
    assert "x" in out.cells[0][0] and "y" in out.cells[0][0]


def test_tablo_aciklamasi_ve_facts_tasinir():
    block = simple_table(description="Sıcaklık limitleri.",
                         facts=["Aralık 0-40 C'dir.", "Birim °C'dir."])
    out = adapt(doc([block]), config=CFG).blocks[0]
    assert out.description == "Sıcaklık limitleri."
    assert out.facts == ["Aralık 0-40 C'dir.", "Birim °C'dir."]
    assert adapt(doc([simple_table()]), config=CFG).blocks[0].facts == []


def test_header_rows_config_varsayilanindan_gelir(tmp_path):
    override = tmp_path / "o.toml"
    override.write_text("[table_split]\ndefault_header_rows = 2\n",
                        encoding="utf-8")
    cfg = load_config(override=override)
    out = adapt(doc([simple_table(rows=(("A",), ("1",), ("2",)))]),
                config=cfg).blocks[0]
    assert out.header_rows == 2


def test_header_rows_satir_sayisina_kirpilir():
    override_yok = adapt(doc([simple_table(rows=(("tek",),))]), config=CFG)
    assert override_yok.blocks[0].header_rows <= 1
    bos = ir.TableBlock(id="t0", table=ir.TableData(n_rows=0, n_cols=0))
    assert adapt(doc([bos]), config=CFG).blocks[0].header_rows == 0


def test_header_rows_ir_alani_config_varsayilanini_ezer():
    # Forward compatibility: when the parser adds header_rows to its IR,
    # the adapter must use it. Simulated by attaching the field externally.
    block = simple_table(rows=(("A", "B"), ("k1", "k2"), ("1", "2")))
    block.header_rows = 2
    out = adapt(doc([block]), config=CFG).blocks[0]
    assert out.header_rows == 2


# ---------------------------------------------------------------------------
# List rebuilding
# ---------------------------------------------------------------------------


def li(block_id, text, *, list_id="L1", level=0, ordered=False, **kwargs):
    return ir.ParagraphBlock(id=block_id, text=text, list_id=list_id,
                             list_level=level, list_ordered=ordered, **kwargs)


def test_bitisik_ayni_list_id_tek_konteyner_olur():
    blocks = [
        ir.ParagraphBlock(id="b0", text="önce"),
        li("b1", "bir", ordered=True),
        li("b2", "iki", ordered=True),
        li("b3", "iki.a", level=1),
        ir.ParagraphBlock(id="b4", text="sonra"),
    ]
    out = adapt(doc(blocks), config=CFG).blocks
    assert [type(b) for b in out] == [Paragraph, ListBlock, Paragraph]
    lst = out[1]
    assert lst.id == "L1"
    assert [(i.level, i.ordered, i.blocks[0].text) for i in lst.items] == [
        (0, True, "bir"), (0, True, "iki"), (1, False, "iki.a")]


def test_kesintiye_ugrayan_liste_ikinci_konteyner_acar():
    blocks = [li("b0", "bir"), ir.ParagraphBlock(id="b1", text="ara"),
              li("b2", "iki")]
    out = adapt(doc(blocks), config=CFG).blocks
    assert [type(b) for b in out] == [ListBlock, Paragraph, ListBlock]
    assert (out[0].id, out[2].id) == ("L1", "L1#2")
    # Reading order preserved; each container carries its own items.
    assert out[0].items[0].blocks[0].text == "bir"
    assert out[2].items[0].blocks[0].text == "iki"


def test_farkli_list_idler_bitisik_olsa_da_ayrilir():
    blocks = [li("b0", "a", list_id="L1"), li("b1", "b", list_id="L2")]
    out = adapt(doc(blocks), config=CFG).blocks
    assert [b.id for b in out] == ["L1", "L2"]


def test_paragraf_olmayan_blok_son_ogeye_eklenir():
    table = simple_table("t0")
    table.list_id, table.list_level = "L1", 1
    img = ir.ImageBlock(id="i0", image_index=1, list_id="L1", list_level=1)
    blocks = [li("b0", "öğe metni"), table, img, li("b1", "sonraki öğe")]
    lst = adapt(doc(blocks), config=CFG).blocks[0]
    assert len(lst.items) == 2
    first = lst.items[0]
    assert [type(b) for b in first.blocks] == [Paragraph, Table, Image]
    assert lst.items[1].blocks[0].text == "sonraki öğe"


def test_liste_basinda_paragraf_olmayan_blok_kendi_ogesini_acar():
    table = simple_table("t0")
    table.list_id = "L1"
    lst = adapt(doc([table]), config=CFG).blocks[0]
    assert len(lst.items) == 1 and isinstance(lst.items[0].blocks[0], Table)


def test_liste_icindeki_baslik_paragrafa_duser():
    h = ir.HeadingBlock(id="b0", text="Sözde başlık", level=3, list_id="L1",
                        list_level=0)
    lst = adapt(doc([h, li("b1", "devam")]), config=CFG).blocks[0]
    assert isinstance(lst.items[0].blocks[0], Paragraph)
    assert lst.items[0].blocks[0].text == "Sözde başlık"
    assert len(lst.items) == 2


def test_konteyner_sayfa_araligi_ve_provenance_ogelerden_turer():
    blocks = [
        li("b0", "bir", span=ir.Span(page=2), provenance="text-layer-verified",
           heading_path=["Kurulum"]),
        li("b1", "iki", span=ir.Span(page=4), provenance="unverified"),
    ]
    lst = adapt(doc(blocks), config=CFG).blocks[0]
    assert (lst.page_start, lst.page_end) == (2, 4)
    assert lst.provenance == Provenance.UNVERIFIED
    assert lst.heading_path == ["Kurulum"]
    # Item blocks keep their own fields.
    assert lst.items[0].blocks[0].provenance == Provenance.VERIFIED


# ---------------------------------------------------------------------------
# Reversibility: chunk-time exclude_types on the SAME IR, only the chunk
# config changes — no re-parse, IR is not touched (the core claim of the
# visual classification plan).
# ---------------------------------------------------------------------------


def test_exclude_types_reversibility_ayni_ir_farkli_config(tmp_path):
    parsed = doc([
        ir.HeadingBlock(id="h0", text="Giriş", level=1),
        ir.ImageBlock(id="b0", image_index=1, image_id="sha1",
                      visual_type="block_diagram", visual_type_source="vlm",
                      alt_text="Blok diyagram: güç dağıtımı"),
        ir.ParagraphBlock(id="p0", text="ilgili paragraf metni",
                          heading_path=["Giriş"]),
    ])
    tok = FakeTokenizer()

    cfg_full = load_config()  # exclude_types=[] default
    ov = tmp_path / "exclude.toml"
    ov.write_text('[visual]\nexclude_types = ["block_diagram"]\n', encoding="utf-8")
    cfg_excluded = load_config(ov)

    # SAME parsed IR, two different chunk configs → two chunking runs —
    # the parsed dict is NEVER touched (no re-parse).
    cs_full = chunk_document(adapt(parsed, config=cfg_full), cfg_full, tok)
    cs_excluded = chunk_document(adapt(parsed, config=cfg_excluded), cfg_excluded, tok)

    text_full = " ".join(n.text for n in cs_full.nodes)
    text_excluded = " ".join(n.text for n in cs_excluded.nodes)

    # Full-include run shows the real alt-text; excluded run has the
    # placeholder instead, neither leaks into the other.
    assert "güç dağıtımı" in text_full
    assert "güç dağıtımı" not in text_excluded
    assert "[Burada bir blok diyagram var" not in text_full
    assert "[Burada bir blok diyagram var" in text_excluded
    # ...but NOTHING else changes: same blocks, same order, same image
    # reference (stays in the structured images list with its type).
    assert [n.source_block_ids for n in cs_full.nodes] == (
        [n.source_block_ids for n in cs_excluded.nodes])
    assert "ilgili paragraf metni" in text_full
    assert "ilgili paragraf metni" in text_excluded
    full_images = [im.visual_type for n in cs_full.nodes for im in n.images]
    excluded_images = [im.visual_type for n in cs_excluded.nodes for im in n.images]
    assert full_images == excluded_images == ["block_diagram"]