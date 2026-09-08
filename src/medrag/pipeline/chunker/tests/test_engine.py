"""Packer (`chunker.core.engine`) tests — offline, fake tokenizer.

The fake tokenizer counts 1 word = 1 token; MD markers (`#`, `|`, `-`,
```` ``` ````) also count as one "word" each (as long as they're
separated by whitespace). Budgets are hand-computed with this arithmetic
(see the same approach in `test_table_split.py`).
"""

from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import (
    Code,
    Document,
    Heading,
    Image,
    LinkRef,
    ListBlock,
    ListItem,
    Paragraph,
    Provenance,
    Table,
)
from medrag.pipeline.chunker.core.engine import chunk_document
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer


def cfg_yap(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return load_config(dosya)


def kirintisiz(cfg):
    """`min_chunk_tokens=0` copy: tests of packing/sectioning stay
    independent of crumb merging (every synthetic chunk would otherwise
    fall under the default threshold with the fake tokenizer; principle
    "tests that need fixed numbers write their own override" applied
    here)."""
    return cfg.model_copy(update={
        "packing": cfg.packing.model_copy(update={"min_chunk_tokens": 0})})


def belge(bloklar, **degisiklik) -> Document:
    alanlar = {"doc_id": "d1", "source_path": "raw/kilavuz.pdf", "fmt": "pdf",
               "blocks": bloklar}
    alanlar.update(degisiklik)
    return Document(**alanlar)


TOK = FakeTokenizer()


# -- sectioning + breadcrumb (hard, default config) ------------------------------


def test_tek_bolum_baslik_md_ic_metinde_breadcrumb_yok():
    doc = belge([
        Heading(id="h1", text="Genel Bakış", level=1),
        Paragraph(id="p1", text="ayrıntı burada", heading_path=["Genel Bakış"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.text == "# Genel Bakış\n\nayrıntı burada"
    assert node.heading_path == ["Genel Bakış"]
    assert node.source_block_ids == ["h1", "p1"]
    assert node.node_id == "d1::c0"


def test_alt_baslikta_breadcrumb_sadece_atalari_gosterir(tmp_path):
    # The section's own heading is already in the text as MD; the
    # breadcrumb doesn't repeat it. section_strategy="hard" pinned by
    # hand: the two units (p1 content, then h2) must stay in separate
    # chunks; default "merge" would merge them.
    cfg = cfg_yap(tmp_path,
                  "[packing]\nsection_strategy = \"hard\"\nmin_chunk_tokens = 0\n")
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="a", heading_path=["Giriş"]),
        Heading(id="h2", text="Detay", level=2, heading_path=["Giriş"]),
        Paragraph(id="p2", text="b", heading_path=["Giriş", "Detay"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 2
    assert cs.nodes[1].text == "**Giriş**\n\n## Detay\n\nb"
    assert cs.nodes[1].heading_path == ["Giriş", "Detay"]


def test_ardisik_basliklar_ayni_boluma_yapisir():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Heading(id="h2", text="Alt", level=2, heading_path=["Giriş"]),
        Paragraph(id="p1", text="içerik", heading_path=["Giriş", "Alt"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].text == "# Giriş\n\n## Alt\n\niçerik"
    assert cs.nodes[0].source_block_ids == ["h1", "h2", "p1"]


def test_hard_stratejide_bolumler_asla_karismaz(tmp_path):
    cfg = cfg_yap(tmp_path, "[limits]\ntarget_tokens = 30\nhard_max_tokens = 100\n"
                            "[packing]\nsection_strategy = \"hard\"\n"
                            "min_chunk_tokens = 0\n")
    doc = belge([
        Heading(id="h1", text="A", level=1),
        Paragraph(id="p1", text="bir iki", heading_path=["A"]),
        Heading(id="h2", text="B", level=1),
        Paragraph(id="p2", text="uc dort", heading_path=["B"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 2
    assert cs.nodes[0].heading_path == ["A"]
    assert cs.nodes[1].heading_path == ["B"]


def test_prev_next_zinciri(tmp_path):
    # section_strategy="hard" pinned: test goal is the prev/next chain
    # between TWO separate chunks (default "merge" would merge into one
    # and trivialize the test).
    cfg = cfg_yap(tmp_path,
                  "[packing]\nsection_strategy = \"hard\"\nmin_chunk_tokens = 0\n")
    doc = belge([
        Heading(id="h1", text="A", level=1),
        Paragraph(id="p1", text="x", heading_path=["A"]),
        Heading(id="h2", text="B", level=1),
        Paragraph(id="p2", text="y", heading_path=["B"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    a, b = cs.nodes
    assert a.prev_node_id is None and a.next_node_id == b.node_id
    assert b.prev_node_id == a.node_id and b.next_node_id is None


def test_bos_belge_sifir_chunk():
    cs = chunk_document(belge([]), load_config(), TOK)
    assert cs.nodes == []


def test_dokuman_baslikla_biter_icerikssiz_chunk(tmp_path):
    cfg = cfg_yap(tmp_path,
                  "[packing]\nsection_strategy = \"hard\"\nmin_chunk_tokens = 0\n")
    doc = belge([
        Paragraph(id="p0", text="önsöz"),
        Heading(id="h1", text="Sonsöz", level=1),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert [n.text for n in cs.nodes] == ["önsöz", "# Sonsöz"]
    assert cs.nodes[1].source_block_ids == ["h1"]


# -- section_strategy = merge ---------------------------------------------------


def test_merge_ardisik_bolumleri_birlestirir(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 30
hard_max_tokens = 100
[packing]
section_strategy = "merge"
""")
    doc = belge([
        Heading(id="h1", text="A", level=1),
        Paragraph(id="p1", text="bir iki", heading_path=["A"]),
        Heading(id="h2", text="B", level=1),
        Paragraph(id="p2", text="uc dort", heading_path=["B"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].text == "# A\n\nbir iki\n\n# B\n\nuc dort"
    # different-root headings: no common ancestor
    assert cs.nodes[0].heading_path == []


def test_merge_kismi_bolum_tasmasi_yok(tmp_path):
    # A+B fits, A+B+C doesn't: C goes into a third chunk as its own section,
    # no "1.5 sections" appear.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 8
hard_max_tokens = 100
[packing]
section_strategy = "merge"
min_chunk_tokens = 0
""")
    doc = belge([
        Heading(id="h1", text="A", level=1),
        Paragraph(id="p1", text="bir", heading_path=["A"]),
        Heading(id="h2", text="B", level=1),
        Paragraph(id="p2", text="iki", heading_path=["B"]),
        Heading(id="h3", text="C", level=1),
        Paragraph(id="p3", text="uc", heading_path=["C"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert [n.source_block_ids for n in cs.nodes] == [
        ["h1", "p1", "h2", "p2"], ["h3", "p3"]]


def test_merge_ortak_atali_basliklar_prefix_korur(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 30
hard_max_tokens = 100
[packing]
section_strategy = "merge"
""")
    doc = belge([
        Heading(id="h1", text="Ana", level=1),
        Heading(id="h2", text="Alt1", level=2, heading_path=["Ana"]),
        Paragraph(id="p1", text="bir", heading_path=["Ana", "Alt1"]),
        Heading(id="h3", text="Alt2", level=2, heading_path=["Ana"]),
        Paragraph(id="p2", text="iki", heading_path=["Ana", "Alt2"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].heading_path == ["Ana"]


# -- table splitting integration --------------------------------------------------


def test_bolunen_tablo_split_alanlari_ve_breadcrumb(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 10
hard_max_tokens = 100
[table_split]
overlap_rows = 0
""")
    tablo = Table(id="t1", header_rows=1, heading_path=["Giriş"],
                  cells=[["H"], ["r1"], ["r2"], ["r3"], ["r4"]])
    doc = belge([Heading(id="h1", text="Giriş", level=1), tablo])
    cs = chunk_document(doc, cfg, TOK)

    assert len(cs.nodes) > 1
    ilk, *devam = cs.nodes
    assert ilk.split_kind == "table" and ilk.split_index == 1
    assert ilk.text.startswith("# Giriş\n\n")  # real heading in the first part
    for parca in devam:
        assert parca.split_kind == "table"
        assert parca.text.startswith("**Giriş**\n\n")  # breadcrumb in continuations
    assert [n.split_index for n in cs.nodes] == list(range(1, len(cs.nodes) + 1))
    assert all(n.split_total == len(cs.nodes) for n in cs.nodes)
    assert all(n.source_block_ids == ["h1", "t1"] or n.source_block_ids == ["t1"]
              for n in cs.nodes)
    # split-block metadata is NOT distributed per part — every part
    # carries the full original block's metadata
    assert all(n.heading_path == ["Giriş"] for n in cs.nodes)


def test_tablo_sigan_boyutta_bolunmez(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    tablo = Table(id="t1", header_rows=1, heading_path=["Giriş"],
                  cells=[["Ad", "Adet"], ["vida", "10"]])
    doc = belge([Heading(id="h1", text="Giriş", level=1), tablo])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].split_kind is None


def test_tablo_facts_bolunmemis_tabloda_compose_ile_girer(tmp_path):
    cfg = cfg_yap(tmp_path, "[table_split]\ninject_facts = true\n")
    tablo = Table(id="t1", header_rows=1, heading_path=["Giriş"],
                  cells=[["Ad", "Adet"], ["vida", "10"]],
                  facts=["Depoda 10 vida vardır."])
    doc = belge([Heading(id="h1", text="Giriş", level=1), tablo])
    cs = chunk_document(doc, cfg, TOK)
    assert cs.nodes[0].text.endswith("Depoda 10 vida vardır.")


def test_kisa_paragraf_tablo_ayni_bolumde_sonra_tablo_kendi_chunklarina_gecer(
        tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 10
hard_max_tokens = 100
[table_split]
overlap_rows = 0
""")
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="kısa metin", heading_path=["Giriş"]),
        Table(id="t1", header_rows=1, heading_path=["Giriş"],
              cells=[["H"], ["r1"], ["r2"], ["r3"], ["r4"]]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert cs.nodes[0].text == "# Giriş\n\nkısa metin"
    assert cs.nodes[0].split_kind is None
    assert all(n.split_kind == "table" for n in cs.nodes[1:])


# -- fallback splitting integration (paragraph/list/code) -------------------------


def test_paragraf_bolunur_ve_overlap_yansir(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[fallback_split]
paragraph_overlap_sentences = 2
""")
    metin = " ".join(f"Cumle{i} yedi kelimelik bir ornek cumledir burada." for
                     i in range(1, 5))
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                Paragraph(id="p1", text=metin, heading_path=["Giriş"])])
    cs = chunk_document(doc, cfg, TOK)

    assert len(cs.nodes) > 1
    assert all(n.split_kind == "paragraph" for n in cs.nodes)
    assert cs.nodes[0].text.startswith("# Giriş\n\n")
    assert all(n.text.startswith("**Giriş**\n\n") for n in cs.nodes[1:])
    assert cs.nodes[0].overlap_units == 0
    assert cs.nodes[1].overlap_units >= 1


def test_liste_uc_seviye_grup_olarak_bolunur(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[fallback_split]
list_overlap_items = 0
""")
    ogeler = [ListItem(level=0, blocks=[Paragraph(id=f"i{i}", text=f"oge {i}")])
              for i in range(1, 8)]
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                ListBlock(id="L1", heading_path=["Giriş"], items=ogeler)])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) > 1
    assert all(n.split_kind == "list" for n in cs.nodes)


def test_kod_satirlarina_bolunur_cit_her_parcada(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 8
hard_max_tokens = 100
[fallback_split]
code_overlap_lines = 0
""")
    kod = "\n".join(f"satir_{i} = {i}" for i in range(1, 6))
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                Code(id="c1", text=kod, language="python",
                     heading_path=["Giriş"])])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) > 1
    for n in cs.nodes:
        assert n.split_kind == "code"
        assert "```python" in n.text


def test_kucultulemeyen_atom_durustce_tasar(tmp_path):
    # A single "sentence" (no punctuation) can overflow even the hard
    # ceiling; it's not split, the flex diagnostic records the overflow.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 5
hard_max_tokens = 20
""")
    dev = " ".join(f"kelime{i}" for i in range(30))
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                Paragraph(id="p1", text=dev, heading_path=["Giriş"])])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.flex_applied is True
    assert node.token_count > node.token_limit
    assert "unsplittable" in node.flex_reason


# -- metadata: cross-ref / image / provenance -------------------------------------


def test_ic_link_cross_ref_dis_link_dislanir():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="bkz", heading_path=["Giriş"], links=[
            LinkRef(text="kurulum", target="#kurulum"),
            LinkRef(text="dış", target="https://example.com"),
        ]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    refs = cs.nodes[0].cross_refs
    assert len(refs) == 1
    assert refs[0].link == "#kurulum"
    assert refs[0].target_chunk_id is None


def test_liste_ici_gorsel_images_listesine_girer():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        ListBlock(id="L1", heading_path=["Giriş"], items=[
            ListItem(blocks=[Image(id="g1", image_id="sha256:abc",
                                   ocr_text="ŞEKİL 1", alt_text="şema")]),
        ]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    images = cs.nodes[0].images
    assert len(images) == 1
    assert images[0].image_id == "sha256:abc"
    assert images[0].ocr_text == "ŞEKİL 1"


def test_gorselden_tureyen_uc_metin_de_images_listesinde_yapili_durur(tmp_path):
    # v6: all THREE image-derived texts must be present verbatim in
    # ImageRef — independent of whether they go into the body, because
    # the structured channel is the consumer's only reliable reference.
    # Two of the three (ocr_text, description) NO LONGER go into the body
    # by default; only alt_text does.
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Image(id="g1", image_id="sha256:abc", alt_text="şema",
              ocr_text="ŞEKİL 1", visual_type="technical_drawing",
              description="The image shows a technical drawing.",
              heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    ref = cs.nodes[0].images[0]
    assert ref.ocr_text == "ŞEKİL 1"
    assert ref.alt_text == "şema"
    assert ref.description == "The image shows a technical drawing."
    metin = cs.nodes[0].text
    assert ref.alt_text in metin, "alt_text stays in body (short, document's own label)"
    assert ref.ocr_text not in metin, "OCR does NOT enter the body"
    assert ref.description not in metin, "VLM description does NOT enter the body"


def test_o13_gorsel_tarifi_varsayilan_configle_govdeye_gomulmez():
    # The fact-extraction input must be a CLEAN chunk set — the 188 fact
    # rows came exactly from VLM descriptions of this pattern leaking
    # into the body. With the default config (no override) the text
    # must NOT enter the chunk body, because facts/run_full.py sends
    # ONLY ChunkRow.text to the LLM.
    aciklama = (
        "On the left side (labeled REAR PANEL) is a connector diagram "
        "showing the DE9 pinout."
    )
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Image(id="g1", image_id="sha256:abc", alt_text="REAR PANEL",
              visual_type="technical_drawing", description=aciklama,
              heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    metin = cs.nodes[0].text
    assert aciklama not in metin, "VLM image description MUST NOT enter the body"
    assert "REAR PANEL" in metin, "short alt_text may stay (document's own label)"
    assert cs.nodes[0].images[0].description == aciklama, "still in structured channel"


def test_description_gomulmese_bile_images_listesinde_durur(tmp_path):
    # [visual] inject_description=false: description does NOT enter the
    # body but the structured reference still carries it — the
    # "don't hide existence" principle holds for description too.
    cfg = cfg_yap(tmp_path, "[visual]\ninject_description = false\n")
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Image(id="g1", image_id="sha256:abc", ocr_text="ŞEKİL 1",
              description="The image shows a chart.", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert "The image shows a chart." not in cs.nodes[0].text
    assert cs.nodes[0].images[0].description == "The image shows a chart."


def test_ocr_varsayilan_konfigle_chunk_govdesine_girmez():
    # End-to-end: with the default config, the engine does NOT put OCR
    # into the body — it's kept in the structured images list.
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Image(id="g1", ocr_text="ŞEKİL 1", provenance=Provenance.UNVERIFIED,
              heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert "ŞEKİL 1" not in cs.nodes[0].text
    assert "*(OCR, doğrulanmamış):*" not in cs.nodes[0].text
    assert cs.nodes[0].images[0].ocr_text == "ŞEKİL 1"


def test_unverified_gorsel_ocru_chunk_metninde_etiketlenir(tmp_path):
    # End-to-end: render_image actually receives cfg and tags unverified
    # OCR when inject_ocr is on.
    cfg = cfg_yap(tmp_path, "[images]\ninject_ocr = true\n")
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Image(id="g1", ocr_text="ŞEKİL 1", provenance=Provenance.UNVERIFIED,
              heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert "*(OCR, doğrulanmamış):*" in cs.nodes[0].text
    assert "ŞEKİL 1" in cs.nodes[0].text
    # structured images list still carries the untagged raw OCR
    assert cs.nodes[0].images[0].ocr_text == "ŞEKİL 1"


def test_provenance_summary_en_dusuk_seviye():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="a", heading_path=["Giriş"],
                  provenance=Provenance.VERIFIED),
        Paragraph(id="p2", text="b", heading_path=["Giriş"],
                  provenance=Provenance.UNVERIFIED),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    # both paragraphs fit together
    assert cs.nodes[0].provenance_summary == Provenance.UNVERIFIED


def test_sayfa_araligi_bolum_bloklarindan_turer():
    doc = belge([
        Heading(id="h1", text="Giriş", level=1, page_start=1, page_end=1),
        Paragraph(id="p1", text="a", heading_path=["Giriş"],
                  page_start=2, page_end=3),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert cs.nodes[0].page_start == 1
    assert cs.nodes[0].page_end == 3


# -- source_path/fmt pass-through ------------------------------------------------


def test_source_path_fmt_gecer():
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                Paragraph(id="p1", text="a", heading_path=["Giriş"])])
    cs = chunk_document(doc, load_config(), TOK)
    assert cs.nodes[0].source_path == "raw/kilavuz.pdf"
    assert cs.nodes[0].fmt == "pdf"


# -- boilerplate filter ---------------------------------------------------------


def test_boilerplate_bolumu_tamamen_dusurulur():
    doc = belge([
        Heading(id="h0", text="Contents", level=1),
        Paragraph(id="p0", text="1. Description ... 2. Hardware ...",
                  heading_path=["Contents"]),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="asıl içerik", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].text == "# Giriş\n\nasıl içerik"
    assert "Contents" not in cs.nodes[0].text


def test_boilerplate_eslesme_casefold_ve_kirpili():
    doc = belge([
        Heading(id="h0", text="  İÇİNDEKİLER  ", level=1),
        Paragraph(id="p0", text="toc", heading_path=["İÇİNDEKİLER"]),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="asıl içerik", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, load_config(), TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].heading_path == ["Giriş"]


def test_boilerplate_listesi_bos_ise_hicbir_sey_dusmez(tmp_path):
    cfg = cfg_yap(tmp_path, "[boilerplate]\ndrop_headings = []\n")
    doc = belge([
        Heading(id="h0", text="Contents", level=1),
        Paragraph(id="p0", text="toc", heading_path=["Contents"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1


def test_boilerplate_normal_baslik_etkilenmez():
    doc = belge([Heading(id="h1", text="Giriş", level=1),
                Paragraph(id="p1", text="a", heading_path=["Giriş"])])
    cs = chunk_document(doc, load_config(), TOK)
    assert len(cs.nodes) == 1


# -- crumb merging --------------------------------------------------------------


def kirinti_cfg(tmp_path, esik: int, ekstra: str = ""):
    return cfg_yap(tmp_path, f"""
[limits]
target_tokens = 30
hard_max_tokens = 100
[packing]
min_chunk_tokens = {esik}
{ekstra}""")


def test_kirinti_sonraki_chunka_katilir(tmp_path):
    # The "önsöz" crumb (1 token) merges into the next section; ids stay
    # consecutive.
    cfg = kirinti_cfg(tmp_path, 3)
    doc = belge([
        Paragraph(id="p0", text="kapak"),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="asıl içerik dört kelime burada",
                  heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.text == "kapak\n\n# Giriş\n\nasıl içerik dört kelime burada"
    assert node.source_block_ids == ["p0", "h1", "p1"]
    assert node.node_id == "d1::c0"
    assert node.token_count == TOK.count(node.text)


def test_bos_metinli_tablo_chunk_komsuya_katilir(tmp_path):
    # A truly content-less section (empty table → "" render) produces an
    # empty chunk that, threshold-independent (empty), joins a neighbor.
    cfg = kirinti_cfg(tmp_path, 1)
    doc = belge([
        Table(id="t1", cells=[]),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="içerik iki üç dört beş", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.text == "# Giriş\n\niçerik iki üç dört beş"  # empty text leaves no trace
    assert node.source_block_ids == ["t1", "h1", "p1"]


def test_siniflandirilmamis_gorsel_artik_kirinti_uretmez_placeholder_kalir(tmp_path):
    # Visual classification plan: an image NEVER renders "" now (at least
    # a placeholder) — what used to count as a crumb no longer is one
    # (placeholder text isn't empty); the "merge" packing strategy
    # bundles it normally with the next section, the image reference
    # doesn't disappear.
    cfg = kirinti_cfg(tmp_path, 1)
    doc = belge([
        Image(id="g1", image_id="sha256:abc"),  # no visual_type → generic placeholder
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="içerik iki üç dört beş", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    node = cs.nodes[0]
    assert node.text == (
        "[Burada tanımlanamayan bir görsel öğe var: Görsel g1]\n\n"
        "# Giriş\n\niçerik iki üç dört beş")
    assert [im.image_id for im in node.images] == ["sha256:abc"]


def test_son_kirinti_oncekine_katilir(tmp_path):
    # Document ends with a crumb ("# Ek\n\nnot" = 3 tokens < 4): can't go
    # forward, joins the previous chunk.
    cfg = kirinti_cfg(tmp_path, 4)
    doc = belge([
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="asıl içerik dört kelime burada",
                  heading_path=["Giriş"]),
        Heading(id="h2", text="Ek", level=1),
        Paragraph(id="p2", text="not", heading_path=["Ek"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].text.endswith("# Ek\n\nnot")
    assert cs.nodes[0].source_block_ids == ["h1", "p1", "h2", "p2"]


def test_kirinti_heading_path_ortak_prefixe_duser(tmp_path):
    # First chunk "# Ana\n\n## Alt1\n\na" = 5 tokens < 6 → crumb.
    cfg = kirinti_cfg(tmp_path, 6)
    doc = belge([
        Heading(id="h1", text="Ana", level=1),
        Heading(id="h2", text="Alt1", level=2, heading_path=["Ana"]),
        Paragraph(id="p1", text="a", heading_path=["Ana", "Alt1"]),
        Heading(id="h3", text="Alt2", level=2, heading_path=["Ana"]),
        Paragraph(id="p2", text="asıl içerik dört kelime burada",
                  heading_path=["Ana", "Alt2"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].heading_path == ["Ana"]


def test_kirinti_bolunmus_parcaya_katilmaz(tmp_path):
    # If the neighbor is a split part, no merging — its split metadata
    # would be corrupted; the crumb stays as-is.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 10
hard_max_tokens = 100
[packing]
min_chunk_tokens = 3
[table_split]
overlap_rows = 0
""")
    tablo = Table(id="t1", header_rows=1, heading_path=["Giriş"],
                  cells=[["H"], ["r1"], ["r2"], ["r3"], ["r4"]])
    doc = belge([
        Paragraph(id="p0", text="ön"),
        Heading(id="h1", text="Giriş", level=1),
        tablo,
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert cs.nodes[0].text == "ön"  # crumb stays
    assert all(n.split_kind == "table" for n in cs.nodes[1:])


def test_kirinti_birlesince_ikinci_parcanin_breadcrumbu_soyulur(tmp_path):
    # Real-corpus pattern (PN1155 c7): two consecutive subsections (H3)
    # share an ancestor; the first is a crumb and joins the second. The
    # second's own breadcrumb ("**Ana > Alt**") must NOT repeat in the
    # merged text — A's context already continues.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 30
hard_max_tokens = 100
[packing]
section_strategy = "hard"
min_chunk_tokens = 8
""")
    doc = belge([
        Heading(id="h0", text="Ana", level=1),
        Heading(id="h1", text="Alt", level=2, heading_path=["Ana"]),
        Heading(id="h2", text="Fiziksel", level=3, heading_path=["Ana", "Alt"]),
        Paragraph(id="p1", text="kisa", heading_path=["Ana", "Alt", "Fiziksel"]),
        Heading(id="h3", text="Ortam", level=3, heading_path=["Ana", "Alt"]),
        Paragraph(id="p2", text="asıl içerik dört kelime burada",
                  heading_path=["Ana", "Alt", "Ortam"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].text == (
        "# Ana\n\n## Alt\n\n### Fiziksel\n\nkisa\n\n"
        "### Ortam\n\nasıl içerik dört kelime burada")
    assert "**Ana > Alt**" not in cs.nodes[0].text
    assert cs.nodes[0].token_count == TOK.count(cs.nodes[0].text)


def test_kirinti_ustuste_zincirleme_birlesir(tmp_path):
    # Consecutive crumbs coalesce in a single pass ("Datasheet" + cover + body).
    cfg = kirinti_cfg(tmp_path, 5)
    doc = belge([
        Paragraph(id="p0", text="Datasheet"),
        Paragraph(id="p1", text="PN1127 kapak satırı"),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p2", text="asıl içerik beş altı yedi sekiz dokuz",
                  heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 1
    assert cs.nodes[0].source_block_ids == ["p0", "p1", "h1", "p2"]


def test_kirinti_birlestirme_sifirla_kapali(tmp_path):
    # section_strategy="hard" pinned: test goal is to show crumb merging
    # is OFF (default "merge" would merge at packing stage and trivialize
    # the test).
    cfg = kirinti_cfg(tmp_path, 0, 'section_strategy = "hard"\n')
    doc = belge([
        Paragraph(id="p0", text="kapak"),
        Heading(id="h1", text="Giriş", level=1),
        Paragraph(id="p1", text="içerik", heading_path=["Giriş"]),
    ])
    cs = chunk_document(doc, cfg, TOK)
    assert len(cs.nodes) == 2