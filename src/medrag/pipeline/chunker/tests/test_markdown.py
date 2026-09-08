"""MD renderer (`chunker.core.markdown`) tests — offline, pure text."""


from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import (
    Code,
    Heading,
    Image,
    ListBlock,
    ListItem,
    Paragraph,
    Provenance,
    Table,
)
from medrag.pipeline.chunker.core.markdown import render_block, render_blocks


def cfg_yap(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return load_config(dosya)


# -- heading / paragraph ---------------------------------------------------------


def test_baslik_seviyesi():
    assert render_block(Heading(id="h1", text="Genel Bakış", level=2)) == \
        "## Genel Bakış"


def test_baslik_seviyesi_6da_kirpilir():
    assert render_block(Heading(id="h1", text="Derin", level=9)).startswith("###### ")


def test_paragraf_escape_edilmez():
    # Body text is canonical: MD special characters stay as-is.
    p = Paragraph(id="p1", text="Fiyat *liste* fiyatıdır # indirimsiz.")
    assert render_block(p) == "Fiyat *liste* fiyatıdır # indirimsiz."


# -- code -----------------------------------------------------------------------


def test_kod_citli_ve_dilli():
    c = Code(id="c1", text="print('merhaba')\n", language="python")
    assert render_block(c) == "```python\nprint('merhaba')\n```"


def test_kod_dilsiz():
    assert render_block(Code(id="c1", text="a = 1")) == "```\na = 1\n```"


def test_kod_icindeki_backtick_citi_uzatir():
    c = Code(id="c1", text="md içinde ```js\nkod\n``` örneği")
    out = render_block(c)
    assert out.startswith("````\n")
    assert out.endswith("\n````")


# -- table ---------------------------------------------------------------------


def test_tablo_pipe_ve_ayrac():
    t = Table(id="t1", header_rows=1,
              cells=[["Ad", "Adet"], ["vida", "10"], ["somun", "4"]])
    assert render_block(t) == (
        "| Ad | Adet |\n"
        "| --- | --- |\n"
        "| vida | 10 |\n"
        "| somun | 4 |")


def test_tablo_cok_satirli_baslik_ayrac_ustune():
    # Decision: the full `header_rows` block goes above the separator.
    t = Table(id="t1", header_rows=2,
              cells=[["Grup", "Grup"], ["Ad", "Adet"], ["vida", "10"]])
    satirlar = render_block(t).split("\n")
    assert satirlar[2] == "| --- | --- |"
    assert satirlar[3] == "| vida | 10 |"


def test_tablo_basliksiz_bos_baslik_uretir():
    # header_rows=0: data rows are not displayed as if they were headers;
    # GFM's mandatory header row is filled with empty cells.
    t = Table(id="t1", header_rows=0, cells=[["a", "b"]])
    assert render_block(t) == "|  |  |\n| --- | --- |\n| a | b |"


def test_tablo_hucre_pipe_escape_ve_br():
    t = Table(id="t1", header_rows=1,
              cells=[["Ad", "Not"], ["boru", "çap|uzunluk\niki satır"]])
    assert "çap\\|uzunluk<br>iki satır" in render_block(t)


def test_tablo_kisa_satir_bos_hucreyle_tamamlanir():
    t = Table(id="t1", header_rows=1, cells=[["A", "B", "C"], ["1"]])
    assert render_block(t).split("\n")[2] == "| 1 |  |  |"


def test_bos_tablo_bos_metin():
    assert render_block(Table(id="t1", cells=[])) == ""


def test_header_rows_satir_sayisina_kirpilir():
    t = Table(id="t1", header_rows=5, cells=[["A"], ["1"]])
    # All rows count as header, separator goes last; no crash.
    assert render_block(t) == "| A |\n| 1 |\n| --- |"


# -- image --------------------------------------------------------------------


def test_gorsel_alt_ve_ocr():
    g = Image(id="g1", alt_text="Bağlantı şeması", ocr_text="ŞEKİL 1: giriş")
    assert render_block(g) == "*Bağlantı şeması*\n\nŞEKİL 1: giriş"


def test_gorsel_bos_metin_tanimlanamayan_placeholder_olur():
    # (visual classification plan): if content is empty and there's no
    # visual_type (unclassified / unprocessed), it falls to a generic
    # placeholder — never silently disappears.
    assert render_block(Image(id="g1", image_id="sha256:abc")) == (
        "[Burada tanımlanamayan bir görsel öğe var: Görsel g1]")


def test_gorsel_cfg_verilmezse_ocr_etiketsiz():
    # cfg=None (default): legacy behavior preserved, no label added.
    g = Image(id="g1", ocr_text="tablo metni", provenance=Provenance.UNVERIFIED)
    assert render_block(g) == "tablo metni"


def test_gorsel_unverified_ocr_etiketlenir(tmp_path):
    # inject_ocr is now OFF by default — open it explicitly to test the
    # labeling behavior.
    cfg = cfg_yap(tmp_path, "[images]\ninject_ocr = true\n")
    g = Image(id="g1", ocr_text="tablo metni", provenance=Provenance.UNVERIFIED)
    metin = render_block(g, cfg)
    assert metin.startswith("*(OCR, doğrulanmamış):*\n")
    assert "tablo metni" in metin


def test_gorsel_verified_ocr_etiketlenmez(tmp_path):
    cfg = cfg_yap(tmp_path, "[images]\ninject_ocr = true\n")
    g = Image(id="g1", ocr_text="tablo metni", provenance=Provenance.VERIFIED)
    assert render_block(g, cfg) == "tablo metni"


def test_gorsel_label_kapatilabilir(tmp_path):
    cfg = cfg_yap(tmp_path, "[images]\ninject_ocr = true\nlabel_unverified_ocr = false\n")
    g = Image(id="g1", ocr_text="tablo metni", provenance=Provenance.UNVERIFIED)
    assert render_block(g, cfg) == "tablo metni"


def test_liste_ici_gorsel_cfg_ile_etiketlenir(tmp_path):
    # cfg must propagate through render_list → _render_item_content →
    # render_block (nested image).
    cfg = cfg_yap(tmp_path, "[images]\ninject_ocr = true\n")
    lb = ListBlock(id="L1", items=[
        ListItem(blocks=[Image(id="g1", ocr_text="satır",
                               provenance=Provenance.UNVERIFIED)]),
    ])
    assert "*(OCR, doğrulanmamış):*" in render_block(lb, cfg)


# -- visual-region type exclusion / placeholder (visual classification plan) ---


def test_gorsel_excluded_at_parse_placeholder():
    # Parser-time excluded (no content produced) — placeholder fires even
    # without cfg (driven by the block's own flag). `image_id` is
    # DELIBERATELY not leaked into the placeholder (structured metadata
    # only); if no `ref`, placeholder shows "—".
    g = Image(id="g1", image_id="sha1", visual_type="block_diagram",
             excluded_at_parse=True)
    assert render_block(g) == "[Burada bir blok diyagram var: Görsel g1, —]"
    assert "sha1" not in render_block(g)


def test_gorsel_chunk_zamani_exclude_types_placeholder(tmp_path):
    cfg = cfg_yap(tmp_path, '[visual]\nexclude_types = ["decorative"]\n')
    g = Image(id="g1", image_id="sha1", visual_type="decorative",
             ocr_text="bu içerik render edilmemeli")
    assert render_block(g, cfg) == "[Burada bir dekoratif görsel var: Görsel g1, —]"


def test_gorsel_exclude_types_disinda_normal_render(tmp_path):
    cfg = cfg_yap(tmp_path,
                  '[images]\ninject_ocr = true\n'
                  '[visual]\nexclude_types = ["decorative"]\n')
    g = Image(id="g1", image_id="sha1", visual_type="product_photo",
             ocr_text="etiket metni")
    assert render_block(g, cfg) == "etiket metni"


def test_gorsel_placeholder_image_id_sizdirmaz_source_crop_kalir():
    # `image_id` lives ONLY in structured metadata (ChunkNode.images/
    # ImageRef); it is not embedded into placeholder text. `source_crop`
    # is a file-path-style reference (NOT an id) — it may appear as `ref`.
    g = Image(id="g1", image_id="sha256:should-not-leak",
             source_crop="crops/page3_img1.png", visual_type="chart",
             excluded_at_parse=True)
    out = render_block(g)
    assert out == "[Burada bir grafik var: Görsel g1, crops/page3_img1.png]"
    assert "sha256" not in out
    assert "should-not-leak" not in out


def test_gorsel_tanimlanamayan_tip_jenerik_placeholder():
    g = Image(id="g1")  # no visual_type, no content
    assert render_block(g) == "[Burada tanımlanamayan bir görsel öğe var: Görsel g1]"


def test_gorsel_bilinen_tip_icerik_bos_yine_placeholder():
    # Processed (not excluded) but no content produced (e.g. SmartArt) —
    # the type still shows through, never silently disappears.
    g = Image(id="g1", visual_type="block_diagram")
    assert render_block(g) == "[Burada bir blok diyagram var: Görsel g1, —]"


def test_gorsel_description_render_edilir():
    g = Image(id="g1", image_id="sha1", visual_type="chart",
             description="Bar chart. Series 'X': A=1, B=2.")
    assert render_block(g) == "Bar chart. Series 'X': A=1, B=2."


def test_gorsel_description_inject_description_kapaliysa_gizlenir(tmp_path):
    """[visual] inject_description is shared with tables — turning it off
    must also hide image descriptions."""
    cfg = cfg_yap(tmp_path, "[visual]\ninject_description = false\n")
    g = Image(id="g1", image_id="sha1", visual_type="chart",
             description="Bar chart. Series 'X': A=1, B=2.")
    # description was the only content; turning it off empties the block
    # → placeholder
    assert render_block(g, cfg) == "[Burada bir grafik var: Görsel g1, —]"


def test_gorsel_ocr_varsayilan_olarak_govdeye_girmez(tmp_path):
    """OCR text is NEVER injected into the chunk BODY — it lives only in
    the structured channel (`ChunkNode.images[].ocr_text`). When OCR is
    the only content the block falls to a placeholder, never silently
    disappears."""
    cfg = cfg_yap(tmp_path, "")  # default: inject_ocr = false
    g = Image(id="g1", image_id="sha1", visual_type="product_photo",
              ocr_text="MPLS 10G VLANs", provenance=Provenance.UNVERIFIED)
    out = render_block(g, cfg)
    assert "MPLS" not in out
    assert "*(OCR, doğrulanmamış):*" not in out
    assert out == "[Burada bir ürün fotoğrafı var: Görsel g1, —]"


def test_gorsel_varsayilanda_yalniz_alt_text_kalir(tmp_path):
    """`inject_image_description` is OFF by default — VLM image
    descriptions don't enter the body, only the structured channel
    (`ChunkNode.images[].description`)."""
    cfg = cfg_yap(tmp_path, "")
    g = Image(id="g1", image_id="sha1", visual_type="chart",
              alt_text="Bağlantı şeması", ocr_text="5V", description="Bar chart.")
    out = render_block(g, cfg)
    assert out == "*Bağlantı şeması*"


def test_gorsel_description_acikca_istenirse_govdeye_girer(tmp_path):
    """The legacy behavior must remain reopenable — both flags on."""
    cfg = cfg_yap(tmp_path, "[visual]\ninject_image_description = true\n")
    g = Image(id="g1", image_id="sha1", visual_type="chart",
              alt_text="Bağlantı şeması", ocr_text="5V", description="Bar chart.")
    out = render_block(g, cfg)
    assert out == "*Bağlantı şeması*\n\nBar chart."


def test_yeni_kapi_tablo_aciklamasini_KAPSAMAZ(tmp_path):
    """The new flag only NARROWS the image path. Table descriptions
    summarize a table that already sits in the chunk (not commentary)
    and stay on by design."""
    cfg = cfg_yap(tmp_path, "")
    assert cfg.visual.inject_description is True
    assert cfg.visual.inject_image_description is False


def test_gorsel_description_inject_description_kapaliyken_ocr_hala_gorunur(tmp_path):
    cfg = cfg_yap(tmp_path,
                  "[images]\ninject_ocr = true\n"
                  "[visual]\ninject_description = false\n")
    g = Image(id="g1", image_id="sha1", visual_type="chart",
             description="Bar chart.", ocr_text="etikette 5V yazıyor")
    out = render_block(g, cfg)
    assert "etikette 5V yazıyor" in out
    assert "Bar chart." not in out


def test_tablo_chunk_zamani_exclude_types_placeholder(tmp_path):
    cfg = cfg_yap(tmp_path, '[visual]\nexclude_types = ["table"]\n')
    t = Table(id="t1", cells=[["A", "B"], ["1", "2"]])
    from medrag.pipeline.chunker.core.table_split import compose_table_text
    assert compose_table_text(t, cfg) == "[Burada bir tablo var: Blok t1, —]"


def test_tablo_exclude_types_kapaliyken_normal_render(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    t = Table(id="t1", cells=[["A", "B"], ["1", "2"]])
    from medrag.pipeline.chunker.core.table_split import compose_table_text
    assert compose_table_text(t, cfg) == render_block(t)


# -- list ---------------------------------------------------------------------


def _mi(text: str, level: int = 0, ordered: bool = False) -> ListItem:
    return ListItem(level=level, ordered=ordered,
                    blocks=[Paragraph(id=f"p-{text}", text=text)])


def test_madde_listesi_ve_girinti():
    lb = ListBlock(id="L1", items=[_mi("bir"), _mi("alt", level=1), _mi("iki")])
    assert render_block(lb) == "- bir\n    - alt\n- iki"


def test_sirali_liste_ardisik_numaralanir():
    lb = ListBlock(id="L1", items=[
        _mi("bir", ordered=True), _mi("iki", ordered=True),
        _mi("alt", level=1, ordered=True), _mi("üç", ordered=True)])
    assert render_block(lb) == "1. bir\n2. iki\n    1. alt\n3. üç"


def test_isaret_tipi_degisince_sayac_sifirlanir():
    lb = ListBlock(id="L1", items=[
        _mi("bir", ordered=True), _mi("madde"), _mi("yeni bir", ordered=True)])
    assert render_block(lb) == "1. bir\n- madde\n1. yeni bir"


def test_cok_bloklu_oge_devam_satirlari_hizali():
    oge = ListItem(blocks=[
        Paragraph(id="p1", text="açıklama"),
        Code(id="c1", text="x = 1", language="python"),
    ])
    out = render_block(ListBlock(id="L1", items=[oge]))
    assert out.split("\n")[0] == "- açıklama"
    # code fence indented by the marker width (2) on continuation lines
    assert "\n  ```python\n  x = 1\n  ```" in out


def test_bos_oge_yalniz_isaret():
    lb = ListBlock(id="L1", items=[ListItem()])
    assert render_block(lb) == "-"


# -- render_blocks -------------------------------------------------------------


def test_render_blocks_bos_satirla_birlesir_ve_bos_atlar():
    bloklar = [
        Heading(id="h1", text="Başlık"),
        Paragraph(id="p2", text=""),  # empty paragraph still skipped
        Paragraph(id="p1", text="Metin."),
    ]
    assert render_blocks(bloklar) == "# Başlık\n\nMetin."