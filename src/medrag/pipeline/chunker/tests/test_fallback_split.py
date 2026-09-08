"""Fallback splitters (`chunker.core.fallback_split`) tests — offline,
fake tokenizer (1 word = 1 token).
"""

from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import Code, ListBlock, ListItem, Paragraph
from medrag.pipeline.chunker.core.fallback_split import (
    sentences,
    split_code,
    split_list,
    split_paragraph,
)
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer


def cfg_yap(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return load_config(dosya)


TOK = FakeTokenizer()


# -- sentences() ------------------------------------------------------------------


def test_sentences_noktalama_sonrasi_boler():
    assert sentences("Bir cümle. İki cümle! Üç mü?") == [
        "Bir cümle.", "İki cümle!", "Üç mü?"]


def test_sentences_noktalamasiz_tek_parca():
    assert sentences("noktalama yok burada") == ["noktalama yok burada"]


def test_sentences_bos_metin():
    assert sentences("") == []
    assert sentences("   ") == []


def test_sentences_ucgen_noktasi():
    assert sentences("Devam ediyor… sonra bitti.") == ["Devam ediyor…", "sonra bitti."]


# -- split_paragraph ----------------------------------------------------------------


def test_paragraf_sigan_boyutta_tek_parca(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    p = Paragraph(id="p1", text="Kısa bir cümle burada.")
    parcalar = split_paragraph(p, cfg, TOK)
    assert len(parcalar) == 1
    assert parcalar[0].split_total == 1
    assert parcalar[0].text == "Kısa bir cümle burada."
    assert parcalar[0].block.id == "p1#p1"


def test_paragraf_cumle_sinirindan_boler(tmp_path):
    # target=6: each sentence = 3 words → 2 sentences=6 fit, 3 don't.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 6
hard_max_tokens = 100
[fallback_split]
paragraph_overlap_sentences = 0
""")
    metin = "Bir iki uc. Dort bes alti. Yedi sekiz dokuz. On onbir oniki."
    p = Paragraph(id="p1", text=metin)
    parcalar = split_paragraph(p, cfg, TOK)

    assert [pt.split_index for pt in parcalar] == [1, 2]
    assert parcalar[0].text == "Bir iki uc. Dort bes alti."
    assert parcalar[1].text == "Yedi sekiz dokuz. On onbir oniki."
    assert all(pt.overlap_units == 0 for pt in parcalar)
    assert [pt.token_count for pt in parcalar] == [6, 6]


def test_paragraf_overlap_cumle_tekrarlar(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 6
hard_max_tokens = 100
[fallback_split]
paragraph_overlap_sentences = 1
""")
    metin = "Bir iki uc. Dort bes alti. Yedi sekiz dokuz."
    parcalar = split_paragraph(Paragraph(id="p1", text=metin), cfg, TOK)
    assert parcalar[0].text == "Bir iki uc. Dort bes alti."
    assert parcalar[1].text == "Dort bes alti. Yedi sekiz dokuz."
    assert parcalar[1].overlap_units == 1


def test_paragraf_dev_cumle_bolunmez_durustce_tasar(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 3
hard_max_tokens = 100
""")
    dev = " ".join(f"kelime{i}" for i in range(20)) + "."
    parcalar = split_paragraph(Paragraph(id="p1", text=dev), cfg, TOK)
    assert len(parcalar) == 1
    assert parcalar[0].token_count == 20


def test_paragraf_id_turetme(tmp_path):
    cfg = cfg_yap(tmp_path, "[limits]\ntarget_tokens = 6\nhard_max_tokens = 100\n"
                            "[fallback_split]\nparagraph_overlap_sentences = 0\n")
    metin = "Bir iki uc. Dort bes alti. Yedi sekiz dokuz."
    parcalar = split_paragraph(Paragraph(id="para", text=metin), cfg, TOK)
    assert [pt.block.id for pt in parcalar] == ["para#p1", "para#p2"]


# -- split_code -------------------------------------------------------------------


def test_kod_sigan_boyutta_tek_parca(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    c = Code(id="c1", text="x = 1\ny = 2", language="python")
    parcalar = split_code(c, cfg, TOK)
    assert len(parcalar) == 1
    assert parcalar[0].text == "```python\nx = 1\ny = 2\n```"


def test_kod_satir_sinirindan_boler_cit_her_parcada(tmp_path):
    # target=6: "```python"+"satir_N = N" = 1(fence)+3(line) = 4 tokens/line,
    # so 1 line fits, 2 lines (4+3=7) don't → 1 line per part.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 6
hard_max_tokens = 100
[fallback_split]
code_overlap_lines = 0
""")
    kod = "satir_1 = 1\nsatir_2 = 2\nsatir_3 = 3"
    parcalar = split_code(Code(id="c1", text=kod, language="python"), cfg, TOK)

    assert len(parcalar) == 3
    for i, pt in enumerate(parcalar, start=1):
        assert pt.text == f"```python\nsatir_{i} = {i}\n```"
        assert pt.split_index == i
    assert all(pt.overlap_units == 0 for pt in parcalar)


def test_kod_overlap_satir_tekrarlar(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 9
hard_max_tokens = 100
[fallback_split]
code_overlap_lines = 1
""")
    kod = "a = 1\nb = 2\nc = 3\nd = 4"
    parcalar = split_code(Code(id="c1", text=kod), cfg, TOK)
    # consecutive parts share 1 line
    assert parcalar[1].overlap_units == 1


def test_kod_sondaki_yeni_satir_yok_sayilir(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    c = Code(id="c1", text="x = 1\n")
    parcalar = split_code(c, cfg, TOK)
    assert len(parcalar) == 1
    assert parcalar[0].text == "```\nx = 1\n```"


# -- split_list -------------------------------------------------------------------


def _oge(text: str, level: int = 0) -> ListItem:
    return ListItem(level=level, blocks=[Paragraph(id=f"i-{text}", text=text)])


def test_liste_sigan_boyutta_tek_parca(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    lb = ListBlock(id="L1", items=[_oge("bir"), _oge("iki")])
    parcalar = split_list(lb, cfg, TOK)
    assert len(parcalar) == 1
    assert parcalar[0].text == "- bir\n- iki"


def test_liste_ust_seviye_gruptan_boler(tmp_path):
    # target=2: each item = 1 word + "- " marker = 1 token → 2 items = 4
    # tokens, doesn't fit → 1 item per part.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 2
hard_max_tokens = 100
[fallback_split]
list_overlap_items = 0
""")
    lb = ListBlock(id="L1", items=[_oge("bir"), _oge("iki"), _oge("uc")])
    parcalar = split_list(lb, cfg, TOK)
    assert [pt.text for pt in parcalar] == ["- bir", "- iki", "- uc"]


def test_liste_ic_ice_oge_grubuyla_birlikte_tasinir(tmp_path):
    # A nested item is not split off its top-level item: the group stays whole.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 100
hard_max_tokens = 200
[fallback_split]
list_overlap_items = 0
""")
    lb = ListBlock(id="L1", items=[
        _oge("bir"), _oge("altbir", level=1), _oge("iki")])
    parcalar = split_list(lb, cfg, TOK)
    assert len(parcalar) == 1  # fits, so one part; group integrity tested below

    cfg2 = cfg_yap(tmp_path, """
[limits]
target_tokens = 5
hard_max_tokens = 100
[fallback_split]
list_overlap_items = 0
""")
    parcalar2 = split_list(lb, cfg2, TOK)
    # "bir" + nested "altbir" together, "iki" separate
    assert parcalar2[0].text == "- bir\n    - altbir"
    assert parcalar2[1].text == "- iki"


def test_liste_overlap_grup_tekrarlar(tmp_path):
    # target=4: window holds 2 items ("- bir\n- iki" = 4 tokens) →
    # overlap=1 is actually applicable (clipped when window is 1 item).
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 4
hard_max_tokens = 100
[fallback_split]
list_overlap_items = 1
""")
    lb = ListBlock(id="L1", items=[_oge("bir"), _oge("iki"), _oge("uc"),
                                    _oge("dort")])
    parcalar = split_list(lb, cfg, TOK)
    assert parcalar[0].text == "- bir\n- iki"
    assert parcalar[1].text == "- iki\n- uc"
    assert parcalar[1].overlap_units == 1


def test_liste_bos_govde_sifir_parca(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    parcalar = split_list(ListBlock(id="L1", items=[]), cfg, TOK)
    assert parcalar == []