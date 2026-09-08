"""Table splitter (`chunker.core.table_split`) tests — offline, fake
tokenizer.

The fake tokenizer counts 1 word = 1 token; a pipe-table row "| vida | 10 |"
is 5 tokens (3 pipes + 2 cells). Budgets in the tests were hand-computed
with this arithmetic.
"""

from medrag.pipeline.chunker.config import load_config
from medrag.pipeline.chunker.core.document import Table
from medrag.pipeline.chunker.core.table_split import compose_table_text, split_table
from medrag.pipeline.chunker.tokenization.fake import FakeTokenizer


def cfg_yap(tmp_path, icerik: str):
    dosya = tmp_path / "override.toml"
    dosya.write_text(icerik, encoding="utf-8")
    return load_config(dosya)


def tek_kolon(n_satir: int) -> Table:
    """Header ["H"] + body ["r1"].."rN" — 3 tokens per row."""
    return Table(id="t1", header_rows=1,
                 cells=[["H"]] + [[f"r{i}"] for i in range(1, n_satir + 1)])


def stok_tablosu(**degisiklik) -> Table:
    alanlar = {
        "id": "t1", "header_rows": 1,
        "cells": [["Ad", "Adet"], ["vida", "10"], ["somun", "4"], ["pul", "25"]],
        "facts": ["Tablo stok durumunu özetler.",   # matches no cell
                  "Depoda 10 vida vardır.",          # → vida row
                  "Somun sayısı dörttür."],          # → somun row
    }
    alanlar.update(degisiklik)
    return Table(**alanlar)


# -- compose_table_text (unsplit table) ------------------------------------------


def test_compose_facts_kapali_sadece_tablo(tmp_path):
    cfg = cfg_yap(tmp_path, "[table_split]\ninject_facts = false\n")
    metin = compose_table_text(stok_tablosu(), cfg)
    assert metin.startswith("| Ad | Adet |")
    assert "vida" in metin
    assert "özetler" not in metin  # inject_facts off


def test_compose_facts_acik_tablonun_altina(tmp_path):
    cfg = cfg_yap(tmp_path, "[table_split]\ninject_facts = true\n")
    metin = compose_table_text(stok_tablosu(), cfg)
    tablo, facts = metin.split("\n\n", 1)
    assert tablo.endswith("| pul | 25 |")
    assert facts == ("Tablo stok durumunu özetler.\n"
                     "Depoda 10 vida vardır.\nSomun sayısı dörttür.")


def test_compose_description_enjekte_edilir(tmp_path):
    # (revision): description now also goes into unsplit tables — the
    # VLM's only natural-language summary of the table was systematically
    # dropped before because most tables never split.
    cfg = cfg_yap(tmp_path, "")
    metin = compose_table_text(stok_tablosu(description="Stok tablosu."), cfg)
    assert metin.startswith("Stok tablosu.\n\n")


def test_compose_description_kapatilabilir(tmp_path):
    cfg = cfg_yap(tmp_path, "[visual]\ninject_description = false\n")
    metin = compose_table_text(stok_tablosu(description="Stok tablosu."), cfg)
    assert "Stok tablosu." not in metin


# -- split_table: row packing + overlap ------------------------------------------


def test_bolme_target_paketleme_ve_overlap(tmp_path):
    # target=15: header(5) + separator(5) + m rows(3m) ≤ 15 → 3 rows per part.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[table_split]
overlap_rows = 1
""")
    parcalar = split_table(tek_kolon(6), cfg, FakeTokenizer())

    assert [p.split_index for p in parcalar] == [1, 2, 3]
    assert all(p.split_total == 3 for p in parcalar)
    # k=1: parts 2 and 3 start with the last row of the previous part
    assert [p.overlap_units for p in parcalar] == [0, 1, 1]
    assert parcalar[0].table.cells == [["H"], ["r1"], ["r2"], ["r3"]]
    assert parcalar[1].table.cells == [["H"], ["r3"], ["r4"], ["r5"]]
    assert parcalar[2].table.cells == [["H"], ["r5"], ["r6"]]
    # size measured from final render and fits target
    assert [p.token_count for p in parcalar] == [15, 15, 12]
    # id derivation: the adapter's "L1#2" convention
    assert [p.table.id for p in parcalar] == ["t1#p1", "t1#p2", "t1#p3"]


def test_baslik_tekrari_kapatilabilir(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[table_split]
overlap_rows = 1
repeat_header_rows = false
""")
    parcalar = split_table(tek_kolon(6), cfg, FakeTokenizer())
    assert parcalar[0].table.cells[0] == ["H"]
    for parca in parcalar[1:]:
        assert parca.table.header_rows == 0
        assert ["H"] not in parca.table.cells


def test_description_her_parcanin_basinda(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
""")
    tablo = tek_kolon(6).model_copy(update={"description": "Stok tablosu."})
    parcalar = split_table(tablo, cfg, FakeTokenizer())
    assert len(parcalar) > 1
    assert all(p.text.startswith("Stok tablosu.\n\n") for p in parcalar)


def test_description_kapatilabilir(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[visual]
inject_description = false
""")
    tablo = tek_kolon(6).model_copy(update={"description": "Stok tablosu."})
    parcalar = split_table(tablo, cfg, FakeTokenizer())
    assert all("Stok tablosu." not in p.text for p in parcalar)


# -- split_table: fact matching -------------------------------------------------


def test_facts_satira_eslesir_genel_her_parcaya(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 25
hard_max_tokens = 100
[table_split]
overlap_rows = 0
inject_facts = true
""")
    parcalar = split_table(stok_tablosu(), cfg, FakeTokenizer())

    # facts are budget-included: row+fact load forces 1 row per part
    assert len(parcalar) == 3
    # row-matched fact only in that row's part
    assert "Depoda 10 vida vardır." in parcalar[0].text
    assert all("vida vardır" not in p.text for p in parcalar[1:])
    assert "Somun sayısı dörttür." in parcalar[1].text
    # table-wide fact (no cell matched) is in every part
    assert all("Tablo stok durumunu özetler." in p.text for p in parcalar)
    # measured == stored: 15 (table) + 4 (general) + 4 (vida fact)
    assert parcalar[0].token_count == 23


def test_facts_kapaliyken_metne_girmez(tmp_path):
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 25
hard_max_tokens = 100
[table_split]
inject_facts = false
""")
    parcalar = split_table(stok_tablosu(), cfg, FakeTokenizer())
    assert all("özetler" not in p.text for p in parcalar)


def test_overlap_satirinin_facti_tekrarlanmaz(tmp_path):
    # A fact goes into the part where its row FIRST appears; overlap
    # repetition is not accompanied by fact repetition.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
[table_split]
overlap_rows = 1
inject_facts = true
""")
    tablo = tek_kolon(4).model_copy(update={"facts": ["r3 bilgisi burada"]})
    parcalar = split_table(tablo, cfg, FakeTokenizer())

    r3_tasiyanlar = [p for p in parcalar if ["r3"] in p.table.cells]
    fact_tasiyanlar = [p for p in parcalar if "r3 bilgisi burada" in p.text]
    assert len(r3_tasiyanlar) == 2      # r3 is new in one part, overlap in another
    assert len(fact_tasiyanlar) == 1    # fact only in the part where it's new
    assert ["r3"] in fact_tasiyanlar[0].table.cells


def test_facts_butceye_dahil(tmp_path):
    # Same table: with facts off, fits in one part; with the general fact
    # on, it eats the budget and forces a split.
    ortak = """
[limits]
target_tokens = 21
hard_max_tokens = 100
[table_split]
overlap_rows = 0
"""
    tablo = tek_kolon(4).model_copy(
        update={"facts": ["Bu tablo genel stok durumunu özetler."]})  # 6 tokens

    kapali = split_table(
        tablo, cfg_yap(tmp_path, ortak + "inject_facts = false\n"), FakeTokenizer())
    assert len(kapali) == 1  # 5+5+12 = 22 > 21? no: 3*(2+4)=18 ≤ 21

    acik = split_table(
        tablo, cfg_yap(tmp_path, ortak + "inject_facts = true\n"), FakeTokenizer())
    assert len(acik) == 2    # 18 + 6 = 24 > 21 → r1..r3 (15+6=21) + r4


# -- split_table: edge cases ------------------------------------------------------


def test_dev_satir_durustce_tasar(tmp_path):
    # Even a one-new-row part can overflow target; the row is NOT
    # sub-split, the part goes out as-is with an honest token_count.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
""")
    dev = " ".join(f"kelime{i}" for i in range(30))
    tablo = Table(id="t1", header_rows=1,
                  cells=[["H"], ["r1"], [dev], ["r3"]])
    parcalar = split_table(tablo, cfg, FakeTokenizer())

    assert [len(p.table.cells) for p in parcalar] == [2, 2, 2]  # header + 1 row
    assert parcalar[1].token_count > 15
    assert parcalar[0].token_count <= 15 and parcalar[2].token_count <= 15


def test_govdesiz_tablo_tek_parca(tmp_path):
    cfg = cfg_yap(tmp_path, "")
    parcalar = split_table(
        Table(id="t1", header_rows=1, cells=[["A", "B"]]), cfg, FakeTokenizer())
    assert len(parcalar) == 1
    assert parcalar[0].split_total == 1
    assert parcalar[0].table.id == "t1#p1"


def test_yeni_satirlar_kaybolmadan_siralanir(tmp_path):
    # When overlap is dropped, the parts' "new" rows cover the body
    # completely and in order.
    cfg = cfg_yap(tmp_path, """
[limits]
target_tokens = 15
hard_max_tokens = 100
""")
    parcalar = split_table(tek_kolon(9), cfg, FakeTokenizer())
    yeni: list[list[str]] = []
    for p in parcalar:
        govde = p.table.cells[1:]  # skip header repetition
        yeni.extend(govde[p.overlap_units:])
    assert yeni == [[f"r{i}"] for i in range(1, 10)]