"""Cikti dizinlerinin ARA dizin yokken de yaratildigi testi.

Gercek olay (2026-08-25, ilk konteyner kosusu): `run_full.run_products`
`results_dir.mkdir(exist_ok=True)` cagiriyordu. `exist_ok` yalnizca "son
dizin zaten varsa sorun degil" der -- ustteki eksik dizini YARATMAZ. Yerelde
`facts/chunk_full_run/` zaten var oldugu icin hata hic gorulmedi; konteynerde
`.dockerignore` `facts/` altindan sadece `db/`yi aldigi icin ara dizin yoktu
ve facts asamasi FileNotFoundError ile dustu.
"""

from __future__ import annotations

import pytest

from medrag.pipeline.facts import run_full


def test_run_products_ara_dizin_yokken_de_yaratir(tmp_path, monkeypatch):
    """Hicbir usti olmayan derin bir yola cikti istendiginde tum zincir
    yaratilmali. Kod yolu kisa tutuluyor: `codes=[]` ile hicbir LLM cagrisi
    yapilmadan yalnizca dizin+ozet olusturma adimi kosar."""
    hedef = tmp_path / "yok" / "hala_yok" / "results"
    assert not hedef.parent.exists()

    ozet = run_full.run_products(
        [], con=None, provider="ollama", model="test", base_url="", api_key="",
        all_chunks_index={}, atoms_bridge_index={}, exclude_doc_types=set(),
        owner_counts={}, chunk_ownership_index={}, chunk_anchor_index={},
        system="test", results_dir=hedef,
    )

    assert hedef.is_dir()
    assert (hedef / "_run_summary.json").is_file()
    assert ozet["n_products_requested"] == 0


@pytest.mark.parametrize("modul, oznitelik", [
    ("medrag.pipeline.facts.load_to_db", "QUEUE_DIR"),
    ("medrag.pipeline.facts.catalog_chunk_ownership.build_all", "RESULTS_DIR"),
])
def test_ayni_hata_kalan_cikti_dizinlerinde_de_yok(modul, oznitelik):
    """Ayni desen uc yerde daha vardi (`load_to_db.QUEUE_DIR`,
    `build_all.RESULTS_DIR` x2). Biri facts zincirinin BIR SONRAKI adimi --
    duzeltilmeseydi kosu birkac dakika sonra ayni sekilde duserdi. Bu test
    kaynakta `mkdir(exist_ok=True)` deseninin geri gelmesini yakalar."""
    import importlib
    import inspect

    kaynak = inspect.getsource(importlib.import_module(modul))
    assert f"{oznitelik}.mkdir(exist_ok=True)" not in kaynak, (
        f"{modul}.{oznitelik} yine parents=True olmadan mkdir ediyor")
