"""N-22 testleri: gece kosusuna baglanan iki KOPUK HALKA --
`catalog_chunk_ownership/build_all.py` (ownership) ve `load_to_db.py` (load).

Kapsam: `argv` parametresi (subprocess YOK, `run_nightly.py`nin cagirdigi
AYNI sekil), `FACTS_RESULTS_DIR`/`CHUNK_OWNERSHIP_DIR` ortam degiskenlerinin
YAZAN/OKUYAN taraflarda AYNI dizine cozulmesi, ve build_all.py'nin
LLM'siz-oldugunda YUKSEK SESLE patlamasi. Tamami OFFLINE -- gercek Ollama/
specs.db'ye hic dokunulmaz (`_chat_ollama`/`_connect_ro`/`DB_PATH` mock'lanir).
"""
from __future__ import annotations

import importlib
import sqlite3
from unittest.mock import MagicMock

import pytest

from medrag.core import paths as core_paths


def test_facts_results_dir_env_overrides_both_writer_and_reader(monkeypatch, tmp_path):
    """`run_full.py::RESULTS_DIR` (facts'in JSON'lari YAZDIGI yer) ve
    `load_to_db.py::RESULTS_DIR` (AYNI JSON'lari OKUYUP specs.db'ye yazan
    yer) `FACTS_RESULTS_DIR` ile AYNI yere cozulmeli -- ayrilirsa `load`
    asamasi facts'in yazdigi hicbir dosyayi BULAMAZ."""
    from medrag.pipeline.facts import load_to_db, run_full

    target = tmp_path / "corpus" / "facts_results"
    monkeypatch.setenv("FACTS_RESULTS_DIR", str(target))
    try:
        run_full_reloaded = importlib.reload(run_full)
        load_to_db_reloaded = importlib.reload(load_to_db)
        assert run_full_reloaded.RESULTS_DIR == target
        assert load_to_db_reloaded.RESULTS_DIR == target
    finally:
        monkeypatch.delenv("FACTS_RESULTS_DIR", raising=False)
        importlib.reload(run_full)
        importlib.reload(load_to_db)


def test_facts_results_dir_default_unchanged_when_unset(monkeypatch):
    """Env tanimsizsa BUGUNKU yol (`facts/chunk_full_run/results`)
    DEGISMEMELI (task kisiti: varsayilanlar davranisi bozmasin)."""
    from medrag.pipeline.facts import load_to_db, run_full

    monkeypatch.delenv("FACTS_RESULTS_DIR", raising=False)
    run_full_reloaded = importlib.reload(run_full)
    load_to_db_reloaded = importlib.reload(load_to_db)

    beklenen = run_full_reloaded._FACTS_DATA_ROOT / "chunk_full_run" / "results"
    assert run_full_reloaded.RESULTS_DIR == beklenen
    assert load_to_db_reloaded.RESULTS_DIR == beklenen


def test_chunk_ownership_dir_env_overrides_writer_and_reader(monkeypatch, tmp_path):
    """`build_all.py::_results_dir()` (YAZAN) ve `discover.py::
    _chunk_ownership_path()` (OKUYAN) `CHUNK_OWNERSHIP_DIR`den AYNI
    `resolve_chunk_ownership_dir()`i (core/paths.py) cagirarak turer --
    ayrilirsa build_all yazar, discover HIC BULAMAZ, cok-sahipli dokumanlar
    yine sessizce disliniyor demektir."""
    from medrag.pipeline.facts import discover
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    target = tmp_path / "corpus" / "ownership_results"
    monkeypatch.setenv("CHUNK_OWNERSHIP_DIR", str(target))

    assert build_all._results_dir() == target
    assert discover._chunk_ownership_path() == target / "chunk_ownership_all.json"
    assert discover.load_chunk_ownership_index.__defaults__ == (None,), (
        "load_chunk_ownership_index varsayilani import-zamanli DONMUS "
        "olmamali -- path=None cagri aninda cozulur"
    )


def test_chunk_ownership_dir_default_unchanged_when_unset(monkeypatch):
    from medrag.pipeline.facts import discover
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    monkeypatch.delenv("CHUNK_OWNERSHIP_DIR", raising=False)

    writer = build_all._results_dir()
    reader_dir = discover._chunk_ownership_path().parent
    assert writer == reader_dir == core_paths.CHUNK_OWNERSHIP_DIR


# --- build_all.main(argv=...) -- LLM/DB tamamen mock'lu -------------------


def _patch_common(monkeypatch, build_all, *, docs, chunks_by_doc, product_info=None):
    fake_config = MagicMock()
    fake_config.scope.exclude_doc_types = []
    monkeypatch.setattr(build_all, "_connect_ro", lambda: MagicMock())
    monkeypatch.setattr(build_all, "load_config", lambda: fake_config)
    monkeypatch.setattr(build_all, "_load_documents", lambda con, exclude: docs)
    monkeypatch.setattr(build_all, "_load_chunks_from_all_chunks_json", lambda: chunks_by_doc)
    monkeypatch.setattr(build_all, "_load_chunk_bridge", dict)
    monkeypatch.setattr(build_all, "_load_product_info", lambda con, codes: product_info or {})


def test_build_all_main_argv_writes_to_env_overridden_results_dir(monkeypatch, tmp_path):
    """`main(argv=[])`, `run_nightly.stage_ownership`in cagirdigi TAM sekil --
    tek-sahipli bir korpusta LLM HIC cagrilmadan deterministik satirlar
    uretilir ve `CHUNK_OWNERSHIP_DIR`in gosterdigi yere yazilir."""
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    monkeypatch.setenv("CHUNK_OWNERSHIP_DIR", str(tmp_path / "ownership"))
    _patch_common(
        monkeypatch, build_all,
        docs=[{"doc_id": "d1", "file_name": "f1.pdf", "doc_type": "DATASHEET",
               "product_codes": ["DE1"]}],
        chunks_by_doc={"d1": [(1, 1, "d1::c0", "govde metni")]},
    )
    monkeypatch.setattr(
        build_all, "_chat_ollama",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("tek-sahipli korpusta LLM cagrilmamali")),
    )

    result = build_all.main(argv=[])

    out_path = tmp_path / "ownership" / "chunk_ownership_all.json"
    assert out_path.is_file()
    assert result["meta"]["n_llm_chunks"] == 0
    assert result["rows"] == [{
        "doc_id": "d1", "file_name": "f1.pdf", "chunk_id": "d1::c0",
        "page_start": 1, "page_end": 1, "product_code": "DE1",
        "confidence": 1.0, "accepted": True, "evidence": None, "source": "deterministic",
    }]


def test_build_all_main_raises_loudly_when_every_llm_call_fails(monkeypatch, tmp_path):
    """N-22 kabul: eksik/yanlis `OWNERSHIP_LLM_BASE_URL` env'inde (konteynerde
    Ollama'ya erisilemiyor) sonuc SESSIZCE hatali/bos-atama satirlarla
    doldurulup 'tamam' basilmamali -- ACIKCA patlamali, tabloya HIC
    YAZILMAMALI (eski tam tablo bozulmasin)."""
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    monkeypatch.setenv("CHUNK_OWNERSHIP_DIR", str(tmp_path / "ownership"))
    _patch_common(
        monkeypatch, build_all,
        docs=[{"doc_id": "d2", "file_name": "f2.pdf", "doc_type": "CATALOGUE",
               "product_codes": ["DE1", "DE2"]}],
        chunks_by_doc={"d2": [(1, 1, "d2::c0", "govde metni")]},
        product_info={"DE1": {"product_code": "DE1"}, "DE2": {"product_code": "DE2"}},
    )

    def _patlayan(system, user):
        raise build_all.ProviderError("cannot reach http://fake-ollama:11434")

    monkeypatch.setattr(build_all, "_chat_ollama", _patlayan)

    with pytest.raises(RuntimeError, match="OWNERSHIP_LLM"):
        build_all.main(argv=[])

    assert not (tmp_path / "ownership" / "chunk_ownership_all.json").exists(), (
        "TUM LLM cagrilari basarisizken eski/yeni tabloya HIC yazilmamali"
    )


# --- load_to_db.main(argv=...) ---------------------------------------------


def test_load_to_db_main_argv_empty_results_dir_returns_zero_products(monkeypatch, tmp_path):
    """`argv=[...]` -- `run_nightly.stage_load`in cagirdigi AYNI sekil,
    orkestratorun kendi sys.argv'sini OKUMADAN calisir. Sonuc dict artik
    (N-22) YAZDIRMAKLA YETINMIYOR, DONUYOR de."""
    from medrag.pipeline.facts import load_to_db

    empty_dir = tmp_path / "results"
    empty_dir.mkdir()
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, unit TEXT, "
                "conditions TEXT, subfields TEXT)")
    con.execute("CREATE TABLE document (doc_id TEXT, doc_type TEXT)")
    con.commit()
    con.close()
    monkeypatch.setattr(load_to_db, "DB_PATH", db_path)
    monkeypatch.setattr(load_to_db, "verify_db_integrity", lambda con: None)
    monkeypatch.setattr(load_to_db, "QUEUE_DIR", tmp_path / "queue")

    result = load_to_db.main(argv=["--results-dir", str(empty_dir), "--no-issues"])

    assert result == {
        "n_products": 0, "n_present": 0, "n_absent": 0, "n_conflicting": 0,
        "n_new_key": 0, "n_skipped_facts": 0, "reports": [],
    }


def test_load_to_db_main_argv_missing_results_dir_raises_system_exit(tmp_path):
    from medrag.pipeline.facts import load_to_db

    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit):
        load_to_db.main(argv=["--results-dir", str(missing)])
