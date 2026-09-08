"""run_nightly.py (N-01) testleri -- tamami OFFLINE (LLM/GPU/gercek dosya
sistemi/subprocess yok). `stage_*` fonksiyonlari mock ile degistirilir;
gercek `run_parse_pipeline`/`run_full`/`vectorize.cli` HIC import EDILMEZ.

Calistirma: `python -m pytest src/medrag/pipeline/cli/tests/test_run_nightly.py`
"""

from __future__ import annotations

import json
import unittest.mock
from unittest.mock import MagicMock

import pytest

from medrag.pipeline.cli import run_nightly


def _mock_backup_ok(monkeypatch, tmp_path):
    """N-05: `main()`in dry-run OLMAYAN yolu artik ilk adim olarak
    `nightly_backup.run_backup()` cagirir -- gercek yedek (specs.db/
    document_nodes.json/vb. gercek dosya sistemi erisimi) gerektirmeyen
    testler bunu SAHTE, hep basarili donen bir surumle degistirir."""
    from medrag.pipeline.cli import nightly_backup

    fake_manifest = nightly_backup.BackupManifest(
        root=tmp_path / "fake-backup", started_at="t0", finished_at="t1",
        duration_seconds=0.01, entries=[])
    monkeypatch.setattr(nightly_backup, "run_backup",
                        lambda **kwargs: fake_manifest)
    # N-25: `restore_backup`/isaretci fonksiyonlari gercek dosya sistemine
    # dokunur (manifest.json okur, dosya kopyalar) -- sahte yedegin diskte
    # gercek icerigi yok, o yuzden bu yardimci onlari da SAHTE/no-op yapar
    # (restore-DAVRANISININ kendisini test eden testler kendi manifest.json'unu
    # kurup bu sahteyi kullanmaz).
    monkeypatch.setattr(nightly_backup, "restore_backup", lambda root: [])
    monkeypatch.setattr(nightly_backup, "mark_in_progress", lambda *a, **kw: None)
    monkeypatch.setattr(nightly_backup, "read_in_progress", lambda **kw: None)
    monkeypatch.setattr(nightly_backup, "clear_in_progress", lambda **kw: None)
    return fake_manifest


def test_plan_full_chain_default():
    """med-rag'de Excel urun katalogu (ve dolayisiyla `products` asamasi)
    kaldirildi; zincir scan'de baslar -- urun katalogu olmadan da tum
    belgeler kaydedilir (catalog-less scan modu)."""
    assert run_nightly.plan() == [
        "scan", "parse", "chunk", "ownership", "facts", "load", "vectorize",
    ]


def test_plan_from_stage():
    assert run_nightly.plan(from_stage="chunk") == [
        "chunk", "ownership", "facts", "load", "vectorize",
    ]


def test_plan_single_stage():
    assert run_nightly.plan(stage="facts") == ["facts"]


def test_plan_stage_and_from_mutually_exclusive():
    with pytest.raises(SystemExit):
        run_nightly.plan(stage="facts", from_stage="chunk")


def test_plan_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        run_nightly.plan(stage="does-not-exist")


def test_from_chunk_never_calls_parse_or_scan(monkeypatch):
    """KURAL (I-26): `--from chunk` ile baslatilan zincirde `scan`/`parse`
    HIC CAGRILMAZ -- parse saatler surebilir, sessizce kosarsa fark
    edilmez ama GPU'yu isgal eder. Her bes `stage_*` da mock'lanir; yalnizca
    chunk/facts/vectorize'in cagrildigi, scan/parse'in HIC dokunulmadigi
    dogrulanir."""
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    executed = run_nightly.run(from_stage="chunk")

    assert executed == ["chunk", "ownership", "facts", "load", "vectorize"]
    mocks["scan"].assert_not_called()
    mocks["parse"].assert_not_called()
    mocks["chunk"].assert_called_once_with(doc_id=None)
    mocks["ownership"].assert_called_once_with(doc_id=None)
    mocks["facts"].assert_called_once_with(doc_id=None)
    mocks["load"].assert_called_once_with(doc_id=None)
    mocks["vectorize"].assert_called_once_with(doc_id=None)


def test_stage_flag_calls_only_that_stage(monkeypatch):
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    executed = run_nightly.run(stage="facts", doc_id="PN1057")

    assert executed == ["facts"]
    mocks["facts"].assert_called_once_with(doc_id="PN1057")
    for name in ("scan", "parse", "chunk", "ownership", "load", "vectorize"):
        mocks[name].assert_not_called()


def test_dry_run_calls_no_real_stage_function(monkeypatch):
    """`--dry-run`, gercek `stage_*` fonksiyonlarina HIC dokunmaz (yazma/ag
    cagrisi riski tasiyanlar bunlar) -- yerine `_dry_run_*` raporlayicilari
    cagrilir (N-03)."""
    real_stage_mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in real_stage_mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)
    dry_run_mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in dry_run_mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._DRY_RUN_FUNC_NAMES[name], mock)

    executed = run_nightly.run(dry_run=True)

    assert executed == run_nightly.STAGES
    for mock in real_stage_mocks.values():
        mock.assert_not_called()
    for name, mock in dry_run_mocks.items():
        mock.assert_called_once_with(doc_id=None)


def test_doc_id_propagates_to_every_called_stage(monkeypatch):
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    run_nightly.run(from_stage="facts", doc_id="PN1337")

    mocks["facts"].assert_called_once_with(doc_id="PN1337")
    mocks["vectorize"].assert_called_once_with(doc_id="PN1337")


def test_cli_from_flag_parses(monkeypatch, tmp_path):
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)
    # gercek (dry-run olmayan) yol N-02 kilidini kullanir -- test kilidini
    # sistem gecici dosyasindan izole eder (paralel testlerle catismasin).
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(tmp_path / "nightly.lock"))
    _mock_backup_ok(monkeypatch, tmp_path)

    run_nightly.main(["--from", "vectorize"])

    mocks["vectorize"].assert_called_once_with(doc_id=None)
    for name in ("scan", "parse", "chunk", "ownership", "facts", "load"):
        mocks[name].assert_not_called()


def test_doc_noop_stages_warn_but_run(monkeypatch, capsys):
    """scan/parse/chunk/vectorize --doc destegi yok -- NO-OP uyarisi
    basilir ama asamanin kendisi yine calisir (sessizce yanlis is yapmaz,
    sessizce hicbir sey de atlamaz)."""
    called = []
    monkeypatch.setattr(run_nightly, "_load_classify_documents",
                        lambda: MagicMock(main=lambda: called.append("scan")))

    run_nightly.stage_scan(doc_id="DE1")

    assert called == ["scan"]
    assert "no-op" in capsys.readouterr().out


# --- N-03: kuru kosu raporlayicilari ------------------------------------


def test_dry_run_scan_calls_classify_documents_with_dry_run_flag(monkeypatch):
    """`_dry_run_scan`, `classify_documents.main(dry_run=True)`yi cagirir --
    `dry_run=False` (gercek yazan) surumu DEGIL."""
    module = MagicMock()
    monkeypatch.setattr(run_nightly, "_load_classify_documents", lambda: module)

    run_nightly._dry_run_scan()

    module.main.assert_called_once_with(dry_run=True)


def test_dry_run_facts_never_calls_run_products(monkeypatch):
    """`_dry_run_facts`, `resolve_codes` disinda hicbir seye dokunmaz --
    `run_products` (LLM cagrisinin gercek tetikleyicisi) HIC cagrilmaz."""
    import medrag.pipeline.facts as facts_pkg

    fake_run_full = MagicMock()
    fake_run_full.ENV_PATH = "irrelevant"
    fake_run_full._connect_ro.return_value = "fake-con"
    fake_run_full.resolve_codes.return_value = ["DE1", "DE2"]
    monkeypatch.setattr(facts_pkg, "run_full", fake_run_full, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)

    run_nightly._dry_run_facts(doc_id="DE1")

    fake_run_full.resolve_codes.assert_called_once_with("fake-con", doc_id="DE1")
    fake_run_full.run_products.assert_not_called()


def test_dry_run_chunk_reads_input_dir_without_subprocess(monkeypatch, tmp_path):
    """`_dry_run_chunk`, girdi klasorundeki dokuman klasorlerini SAYAR ama
    chunker'i subprocess olarak HIC baslatmaz (registry'ye de yazmaz)."""
    input_dir = tmp_path / "parsed"
    for name in ("docA", "docB", "docC"):
        (input_dir / name).mkdir(parents=True)
    (input_dir / "not_a_dir.txt").write_text("x", encoding="utf-8")

    monkeypatch.setenv("CHUNKER_INPUT_DIR", str(input_dir))
    monkeypatch.delenv("PARSED_OUTPUT_DIR", raising=False)

    with unittest.mock.patch("subprocess.run") as fake_run:
        run_nightly._dry_run_chunk()
        fake_run.assert_not_called()


def test_dry_run_vectorize_only_discovers_never_builds_embedder(monkeypatch, tmp_path):
    """`_dry_run_vectorize`, embedder/Qdrant istemcisini HIC kurmaz --
    yalniz `discover_chunk_scopes` (disk okuma) cagrilir."""
    input_dir = tmp_path / "chunks"
    input_dir.mkdir()
    monkeypatch.setenv("VECTORIZE_INPUT_DIR", str(input_dir))

    fake_scopes = ["scope1", "scope2"]
    with unittest.mock.patch(
            "medrag.pipeline.vectorize.discover.discover_chunk_scopes",
            return_value=fake_scopes) as fake_discover, \
        unittest.mock.patch(
            "medrag.pipeline.vectorize.cli.embedder_from_env") as fake_embedder:
        run_nightly._dry_run_vectorize()
        fake_discover.assert_called_once()
        fake_embedder.assert_not_called()


def test_main_rejects_second_run_while_lock_held(monkeypatch, tmp_path, capsys):
    """N-02: `main()` (dry-run OLMAYAN yol) NightlyLock kullanir -- kilit
    tutulurken ikinci `main()` cagrisi REDDEDILIR (SystemExit(1)), ilk
    surecin `stage_*`lari HIC etkilenmez."""
    from medrag.pipeline.cli.nightly_lock import NightlyLock

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    _mock_backup_ok(monkeypatch, tmp_path)
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    holder = NightlyLock(lock_path)
    holder.acquire()
    try:
        with pytest.raises(SystemExit):
            run_nightly.main(["--stage", "facts"])
        assert "REDDEDILDI" in capsys.readouterr().out
        mocks["facts"].assert_not_called()
    finally:
        holder.release()


def test_main_acquires_and_releases_lock_around_real_run(monkeypatch, tmp_path):
    """Kilit tutulmuyorken `main()` kilidi alir, kosuyu yapar, sonunda
    SERBEST birakir (bir sonraki gecenin kosabilmesi icin)."""
    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    _mock_backup_ok(monkeypatch, tmp_path)
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    run_nightly.main(["--stage", "facts"])

    mocks["facts"].assert_called_once_with(doc_id=None)
    assert not lock_path.exists(), "kosu bitince kilit SERBEST kalmali"


# --- N-05: yedek-once adimi ----------------------------------------------


def test_main_calls_backup_before_any_stage(monkeypatch, tmp_path):
    """N-05: `main()` (dry-run OLMAYAN yol) her `stage_*`den ONCE
    `nightly_backup.run_backup()`u cagirir."""
    from medrag.pipeline.cli import nightly_backup

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    # N-21: `main([])` (tam zincir) artik kosu sonunda bir rapor YAZAR --
    # gercek repo `reports/` dizinine sizmasin diye tmp_path'e yonlendirilir.
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))
    calls = []
    fake_manifest = nightly_backup.BackupManifest(
        root=tmp_path / "backup", started_at="t0", finished_at="t1",
        duration_seconds=0.01, entries=[])
    monkeypatch.setattr(nightly_backup, "run_backup",
                        lambda **kwargs: (calls.append("backup"), fake_manifest)[1])
    mocks = {name: MagicMock(side_effect=lambda n=name, **kw: calls.append(n))
             for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    run_nightly.main([])

    assert calls[0] == "backup", "yedek HER ZAMAN ilk asamadan once gelmeli"
    assert calls[1:] == run_nightly.STAGES
    assert (tmp_path / "reports").exists(), "tam zincir kosusu bir rapor YAZMALI (N-21)"


def test_main_aborts_with_no_stage_calls_when_backup_fails(monkeypatch, tmp_path):
    """N-05'in kabul kriteri: yedek basarisiz olursa HICBIR asama
    calistirilmaz ve gece BASARISIZ raporlanir (burada SystemExit(1) ile)."""
    from medrag.pipeline.cli import nightly_backup

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))

    def _patlayan_yedek(**kwargs):
        raise nightly_backup.BackupError("test: yedek hedefi salt-okunur")

    monkeypatch.setattr(nightly_backup, "run_backup", _patlayan_yedek)
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    with pytest.raises(SystemExit):
        run_nightly.main([])

    for name, mock in mocks.items():
        mock.assert_not_called()
    assert not lock_path.exists(), "yedek basarisiz olsa bile kilit SERBEST kalmali"


def test_full_dry_run_plan_uses_only_dry_run_reporters(monkeypatch):
    """`run(dry_run=True)`, plandaki HER asama icin gercek `stage_*` yerine
    `_dry_run_*` raporlayicisini cagirir -- tam zincirde de dogru."""
    real_mocks = {name: MagicMock() for name in run_nightly.STAGES}
    dry_mocks = {name: MagicMock() for name in run_nightly.STAGES}
    for name in run_nightly.STAGES:
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], real_mocks[name])
        monkeypatch.setattr(run_nightly, run_nightly._DRY_RUN_FUNC_NAMES[name], dry_mocks[name])

    executed = run_nightly.run(dry_run=True, doc_id="DE7")

    assert executed == run_nightly.STAGES
    for name in run_nightly.STAGES:
        real_mocks[name].assert_not_called()
        dry_mocks[name].assert_called_once_with(doc_id="DE7")


# --- N-19: forget_deleted_source gece kosusuna baglaniyor -----------------


def _write_document_nodes(path, *records):
    path.write_text(json.dumps({"documents": list(records)}, ensure_ascii=False), encoding="utf-8")


def test_forgettable_scan_records_returns_deleted_and_modified(monkeypatch, tmp_path):
    """K-97 (2026-08-26 fix): `MODIFIED` de `DELETED` gibi unutulacak kayitlar
    listesine GIRMELI (kaynak degisimi silme+ekleme olarak islenir). `MOVED`
    ve `is_active=false` (editoryal devre disi, FARKLI niyet -- K-96) bu
    listeye GIRMEMELI. K-100: donus artik `content_hash`i DORDUNCU alan
    olarak tasir (cagiran taraf `forgotten_hash` yazabilsin diye)."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-deleted"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "DELETED", "content_hash": "sha256:d1"}},
        {"identity": {"doc_id": "doc-modified"}, "doc_type": "CATALOGUE",
         "is_active": True, "scan": {"scan_status": "MODIFIED", "content_hash": "sha256:m1"}},
        {"identity": {"doc_id": "doc-moved"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "MOVED", "content_hash": "sha256:mv1"}},
        {"identity": {"doc_id": "doc-inactive"}, "doc_type": "DATASHEET",
         "is_active": False, "scan": {"scan_status": "UNCHANGED", "content_hash": "sha256:i1"}},
        {"identity": {"doc_id": "doc-live"}, "doc_type": "BROCHURE",
         "is_active": True, "scan": {"scan_status": "UNCHANGED", "content_hash": "sha256:l1"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)

    result = run_nightly._forgettable_scan_records()

    assert result == [
        ("doc-deleted", "DATASHEET", "DELETED", "sha256:d1"),
        ("doc-modified", "CATALOGUE", "MODIFIED", "sha256:m1"),
    ]


def test_forgettable_scan_records_skips_already_forgotten_same_hash(monkeypatch, tmp_path):
    """K-100 (2026-08-27 fix): `forgotten_hash == content_hash` -- bu kayit
    daha once, AYNI icerikle, zaten unutulmus (turevleri silinmis) --
    tekrar secilmemeli (birikme sorununun kok duzeltmesi)."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-already-forgotten"}, "doc_type": "DATASHEET",
         "is_active": True,
         "scan": {"scan_status": "DELETED", "content_hash": "sha256:d1",
                   "forgotten_hash": "sha256:d1"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)

    assert run_nightly._forgettable_scan_records() == []


def test_forgettable_scan_records_reselects_when_hash_changed_since_forgotten(
        monkeypatch, tmp_path):
    """K-100: `forgotten_hash` KAYITLI ama guncel `content_hash`ten FARKLI --
    icerik unutulduktan SONRA tekrar degisti (yeni bir MODIFIED olayi), bu
    yeni icerigin ESKI turevleri de silinmeli -- tekrar secilmeli."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-changed-again"}, "doc_type": "DATASHEET",
         "is_active": True,
         "scan": {"scan_status": "MODIFIED", "content_hash": "sha256:new",
                   "forgotten_hash": "sha256:old"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)

    assert run_nightly._forgettable_scan_records() == [
        ("doc-changed-again", "DATASHEET", "MODIFIED", "sha256:new"),
    ]


def test_forgettable_scan_records_missing_file_is_empty(monkeypatch, tmp_path):
    from medrag.pipeline.cli import run_parse_pipeline

    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH",
                        str(tmp_path / "yok.json"), raising=False)

    assert run_nightly._forgettable_scan_records() == []


def test_forget_deleted_sources_dry_run_deletes_nothing(monkeypatch, tmp_path, capsys):
    """N-19 kabul: `--dry-run` yolunda HICBIR SEY SILINMEZ, yalniz rapor."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-1"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "DELETED", "content_hash": "sha256:d1"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)
    forget_mock = MagicMock()
    monkeypatch.setattr("medrag.pipeline.forget_deleted_source.forget_deleted_source", forget_mock)

    result = run_nightly._forget_deleted_sources(dry_run=True)

    assert result == [("doc-1", "DATASHEET", "DELETED", "sha256:d1")]
    forget_mock.assert_not_called()
    assert "YAZILMAYACAK" in capsys.readouterr().out


def test_forget_deleted_sources_dry_run_noop_when_nothing_deleted(monkeypatch, tmp_path, capsys):
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(nodes_path)
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)
    forget_mock = MagicMock()
    monkeypatch.setattr("medrag.pipeline.forget_deleted_source.forget_deleted_source", forget_mock)

    result = run_nightly._forget_deleted_sources(dry_run=True)

    assert result == []
    forget_mock.assert_not_called()
    assert "DELETED/MODIFIED kayit yok" in capsys.readouterr().out


def test_forget_deleted_sources_real_run_calls_forget_deleted_source_per_doc(
        monkeypatch, tmp_path):
    """N-19 kabul + K-97 (2026-08-26 fix): gercek yol (dry_run=False) her
    DELETED **VE MODIFIED** doc_id icin `forget_deleted_source`u cagirir --
    kaynak degisimi silme+ekleme olarak islenir. Gercek Qdrant/DB baglantisi
    YOK, hepsi mock'lanir (bu makinede LLM/ag cagrisi YASAK, ayrica gercek
    specs.db'ye YAZILMAZ kurali).

    K-100 (2026-08-27 fix): artik DELETED de MODIFIED gibi pending-rework
    kuyruguna girer (eskiden yalniz MODIFIED girerdi -- bkz. `_forget_
    deleted_sources`in K-100 notu) VE her ikisi de `document_nodes.json`a
    `scan.forgotten_hash` + sifirlanmis `parse` bloguyla geri yazilir."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-1"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "DELETED", "content_hash": "sha256:d1"}},
        {"identity": {"doc_id": "doc-2"}, "doc_type": None,
         "is_active": True, "scan": {"scan_status": "DELETED", "content_hash": "sha256:d2"}},
        {"identity": {"doc_id": "doc-3"}, "doc_type": "CATALOGUE",
         "is_active": True, "scan": {"scan_status": "MODIFIED", "content_hash": "sha256:m3"}},
        {"identity": {"doc_id": "doc-moved"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "MOVED", "content_hash": "sha256:mv1"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)
    monkeypatch.setattr(run_parse_pipeline, "PARSED_OUTPUT_DIR", str(tmp_path / "parsed"), raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant:6333")

    fake_result = MagicMock(parse_dirs_removed=[], chunk_entry_removed=False,
                            spec_evidence={"rows_deleted": 0})
    forget_mock = MagicMock(return_value=fake_result)
    monkeypatch.setattr("medrag.pipeline.forget_deleted_source.forget_deleted_source", forget_mock)

    fake_vec_cfg = MagicMock()
    fake_vec_cfg.qdrant.collection_name = "col"
    fake_vec_cfg.qdrant.distance = "cosine"
    fake_vec_cfg.qdrant.on_dim_mismatch = "error"
    fake_vec_cfg.qdrant.upsert_batch_size = 128
    monkeypatch.setattr("medrag.pipeline.vectorize.config.load_config",
                        lambda *a, **k: fake_vec_cfg)
    fake_store = MagicMock()
    monkeypatch.setattr("medrag.pipeline.vectorize.store.QdrantVectorStore.from_env",
                        classmethod(lambda cls, **kw: fake_store))
    monkeypatch.setattr("medrag.pipeline.facts.discover.DB_PATH", ":memory:")
    pending_path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(pending_path))

    result = run_nightly._forget_deleted_sources(dry_run=False)

    assert len(result) == 3
    assert forget_mock.call_count == 3
    called_doc_ids = {kw["doc_id"] for _, kw in forget_mock.call_args_list}
    assert called_doc_ids == {"doc-1", "doc-2", "doc-3"}, "MOVED (doc-moved) unutulmamali"
    # K-100: DELETED (doc-1/doc-2) VE MODIFIED (doc-3) -- ucu de kuyruga girer,
    # sonsuza kadar gitmis DELETED'ler `_settle_pending_rework` ile kendiliginden
    # 'orphaned' olup duser (bkz. o fonksiyonun docstring'i).
    pending_events = [json.loads(line) for line in pending_path.read_text(encoding="utf-8").splitlines()]
    assert {e["doc_id"] for e in pending_events} == {"doc-1", "doc-2", "doc-3"}
    assert all(e["status"] == "pending" for e in pending_events)

    written = json.loads(nodes_path.read_text(encoding="utf-8"))
    by_id = {rec["identity"]["doc_id"]: rec for rec in written["documents"]}
    for doc_id, content_hash in (("doc-1", "sha256:d1"), ("doc-2", "sha256:d2"), ("doc-3", "sha256:m3")):
        assert by_id[doc_id]["scan"]["forgotten_hash"] == content_hash
        assert by_id[doc_id]["parse"]["status"] == "PENDING"
        assert by_id[doc_id]["parse"]["parsed_from_hash"] is None
    # MOVED (doc-moved) hic unutulmadi -- forgotten_hash yazilmamali.
    assert "forgotten_hash" not in by_id["doc-moved"]["scan"]


def test_forget_deleted_sources_real_run_requires_qdrant_url(monkeypatch, tmp_path):
    """QDRANT_URL tanimsizsa gercek yol acikca patlamali -- silinen kaynagin
    Qdrant temizligi sessizce ATLANMAMALI."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-1"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "DELETED"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)

    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        run_nightly._forget_deleted_sources(dry_run=False)


def test_stage_parse_calls_forget_deleted_sources_before_real_parse(monkeypatch):
    """K-96: scan'DAN SONRA, parse'DAN ONCE -- `stage_parse` gercek parse'i
    cagirmadan ONCE `_forget_deleted_sources`u cagirmali."""
    from medrag.pipeline.cli import run_parse_pipeline

    calls = []
    monkeypatch.setattr(run_nightly, "_forget_deleted_sources",
                        lambda **kw: calls.append(("forget", kw)))
    monkeypatch.setattr(run_parse_pipeline, "main",
                        lambda argv: calls.append(("parse", argv)))

    run_nightly.stage_parse()

    assert calls[0] == ("forget", {"dry_run": False})
    assert calls[1] == ("parse", [])


def test_dry_run_parse_calls_forget_deleted_sources_in_dry_run_mode(monkeypatch):
    from medrag.pipeline.cli import run_parse_pipeline

    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOC_TYPES", set(), raising=False)
    monkeypatch.setattr(run_parse_pipeline, "compute_pending",
                        lambda: ([], {"total": 0}))
    calls = []
    monkeypatch.setattr(run_nightly, "_forget_deleted_sources",
                        lambda **kw: calls.append(kw))

    run_nightly._dry_run_parse()

    assert calls == [{"dry_run": True}]


# --- K-98 (2026-08-26 fix, takip 1): pending-rework kuyrugu ----------------


def test_pending_rework_doc_ids_empty_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(tmp_path / "yok.jsonl"))
    assert run_nightly._pending_rework_doc_ids() == []


def test_pending_rework_doc_ids_reduces_to_latest_status_per_doc(monkeypatch, tmp_path):
    """Append-only, `chunk_owner_progress.jsonl` ile AYNI disiplin: HER
    doc_id icin SON kayit kazanir -- 'pending' sonra 'done' gorulurse artik
    pending SAYILMAZ, ikinci kez 'pending' yazilirsa (yeni bir forget) TEKRAR
    pending sayilir."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text(
        '{"doc_id": "doc-a", "status": "pending"}\n'
        '{"doc_id": "doc-b", "status": "pending"}\n'
        '{"doc_id": "doc-a", "status": "done"}\n'
        '{"doc_id": "doc-c", "status": "pending"}\n'
        '{"doc_id": "doc-c", "status": "done"}\n'
        '{"doc_id": "doc-c", "status": "pending"}\n',
        encoding="utf-8",
    )

    assert run_nightly._pending_rework_doc_ids() == ["doc-b", "doc-c"]


def test_append_pending_rework_events_is_append_only(monkeypatch, tmp_path):
    """Dosya HICBIR ZAMAN uzerine yazilmaz/kisaltilmaz -- yalniz eklenir,
    ONCEKI satirlar korunur."""
    path = tmp_path / "sub" / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))

    run_nightly._append_pending_rework_events([{"doc_id": "doc-a", "status": "pending"}])
    run_nightly._append_pending_rework_events([{"doc_id": "doc-b", "status": "pending"}])

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line) for line in lines] == [
        {"doc_id": "doc-a", "status": "pending"},
        {"doc_id": "doc-b", "status": "pending"},
    ]


def test_append_pending_rework_events_noop_for_empty_list(monkeypatch, tmp_path):
    """Bos olay listesi dosyayi HIC YARATMAZ -- gereksiz bos kuyruk dosyasi
    olusmaz."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))

    run_nightly._append_pending_rework_events([])

    assert not path.is_file()


def test_settle_pending_rework_drops_doc_id_when_all_its_products_succeed(monkeypatch, tmp_path):
    """K-98 kabul: bir pending doc_id'nin TUM urunleri bu kosuda hatasiz
    yuklendiyse (`load_report['reports']`de `error` YOK) doc_id kuyruktan
    duser (append-only 'done' olayi)."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text('{"doc_id": "doc-a", "status": "pending"}\n', encoding="utf-8")

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.return_value = ["DE1", "DE2"]
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    load_report = {"reports": [
        {"model_code": "DE1", "written_present": 1},
        {"model_code": "DE2", "written_present": 2},
    ]}
    run_nightly._settle_pending_rework("fake-con", load_report)

    assert run_nightly._pending_rework_doc_ids() == []


def test_settle_pending_rework_keeps_doc_id_on_partial_failure(monkeypatch, tmp_path):
    """K-98 kabul: pending doc_id'nin urunlerinden BIRI bu kosuda hata
    verdiyse (`error` alani) ya da hic denenmediyse doc_id kuyrukta KALIR --
    kismi basari YETERLI DEGIL."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text('{"doc_id": "doc-a", "status": "pending"}\n', encoding="utf-8")

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.return_value = ["DE1", "DE2"]
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    load_report = {"reports": [
        {"model_code": "DE1", "written_present": 1},
        {"model_code": "DE2", "error": "IntegrityError: ..."},
    ]}
    run_nightly._settle_pending_rework("fake-con", load_report)

    assert run_nightly._pending_rework_doc_ids() == ["doc-a"]


def test_settle_pending_rework_keeps_doc_id_when_product_not_attempted(monkeypatch, tmp_path):
    """Pending doc_id'nin urunlerinden biri bu kosunun `reports[]`inde HIC
    GORUNMEDIYSE (bu doc_id'nin urunleri bu kosuda islenmedi) doc_id
    kuyrukta KALIR."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text('{"doc_id": "doc-a", "status": "pending"}\n', encoding="utf-8")

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.return_value = ["DE1", "DE2"]
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    load_report = {"reports": [{"model_code": "DE1", "written_present": 1}]}
    run_nightly._settle_pending_rework("fake-con", load_report)

    assert run_nightly._pending_rework_doc_ids() == ["doc-a"]


def test_settle_pending_rework_noop_when_nothing_pending(monkeypatch, tmp_path):
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    fake_run_full = MagicMock()
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    run_nightly._settle_pending_rework("fake-con", {"reports": []})

    fake_run_full.resolve_codes.assert_not_called()
    assert not path.is_file()


def test_settle_pending_rework_drops_ownerless_doc_as_orphaned(monkeypatch, tmp_path):
    """Sahipsiz doc (resolve_codes BOS doner): kayit TATMIN EDILEMEZ -- "bu
    doc'in urunlerini yeniden isle" kurali urun yoksa hicbir kosuda yerine
    getirilemez, `continue` ile atlanirsa kuyrukta sonsuza kadar kalir (2026-
    08-26 gece kosusu incelemesi sizinti). 'orphaned' olayiyle duser ('done'
    DEGIL -- hicbir urun islenmedi), URUNLU pending doc ayni kosuda kalmaya
    devam eder."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text(
        '{"doc_id": "doc-sahipsiz", "status": "pending"}\n'
        '{"doc_id": "doc-a", "status": "pending"}\n',
        encoding="utf-8",
    )

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.side_effect = (
        lambda con, doc_id=None: [] if doc_id == "doc-sahipsiz" else ["DE1"]
    )
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    # Rapor doc-a'nin urununu ICERMEZ (DE9 baska bir doc'un urunu) -- sahipsiz
    # duserken urunlu pending doc'yun KALDIGINI ayirt edebilmek icin.
    load_report = {"reports": [{"model_code": "DE9", "written_present": 1}]}
    run_nightly._settle_pending_rework("fake-con", load_report)

    assert run_nightly._pending_rework_doc_ids() == ["doc-a"]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
    orphaned = [rec for rec in lines if rec["doc_id"] == "doc-sahipsiz"][-1]
    assert orphaned["status"] == "orphaned"


def test_settle_pending_rework_settles_ownerless_even_with_empty_report(monkeypatch, tmp_path):
    """Even with a completely empty `load_report` (e.g. `stage_load` had an
    empty `codes` and so never called `load_to_db.main()`), ownerless records
    are SETTLED; product-bearing pending docs STAY because they were not
    tried."""
    path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(path))
    path.write_text(
        '{"doc_id": "doc-sahipsiz", "status": "pending"}\n'
        '{"doc_id": "doc-a", "status": "pending"}\n',
        encoding="utf-8",
    )

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.side_effect = (
        lambda con, doc_id=None: [] if doc_id == "doc-sahipsiz" else ["DE1"]
    )
    monkeypatch.setattr("medrag.pipeline.facts.run_full", fake_run_full, raising=False)

    run_nightly._settle_pending_rework("fake-con", {"reports": []})

    assert run_nightly._pending_rework_doc_ids() == ["doc-a"]


def test_pending_rework_full_cycle_survives_scan_status_flip_to_unchanged(monkeypatch, tmp_path):
    """Item 1'in TAM kabul testi: bir MODIFIED dokuman icin `forget`
    uygulanir, sonra facts/load kosmadan surec KESILIR (`_settle_pending_
    rework` HIC cagrilmaz). Ikinci gece: `_incremental_doc_ids()` bu
    dokumani ARTIK gormez (scan UNCHANGED dedi -- `classify_documents.py::
    scan_status_for`in gercek davranisi, hash bir onceki taramada
    guncellendi) AMA urun YINE de secilir, cunku pending-rework kuyrugunda
    hala 'pending' duruyor."""
    from medrag.pipeline.cli import run_parse_pipeline

    nodes_path = tmp_path / "document_nodes.json"
    pending_path = tmp_path / "pending_rework.jsonl"
    monkeypatch.setenv("FACTS_PENDING_REWORK_PATH", str(pending_path))

    # --- Gece 1: dokuman MODIFIED, forget calisir, facts/load HIC kosmaz. ---
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-mod"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "MODIFIED"}},
    )
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)
    monkeypatch.setattr(run_parse_pipeline, "PARSED_OUTPUT_DIR", str(tmp_path / "parsed"), raising=False)
    monkeypatch.setenv("QDRANT_URL", "http://fake-qdrant:6333")

    fake_result = MagicMock(parse_dirs_removed=[], chunk_entry_removed=False,
                            spec_evidence={"rows_deleted": 0})
    monkeypatch.setattr("medrag.pipeline.forget_deleted_source.forget_deleted_source",
                        MagicMock(return_value=fake_result))
    fake_vec_cfg = MagicMock()
    fake_vec_cfg.qdrant.collection_name = "col"
    fake_vec_cfg.qdrant.distance = "cosine"
    fake_vec_cfg.qdrant.on_dim_mismatch = "error"
    fake_vec_cfg.qdrant.upsert_batch_size = 128
    monkeypatch.setattr("medrag.pipeline.vectorize.config.load_config", lambda *a, **k: fake_vec_cfg)
    monkeypatch.setattr("medrag.pipeline.vectorize.store.QdrantVectorStore.from_env",
                        classmethod(lambda cls, **kw: MagicMock()))
    monkeypatch.setattr("medrag.pipeline.facts.discover.DB_PATH", ":memory:")

    run_nightly._forget_deleted_sources(dry_run=False)
    # KOSU BURADA PATLADI -- facts/load hic calismadi, _settle_pending_rework
    # cagrilmadi.

    assert run_nightly._pending_rework_doc_ids() == ["doc-mod"]

    # --- Gece 2: scan artik UNCHANGED gormus olsun (hash bir onceki tarama- --
    # da kaydedildi) -- eski (yanlis) varsayimin test ettigi TAM senaryo.
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-mod"}, "doc_type": "DATASHEET",
         "is_active": True, "scan": {"scan_status": "UNCHANGED"}},
    )
    assert run_nightly._incremental_doc_ids() == [], "scan artik UNCHANGED gormeli"

    # DIKKAT: `_fake_run_full` KULLANILMAZ -- o yardimci `_pending_rework_
    # doc_ids`i VARSAYILAN olarak BOS doner (bu testin konusu OLMAYAN diger
    # testleri kolaylastirmak icin), tam da bu testin dogrulamak istedigi
    # GERCEK kuyruk-okuma davranisini MASKELERDI.
    import medrag.pipeline.facts as facts_pkg

    fake_run_full = MagicMock()
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: {"doc-mod": ["DE7"]}[doc_id]
    monkeypatch.setattr(facts_pkg, "run_full", fake_run_full, raising=False)
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    monkeypatch.setattr("medrag.pipeline.nightly_report.current_pipeline_versions",
                        lambda: {"facts": "1.0.0"})
    monkeypatch.setattr("medrag.pipeline.facts.staleness_audit.stale_product_codes",
                        lambda *a, **k: [])

    codes, _process_all = run_nightly._resolve_incremental_codes("fake-con", doc_id=None)

    assert codes == ["DE7"], "MODIFIED gorunmese bile pending-rework urunu kacirmamali"


# --- K-97 (2026-08-26 fix): NEW+MODIFIED dokuman birlesimi stage_facts'e --
# baglaniyor (eski N-20 bayatlik-sezgisi kapisinin YERINE) ------------------


def _fake_run_full(monkeypatch, *, resolve_codes_return=None):
    """`_dry_run_facts` testindeki AYNI desen: `medrag.pipeline.facts.run_full`
    yerine sahte bir modul koyar -- gercek discover.py/specs.db/LLM'e hic
    dokunulmaz.

    K-98/takip 2+3 (2026-08-26 fix): `_resolve_incremental_codes` artik
    `_pending_rework_doc_ids()` VE `staleness_audit.stale_product_codes`i de
    (ikincil kaynaklar) cagiriyor -- ikisi de burada VARSAYILAN olarak BOS
    doner (`con` bu testlerde "fake-con" bir STRING, gercek `.execute()`
    cagrisi patlar) ki bu yardimci ile yazilan HER test kendi konusu
    OLMAYAN bu iki kaynagi ayrica mock'lamak ZORUNDA kalmasin. Bu iki
    kaynagin KENDI davranisini test eden testler kendi monkeypatch'lerini
    UZERINE yazar (bkz. `test_stage_facts_unions_pending_rework_doc_ids`/
    `test_stage_facts_unions_stale_product_codes`)."""
    import medrag.pipeline.facts as facts_pkg

    fake = MagicMock()
    fake.ENV_PATH = "irrelevant"
    fake._connect_ro.return_value = "fake-con"
    fake._resolve_llm_env.return_value = ("ollama", "m", "http://x", None)
    fake.build_system_message.return_value = "sys"
    fake.load_indexes.return_value = ({}, {}, frozenset(), {}, {})
    fake.owner_count_by_doc.return_value = {}
    if resolve_codes_return is not None:
        fake.resolve_codes.return_value = resolve_codes_return
    fake.run_products.return_value = {"n_processed": 0}
    monkeypatch.setattr(facts_pkg, "run_full", fake, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(run_nightly, "_pending_rework_doc_ids", list)
    monkeypatch.setattr("medrag.pipeline.facts.staleness_audit.stale_product_codes",
                        lambda *a, **k: [])
    return fake


def test_stage_facts_unions_resolve_codes_over_new_and_modified_docs(monkeypatch):
    """K-97 (2026-08-26 fix): `--doc` yok, `FACTS_PROCESS_ALL` yok -- urun
    secimi bir bayatlik SEZGISI DEGIL, `_incremental_doc_ids()`in dondurdugu
    HER (NEW/MODIFIED) doc_id icin `resolve_codes(con, doc_id=...)`
    cagirilip sonuclarin BIRLESIMI (DE2 iki dokumanda da var, TEKRARSIZ)."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    by_doc = {"doc-a": ["DE1", "DE2"], "doc-b": ["DE2", "DE3"]}
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: by_doc[doc_id]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", lambda: ["doc-a", "doc-b"])

    run_nightly.stage_facts()

    assert fake_run_full.resolve_codes.call_count == 2
    fake_run_full.run_products.assert_called_once()
    assert fake_run_full.run_products.call_args.args[0] == ["DE1", "DE2", "DE3"]


def test_stage_facts_processes_zero_products_when_nothing_changed(monkeypatch):
    """Kabul kriteri (K-97): hicbir dokuman degismedigi (NEW/MODIFIED yok)
    bir gecede facts SIFIR urun isler -- `resolve_codes` HIC cagrilmaz."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", list)

    run_nightly.stage_facts()

    fake_run_full.resolve_codes.assert_not_called()
    assert fake_run_full.run_products.call_args.args[0] == []


def test_stage_facts_unions_pending_rework_doc_ids(monkeypatch):
    """K-98 (2026-08-26 fix, takip 1) kabul testi: `_incremental_doc_ids()`
    (NEW/MODIFIED) BOS olsa bile pending-rework kuyrugundaki doc_id'lerin
    urunleri secime GIRER -- bir onceki gece `forget` uygulanip facts/load
    tamamlanamayan bir dokuman, ikinci gece scan_status'u ne olursa olsun
    (ORNEGIN artik UNCHANGED gorunse bile) BIR DAHA KACIRILMAZ."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: {
        "doc-pending": ["DE7"],
    }[doc_id]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", list)
    monkeypatch.setattr(run_nightly, "_pending_rework_doc_ids", lambda: ["doc-pending"])

    run_nightly.stage_facts()

    fake_run_full.resolve_codes.assert_called_once_with("fake-con", doc_id="doc-pending")
    assert fake_run_full.run_products.call_args.args[0] == ["DE7"]


def test_stage_facts_unions_stale_product_codes(monkeypatch):
    """Takip 2+3 kabul testi (K-96'nin denetim ucunun ikincil AG olarak GERI
    EKLENMESI): `_incremental_doc_ids()` VE pending-rework kuyrugu BOS olsa
    bile `staleness_audit.stale_product_codes`in dondurdugu urunler secime
    GIRER -- ne bir dokuman degisti ne kuyrukta bir sey var, ama `stale_
    extractor_version`/`dangling_evidence`/`empty_evidence` mutabakat agi
    bir urunu isaretledi."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", list)
    monkeypatch.setattr("medrag.pipeline.facts.staleness_audit.stale_product_codes",
                        lambda *a, **k: ["DE99"])

    run_nightly.stage_facts()

    fake_run_full.resolve_codes.assert_not_called()
    assert fake_run_full.run_products.call_args.args[0] == ["DE99"]


def test_stage_facts_doc_id_bypasses_the_gate(monkeypatch):
    """`--doc` (elle mudahale) K-96'nin kacis kapilarindan biri -- gate hic
    cagrilmamali, eski `resolve_codes(con, doc_id=...)` yolu kullanilmali."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch, resolve_codes_return=["DE9"])
    gate_mock = MagicMock()
    monkeypatch.setattr("medrag.pipeline.facts.staleness_audit.stale_product_codes", gate_mock)

    run_nightly.stage_facts(doc_id="DE9")

    gate_mock.assert_not_called()
    fake_run_full.resolve_codes.assert_called_once_with("fake-con", doc_id="DE9")
    assert fake_run_full.run_products.call_args.args[0] == ["DE9"]


def test_stage_facts_process_all_env_var_bypasses_the_gate(monkeypatch):
    """Escape hatch: `FACTS_PROCESS_ALL=1` fully bypasses the gate and reverts
    to the old "all products" behavior -- no new CLI flag was added
    (repo convention: settings live in env)."""
    monkeypatch.setenv("FACTS_PROCESS_ALL", "1")
    fake_run_full = _fake_run_full(monkeypatch, resolve_codes_return=["DE1", "DE2", "DE3"])
    gate_mock = MagicMock()
    monkeypatch.setattr("medrag.pipeline.facts.staleness_audit.stale_product_codes", gate_mock)

    run_nightly.stage_facts()

    gate_mock.assert_not_called()
    fake_run_full.resolve_codes.assert_called_once_with("fake-con", doc_id=None)
    assert fake_run_full.run_products.call_args.args[0] == ["DE1", "DE2", "DE3"]


def test_stage_facts_does_not_skip_existing_by_default(monkeypatch):
    """VARSAYILAN KAPALI: `FACTS_SKIP_EXISTING` verilmediginde
    `run_products(..., skip_existing=False)` -- yani NEW/MODIFIED bir
    dokumanin urunu, onceki geceden kalma TAM bir `results/<CODE>.json`i
    olsa bile YENIDEN uretilir."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    monkeypatch.delenv("FACTS_SKIP_EXISTING", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: ["DE1"]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", lambda: ["doc-a"])

    run_nightly.stage_facts()

    assert fake_run_full.run_products.call_args.kwargs["skip_existing"] is False


def test_stage_facts_skip_existing_env_var_is_forwarded(monkeypatch):
    """Yarida kalan bir kosuyu ELLE surdurme kapisi: `FACTS_SKIP_EXISTING=1`
    `run_products`a gecirilir (URUN SECIMINI degistirmez -- NEW+MODIFIED
    dokuman birlesimi yine kosar, yalniz ic dongu tam JSON'i olan urunu
    atlar)."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    monkeypatch.setenv("FACTS_SKIP_EXISTING", "1")
    fake_run_full = _fake_run_full(monkeypatch)
    by_doc = {"doc-a": ["DE1", "DE2"]}
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: by_doc[doc_id]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", lambda: ["doc-a"])

    run_nightly.stage_facts()

    assert fake_run_full.run_products.call_args.kwargs["skip_existing"] is True
    assert fake_run_full.run_products.call_args.args[0] == ["DE1", "DE2"]


# --- N-22: ownership/load asamalari zincire baglaniyor --------------------


def test_stage_ownership_calls_build_all_with_empty_argv(monkeypatch):
    """`stage_ownership`, `build_all.main(argv=[])`i PROGRAMATIK cagirir --
    subprocess ACMAZ -- ve donusunu (meta+rows) aynen geri verir (gecelik
    raporun ownership sayilarinin TEK kaynagi)."""
    import medrag.pipeline.facts.catalog_chunk_ownership as ownership_pkg

    fake_build_all = MagicMock()
    fake_build_all.main.return_value = {"meta": {"n_rows": 3}, "rows": []}
    monkeypatch.setattr(ownership_pkg, "build_all", fake_build_all, raising=False)

    result = run_nightly.stage_ownership()

    fake_build_all.main.assert_called_once_with(argv=[])
    assert result == {"meta": {"n_rows": 3}, "rows": []}


def test_stage_ownership_doc_noop_but_still_runs(monkeypatch, capsys):
    """`--doc` bu asamada NO-OP -- uyari basilir ama build_all.main() YINE
    cagrilir (tum korpus taranir, sessizce atlanmaz)."""
    import medrag.pipeline.facts.catalog_chunk_ownership as ownership_pkg

    fake_build_all = MagicMock()
    fake_build_all.main.return_value = None
    monkeypatch.setattr(ownership_pkg, "build_all", fake_build_all, raising=False)

    run_nightly.stage_ownership(doc_id="DE1")

    fake_build_all.main.assert_called_once_with(argv=[])
    assert "no-op" in capsys.readouterr().out


def test_dry_run_ownership_never_touches_chat_ollama(monkeypatch):
    """`_dry_run_ownership`, `split_ownership_work` disinda hicbir seye
    dokunmamali -- `_chat_ollama`in KENDISI patlayan bir surumle degistirilir,
    cagrilirsa test kendisi patlar (mock'un `assert_not_called()`inden daha
    sert bir garanti: `_tag_chunk_llm` uzerinden DOLAYLI bir cagriyi da
    yakalar)."""
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    def _patlayan(*a, **k):
        raise AssertionError("dry-run LLM'e dokunmamali")

    monkeypatch.setattr(build_all, "_chat_ollama", _patlayan)
    monkeypatch.setattr(build_all, "_connect_ro", lambda: MagicMock())
    fake_config = MagicMock()
    fake_config.scope.exclude_doc_types = []
    monkeypatch.setattr(build_all, "load_config", lambda: fake_config)
    monkeypatch.setattr(build_all, "_load_documents", lambda con, exclude: [])
    monkeypatch.setattr(build_all, "_load_chunks_from_all_chunks_json", dict)
    monkeypatch.setattr(build_all, "_load_chunk_bridge", dict)
    monkeypatch.setattr(build_all, "_load_product_info", lambda con, codes: {})

    run_nightly._dry_run_ownership()  # patlamazsa test gecer


def _fake_load_to_db(monkeypatch, *, main_return=None):
    """`stage_load`in `load_to_db` modulu yerine sahte bir modul koyar --
    gercek specs.db'ye HIC yazilmaz."""
    import medrag.pipeline.facts as facts_pkg

    fake = MagicMock()
    fake.main.return_value = main_return if main_return is not None else {"n_products": 0}
    monkeypatch.setattr(facts_pkg, "load_to_db", fake, raising=False)
    return fake


def test_stage_load_doc_id_delegates_to_load_to_db_doc_id(monkeypatch):
    """`--doc` verilmisse `stage_load`, `_resolve_incremental_codes`e HIC
    UGRAMADAN direkt `load_to_db.main(argv=["--doc-id", ...])`ye devreder --
    `load_product_incremental` (O-12) yolu. K-98: `run_full`/`_settle_
    pending_rework` mock'lanir -- bu testin konusu DELEGASYON, gercek
    specs.db'ye/kuyruk dosyasina DOKUNULMAZ."""
    _fake_run_full(monkeypatch)
    fake_load_to_db = _fake_load_to_db(monkeypatch, main_return={"n_present": 1})
    settle_calls = []
    monkeypatch.setattr(run_nightly, "_settle_pending_rework",
                        lambda con, report: settle_calls.append((con, report)))

    result = run_nightly.stage_load(doc_id="DE9")

    fake_load_to_db.main.assert_called_once_with(argv=["--doc-id", "DE9"])
    assert result == {"n_present": 1}
    assert settle_calls == [("fake-con", {"n_present": 1})]


def test_stage_load_uses_same_codes_as_facts_and_forces(monkeypatch):
    """N-22 kritik kisit: `--force` OLMADAN `load_to_db.py` zaten-yuklenmis
    urunleri "already loaded" diye ATLAR -- `stage_load` HER ZAMAN `--force`
    gecirmeli, VE facts'in kullandigi AYNI NEW+MODIFIED dokuman birlesimini
    (`_resolve_incremental_codes`, K-97) kullanmali."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    by_doc = {"doc-a": ["DE1", "DE2"]}
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: by_doc[doc_id]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", lambda: ["doc-a"])
    fake_load_to_db = _fake_load_to_db(monkeypatch, main_return={"n_present": 4})

    result = run_nightly.stage_load()

    fake_load_to_db.main.assert_called_once_with(argv=["--models", "DE1,DE2", "--force"])
    assert result == {"n_present": 4}


def test_stage_load_skips_load_to_db_when_no_stale_codes(monkeypatch):
    """Hicbir dokuman NEW/MODIFIED degilse `load_to_db.main()` HIC cagrilmaz
    -- bos `--models ""` anlamsiz bir hataya yol acardi (K-97 kabul kriteri:
    hicbir dokuman degismedigi gece SIFIR urun islenir)."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    _fake_run_full(monkeypatch)
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", list)
    fake_load_to_db = _fake_load_to_db(monkeypatch)

    result = run_nightly.stage_load()

    fake_load_to_db.main.assert_not_called()
    assert result == {"n_products": 0}


def test_stage_load_settles_pending_rework_even_when_no_codes(monkeypatch):
    """codes BOS oldugu dalda bile `_settle_pending_rework` cagrilir (2026-08-
    26 gece kosusu incelemesi): sahipsiz pending kayitlar (resolve_codes BOS
    donen doc'ler) hicbir kosunun `codes` kumesine giremedigi icin ancak bu
    dalda duser -- settle edilmezlerse her gece sifir isle yeniden denenip
    kuyruk sonsuza kadar buyur. Bos `reports` verilir: urunlu pending
    doc'ler denenmedikleri icin kuyrukta KALIR."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    _fake_run_full(monkeypatch)
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", list)
    fake_load_to_db = _fake_load_to_db(monkeypatch)
    settle_calls = []
    monkeypatch.setattr(run_nightly, "_settle_pending_rework",
                        lambda con, report: settle_calls.append((con, report)))

    result = run_nightly.stage_load()

    fake_load_to_db.main.assert_not_called()
    assert result == {"n_products": 0}
    assert settle_calls == [("fake-con", {"reports": []})]


def test_dry_run_load_never_calls_load_to_db_main(monkeypatch):
    """`_dry_run_load`, `_resolve_incremental_codes` disinda hicbir seye
    dokunmaz -- `load_to_db.main` (gercek yazmanin tetikleyicisi) HIC
    cagrilmaz."""
    monkeypatch.delenv("FACTS_PROCESS_ALL", raising=False)
    fake_run_full = _fake_run_full(monkeypatch)
    fake_run_full.resolve_codes.side_effect = lambda con, doc_id: ["DE1"]
    monkeypatch.setattr(run_nightly, "_incremental_doc_ids", lambda: ["doc-a"])
    fake_load_to_db = _fake_load_to_db(monkeypatch)

    run_nightly._dry_run_load()

    fake_load_to_db.main.assert_not_called()


def test_dry_run_load_doc_id_does_not_call_resolve_incremental_codes(monkeypatch):
    calls = []
    monkeypatch.setattr(run_nightly, "_resolve_incremental_codes",
                        lambda *a, **k: calls.append(1))

    run_nightly._dry_run_load(doc_id="DE1")

    assert calls == []


def test_report_computes_real_ownership_counts_when_stage_returns_rows():
    """N-22: `stage_ownership`in gercek donusu (rows) verildiginde
    `_build_nightly_report`, deterministik/model-yonlendirilen/cozulemeyen
    chunk sayilarini GERCEKTEN hesaplamali (SAYI UYDURMADAN)."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    collect = {"stages": {"ownership": {
        "meta": {"n_llm_chunks": 2},
        "rows": [
            {"source": "deterministic", "doc_id": "d1", "chunk_id": "c0", "accepted": True},
            {"source": "deterministic", "doc_id": "d1", "chunk_id": "c1", "accepted": True},
            {"source": "llm", "doc_id": "d2", "chunk_id": "c0", "accepted": True},
            {"source": "llm", "doc_id": "d2", "chunk_id": "c1", "accepted": False},
        ],
    }}}
    now = datetime.now(UTC)
    report = run_nightly._build_nightly_report(
        collect=collect, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert report.ownership.deterministic_count == 2
    assert report.ownership.model_routed_count == 2
    assert report.ownership.unresolved_count == 1


def test_report_leaves_ownership_none_when_stage_did_not_run():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    report = run_nightly._build_nightly_report(
        collect={"stages": {}}, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert report.ownership.deterministic_count is None
    assert report.ownership.model_routed_count is None
    assert report.ownership.unresolved_count is None


def test_report_leaves_ownership_none_when_interrupted():
    """Ctrl+C ile kesilmis bir ownership kosusu ("interrupted": True) YARIM
    veri tasir -- SAYI UYDURULMAZ, None kalir."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    collect = {"stages": {"ownership": {"interrupted": True, "n_llm_jobs": 10}}}
    report = run_nightly._build_nightly_report(
        collect=collect, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert report.ownership.deterministic_count is None


def test_report_turns_parse_failed_docs_into_failure_entries():
    """N-10 fix (2026-08-26): `stage_parse`in `failed_docs`i (asama kendisi
    exception FIRLATMADI, tek tek dokuman basarisiz oldu) `NightlyReport.
    failures`e girmeli -- `outcome()` bunu gorunce artik `full_success`
    DIYEMEZ."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    collect = {"stages": {"parse": {
        "processed_count": 5, "duration_seconds": 1.0,
        "failed_docs": [{"doc_id": "doc-bad", "file_name": "bad.pdf", "error": "OCR timeout"}],
    }}}
    report = run_nightly._build_nightly_report(
        collect=collect, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert len(report.failures) == 1
    assert report.failures[0].stage == "parse"
    assert report.failures[0].file == "bad.pdf"
    assert "OCR timeout" in report.failures[0].error_text
    assert report.outcome() != "full_success"


def test_report_turns_load_product_errors_into_failure_entries():
    """N-10 fix (2026-08-26): `stage_load`in `reports[]`indeki urun-basina
    `error` alani (asama kendisi exception FIRLATMADI, `load_to_db.main()`
    o urunu rollback edip devam etti) `NightlyReport.failures`e girmeli --
    2026-08-26 kosusunda 34 urun boyle yazilamadi ama rapor `full_success`
    diyordu."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    collect = {"stages": {"load": {"n_present": 3, "reports": [
        {"model_code": "DE1", "written_present": 3},
        {"model_code": "DE2", "error": "IntegrityError: UNIQUE constraint failed"},
    ]}}}
    report = run_nightly._build_nightly_report(
        collect=collect, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert len(report.failures) == 1
    assert report.failures[0].stage == "load"
    assert report.failures[0].file == "DE2"
    assert "UNIQUE constraint" in report.failures[0].error_text
    assert report.outcome() != "full_success"


def test_report_computes_features_added_from_load_stage():
    """`stage_load`in dondurdugu `n_present`, `product_features.
    features_added`e GERCEK deger olarak gecer -- diger uc alan (evidence_*)
    HALA None kalmali (load_to_db raporu onlari TASIMIYOR)."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    now = datetime.now(UTC)
    collect = {"stages": {"load": {"n_present": 7, "n_absent": 1}}}
    report = run_nightly._build_nightly_report(
        collect=collect, started_at=now, finished_at=now,
        backup_manifest=SimpleNamespace(root="/tmp/fake"),
    )

    assert report.product_features.features_added == 7
    assert report.product_features.evidence_added is None
    assert report.product_features.evidence_removed is None
    assert report.product_features.features_removed_for_no_evidence is None


# --- N-21: gecelik rapor -- run_nightly.py nightly_report.py'yi cagiriyor -


def test_stage_scan_returns_the_scan_output_dict(monkeypatch):
    """N-21: `module.main()`in donusu artik ATILMIYOR -- `stage_scan`
    onu geri veriyor (gecelik raporun Kaynak farki/Atlananlar'inin TEK
    kaynagi)."""
    fake_output = {"summary": {"scan_status_counts": {"NEW": 1}}, "documents": []}
    monkeypatch.setattr(run_nightly, "_load_classify_documents",
                        lambda: MagicMock(main=lambda: fake_output))

    assert run_nightly.stage_scan() is fake_output


def test_stage_parse_returns_real_processed_count_and_measured_duration(monkeypatch, tmp_path):
    from medrag.pipeline.cli import run_parse_pipeline

    monkeypatch.setattr(run_nightly, "_forget_deleted_sources", lambda **kw: [])
    monkeypatch.setattr(run_parse_pipeline, "main", lambda argv: 7)
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH",
                        str(tmp_path / "yok.json"), raising=False)

    result = run_nightly.stage_parse()

    assert result["processed_count"] == 7
    assert result["duration_seconds"] >= 0.0
    assert result["failed_docs"] == []


def test_stage_parse_surfaces_failed_docs(monkeypatch, tmp_path):
    """N-10 fix (2026-08-26): `document_nodes.json`daki `parse.status ==
    'FAILED'` kayitlar `stage_parse`in donusune GIRMELI -- eskiden
    `run_parse_pipeline.main()`in donus degeri yalniz TOPLAM sayiydi, tek tek
    basarisizlik kayboluyordu. K-99: `last_parsed` kosu BASLANGICINDAN
    SONRA olmali (bkz. `test_stage_parse_excludes_stale_failed_docs_from_
    previous_runs` -- ONCEKI gecelerden kalma FAILED kayitlar surafcelenmemeli)."""
    from datetime import datetime, timedelta

    from medrag.pipeline.cli import run_parse_pipeline

    recent = (datetime.now().astimezone() + timedelta(seconds=30)).isoformat(timespec="seconds")
    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-ok", "file_name": "ok.pdf"},
         "parse": {"status": "DONE"}},
        {"identity": {"doc_id": "doc-bad", "file_name": "bad.pdf"},
         "parse": {"status": "FAILED", "error": "OCR timeout", "last_parsed": recent}},
    )
    monkeypatch.setattr(run_nightly, "_forget_deleted_sources", lambda **kw: [])
    monkeypatch.setattr(run_parse_pipeline, "main", lambda argv: 2)
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)

    result = run_nightly.stage_parse()

    assert result["failed_docs"] == [
        {"doc_id": "doc-bad", "file_name": "bad.pdf", "error": "OCR timeout"}
    ]


def test_stage_parse_excludes_stale_failed_docs_from_previous_runs(monkeypatch, tmp_path):
    """A "tombstone" FAILED record from a previous run (e.g. source file
    later deleted, the record stuck at FAILED forever) must NOT enter this
    run's `failed_docs` -- otherwise the report would NEVER be `full_success`
    even on a perfectly clean night (the inverse of N-10's fix)."""
    from datetime import datetime, timedelta

    from medrag.pipeline.cli import run_parse_pipeline

    stale = (datetime.now().astimezone() - timedelta(days=3)).isoformat(timespec="seconds")
    nodes_path = tmp_path / "document_nodes.json"
    _write_document_nodes(
        nodes_path,
        {"identity": {"doc_id": "doc-tombstone", "file_name": "gone.pdf"},
         "parse": {"status": "FAILED", "error": "file not found", "last_parsed": stale}},
    )
    monkeypatch.setattr(run_nightly, "_forget_deleted_sources", lambda **kw: [])
    monkeypatch.setattr(run_parse_pipeline, "main", lambda argv: 0)
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes_path), raising=False)

    result = run_nightly.stage_parse()

    assert result["failed_docs"] == []


def test_stage_chunk_computes_produced_and_removed_counts_from_diff(monkeypatch, tmp_path):
    """N-21: `run_chunk_pipeline.main()` (subprocess'liyor) bu sayilari
    HIC DONDURMUYOR -- `stage_chunk` `all_chunks.json`in KOSU ONCESI/SONRASI
    halini karsilastirarak gercek sayi uretir."""
    from medrag.pipeline.cli import run_chunk_pipeline

    all_chunks = tmp_path / "all_chunks.json"
    all_chunks.write_text(json.dumps({"documents": {
        "doc-A": {"nodes": [{}, {}]},
        "doc-B": {"nodes": [{}]},
    }}), encoding="utf-8")
    monkeypatch.setattr(run_chunk_pipeline, "resolve_all_chunks_path", lambda: all_chunks)

    def _sahte_main():
        # doc-B artik yok (silinmis kaynak), doc-A 3 chunk'a cikti.
        all_chunks.write_text(json.dumps({"documents": {
            "doc-A": {"nodes": [{}, {}, {}]},
        }}), encoding="utf-8")
        return 0

    monkeypatch.setattr(run_chunk_pipeline, "main", _sahte_main)

    result = run_nightly.stage_chunk()

    assert result["produced_count"] == 3
    assert result["per_document_counts"] == {"doc-A": 3}
    assert result["removed_count"] == 1, "doc-B'nin ONCEKI 1 chunk'i kayboldu"


def test_stage_vectorize_passes_through_calistir_stats(monkeypatch):
    """`stage_vectorize` artik duz int DEGIL, `VectorizeRunStats` dondurur --
    `calistir()`in kendisi degisti (N-21), burada yalniz gecirgenlik
    dogrulanir."""
    from medrag.pipeline.vectorize import cli as vectorize_cli

    stats = vectorize_cli.VectorizeRunStats(exit_code=0, points_written=42,
                                            embedding_model="bge-m3")
    monkeypatch.setattr(vectorize_cli, "calistir", lambda **kw: stats)

    assert run_nightly.stage_vectorize() is stats


def _stage_returns(monkeypatch, **by_stage):
    # med-rag: `products` asamasi kaldirildi; zincir `scan`de baslar. Testin
    # acikca vermedigi asamalar mock'lanir -- gercekleri kosarsa env/disk arar
    # ve tam-zincir testleri dusardi.
    for name, value in by_stage.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name],
                            MagicMock(return_value=value))


_FAKE_SCAN_OUTPUT = {
    "summary": {"scan_status_counts": {"NEW": 1, "MODIFIED": 1, "DELETED": 1,
                                        "MOVED": 1, "UNCHANGED": 2}},
    "documents": [
        {"identity": {"doc_id": "d-new"}, "location": {"rel_path": "new.pdf"},
         "scan": {"scan_status": "NEW"}},
        {"identity": {"doc_id": "d-mod"}, "location": {"rel_path": "mod.pdf"},
         "scan": {"scan_status": "MODIFIED"}},
        {"identity": {"doc_id": "d-del"}, "location": {"rel_path": "del.pdf"},
         "scan": {"scan_status": "DELETED"}},
        {"identity": {"doc_id": "d-mov"}, "location": {"rel_path": "mov.pdf"},
         "scan": {"scan_status": "MOVED"}},
        {"identity": {"doc_id": "d-u1"}, "location": {"rel_path": "u1.pdf"},
         "scan": {"scan_status": "UNCHANGED"}},
        {"identity": {"doc_id": "d-u2"}, "location": {"rel_path": "u2.pdf"},
         "scan": {"scan_status": "UNCHANGED"}},
    ],
}


def test_full_chain_success_writes_report_with_real_stage_numbers(monkeypatch, tmp_path):
    """N-21 kabul: tam zincir basariyla biterse rapor GERCEK sayilarla
    (mock'lanan `stage_*` donuslerinden) yazilir, JSON/markdown AYNI
    sayilari gosterir."""
    from medrag.pipeline.vectorize.cli import VectorizeRunStats

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(reports_dir))
    _mock_backup_ok(monkeypatch, tmp_path)

    _stage_returns(
        monkeypatch,
        scan=_FAKE_SCAN_OUTPUT,
        parse={"processed_count": 2, "duration_seconds": 12.5},
        chunk={"produced_count": 17, "per_document_counts": {"d-new": 17}, "removed_count": 3},
        ownership=None,  # N-22: Ctrl+C ile kesilmis/hic PLANA girmemis gibi -- ucu de "olculmedi" kalmali
        facts={"n_processed": 5},
        load=None,  # N-22: features_added da "olculmedi" kalmali (asagida dogrulanir)
        vectorize=VectorizeRunStats(exit_code=0, points_written=17, embedding_model="bge-m3"),
    )

    run_nightly.main([])

    files = list(reports_dir.glob("nightly_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))

    assert data["outcome"] == "full_success"
    assert data["source_diff"]["added"] == ["new.pdf"]
    assert data["source_diff"]["changed"] == ["mod.pdf"]
    assert data["source_diff"]["removed"] == ["del.pdf"]
    assert data["source_diff"]["moved"] == ["mov.pdf"]
    assert data["skipped"]["unchanged_count"] == 2
    assert data["parse"]["processed_count"] == 2
    assert data["parse"]["duration_seconds"] == 12.5
    assert data["parse"]["model_call_count"] is None, "olculmuyor -- None kalmali, 0 UYDURULMAMALI"
    assert data["chunk"]["produced_count"] == 17
    assert data["chunk"]["removed_count"] == 3
    assert data["vectorize"]["points_written"] == 17
    assert data["vectorize"]["points_deleted"] is None
    assert data["vectorize"]["total_points_after"] is None
    assert data["vectorize"]["embedding_model"] == "bge-m3"
    assert data["ownership"]["deterministic_count"] is None
    assert data["product_features"]["features_added"] is None
    assert data["failures"] == []

    md_files = list(reports_dir.glob("nightly_*.md"))
    assert len(md_files) == 1
    md = md_files[0].read_text(encoding="utf-8")
    assert "17" in md
    assert "olculmedi" in md, "None alanlar markdown'da acikca 'olculmedi' demeli"


def test_full_chain_partial_failure_writes_report_and_exits_nonzero(monkeypatch, tmp_path):
    """N-21 kabul (K-62 niyeti): bir asama YARIDA patlarsa kosu durur AMA
    o ana kadarki rapor YINE yazilir -- bugune kadar traceback'le cikiliyor,
    HICBIR IZ kalmiyordu."""
    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(reports_dir))
    _mock_backup_ok(monkeypatch, tmp_path)

    _stage_returns(monkeypatch, scan=_FAKE_SCAN_OUTPUT,
                   parse={"processed_count": 2, "duration_seconds": 1.0})
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["chunk"],
                        MagicMock(side_effect=RuntimeError("disk full")))
    ownership_mock = MagicMock()
    facts_mock = MagicMock()
    load_mock = MagicMock()
    vectorize_mock = MagicMock()
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["ownership"], ownership_mock)
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["facts"], facts_mock)
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["load"], load_mock)
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["vectorize"], vectorize_mock)

    with pytest.raises(SystemExit) as exc_info:
        run_nightly.main([])
    assert exc_info.value.code == 1

    ownership_mock.assert_not_called()
    facts_mock.assert_not_called()
    load_mock.assert_not_called()
    vectorize_mock.assert_not_called()

    files = list(reports_dir.glob("nightly_*.json"))
    assert len(files) == 1, "asama patlasa BILE rapor yazilmis olmali"
    data = json.loads(files[0].read_text(encoding="utf-8"))

    assert data["outcome"] in ("partial", "failed")
    # scan/parse GERCEKTEN calisti -- sayilari koruyor.
    assert data["source_diff"]["added"] == ["new.pdf"]
    assert data["parse"]["processed_count"] == 2
    # chunk hic tamamlanamadi -- 0'a duser (ama asagidaki failure kaydi
    # bunun "olculdu, sifir cikti" degil "hic calismadi" oldugunu acikca soyler).
    assert data["chunk"]["produced_count"] == 0

    failure_stages = {f["stage"] for f in data["failures"]}
    assert failure_stages == {"chunk", "ownership", "facts", "load", "vectorize"}
    chunk_failure = next(f for f in data["failures"] if f["stage"] == "chunk")
    assert chunk_failure["reason"] == "RuntimeError"
    assert "disk full" in chunk_failure["error_text"]
    facts_failure = next(f for f in data["failures"] if f["stage"] == "facts")
    assert facts_failure["reason"] == "skipped_upstream_failure"


def test_dry_run_full_chain_writes_no_report(monkeypatch, tmp_path):
    """N-03/N-21: `--dry-run` HICBIR VERI YAZMAZ -- rapor da bunun disinda
    degil."""
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(reports_dir))
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._DRY_RUN_FUNC_NAMES[name], mock)

    run_nightly.main(["--dry-run"])

    assert not reports_dir.exists(), "--dry-run bir rapor dosyasi bile YARATMAMALI"


def test_manual_stage_flag_writes_no_report(monkeypatch, tmp_path):
    """N-21: rapor uretimi BILINCLI olarak yalniz TAM zincire (--stage/--from
    YOK) baglandi -- K-60'in elle mudahale araci (`--stage facts`) rapor
    beklemez (bkz. `_run_full_chain_with_report` docstring'i)."""
    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(reports_dir))
    _mock_backup_ok(monkeypatch, tmp_path)
    mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    run_nightly.main(["--stage", "facts"])

    assert not reports_dir.exists()


def test_kismi_kosuda_yigin_izi_da_basilir(tmp_path, monkeypatch, capsys):
    """Bir asama patladiginda rapor yazilir AMA yigin izi de log'a dusmeli.

    `SystemExit` traceback YAZDIRMAZ; `raise SystemExit(1) from exc` yazmak
    raporu kazanip teshisi kaybetmek olurdu. 2026-08-25'te iki ariza (VLM
    canary'si, facts'in `mkdir(parents=True)` eksigi) dogrudan log'daki
    traceback'in son satirindan teshis edildi -- o izi koruyoruz.
    """
    import medrag.pipeline.cli.run_nightly as rn

    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))

    def _patlayan(**_kw):
        raise RuntimeError("ORNEK-ARIZA-METNI")

    # med-rag'de `products` asamasi kaldirildi; zincir `scan`de baslar --
    # bu test ARIZAYI `scan`da olcuyor.
    monkeypatch.setattr(rn, "stage_scan", _patlayan)

    from medrag.pipeline.cli import nightly_backup
    # N-25: bir asama patladiginda artik `restore_backup` cagriliyor -- bu
    # test yigin izi/rapor davranisini olcuyor, restore'un KENDISINI degil,
    # o yuzden no-op'a sahtelenir (davranisi `test_full_chain_partial_
    # failure_writes_report_and_exits_nonzero` gibi testler ayrica kapsar).
    monkeypatch.setattr(nightly_backup, "restore_backup", lambda root: [])

    from types import SimpleNamespace
    sahte_yedek = SimpleNamespace(root=tmp_path / "yedek")

    with pytest.raises(SystemExit):
        rn._run_full_chain_with_report(doc_id=None, backup_manifest=sahte_yedek)

    cikti = capsys.readouterr()
    hepsi = cikti.out + cikti.err
    assert "ORNEK-ARIZA-METNI" in hepsi
    assert "Traceback" in hepsi, "yigin izi basilmali -- SystemExit tek basina basmaz"
    assert "rapor yazildi" in hepsi, "kismi rapor yine de yazilmali"


# --- N-25: otomatik geri yukleme --------------------------------------------


def test_stage_failure_triggers_restore_to_this_runs_own_backup(monkeypatch, tmp_path):
    """N-25 cekirdek kabul: bir asama YARIDA patlarsa, `restore_backup`
    TAM OLARAK bu kosunun kendi (kosu-oncesi) yedegine cagrilir -- eski bir
    yedege ya da baska bir seye DEGIL."""
    from medrag.pipeline.cli import nightly_backup

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))
    fake_manifest = _mock_backup_ok(monkeypatch, tmp_path)

    restore_mock = MagicMock(return_value=[])
    monkeypatch.setattr(nightly_backup, "restore_backup", restore_mock)

    _stage_returns(monkeypatch, scan=_FAKE_SCAN_OUTPUT)
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["parse"],
                        MagicMock(side_effect=RuntimeError("ORNEK-ARIZA")))

    with pytest.raises(SystemExit) as exc_info:
        run_nightly.main([])
    assert exc_info.value.code == 1

    restore_mock.assert_called_once_with(fake_manifest.root)

    files = list((tmp_path / "reports").glob("nightly_*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert "GERI ALINDI" in data["integrity"]["notes"]


def test_main_restores_from_marker_when_previous_run_crashed(monkeypatch, tmp_path):
    """N-25: onceki kosu `kill -9`/cokme ile TEMIZ CIKMAMISSA (diskte
    'devam ediyor' isaretcisi kalmis), bir sonraki `main()` cagrisi YENI bir
    yedek almadan ONCE o kosunun yedegine geri doner."""
    from medrag.pipeline.cli import nightly_backup

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))

    inprogress_path = lock_path.with_name(lock_path.stem + ".inprogress.json")
    prev_backup_root = tmp_path / "prev-backup-from-killed-run"
    nightly_backup.mark_in_progress(prev_backup_root, path=inprogress_path)

    # `_mock_backup_ok` no-ops the marker helpers too (see that helper's
    # docstring) -- this test specifically measures the marker flow, so we
    # stub NOTHING outside `run_backup`; only `restore_backup` is wired to
    # an observable mock.
    fake_manifest = nightly_backup.BackupManifest(
        root=tmp_path / "fake-backup", started_at="t0", finished_at="t1",
        duration_seconds=0.01, entries=[])
    monkeypatch.setattr(nightly_backup, "run_backup", lambda **kwargs: fake_manifest)
    restore_mock = MagicMock(return_value=[])
    monkeypatch.setattr(nightly_backup, "restore_backup", restore_mock)

    _stage_returns(monkeypatch, scan=_FAKE_SCAN_OUTPUT,
                   parse={"processed_count": 1, "duration_seconds": 0.1},
                   chunk={"produced_count": 0, "per_document_counts": {}, "removed_count": 0},
                   ownership=None, facts={"n_processed": 0}, load=None,
                   vectorize=unittest.mock.MagicMock(exit_code=0, points_written=0,
                                                      embedding_model="x",
                                                      points_deleted=None,
                                                      total_points_after=None))

    run_nightly.main([])

    restore_mock.assert_called_once_with(str(prev_backup_root))
    # isaretci hem kurtarma sonrasi hem bu kosunun kendisi bittiginde
    # temizlenir -- sonunda diskte KALMAMALI.
    assert nightly_backup.read_in_progress(path=inprogress_path) is None


def test_restore_error_leaves_marker_for_next_attempt(monkeypatch, tmp_path):
    """N-25: geri yuklemenin KENDISI patlarsa (`RestoreError`), isaretci
    BILEREK diskte birakilir -- bir sonraki cagri kurtarmayi TEKRAR
    denesin diye. Yutulup temizlenirse bu koruma tek kullanimlik olurdu."""
    from medrag.pipeline.cli import nightly_backup

    lock_path = tmp_path / "nightly.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(lock_path))
    monkeypatch.setenv("NIGHTLY_REPORTS_DIR", str(tmp_path / "reports"))
    fake_manifest = nightly_backup.BackupManifest(
        root=tmp_path / "fake-backup", started_at="t0", finished_at="t1",
        duration_seconds=0.01, entries=[])
    monkeypatch.setattr(nightly_backup, "run_backup", lambda **kwargs: fake_manifest)

    def _patlayan_restore(root):
        raise nightly_backup.RestoreError("test: yedek de bozuk cikti")

    monkeypatch.setattr(nightly_backup, "restore_backup", _patlayan_restore)

    _stage_returns(monkeypatch, scan=_FAKE_SCAN_OUTPUT)
    monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES["parse"],
                        MagicMock(side_effect=RuntimeError("ORNEK-ARIZA")))

    inprogress_path = lock_path.with_name(lock_path.stem + ".inprogress.json")

    with pytest.raises(nightly_backup.RestoreError):
        run_nightly.main([])

    pending = nightly_backup.read_in_progress(path=inprogress_path)
    assert pending == {"backup_root": str(fake_manifest.root)}
    # RestoreError bir asama basarisizligi degil -- rapor YAZILAMAZ (veri
    # belirsiz durumda, o ana kadarki durumu bile guvenle raporlayamayiz).
    assert not (tmp_path / "reports").exists()


# --- gecelik rapor maili ---------------------------------------------------


def _fake_report(outcome="full_success"):
    from types import SimpleNamespace
    return SimpleNamespace(
        run_id="2026-08-26T00-00-00Z", outcome=lambda: outcome,
        source_diff=SimpleNamespace(added=["a.pdf"], changed=[], removed=["b.pdf"], moved=[]),
        skipped=SimpleNamespace(unchanged_count=5),
        chunk=SimpleNamespace(produced_count=120, removed_count=3),
        vectorize=SimpleNamespace(points_written=120),
        product_features=SimpleNamespace(features_added=8),
        failures=[],
    )


def test_send_nightly_email_skips_when_no_recipient(monkeypatch, tmp_path, capsys):
    """`NIGHTLY_REPORT_EMAIL_TO` tanimsizsa mail HIC ATILMAZ -- SMTP hic
    kurulmamali (varsayilan kapali, kimse istemeden mail akmasin)."""
    monkeypatch.delenv("NIGHTLY_REPORT_EMAIL_TO", raising=False)
    called = []
    monkeypatch.setattr("smtplib.SMTP", lambda *a, **kw: called.append(1) or MagicMock())

    run_nightly._send_nightly_email(
        report=_fake_report(), markdown_text="kisa rapor",
        md_path=tmp_path / "nightly_2026-08-26.md",
    )

    assert called == []
    assert "atlaniyor" in capsys.readouterr().out


def test_build_nightly_email_body_is_short_numeric_summary(tmp_path):
    """Govde RAPOR degil -- kullanicinin istedigi kisa sayisal ozet
    (sonuc/yeni/kaldirilan/vb.), tam rapor HER ZAMAN ek olarak gider."""
    uzun_rapor = "# Gecelik Kosu Raporu\n" + ("x" * 5000)
    msg = run_nightly.build_nightly_email(
        report=_fake_report(), markdown_text=uzun_rapor,
        md_path=tmp_path / "nightly_2026-08-26.md",
        from_addr="nightly@test", to_addrs=["a@example.com"],
    )

    body = msg.get_body(preferencelist=("plain",)).get_content()
    assert "Sonuc: full_success" in body
    assert "+1 yeni" in body
    # K-101: etiket artik "sistemde toplam DELETED (sticky)" diyor, "bu gece
    # kaldirilan" degil -- bkz. `build_nightly_email_summary`in K-101 notu.
    assert "1 DELETED (sistemde toplam, sticky" in body
    assert uzun_rapor not in body, "tam rapor govdeye GOMULMEMELI, sadece ek olarak"
    assert len(body) < 500, "govde kisa sayisal ozet olmali, rapor DEGIL"

    attachments = list(msg.iter_attachments())
    assert len(attachments) == 1
    assert attachments[0].get_content().strip() == uzun_rapor
    assert attachments[0].get_filename() == "nightly_2026-08-26.md"


def test_send_nightly_email_reports_smtp_failure_without_raising(monkeypatch, tmp_path, capsys):
    """SMTP hatasi gece kosusunu DUSURMEMELI -- rapor zaten diske yazildi."""
    monkeypatch.setenv("NIGHTLY_REPORT_EMAIL_TO", "a@example.com")

    class _PatlayanSMTP:
        def __init__(self, *a, **kw):
            raise OSError("connection refused")

    monkeypatch.setattr("smtplib.SMTP", _PatlayanSMTP)

    run_nightly._send_nightly_email(
        report=_fake_report(), markdown_text="kisa rapor",
        md_path=tmp_path / "nightly_2026-08-26.md",
    )

    assert "gonderilemedi" in capsys.readouterr().out
