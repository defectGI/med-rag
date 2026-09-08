"""nightly_backup.py (N-05) testleri -- tamami OFFLINE (gercek Qdrant/ag
cagrisi YOK). Gercek dosya sistemi kullanilir (`tmp_path`) ama yalniz test
kapsaminda uydurma kucuk dosyalar uzerinde -- gercek `facts/db/specs.db`ye
ya da gercek `chatbot-corpus/`e HIC dokunulmaz (`run_parse_pipeline._bootstrap`
ve `discover.DB_PATH`/`discover.ALL_CHUNKS_PATH` monkeypatch'lenir).

Calistirma: `python -m pytest src/medrag/pipeline/cli/tests/test_nightly_backup.py`
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from medrag.pipeline.cli import nightly_backup as nb

# --- kalem-bazli birim testleri ------------------------------------------


def test_backup_sqlite_survives_wal_mode_uncommitted_pages(tmp_path):
    """WAL modunda calisan, henuz checkpoint edilmemis bir SQLite'i duz
    `shutil.copy` ile almak BOZUK/eksik bir dosya uretebilir -- `.backup()`
    API'si canli baglantidan okur, checkpoint beklemeden TUTARLI bir goruntu
    verir. Bu test WAL'i acik birakip (checkpoint YOK) `_backup_sqlite`in
    yine de TUM satirlari gordugunu kanitlar."""
    src = tmp_path / "specs.db"
    con = sqlite3.connect(str(src))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE t (x INTEGER)")
    con.execute("INSERT INTO t VALUES (1), (2), (3)")
    con.commit()  # WAL dosyasina yazildi, ANA .db dosyasina HENUZ checkpoint edilmedi
    assert (tmp_path / "specs.db-wal").exists(), "test on-kosulu: WAL dosyasi olusmali"

    dest = tmp_path / "out" / "specs.db"
    size = nb._backup_sqlite(src, dest)
    con.close()

    assert dest.is_file()
    assert size > 0
    out_con = sqlite3.connect(str(dest))
    rows = out_con.execute("SELECT x FROM t ORDER BY x").fetchall()
    out_con.close()
    assert rows == [(1,), (2,), (3,)]


def test_backup_sqlite_missing_source_raises():
    with pytest.raises(nb.BackupError):
        nb._backup_sqlite(nb.Path("does-not-exist.db"), nb.Path("dest.db"))


def test_backup_file_copies_content(tmp_path):
    src = tmp_path / "document_nodes.json"
    src.write_text('{"a": 1}', encoding="utf-8")
    dest = tmp_path / "out" / "document_nodes.json"

    size = nb._backup_file(src, dest)

    assert dest.read_text(encoding="utf-8") == '{"a": 1}'
    assert size == len(b'{"a": 1}')


def test_backup_file_missing_source_raises(tmp_path):
    with pytest.raises(nb.BackupError):
        nb._backup_file(tmp_path / "yok.json", tmp_path / "out" / "yok.json")


def test_backup_dir_copies_tree(tmp_path):
    src = tmp_path / "parsed"
    (src / "docA").mkdir(parents=True)
    (src / "docA" / "ir.json").write_text("x" * 10, encoding="utf-8")
    (src / "docB").mkdir()
    (src / "docB" / "ir.json").write_text("y" * 5, encoding="utf-8")
    dest = tmp_path / "out" / "parsed"

    size = nb._backup_dir(src, dest)

    assert (dest / "docA" / "ir.json").read_text(encoding="utf-8") == "x" * 10
    assert (dest / "docB" / "ir.json").read_text(encoding="utf-8") == "y" * 5
    assert size == 15


def test_backup_dir_missing_source_raises(tmp_path):
    with pytest.raises(nb.BackupError):
        nb._backup_dir(tmp_path / "yok", tmp_path / "out" / "yok")


# --- qdrant snapshot: HICBIR gercek baglanti kurulmaz --------------------


def test_qdrant_snapshot_skipped_when_url_missing(tmp_path):
    entry = nb.backup_qdrant_snapshot(
        dest_dir=tmp_path / "qdrant", collection="urun_specs", url=None, api_key=None)

    assert entry.skipped is True
    assert "QDRANT_URL" in entry.note


def test_qdrant_snapshot_uses_injected_client_never_touches_network(tmp_path):
    """Gercek `QdrantClient` HIC KURULMAZ -- `client_factory` ile enjekte
    edilen sahte istemci kullanilir (bkz. modul docstring'i: bu makinede
    Qdrant calismiyor, gercek baglanti test EDILEMEZ)."""
    calls = []

    class SahteSnapshotSonucu:
        name = "urun_specs-snapshot-2026-08-20.snapshot"

    class SahteQdrantClient:
        def create_snapshot(self, *, collection_name):
            calls.append(collection_name)
            return SahteSnapshotSonucu()

    def factory(url, api_key):
        assert url == "http://fake-qdrant:6333"
        assert api_key == "fake-key"
        return SahteQdrantClient()

    entry = nb.backup_qdrant_snapshot(
        dest_dir=tmp_path / "qdrant", collection="urun_specs",
        url="http://fake-qdrant:6333", api_key="fake-key", client_factory=factory)

    assert entry.skipped is False
    assert calls == ["urun_specs"]
    assert "urun_specs-snapshot-2026-08-20.snapshot" in entry.note
    assert (tmp_path / "qdrant" / "snapshot_name.txt").read_text(encoding="utf-8") \
        == "urun_specs-snapshot-2026-08-20.snapshot"


def test_qdrant_snapshot_client_error_is_skipped_not_raised(tmp_path):
    def kizgin_factory(url, api_key):
        raise ConnectionError("baglanti reddedildi")

    entry = nb.backup_qdrant_snapshot(
        dest_dir=tmp_path / "qdrant", collection="urun_specs",
        url="http://fake-qdrant:6333", api_key=None, client_factory=kizgin_factory)

    assert entry.skipped is True
    assert "baglanti reddedildi" in entry.note


# --- run_backup(): tam akis, sahte kaynaklarla ---------------------------


@pytest.fixture
def fake_sources(tmp_path, monkeypatch):
    """`run_backup`'in okudugu 4 ZORUNLU kaynagi (specs.db, document_nodes.json,
    all_chunks.json, parse cikti dizini) gercek `facts/`/`chatbot-corpus/`e
    DOKUNMADAN sahte kucuk dosyalarla degistirir."""
    from medrag.pipeline.cli import run_parse_pipeline
    from medrag.pipeline.facts import discover

    specs_db = tmp_path / "src" / "specs.db"
    specs_db.parent.mkdir(parents=True)
    con = sqlite3.connect(str(specs_db))
    con.execute("CREATE TABLE product (product_code TEXT)")
    con.execute("INSERT INTO product VALUES ('DE1')")
    con.commit()
    con.close()

    nodes = tmp_path / "src" / "document_nodes.json"
    nodes.write_text('{"nodes": []}', encoding="utf-8")

    chunks = tmp_path / "src" / "all_chunks.json"
    chunks.write_text('{"documents": {}}', encoding="utf-8")

    parsed_dir = tmp_path / "src" / "parsed_output"
    (parsed_dir / "doc1").mkdir(parents=True)
    (parsed_dir / "doc1" / "ir.json").write_text("{}", encoding="utf-8")

    # `DOCUMENT_NODES_PATH`/`PARSED_OUTPUT_DIR` yalniz `_bootstrap()`
    # cagrildiktan sonra modul globali olarak var oluyor (D-43/K-45) --
    # `raising=False` bu yuzden gerekli.
    monkeypatch.setattr(run_parse_pipeline, "_bootstrap", lambda: None)
    monkeypatch.setattr(run_parse_pipeline, "DOCUMENT_NODES_PATH", str(nodes),
                        raising=False)
    monkeypatch.setattr(run_parse_pipeline, "PARSED_OUTPUT_DIR", str(parsed_dir),
                        raising=False)
    monkeypatch.setattr(discover, "DB_PATH", specs_db)
    monkeypatch.setattr(discover, "ALL_CHUNKS_PATH", chunks)

    return {"specs_db": specs_db, "nodes": nodes, "chunks": chunks, "parsed_dir": parsed_dir}


def test_run_backup_succeeds_and_writes_manifest(fake_sources, tmp_path):
    manifest = nb.run_backup(backup_root=tmp_path / "backups",
                              qdrant_url=None, qdrant_collection="urun_specs")

    names = {e.name for e in manifest.entries}
    assert names == {"specs.db", "document_nodes.json", "all_chunks.json",
                      "parsed_output_dir", "qdrant"}
    for e in manifest.entries:
        if e.name == "qdrant":
            assert e.skipped is True  # QDRANT_URL yok -> best-effort atlandi
            continue
        assert e.skipped is False
        assert e.size_bytes > 0
        assert nb.Path(e.destination).exists()
    assert (manifest.root / "manifest.json").is_file()
    assert manifest.total_size_bytes > 0


def test_run_backup_missing_required_source_raises_and_does_not_leave_partial_dir(
        fake_sources, tmp_path):
    """N-05'in cekirdek kabul kriteri: ZORUNLU bir kalem (burada
    document_nodes.json) eksikse `BackupError` firlatilir VE yarim kalan
    yedek dizini diskte BIRAKILMAZ."""
    fake_sources["nodes"].unlink()

    with pytest.raises(nb.BackupError):
        nb.run_backup(backup_root=tmp_path / "backups", qdrant_url=None)

    # specs.db yedegi document_nodes.json'dan ONCE denendigi icin basarili
    # olmus olabilir, ama BASARISIZ kosunun sonunda kok dizin TEMIZLENIR --
    # diskte "yarim ama var gibi duran" bir yedek KALMAMALI.
    assert not (tmp_path / "backups").exists() or not any(
        (tmp_path / "backups").iterdir())


def test_run_backup_fails_when_destination_root_is_unwritable(fake_sources, tmp_path):
    """N-05 kabul kosulu: yedek hedefi yazilamaz durumdaysa (burada gercek
    dosya-sistemi engeli: hedef YOLUN kendisi zaten duz bir DOSYA olarak
    var -- Windows'ta dizin izin bitleri sahibi icin guvenilir sekilde
    yazmayi ENGELLEMEDIGI icin bu, 'salt-okunur dizin'in platformdan bagimsiz
    calisan esdegeri) `run_backup` `BackupError` firlatir."""
    blocker = tmp_path / "backups"
    blocker.write_text("ben bir dizin degilim", encoding="utf-8")  # dizin degil, DOSYA

    with pytest.raises(nb.BackupError):
        nb.run_backup(backup_root=blocker, qdrant_url=None)


def test_run_backup_aborts_nightly_stages_when_write_mocked_to_fail(
        fake_sources, tmp_path, monkeypatch):
    """Alternatif kanit (task metninin 'ya da yazma fonksiyonunu mock'la
    hata firlat' secenegi): `_backup_sqlite` dogrudan hata firlatacak sekilde
    mock'lanir, `run_nightly.main()`in HICBIR `stage_*`i cagirmadigini ve
    SystemExit(1) ile basarisiz raporladigini dogrular."""
    from unittest.mock import MagicMock

    from medrag.pipeline.cli import run_nightly

    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(tmp_path / "nightly.lock"))

    def _patlayan(*a, **k):
        raise PermissionError("yedek hedefi yazilamiyor (mock)")

    monkeypatch.setattr(nb, "_backup_sqlite", _patlayan)
    stage_mocks = {name: MagicMock(return_value=None) for name in run_nightly.STAGES}
    for name, mock in stage_mocks.items():
        monkeypatch.setattr(run_nightly, run_nightly._STAGE_FUNC_NAMES[name], mock)

    with pytest.raises(SystemExit):
        run_nightly.main([])

    for name, mock in stage_mocks.items():
        mock.assert_not_called()


# --- restore_backup() (N-25) -----------------------------------------------


def test_restore_backup_reverts_live_files_to_backed_up_state(fake_sources, tmp_path):
    """N-25 cekirdek kabul kriteri: backup alindiktan SONRA canli dosyalar
    (specs.db/document_nodes.json/all_chunks.json/parsed_output_dir) BOZUK
    bir kosu tarafindan degistirilmis gibi davranir -- `restore_backup`
    hepsini yedekteki HALINE geri dondurur."""
    manifest = nb.run_backup(backup_root=tmp_path / "backups",
                              qdrant_url=None, qdrant_collection="urun_specs")

    # "Bozuk kosu" simulasyonu: canli dosyalarin uzerine YARIM/BOZUK veri yaz.
    con = sqlite3.connect(str(fake_sources["specs_db"]))
    con.execute("DELETE FROM product")
    con.execute("INSERT INTO product VALUES ('BOZUK')")
    con.commit()
    con.close()
    fake_sources["nodes"].write_text('{"nodes": ["BOZUK"]}', encoding="utf-8")
    fake_sources["chunks"].write_text('{"documents": {"BOZUK": {}}}', encoding="utf-8")
    (fake_sources["parsed_dir"] / "doc1" / "ir.json").write_text('{"BOZUK": true}',
                                                                   encoding="utf-8")
    (fake_sources["parsed_dir"] / "doc2").mkdir()
    (fake_sources["parsed_dir"] / "doc2" / "extra.json").write_text("{}", encoding="utf-8")

    restored = nb.restore_backup(manifest.root)

    names = {e.name for e in restored}
    assert names == {"specs.db", "document_nodes.json", "all_chunks.json",
                      "parsed_output_dir", "qdrant"}

    con = sqlite3.connect(str(fake_sources["specs_db"]))
    rows = con.execute("SELECT product_code FROM product").fetchall()
    con.close()
    assert rows == [("DE1",)]
    assert fake_sources["nodes"].read_text(encoding="utf-8") == '{"nodes": []}'
    assert fake_sources["chunks"].read_text(encoding="utf-8") == '{"documents": {}}'
    assert (fake_sources["parsed_dir"] / "doc1" / "ir.json").read_text(
        encoding="utf-8") == "{}"
    # restore oncesi eklenen "doc2" -- yedekte hic yoktu, agac tamamen
    # yedegin HALINE donduruldugu icin silinmis olmali.
    assert not (fake_sources["parsed_dir"] / "doc2").exists()


def test_restore_backup_qdrant_entry_always_skipped(fake_sources, tmp_path):
    manifest = nb.run_backup(backup_root=tmp_path / "backups", qdrant_url=None)

    restored = nb.restore_backup(manifest.root)

    qdrant_entry = next(e for e in restored if e.name == "qdrant")
    assert qdrant_entry.skipped is True


def test_restore_backup_missing_manifest_raises(tmp_path):
    with pytest.raises(nb.RestoreError):
        nb.restore_backup(tmp_path / "does-not-exist")


def test_restore_backup_raises_on_skipped_required_entry(tmp_path):
    """Manifest'te ZORUNLU bir kalem `skipped=True` isaretliyse (normalde
    olmamali -- byle bir yedek `BackupError` firlatip diskte kalmamali,
    ama elle bozulmus/eski bir manifest.json'a karsi savunma) geri yukleme
    GUVENSIZ sayilir, `RestoreError` firlatilir."""
    root = tmp_path / "backup"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"entries": [
        {"name": "specs.db", "source": str(tmp_path / "live.db"),
         "destination": str(root / "specs.db"), "skipped": True, "note": "x"},
    ]}), encoding="utf-8")

    with pytest.raises(nb.RestoreError):
        nb.restore_backup(root)


# --- "devam ediyor" isaretcisi (N-25) ---------------------------------------


def test_in_progress_marker_roundtrip(tmp_path):
    marker = tmp_path / "marker.json"
    assert nb.read_in_progress(path=marker) is None

    nb.mark_in_progress(tmp_path / "some-backup", path=marker)
    pending = nb.read_in_progress(path=marker)
    assert pending == {"backup_root": str(tmp_path / "some-backup")}

    nb.clear_in_progress(path=marker)
    assert nb.read_in_progress(path=marker) is None


def test_clear_in_progress_missing_marker_is_a_noop(tmp_path):
    nb.clear_in_progress(path=tmp_path / "never-existed.json")  # patlamamali
