"""runner: process/delete akışı -- ağır bağımlılıklar (parse fazları,
chunker, vectorize, Qdrant) monkeypatch ile sahtelenir; akış ve durum
geçişleri gerçek kod üzerinden test edilir."""

import json
from types import SimpleNamespace

import pytest

from medrag.pipeline.lifecycle import durum as durum_store
from medrag.pipeline.lifecycle import registry_rw, runner
from medrag.pipeline.lifecycle.paths import durum_dir


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    belgeler = tmp_path / "BELGELER"
    belgeler.mkdir()
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    registry = tmp_path / "document_nodes.json"
    registry.write_text(json.dumps({"documents": []}), encoding="utf-8")
    monkeypatch.setenv("BELGELER_DIR", str(belgeler))
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(registry))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(parsed))
    monkeypatch.setenv("QDRANT_URL", "http://qdrant-sahte:6333")
    return belgeler, registry, parsed


class _SahteStore:
    def __init__(self):
        self.cleared = []

    def replace_scope(self, scope_id, points):
        self.cleared.append((scope_id, points))


@pytest.fixture()
def sahte_vektor(monkeypatch):
    store = _SahteStore()

    def _con():
        import sqlite3

        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE spec_value "
                    "(value_id TEXT PRIMARY KEY, evidence TEXT NOT NULL DEFAULT '[]')")
        return con

    monkeypatch.setattr(runner, "_vector_store", lambda: store)
    monkeypatch.setattr(runner, "_specs_connection", _con)
    return store


def test_process_document_happy_path(corpus, sahte_vektor, monkeypatch):
    belgeler, registry, parsed = corpus
    src = belgeler / "kitap.md"
    src.write_text("# Başlık\n\nmetin", encoding="utf-8")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]

    # parse fazları: SUCCESS + IR yolu yazılır (gerçek faz fonksiyonlarının
    # registry'ye bıraktığı etkinin minimum taklidi)
    def _faz0(record):
        pass

    def _faz1(record):
        record["parse"]["status"] = "SUCCESS"

    def _faz2(record):
        record["parse"].update(
            status="SUCCESS",
            parsed_from_hash=record["scan"]["content_hash"],
            parsed_json_path=str(parsed / f"{doc_id}.json"),
        )

    cli = "medrag.pipeline.cli.run_parse_pipeline"
    monkeypatch.setattr(f"{cli}.phase0_classify_one", _faz0)
    monkeypatch.setattr(f"{cli}.phase1_one", _faz1)
    monkeypatch.setattr(f"{cli}.phase2_one", _faz2)

    all_chunks = parsed.parent / "chunks" / "all_chunks.json"
    monkeypatch.setattr(runner, "_chunk_output_root",
                        lambda: all_chunks.parent)

    # chunker sahtesi: gerçek chunker all_chunks.json'u sıfırdan yeniden
    # üretir (KARAR-012) -- sahte de aynısını yapar; çünkü process akışı
    # ÖNCE eski türevleri unutturur (kayı düşer), SONRA chunker çıktısını
    # okur.
    def _sahte_chunker(env=None, **_kw):
        all_chunks.parent.mkdir(parents=True, exist_ok=True)
        all_chunks.write_text(json.dumps({"documents": {doc_id: {
            "doc_id": doc_id,
            "provenance": {"generated_at": "2026-09-08T00:00:00+00:00",
                           "chunker_version": "v-test",
                           "source": {"raw_sha256": "sha256:abc"}},
        }}}), encoding="utf-8")
        return 0

    monkeypatch.setattr("medrag.pipeline.chunker.cli.calistir", _sahte_chunker)

    # vectorize: bayatlık kapısı sahtesi
    monkeypatch.setattr(
        "medrag.pipeline.vectorize.cli.calistir", lambda env=None: SimpleNamespace(
            exit_code=0, points_written=7, embedding_model="sahte"))

    gecen_durumlar = []
    sonuc = runner.process_document(
        doc_id, status=lambda s, d=None, **kw: gecen_durumlar.append(s))
    assert sonuc == "ready"
    assert "parsing" in gecen_durumlar and "chunking" in gecen_durumlar \
        and "vectorizing" in gecen_durumlar

    final = durum_store.read_status(durum_dir(), doc_id)
    assert final["state"] == "ready"
    assert "7 parça" in final["detail"]

    # registry: parse + chunk blokları yazılmış
    kayit = registry_rw.find(registry_rw.load(registry), doc_id)
    assert kayit["parse"]["status"] == "SUCCESS"
    assert kayit["chunk"]["status"] == "SUCCESS"
    assert kayit["chunk"]["chunks_path"].endswith("all_chunks.json")


def test_process_document_unsupported_format_is_error(corpus, sahte_vektor, monkeypatch):
    belgeler, _, _ = corpus
    src = belgeler / "resim.png"
    src.write_bytes(b"\x89PNG", )
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]

    cli = "medrag.pipeline.cli.run_parse_pipeline"
    monkeypatch.setattr(f"{cli}.phase0_classify_one", lambda r: None)
    monkeypatch.setattr(f"{cli}.phase1_one", lambda r: r["parse"].update(
        status="SKIPPED", parsed_from_hash=r["scan"]["content_hash"]))
    monkeypatch.setattr(f"{cli}.phase2_one", lambda r: None)

    cagriladi = []
    monkeypatch.setattr("medrag.pipeline.chunker.cli.calistir",
                        lambda env=None: cagriladi.append(1) or 0)

    sonuc = runner.process_document(doc_id, status=lambda s, d=None, **kw: None)
    assert sonuc == "error"
    from medrag.pipeline.lifecycle.paths import durum_dir

    final = durum_store.read_status(durum_dir(), doc_id)
    assert final["state"] == "error"
    assert "ayrıştırıcı" in final["detail"]
    assert not cagriladi, "SKIPPED belge için chunk aşaması çalışmamalı"


def test_delete_document_removes_registry_and_status(corpus, sahte_vektor, monkeypatch):
    belgeler, registry, _ = corpus
    src = belgeler / "eski.md"
    src.write_text("x", encoding="utf-8")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]
    from medrag.pipeline.lifecycle.paths import durum_dir

    durum_store.write_status(durum_dir(), doc_id, "ready")

    sonuc = runner.delete_document(doc_id, status=lambda s, d=None, **kw: None)
    assert sonuc == "deleted"
    assert registry_rw.find(registry_rw.load(registry), doc_id) is None
    assert durum_store.read_status(durum_dir(), doc_id) is None
    # idempotent: ikinci silme de "deleted" (kayıt yok ama patlamaz)
    assert runner.delete_document(
        doc_id, status=lambda s, d=None, **kw: None) == "deleted"


def test_forget_derivatives_uses_store(corpus, sahte_vektor, monkeypatch):
    belgeler, _, parsed = corpus
    src = belgeler / "a.md"
    src.write_text("x", encoding="utf-8")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    # Sahte all_chunks dosyası: kayıt düşürme yolu denensin
    all_chunks = parsed.parent / "chunks" / "all_chunks.json"
    all_chunks.parent.mkdir(parents=True, exist_ok=True)
    all_chunks.write_text(json.dumps({"documents": {
        rec["identity"]["doc_id"]: {"nodes": []}}}), encoding="utf-8")
    monkeypatch.setattr(runner, "_chunk_output_root", lambda: all_chunks.parent)

    sonuc = runner.forget_derivatives(rec)
    assert sonuc["chunk_entry_removed"] is True
    assert sonuc["qdrant_scope_cleared"] is True
    assert sahte_vektor.cleared and sahte_vektor.cleared[0][1] == []


def test_process_document_failed_parse_is_error(corpus, sahte_vektor, monkeypatch):
    belgeler, _, _ = corpus
    src = belgeler / "kirik.pdf"
    src.write_bytes(b"%PDF")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]

    cli = "medrag.pipeline.cli.run_parse_pipeline"
    monkeypatch.setattr(f"{cli}.phase0_classify_one", lambda r: None)
    monkeypatch.setattr(f"{cli}.phase1_one", lambda r: None)
    monkeypatch.setattr(f"{cli}.phase2_one", lambda r: r["parse"].update(
        status="FAILED", error="VLM zaman aşımı"))

    chunker_cagrildi = []
    monkeypatch.setattr("medrag.pipeline.chunker.cli.calistir",
                        lambda env=None: chunker_cagrildi.append(1) or 0)

    sonuc = runner.process_document(doc_id, status=lambda s, d=None, **kw: None)
    assert sonuc == "error"
    assert durum_store.read_status(durum_dir(), doc_id)["state"] == "error"
    assert "VLM zaman aşımı" in durum_store.read_status(durum_dir(), doc_id)["detail"]
    assert not chunker_cagrildi, "FAILED parse sonrası chunk aşaması çalışmamalı"


def test_process_document_partial_parse_still_reaches_ready(corpus, sahte_vektor, monkeypatch):
    belgeler, _, parsed = corpus
    src = belgeler / "yari.md"
    src.write_text("x", encoding="utf-8")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]

    cli = "medrag.pipeline.cli.run_parse_pipeline"
    monkeypatch.setattr(f"{cli}.phase0_classify_one", lambda r: None)
    monkeypatch.setattr(f"{cli}.phase1_one", lambda r: None)
    monkeypatch.setattr(f"{cli}.phase2_one", lambda r: r["parse"].update(
        status="PARTIAL", parsed_from_hash=r["scan"]["content_hash"]))

    all_chunks = parsed.parent / "chunks" / "all_chunks.json"
    monkeypatch.setattr(runner, "_chunk_output_root", lambda: all_chunks.parent)

    def _chunker(env=None, **_kw):
        all_chunks.parent.mkdir(parents=True, exist_ok=True)
        all_chunks.write_text(json.dumps({"documents": {doc_id: {
            "provenance": {"generated_at": "t", "chunker_version": "v",
                           "source": {"raw_sha256": "sha256:abc"}}}}}), encoding="utf-8")
        return 0

    monkeypatch.setattr("medrag.pipeline.chunker.cli.calistir", _chunker)
    monkeypatch.setattr("medrag.pipeline.vectorize.cli.calistir",
                        lambda env=None: SimpleNamespace(
                            exit_code=0, points_written=0, embedding_model="s"))

    sonuc = runner.process_document(doc_id, status=lambda s, d=None, **kw: None)
    assert sonuc == "ready"
    # 0 parça -> "gömülecek yeni parça yok" notu
    assert "gömülecek yeni parça yok" in durum_store.read_status(
        durum_dir(), doc_id)["detail"]


def test_process_document_missing_registry_record_is_error(corpus, sahte_vektor):
    sonuc = runner.process_document("olmayan-doc",
                                    status=lambda s, d=None, **kw: None)
    assert sonuc == "error"
    assert durum_store.read_status(durum_dir(), "olmayan-doc")["state"] == "error"


def test_process_document_missing_source_file_is_error(corpus, sahte_vektor):
    belgeler, _, _ = corpus
    src = belgeler / "kaybolan.md"
    src.write_text("x", encoding="utf-8")
    rec = registry_rw.add_upload(file_path=src, belgeler_dir=belgeler)
    doc_id = rec["identity"]["doc_id"]
    src.unlink()  # kayıt var, disk yok

    sonuc = runner.process_document(doc_id, status=lambda s, d=None, **kw: None)
    assert sonuc == "error"
    assert "kaynak dosya yok" in durum_store.read_status(
        durum_dir(), doc_id)["detail"]


def test_bootstrap_env_loads_cli_dotenv(corpus, monkeypatch):
    cli = "medrag.pipeline.cli.run_parse_pipeline"
    cagrildi = []
    monkeypatch.setattr(f"{cli}._bootstrap", lambda: cagrildi.append(1))
    runner._bootstrap_env()
    assert cagrildi == [1]


def test_chunk_output_root_from_env(corpus, monkeypatch):
    monkeypatch.setenv("CHUNKS_OUTPUT_DIR", str(corpus[2].parent / "özel" / "chunks"))
    from medrag.pipeline.cli import run_chunk_pipeline as rcp
    monkeypatch.setattr(rcp, "resolve_all_chunks_path",
                        lambda: __import__("pathlib").Path("/x/all_chunks.json"))
    # _chunk_output_root, resolve_all_chunks_path'in AYNI çözümlemesine güvenir
    assert runner._chunk_output_root() == __import__("pathlib").Path("/x")


def test_delete_document_without_registry_clears_status(corpus, sahte_vektor):
    # registry kaydı yok ama durum dosyası asılı kalmış
    durum_store.write_status(durum_dir(), "hayalet", "ready")
    assert runner.delete_document("hayalet",
                                  status=lambda s, d=None, **kw: None) == "deleted"
    assert durum_store.read_status(durum_dir(), "hayalet") is None

