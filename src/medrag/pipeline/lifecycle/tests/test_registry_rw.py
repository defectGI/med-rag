"""registry_rw: upload kaydı (NEW -> MODIFIED), silme, kilitli güncelleme."""

import json

import pytest

from medrag.pipeline.lifecycle import registry_rw


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    belgeler = tmp_path / "BELGELER"
    belgeler.mkdir()
    (belgeler / "notlar").mkdir()
    registry = tmp_path / "document_nodes.json"
    registry.write_text(json.dumps({"documents": []}), encoding="utf-8")
    monkeypatch.setenv("BELGELER_DIR", str(belgeler))
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(registry))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    return belgeler, registry


def _dosya_yaz(belgeler, ad, icerik=b"icerik"):
    p = belgeler / ad
    p.write_bytes(icerik)
    return p


def test_add_upload_new_then_modified_same_doc_id(corpus):
    belgeler, _ = corpus
    f = _dosya_yaz(belgeler, "kitap.pdf", b"v1")
    r1 = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler)
    assert r1["scan"]["scan_status"] == "NEW"
    assert r1["location"]["rel_path"] == "kitap.pdf"

    f.write_bytes(b"v2")
    r2 = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler)
    assert r2["identity"]["doc_id"] == r1["identity"]["doc_id"], \
        "aynı yol = aynı kimlik (A4 değişiklik semantiği)"
    assert r2["scan"]["scan_status"] == "MODIFIED"
    assert r2["scan"]["content_hash"] != r1["scan"]["content_hash"]


def test_remove_and_update_record(corpus):
    belgeler, _ = corpus
    f = _dosya_yaz(belgeler, "not.md")
    rec = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler,
                                 doc_type="NOTE", note_title="Deneme")
    doc_id = rec["identity"]["doc_id"]

    guncel = registry_rw.update_record(doc_id, lambda r: r.update(
        {"parse": {**r["parse"], "status": "SUCCESS"}}))
    assert guncel["parse"]["status"] == "SUCCESS"

    kaldirilan = registry_rw.remove(doc_id)
    assert kaldirilan is not None
    assert registry_rw.remove(doc_id) is None  # idempotent


def test_load_missing_registry_raises(corpus, monkeypatch):
    _, registry = corpus
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(registry.parent / "yok.json"))
    with pytest.raises(registry_rw.RegistryError):
        registry_rw.load()


def test_load_corrupt_registry_raises(corpus):
    _, registry = corpus
    registry.write_text("{bozuk", encoding="utf-8")
    with pytest.raises(registry_rw.RegistryError):
        registry_rw.load(registry)


def test_save_writes_and_find(corpus):
    belgeler, registry = corpus
    f = _dosya_yaz(belgeler, "x.md")
    rec = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler)
    data = registry_rw.load(registry)
    assert registry_rw.find(data, rec["identity"]["doc_id"]) is not None
    # save idempotent -- aynı veriyi yeniden yazar
    registry_rw.save(data, registry)
    assert registry_rw.load(registry)["documents"]


def test_update_record_missing_returns_none(corpus):
    assert registry_rw.update_record("yok", lambda r: None) is None


def test_content_hash_is_sha256_prefixed_and_stable(corpus):
    belgeler, _ = corpus
    f = _dosya_yaz(belgeler, "h.pdf", "aynı içerik".encode())
    h1 = registry_rw.content_hash_of(f)
    h2 = registry_rw.content_hash_of(f)
    assert h1 == h2 and h1.startswith("sha256:") and len(h1) == 7 + 64


def test_add_upload_updates_note_title_on_modified(corpus):
    belgeler, _ = corpus
    f = _dosya_yaz(belgeler, "notlar/deneme.md", b"v1")
    r1 = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler,
                                doc_type="NOTE", note_title="İlk başlık")
    f.write_bytes(b"v2")
    r2 = registry_rw.add_upload(file_path=f, belgeler_dir=belgeler,
                                doc_type="NOTE", note_title="Yeni başlık")
    assert r2["identity"]["doc_id"] == r1["identity"]["doc_id"]
    assert r2["note"]["title"] == "Yeni başlık"


def test_blank_pipeline_state_shape(corpus):
    bp = registry_rw.blank_pipeline_state()
    assert set(bp) == {"parse", "chunk", "facts"}
    assert bp["parse"]["status"] == "PENDING"
    assert bp["chunk"]["status"] == "PENDING"
    assert bp["facts"]["status"] == "PENDING"
