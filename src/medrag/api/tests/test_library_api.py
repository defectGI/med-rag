"""Kütüphane API uçları + auth: gerçek Flask uygulaması (orchestrator'sız).

Bu testler yalnız `library` blueprint'ini ve auth'u bağlar -- chat
uçları (orchestrator/factory gerektiren) burada kapsanmaz, onlar
api/tests'in mevcut kapsamında.
"""

import json

import pytest
from flask import Flask

from medrag.api.auth import configure_auth
from medrag.api.library import bp as library_bp


@pytest.fixture()
def corpus(tmp_path, monkeypatch):
    belgeler = tmp_path / "BELGELER"
    belgeler.mkdir()
    monkeypatch.setenv("BELGELER_DIR", str(belgeler))
    monkeypatch.setenv("DOCUMENT_NODES_PATH", str(tmp_path / "document_nodes.json"))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setenv("ISLER_DIR", str(tmp_path / "isler"))
    monkeypatch.setenv("DURUM_DIR", str(tmp_path / "durum"))
    monkeypatch.delenv("MEDRAG_SIFRE", raising=False)
    (tmp_path / "document_nodes.json").write_text(
        json.dumps({"documents": []}), encoding="utf-8")
    return belgeler, tmp_path


@pytest.fixture()
def app(corpus):
    app = Flask(__name__)
    app.secret_key = "test"
    configure_auth(app)
    app.register_blueprint(library_bp)
    return app


def test_upload_list_delete_roundtrip(corpus, app):
    belgeler, tmp_path = corpus
    client = app.test_client()

    resp = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO(b"# kitap\nmetin"), "kitap.md"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["accepted"]) == 1
    doc = body["accepted"][0]
    assert doc["status"]["state"] == "queued"
    assert (belgeler / "kitap.md").is_file()

    # iş kuyruğuna 'process' işi girdi
    isler = list((tmp_path / "isler").glob("*.json"))
    assert len(isler) == 1 and json.loads(isler[0].read_text())["action"] == "process"

    # liste: durum dosyasından queued geliyor
    liste = client.get("/api/library/documents").get_json()["documents"]
    assert len(liste) == 1 and liste[0]["doc_id"] == doc["doc_id"]
    assert liste[0]["status"]["state"] == "queued"

    # silme: delete işi kuyruğa girer
    resp = client.delete(f"/api/library/documents/{doc['doc_id']}")
    assert resp.status_code == 200
    isler = list((tmp_path / "isler").glob("*.json"))
    assert len(isler) == 2
    eylemler = {json.loads(p.read_text())["action"] for p in isler}
    assert eylemler == {"process", "delete"}


def test_upload_rejects_unknown_extension(corpus, app):
    _, tmp_path = corpus
    client = app.test_client()
    resp = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO(b"MZ"), "virus.exe"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["rejected"][0]["reason"].startswith("desteklenmeyen")
    assert list((tmp_path / "isler").glob("*.json")) == []


def test_reupload_same_name_modifies_same_doc_id(corpus, app):
    client = app.test_client()
    r1 = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO(b"v1"), "a.md"),
    }, content_type="multipart/form-data").get_json()["accepted"][0]
    r2 = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO(b"v2"), "a.md"),
    }, content_type="multipart/form-data").get_json()["accepted"][0]
    assert r1["doc_id"] == r2["doc_id"]


def test_document_content_404_when_not_parsed(corpus, app):
    client = app.test_client()
    doc = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO("içerik".encode()), "b.md"),
    }, content_type="multipart/form-data").get_json()["accepted"][0]
    resp = client.get(f"/api/library/documents/{doc['doc_id']}/content")
    assert resp.status_code == 404


def test_notes_crud(corpus, app):
    belgeler, tmp_path = corpus
    client = app.test_client()

    not_ = client.post("/api/library/notes", json={
        "title": "Hasta Notu", "content": "doz: 2x1"}).get_json()["note"]
    assert not_["doc_type"] == "NOTE"
    assert not_["note_title"] == "Hasta Notu"
    dosyalar = list((belgeler / "notlar").glob("*.md"))
    assert len(dosyalar) == 1

    liste = client.get("/api/library/notes").get_json()["notes"]
    assert len(liste) == 1

    resp = client.put(f"/api/library/notes/{not_['doc_id']}",
                      json={"content": "doz: 3x1"})
    assert resp.status_code == 200
    assert "doz: 3x1" in dosyalar[0].read_text(encoding="utf-8")
    isler = list((tmp_path / "isler").glob("*.json"))
    assert len(isler) == 2  # create + update process işleri

    resp = client.delete(f"/api/library/notes/{not_['doc_id']}")
    assert resp.status_code == 200


def test_auth_guard_and_login(corpus, monkeypatch):
    monkeypatch.setenv("MEDRAG_SIFRE", "gizli123")
    app = Flask(__name__)
    app.secret_key = "test"
    configure_auth(app)
    app.register_blueprint(library_bp)
    client = app.test_client()

    assert client.get("/api/library/documents").status_code == 401
    assert client.post("/api/auth/login",
                       json={"password": "yanlis"}).status_code == 401
    assert client.post("/api/auth/login",
                       json={"password": "gizli123"}).status_code == 200
    assert client.get("/api/library/documents").status_code == 200
    assert client.get("/api/auth/session").get_json()["authenticated"] is True
    client.post("/api/auth/logout")
    assert client.get("/api/library/documents").status_code == 401


def test_auth_disabled_when_password_unset(corpus, app):
    assert app.test_client().get("/api/library/documents").status_code == 200


def test_auth_exempt_and_non_api_paths(corpus, monkeypatch):
    monkeypatch.setenv("MEDRAG_SIFRE", "gizli")
    app = Flask(__name__)
    app.secret_key = "t"
    configure_auth(app)
    app.register_blueprint(library_bp)
    client = app.test_client()

    # /api/auth/* muaf: session ve login şifresiz de ulaşılabilir
    assert client.get("/api/auth/session").status_code == 200
    # /api dışı yol (SPA) guard'a girmez
    client.post("/api/auth/login", json={"password": "gizli"})
    assert client.get("/").status_code == 404  # bu app'te rota yok ama 401 DEĞİL


def test_document_content_returns_markdown_when_parsed(corpus, app):
    _, tmp_path = corpus
    client = app.test_client()
    doc = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO("içerik".encode()), "c.md"),
    }, content_type="multipart/form-data").get_json()["accepted"][0]
    # parse çıktısını taklit et: parsed/<doc_type>_<doc_id>/<doc_id>.md
    md_dir = tmp_path / "parsed" / f"{doc['doc_type']}_{doc['doc_id']}"
    md_dir.mkdir(parents=True)
    (md_dir / f"{doc['doc_id']}.md").write_text("# Başlık\nmetin", encoding="utf-8")

    resp = client.get(f"/api/library/documents/{doc['doc_id']}/content")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["markdown"].startswith("# Başlık")
    assert body["file_name"] == "c.md"


def test_document_file_download_and_404(corpus, app):
    _, _ = corpus
    client = app.test_client()
    doc = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO("PİDF".encode()), "d.pdf"),
    }, content_type="multipart/form-data").get_json()["accepted"][0]
    resp = client.get(f"/api/library/documents/{doc['doc_id']}/file")
    assert resp.status_code == 200
    assert resp.data == "PİDF".encode()
    assert resp.headers["Content-Disposition"].find("d.pdf") != -1
    assert client.get("/api/library/documents/olmayan/file").status_code == 404


def test_delete_nonexistent_returns_404(corpus, app):
    assert app.test_client().delete("/api/library/documents/yok").status_code == 404


def test_config_error_returns_503(corpus, app, monkeypatch):
    monkeypatch.delenv("BELGELER_DIR", raising=False)
    monkeypatch.delenv("DOCUMENT_NODES_PATH", raising=False)
    monkeypatch.delenv("PARSED_OUTPUT_DIR", raising=False)
    monkeypatch.delenv("ISLER_DIR", raising=False)
    monkeypatch.delenv("DURUM_DIR", raising=False)
    assert app.test_client().get("/api/library/documents").status_code == 503


def test_upload_size_limit(corpus, app, monkeypatch):
    monkeypatch.setenv("MEDRAG_MAX_YUKLEME_MB", "0")  # 0 bayt -> her şey red
    client = app.test_client()
    resp = client.post("/api/library/documents", data={
        "files": (__import__("io").BytesIO(b"x"), "buyuk.pdf"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert "sınırını aşıyor" in resp.get_json()["rejected"][0]["reason"]


def test_note_update_title_only(corpus, app):
    belgeler, _ = corpus
    client = app.test_client()
    not_ = client.post("/api/library/notes", json={
        "title": "Eski", "content": "sabit içerik"}).get_json()["note"]
    resp = client.put(f"/api/library/notes/{not_['doc_id']}",
                      json={"title": "Yeni başlık"})
    assert resp.status_code == 200
    # dosya içeriği değişmedi, başlık registry'de güncellendi
    dosya = next(iter((belgeler / "notlar").glob("*.md")))
    assert "sabit içerik" in dosya.read_text(encoding="utf-8")
    liste = client.get("/api/library/notes").get_json()["notes"]
    assert liste[0]["note_title"] == "Yeni başlık"


def test_fallback_state_from_registry(corpus, app):
    """Durum dosyası yoksa durum registry'den türetilir (eski belgeler)."""
    import json as _json

    belgeler, tmp_path = corpus
    registry = tmp_path / "document_nodes.json"
    data = _json.loads(registry.read_text(encoding="utf-8"))
    data["documents"].append({
        "identity": {"doc_id": "eski", "file_name": "eski.pdf", "extension": ".pdf"},
        "location": {"rel_path": "eski.pdf", "file_path": str(belgeler / "eski.pdf")},
        "scan": {"content_hash": "sha256:x", "size_bytes": 1,
                 "last_modified_time": "2026-01-01T00:00:00+00:00",
                 "scan_status": "NEW"},
        "doc_type": "BELGE", "is_active": True,
        "links": {}, "parse": {"status": "SUCCESS"}, "chunk": {"status": "SUCCESS"},
        "facts": {"status": "PENDING"},
    })
    registry.write_text(_json.dumps(data), encoding="utf-8")
    docs = app.test_client().get("/api/library/documents").get_json()["documents"]
    eski = next(d for d in docs if d["doc_id"] == "eski")
    assert eski["status"]["state"] == "ready"


def test_events_stream_emits_snapshot_on_change(corpus, app, monkeypatch):
    import medrag.api.library as lib

    seen = {"docs": []}

    class _Durdur(Exception):
        pass

    def _sahte_liste():
        return seen["docs"]

    def _sahte_sleep(_s):
        raise _Durdur  # ilk kare yayınlandıktan sonra döngüyü kır

    monkeypatch.setattr(lib, "_list_documents", _sahte_liste)
    monkeypatch.setattr(lib.time, "sleep", _sahte_sleep)

    gen = lib._events_stream()
    # İlk kare: başlangıç anlık görüntüsü (değişim) yayınlanır
    kare = next(gen)
    assert "event: documents" in kare

    # Liste değişince yeni kare gelir
    seen["docs"] = [{"doc_id": "x", "file_name": "x.md", "doc_type": "BELGE",
                     "rel_path": "x.md", "size_bytes": 1, "uploaded_at": None,
                     "note_title": None, "status": {"state": "ready",
                                                    "detail": None, "updated_at": None}}]
    with pytest.raises(_Durdur):
        gen.send(None)

