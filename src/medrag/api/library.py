"""Kütüphane API'si (B3/B4 arka yüzü): belge yükleme, listeleme, silme,
durum akışı (SSE), içerik/orijinal dosya servisi ve notlar (A7).

## Katman yeri

`medrag.api` pipeline kodunu import EDEMEZ (katman kuralı + ruff TID251).
Bu modül pipeline tarafının YAZDIĞI DOSYA SÖZLEŞMELERİYLE çalışır
(`medrag.pipeline.lifecycle` paketiyle aynı şekiller, ayrı küçük
implementasyonlar -- bilinçli yineleme, bileşen yinelemeyen tek bağın
"dosya sistemi" olmasını sağlar):

  - iş kuyruğu: `ISLER_DIR/<job_id>.json` (jobs sözleşmesi)
  - durum:      `DURUM_DIR/<doc_id>.json` (durum sözleşmesi)
  - registry:   `DOCUMENT_NODES_PATH` (şema: classify_documents kayıt
                şekli; yazımlar `<registry>.lock` flock'uyla)

## Yollar (env)

BELGELER_DIR, DOCUMENT_NODES_PATH, PARSED_OUTPUT_DIR zorunlu (eksikse
ilgili uç 503 döner -- yapılandırma hatası, çökme değil); ISLER_DIR/
DURUM_DIR varsayılanları korpus kökünün kardeşidir (pipeline tarafıyla
aynı kural).

## Notlar (A7)

Notlar `<BELGELER_DIR>/notlar/` altında **.md** dosyalarıdır (parser'ın
markdown okuyucusundan geçebilsin diye; düz .txt ayrıştırıcıya takılır).
doc_type=NOTE ile AYNI registry/iş akışına girer -- chatbot onları başka
bir belgeden farksız kaynak olarak kullanır. Düzenleme = dosyayı yeniden
yaz + scan hash güncelle + process işi (değişiklik semantiği A4).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from flask import Blueprint, Response, jsonify, request, send_file, stream_with_context

logger = logging.getLogger("medrag.api.library")

#: Parser'ın gerçekten okuyabildiği uzantılar (parsers/registry). UI
#: whitelist'i budur; listede olmayanlar 415 ile reddedilir.
ALLOWED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".markdown",
     ".jpg", ".jpeg", ".png"})

#: Dosya başına üst limit (K16/Açık-5). `MEDRAG_MAX_YUKLEME_MB` ile aşılır.
DEFAULT_MAX_UPLOAD_MB = 200

#: Not başlıklarından dosya adı (slug) üretimi için güvenli karakter kümesi.
_SLUG_RE = re.compile(r"[^0-9A-Za-zçğıöşüÇĞİÖŞÜ_-]+")

bp = Blueprint("library", __name__)


# --- küçük yardımcılar (pipeline tarafıyla paralel sözleşmeler) -------------

class _ConfigError(RuntimeError):
    pass


def _env_path(name: str, default: str | None = None) -> Path:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        if default is None:
            raise _ConfigError(f"{name} tanımsız")
        base = _env_path("BELGELER_DIR")
        return (base.parent / default).resolve()
    return Path(raw).expanduser().resolve()


def _belgeler() -> Path:
    return _env_path("BELGELER_DIR")


def _registry() -> Path:
    return _env_path("DOCUMENT_NODES_PATH")


def _parsed_output() -> Path:
    return _env_path("PARSED_OUTPUT_DIR")


def _isler() -> Path:
    return _env_path("ISLER_DIR", default="isler")


def _durum() -> Path:
    return _env_path("DURUM_DIR", default="durum")


def _notlar() -> Path:
    return _belgeler() / "notlar"


def _max_upload_bytes() -> int:
    try:
        return int((os.environ.get("MEDRAG_MAX_YUKLEME_MB") or "").strip()
                   or DEFAULT_MAX_UPLOAD_MB) * 1024 * 1024
    except ValueError:
        return DEFAULT_MAX_UPLOAD_MB * 1024 * 1024


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def _registry_lock(registry: Path):
    """`fslock.file_lock` ile AYNI kilit dosyası (`<registry>.lock`)."""
    lp = registry.with_name(registry.name + ".lock")
    lp.parent.mkdir(parents=True, exist_ok=True)
    return lp


def _registry_load(registry: Path) -> dict:
    try:
        return json.loads(registry.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"documents": []}
    except json.JSONDecodeError as exc:
        raise _ConfigError(f"registry bozuk: {registry}: {exc}") from exc


def _job_write(action: str, doc_id: str, rel_path: str) -> str:
    """`jobs.enqueue` ile AYNI dosya sözleşmesi."""
    jdir = _isler()
    jdir.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" \
        + uuid.uuid4().hex[:8]
    _atomic_write_json(jdir / f"{job_id}.json", {
        "job_id": job_id, "action": action, "doc_id": doc_id,
        "rel_path": rel_path,
        "enqueued_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    logger.info("iş kuyruğa yazıldı: %s %s (%s)", action, doc_id, job_id)
    return job_id


def _durum_write(doc_id: str, state: str, detail: str | None = None,
                 rel_path: str | None = None) -> None:
    """`durum.write_status` ile AYNI dosya sözleşmesi (yalnız api'nin
    yazdığı `queued` durumu için)."""
    ddir = _durum()
    ddir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(ddir / f"{doc_id}.json", {
        "doc_id": doc_id, "state": state, "detail": detail,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **({"rel_path": rel_path} if rel_path else {}),
    })


def _durum_read_all() -> dict[str, dict]:
    ddir = _durum()
    out: dict[str, dict] = {}
    if not ddir.is_dir():
        return out
    for path in ddir.glob("*.json"):
        if path.name.startswith(".tmp_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("doc_id"):
            out[str(data["doc_id"])] = data
    return out


# --- kayıt görünümü ----------------------------------------------------------

def _fallback_state(record: dict) -> tuple[str, str | None]:
    """Durum dosyası yokken registry'den türetilen durum (eski kurulumlar/
    el ile işlenmiş belgeler için)."""
    parse_status = (record.get("parse") or {}).get("status")
    if parse_status in ("SUCCESS", "PARTIAL"):
        return "ready", None
    if parse_status in ("FAILED", "SKIPPED"):
        return "error", (record.get("parse") or {}).get("error") \
            or "ayrıştırıcı bu formatı işlemedi"
    return "queued", None


def _item(record: dict, durum: dict | None) -> dict:
    scan = record.get("scan") or {}
    state_detail = durum.get("detail") if durum else None
    if durum:
        state = str(durum.get("state"))
    else:
        state, state_detail = _fallback_state(record)
    return {
        "doc_id": record["identity"]["doc_id"],
        "file_name": record["identity"]["file_name"],
        "doc_type": record.get("doc_type"),
        "rel_path": (record.get("location") or {}).get("rel_path"),
        "size_bytes": scan.get("size_bytes"),
        "uploaded_at": scan.get("last_modified_time"),
        "note_title": (record.get("note") or {}).get("title"),
        "status": {
            "state": state,
            "detail": state_detail,
            "updated_at": (durum or {}).get("updated_at"),
        },
    }


def _visible_records(data: dict) -> list[dict]:
    out = []
    for record in data.get("documents", []):
        if not record.get("is_active", True):
            continue
        if (record.get("scan") or {}).get("scan_status") == "DELETED":
            continue
        out.append(record)
    return out


def _list_documents() -> list[dict]:
    registry = _registry()
    if not registry.is_file():
        return []
    lp = _registry_lock(registry)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        data = _registry_load(registry)
    finally:
        os.close(fd)
    dmap = _durum_read_all()
    items = [_item(r, dmap.get(r["identity"]["doc_id"]))
             for r in _visible_records(data)]
    items.sort(key=lambda i: (i.get("uploaded_at") or ""), reverse=True)
    return items


# --- uçlar: belgeler ---------------------------------------------------------

@bp.get("/api/library/documents")
def list_documents():
    try:
        return jsonify({"documents": _list_documents()})
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503


@bp.post("/api/library/documents")
def upload_documents():
    files = request.files.getlist("files") or []
    if not files:
        return jsonify({"error": "yüklenecek dosya yok (alan adı: files)"}), 400
    try:
        belgeler = _belgeler()
        registry = _registry()
        max_bytes = _max_upload_bytes()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503

    belgeler.mkdir(parents=True, exist_ok=True)
    kabul: list[dict] = []
    reddedilen: list[dict] = []
    for storage in files:
        name = Path(storage.filename or "").name
        if not name:
            reddedilen.append({"file_name": name or "?", "reason": "dosya adı yok"})
            continue
        uzanti = Path(name).suffix.lower()
        if uzanti not in ALLOWED_EXTENSIONS:
            reddedilen.append({"file_name": name,
                               "reason": f"desteklenmeyen uzantı: {uzanti}"})
            continue
        content = storage.read()
        if len(content) > max_bytes:
            reddedilen.append({"file_name": name,
                               "reason": f"dosya {max_bytes // (1024 * 1024)} MB sınırını aşıyor"})
            continue

        # Aynı yol üzerine yaz: registry'de MODIFIED (A4 değişiklik semantiği).
        hedef = belgeler / name
        hedef.write_bytes(content)

        try:
            record = _registry_add_upload(hedef, belgeler, registry,
                                          doc_type="BELGE")
        except Exception as exc:
            logger.exception("registry kaydı yazılamadı: %s", name)
            reddedilen.append({"file_name": name, "reason": f"kayıt hatası: {exc}"})
            continue
        _durum_write(record["identity"]["doc_id"], "queued",
                     detail="kuyrukta", rel_path=record["location"]["rel_path"])
        _job_write("process", record["identity"]["doc_id"],
                   record["location"]["rel_path"])
        kabul.append(_item(record, None))

    kod = 200 if kabul else 400
    return jsonify({"accepted": kabul, "rejected": reddedilen}), kod


def _registry_add_upload(file_path: Path, belgeler: Path, registry: Path,
                         *, doc_type: str, note_title: str | None = None) -> dict:
    """`registry_rw.add_upload` ile AYNI işlem (kilit + NEW/MODIFIED)."""
    import hashlib
    from datetime import datetime as _dt

    rel_path = file_path.relative_to(belgeler).as_posix()
    stat = file_path.stat()
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(blok)

    lp = _registry_lock(registry)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = _registry_load(registry)
        data.setdefault("documents", [])
        eski = next((r for r in data["documents"]
                     if (r.get("location") or {}).get("rel_path") == rel_path), None)
        if eski is not None:
            eski["scan"] = {
                "content_hash": "sha256:" + digest.hexdigest(),
                "size_bytes": stat.st_size,
                "last_modified_time": _dt.fromtimestamp(stat.st_mtime)
                .astimezone().isoformat(timespec="seconds"),
                "scan_status": "MODIFIED",
            }
            eski["is_active"] = True
            eski["location"]["file_path"] = str(file_path)
            if note_title is not None and isinstance(eski.get("note"), dict):
                eski["note"]["title"] = note_title
            record = eski
        else:
            record = {
                "identity": {
                    "doc_id": str(uuid.uuid4()),
                    "file_name": file_path.name,
                    "extension": file_path.suffix.lower(),
                },
                "location": {"rel_path": rel_path, "file_path": str(file_path)},
                "scan": {
                    "content_hash": "sha256:" + digest.hexdigest(),
                    "size_bytes": stat.st_size,
                    "last_modified_time": _dt.fromtimestamp(stat.st_mtime)
                    .astimezone().isoformat(timespec="seconds"),
                    "scan_status": "NEW",
                },
                "doc_type": doc_type,
                "is_active": True,
                "links": {"owner_ids": [], "link_count": 0,
                          "matched_node_id": None, "resolution_method": "upload"},
                "parse": {"parser": None, "parser_version": None,
                          "status": "PENDING", "parsed_from_hash": None,
                          "parsed_json_path": None, "last_parsed": None,
                          "error": None, "stages": None},
                "chunk": {"status": "PENDING", "chunked_at": None,
                          "chunked_from_hash": None, "chunks_path": None,
                          "chunker_version": None},
                "facts": {"status": "PENDING", "extracted_at": None,
                          "extracted_from_hash": None,
                          "extractor_version": None, "prompt_version": None},
            }
            if note_title is not None:
                record["note"] = {"title": note_title}
            data["documents"].append(record)
        registry.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(registry, data)
        return record
    finally:
        os.close(fd)


@bp.delete("/api/library/documents/<doc_id>")
def delete_document(doc_id: str):
    try:
        data = json.loads(_registry().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return jsonify({"error": "registry yok"}), 503
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"registry okunamadı: {exc}"}), 503
    record = next((r for r in data.get("documents", [])
                   if r.get("identity", {}).get("doc_id") == doc_id), None)
    if record is None:
        return jsonify({"error": "belge bulunamadı"}), 404
    rel_path = (record.get("location") or {}).get("rel_path") or ""
    _durum_write(doc_id, "queued", detail="siliniyor", rel_path=rel_path)
    _job_write("delete", doc_id, rel_path)
    return jsonify({"ok": True})


@bp.get("/api/library/documents/<doc_id>/content")
def document_content(doc_id: str):
    try:
        registry = _registry()
        parsed_output = _parsed_output()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"registry okunamadı: {exc}"}), 503
    record = next((r for r in data.get("documents", [])
                   if r.get("identity", {}).get("doc_id") == doc_id), None)
    if record is None:
        return jsonify({"error": "belge bulunamadı"}), 404
    md_path = parsed_output / f"{record.get('doc_type')}_{doc_id}" / f"{doc_id}.md"
    if not md_path.is_file():
        return jsonify({"error": "ayrıştırılmış içerik henüz yok"}), 404
    return jsonify({
        "doc_id": doc_id,
        "file_name": record["identity"]["file_name"],
        "markdown": md_path.read_text(encoding="utf-8"),
    })


@bp.get("/api/library/documents/<doc_id>/file")
def document_file(doc_id: str):
    try:
        registry = _registry()
        belgeler = _belgeler()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"registry okunamadı: {exc}"}), 503
    record = next((r for r in data.get("documents", [])
                   if r.get("identity", {}).get("doc_id") == doc_id), None)
    if record is None:
        return jsonify({"error": "belge bulunamadı"}), 404
    path = belgeler / (record["location"]["rel_path"])
    if not path.is_file():
        return jsonify({"error": "kaynak dosya diskte yok"}), 404
    return send_file(path, download_name=record["identity"]["file_name"])


# --- uçlar: durum SSE --------------------------------------------------------

@bp.get("/api/library/events")
def events():
    """Durum akışı: `DURUM_DIR` + registry anlık görüntüsü değiştikçe
    `documents` olayı yayınlanır; 15 sn'de bir yorum-heartbeat (proxy
    zaman aşımına karşı). 1 sn'lik yoklama tek kullanıcı için yeterli ve
    ucuzdur (dizin listesi + dosya mtime karşılaştırması)."""
    resp = Response(stream_with_context(_events_stream()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


def _events_stream():
    """SSE çerçevelerini üreten jeneratör (modül-düzeyi -- test edilebilir).

    Anlık görüntü değiştikçe `event: documents`; aksi halde 15 sn'de bir
    yorum-heartbeat. `time.sleep` ve `_list_documents` modül-global'leri
    üzerinden çağrıldığı için testler monkeypatch ile kısaltabilir."""
    last_signature = None
    last_beat = 0.0
    while True:
        try:
            docs = _list_documents()
        except Exception:  # noqa: BLE001 -- yapılandırma eksikliği akışı öldürmesin
            docs = []
        signature = json.dumps(docs, ensure_ascii=False, sort_keys=True)
        now = time.monotonic()
        if signature != last_signature:
            last_signature = signature
            yield ("event: documents\n"
                   f"data: {json.dumps({'documents': docs}, ensure_ascii=False)}\n\n")
            last_beat = now
        elif now - last_beat > 15:
            yield ": heartbeat\n\n"
            last_beat = now
        time.sleep(1.0)


# --- uçlar: notlar -----------------------------------------------------------

def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.strip()).strip("-") or "not"
    return slug.lower()[:60]


@bp.get("/api/library/notes")
def list_notes():
    try:
        items = [i for i in _list_documents() if i.get("doc_type") == "NOTE"]
        return jsonify({"notes": items})
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503


@bp.post("/api/library/notes")
def create_note():
    data = request.get_json(force=True, silent=True) or {}
    title = (data.get("title") or "").strip() or "Adsız not"
    content = data.get("content") or ""
    try:
        notlar = _notlar()
        registry = _registry()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503
    notlar.mkdir(parents=True, exist_ok=True)

    base = _slugify(title)
    hedef = notlar / f"{base}.md"
    n = 1
    while hedef.exists():
        n += 1
        hedef = notlar / f"{base}-{n}.md"
    hedef.write_text(content if content.endswith("\n") else content + "\n",
                     encoding="utf-8")

    try:
        record = _registry_add_upload(hedef, _belgeler(), registry,
                                      doc_type="NOTE", note_title=title)
    except Exception as exc:
        logger.exception("not kaydı yazılamadı: %s", title)
        return jsonify({"error": f"kayıt hatası: {exc}"}), 500
    _durum_write(record["identity"]["doc_id"], "queued", detail="kuyrukta",
                 rel_path=record["location"]["rel_path"])
    _job_write("process", record["identity"]["doc_id"],
               record["location"]["rel_path"])
    return jsonify({"note": _item(record, None)}), 201


@bp.put("/api/library/notes/<note_id>")
def update_note(note_id: str):
    data = request.get_json(force=True, silent=True) or {}
    try:
        registry = _registry()
        belgeler = _belgeler()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        full = json.loads(registry.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"registry okunamadı: {exc}"}), 503
    record = next((r for r in full.get("documents", [])
                   if r.get("identity", {}).get("doc_id") == note_id
                   and r.get("doc_type") == "NOTE"), None)
    if record is None:
        return jsonify({"error": "not bulunamadı"}), 404
    path = belgeler / record["location"]["rel_path"]
    if not path.is_file():
        return jsonify({"error": "not dosyası diskte yok"}), 404
    if "content" in data:
        content = data["content"] or ""
        path.write_text(content if content.endswith("\n") else content + "\n",
                        encoding="utf-8")
    title = (data.get("title") or "").strip()
    if title:
        record.setdefault("note", {})["title"] = title
    # hash'i yenile + process işi (değişiklik semantiği, A4)
    lp = _registry_lock(registry)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        full = json.loads(registry.read_text(encoding="utf-8"))
        rec = next((r for r in full.get("documents", [])
                    if r.get("identity", {}).get("doc_id") == note_id), None)
        if rec is not None:
            rec["scan"]["content_hash"] = _sha256_file(path)
            rec["scan"]["size_bytes"] = path.stat().st_size
            rec["scan"]["scan_status"] = "MODIFIED"
            rec["scan"]["last_modified_time"] = datetime.fromtimestamp(
                path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            if title:
                rec.setdefault("note", {})["title"] = title
            registry.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(registry, full)
    finally:
        os.close(fd)
    _durum_write(note_id, "queued", detail="kuyrukta",
                 rel_path=record["location"]["rel_path"])
    _job_write("process", note_id, record["location"]["rel_path"])
    return jsonify({"ok": True})


@bp.delete("/api/library/notes/<note_id>")
def delete_note(note_id: str):
    return delete_document(note_id)


# --- notları markdown'a uygun biçimlendir (LLM, A7) -------------------------

_FORMAT_SYSTEM = (
    "Rol: bir notu Markdown'a uygun biçimlendirmek.\n"
    "Görev: verilen ham not metninde YALNIZCA biçimlendirme yap; içeriği, "
    "ifadeleri, sözcükleri ve sırayı HİÇ DEĞİŞTİRME.\n"
    "Kurallar:\n"
    "- Uygun yerlerde başlık '# ' / '## ' / '### ', madde listesi '- ' / "
    "'| ', numaralı liste '1. ' kullan.\n"
    "- Vurgu için **bold** / *italic*, kod için ```blok```, alıntı için '> ' "
    "kullan.\n"
    "- Hiçbir cümle, madde veya bilgi ekleme/silme; yalnız biçimlendirmek için "
    "sarmala.\n"
    "- Sadece sonuç Markdown'ı yaz; başka açıklama, karşılama veya yorum ekleme."
)


def _format_note_text(content: str) -> str:
    """Notu LLM ile biçimlendirir (içerik değişmez, yalnız Markdown yapısı).

    `CHATBOT_LLM_*` yapılandırması kullanılır (chatbot'un kendi modeli).
    LLM hata verir/yapılandırılmamışsa `ProviderError` yükseltir -- çağıran
    bunu 502 olarak döndürür. Anthropic native / diğerleri OpenAI-uyumlu
    `/v1/chat/completions` üzerinden çağrılır.
    """
    from medrag.api.answering_model import answering_model_from_env
    from medrag.core.llm.http import post_json

    messages = [
        {"role": "system", "content": _FORMAT_SYSTEM},
        {"role": "user", "content": content or ""},
    ]
    am = answering_model_from_env()
    if am.provider == "anthropic":
        from medrag.core.llm.anthropic import call_anthropic

        return call_anthropic(
            messages, model=am.model, base_url=am.base_url,
            api_key=am.api_key, timeout=am.timeout,
        ).strip()
    resp = post_json(
        f"{am.base_url}/chat/completions",
        {"model": am.model, "messages": messages},
        api_key=am.api_key, timeout=am.timeout,
    )
    return ((resp.get("choices") or [{}])[0].get("message") or {}).get(
        "content", "").strip()


def _persist_note(note_id: str, content: str) -> str:
    """Not içeriğini dosyaya yazar + registry hash/scan günceller + process
    işini kuyruğa atar (A4 değişiklik semantiği). `format` ucu bu ortak
    yazımı kullanır (update_note'taki mantığın aynısı)."""
    registry = _registry()
    belgeler = _belgeler()
    full = json.loads(registry.read_text(encoding="utf-8"))
    record = next((r for r in full.get("documents", [])
                   if r.get("identity", {}).get("doc_id") == note_id), None)
    if record is None:
        raise LookupError("not bulunamadı")
    path = belgeler / record["location"]["rel_path"]
    if not path.is_file():
        raise LookupError("not dosyası diskte yok")

    path.write_text(content if content.endswith("\n") else content + "\n",
                    encoding="utf-8")

    lp = _registry_lock(registry)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        full = json.loads(registry.read_text(encoding="utf-8"))
        rec = next((r for r in full.get("documents", [])
                    if r.get("identity", {}).get("doc_id") == note_id), None)
        if rec is not None:
            rec["scan"]["content_hash"] = _sha256_file(path)
            rec["scan"]["size_bytes"] = path.stat().st_size
            rec["scan"]["scan_status"] = "MODIFIED"
            rec["scan"]["last_modified_time"] = datetime.fromtimestamp(
                path.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            registry.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(registry, full)
    finally:
        os.close(fd)

    _durum_write(note_id, "queued", detail="kuyrukta",
                 rel_path=record["location"]["rel_path"])
    _job_write("process", note_id, record["location"]["rel_path"])
    return content


@bp.post("/api/library/notes/<note_id>/format")
def format_note(note_id: str):
    """Notu LLM ile Markdown'a uygun biçimlendirir ve yeniden işler (A7).

    İçerik DEĞİŞMEZ -- yalnız biçimlendirme. Butonla tetiklenen isteğe bağlı
    bir rafine adımı; hata olursa not olduğu gibi kalır (içerik dosyaya
    yazılmadan önce LLM çıktısı doğrulanır)."""
    try:
        registry = _registry()
    except _ConfigError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        record = next((r for r in json.loads(
            registry.read_text(encoding="utf-8")).get("documents", [])
            if r.get("identity", {}).get("doc_id") == note_id
            and r.get("doc_type") == "NOTE"), None)
    except (OSError, json.JSONDecodeError) as exc:
        return jsonify({"error": f"registry okunamadı: {exc}"}), 503
    if record is None:
        return jsonify({"error": "not bulunamadı"}), 404

    belgeler = _belgeler()
    path = belgeler / record["location"]["rel_path"]
    if not path.is_file():
        return jsonify({"error": "not dosyası diskte yok"}), 404

    current = path.read_text(encoding="utf-8")
    if not current.strip():
        return jsonify({"ok": True, "content": current})

    try:
        formatted = _format_note_text(current)
    except Exception as exc:  # noqa: BLE001 -- LLM/sağlayıcı hatasını 502'ye çevir
        logger.warning("not biçimlendirme başarısız (%s): %s", note_id, exc)
        return jsonify({"error": f"biçimlendirme başarısız: {exc}"}), 502
    if not formatted:
        return jsonify({"error": "biçimlendirme boş döndü"}), 502

    try:
        _persist_note(note_id, formatted)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify({"ok": True, "content": formatted})


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for blok in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(blok)
    return "sha256:" + digest.hexdigest()


__all__ = ["ALLOWED_EXTENSIONS", "bp"]
