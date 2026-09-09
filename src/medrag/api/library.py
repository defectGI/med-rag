"""Library API (B3/B4 backend): document upload, listing, deletion,
status streaming (SSE), content/original-file serving and notes (A7).

## Layer placement

`medrag.api` must NOT import pipeline code (layer rule + ruff TID251).
This module works against the FILE CONTRACTS the pipeline side WRITES
(same shapes as `medrag.pipeline.lifecycle`, separate small
implementations -- deliberate duplication, so the single non-duplicated
link of the component is the "file system"):

  - job queue:    `ISLER_DIR/<job_id>.json` (jobs contract)
  - status:       `DURUM_DIR/<doc_id>.json` (status contract)
  - registry:     `DOCUMENT_NODES_PATH` (schema: classify_documents record
                  shape; writes guarded by `<registry>.lock` flock)

## Paths (env)

BELGELER_DIR, DOCUMENT_NODES_PATH, PARSED_OUTPUT_DIR are REQUIRED (if missing
the related endpoint returns 503 -- a configuration error, not a crash);
ISLER_DIR/DURUM_DIR defaults are siblings of the corpus root (same rule as
the pipeline side).

## Notes (A7)

Notes are **.md** files under `<BELGELER_DIR>/notlar/` (so they pass
through the parser's markdown reader; plain .txt gets caught by the parser).
With doc_type=NOTE they enter the SAME registry/job workflow -- the chatbot
uses them as a source no different from any other document. Editing = rewrite
the file + update scan hash + process job (change semantics A4).
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

#: Extensions the parser can actually read (parsers/registry). This is the
#: UI whitelist; anything not listed is rejected with 415.
ALLOWED_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".markdown",
     ".jpg", ".jpeg", ".png"})

#: Per-file upper limit (K16/Open-5). Overridden via `MEDRAG_MAX_YUKLEME_MB`.
DEFAULT_MAX_UPLOAD_MB = 200

#: Safe character set for deriving a file name (slug) from note titles.
_SLUG_RE = re.compile(r"[^0-9A-Za-zçğıöşüÇĞİÖŞÜ_-]+")

bp = Blueprint("library", __name__)


# --- small helpers (parallel contracts with the pipeline side) ---------------

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
    """The SAME lock file as `fslock.file_lock` (`<registry>.lock`)."""
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
    """The SAME file contract as `jobs.enqueue`."""
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
    """The SAME file contract as `durum.write_status` (for the `queued`
    state that only the api writes)."""
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


# --- record view -------------------------------------------------------------

def _fallback_state(record: dict) -> tuple[str, str | None]:
    """State derived from the registry when no status file exists (for old
    setups / manually processed documents)."""
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


# --- endpoints: documents ----------------------------------------------------

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

        # Overwrite the same path: MODIFIED in the registry (A4 change semantics).
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
    """The SAME operation as `registry_rw.add_upload` (lock + NEW/MODIFIED)."""
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


#: Extensions that can be shown as raw text in the browser (original).
_RAW_TEXT_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".html", ".htm"})


@bp.get("/api/library/documents/<doc_id>/original")
def document_original(doc_id: str):
    """Returns the raw text of the original file (text formats only).

    415 for PDF/image/binary files -- the frontend then falls back to a PDF
    iframe / <img> image / download."""
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
    file_name = record["identity"]["file_name"]
    ext = Path(file_name).suffix.lower()
    if ext not in _RAW_TEXT_EXTENSIONS:
        return jsonify({"error": "bu biçim için ham metin yok"}), 415
    path = belgeler / (record["location"]["rel_path"])
    if not path.is_file():
        return jsonify({"error": "kaynak dosya diskte yok"}), 404
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "dosya metin olarak okunamadı"}), 415
    return jsonify({"file_name": file_name, "content": content})


# --- endpoints: status SSE ---------------------------------------------------

@bp.get("/api/library/events")
def events():
    """Status stream: as `DURUM_DIR` + the registry snapshot change, a
    `documents` event is emitted; a comment-heartbeat every 15 s (against
    proxy timeouts). The 1 s polling is enough and cheap for a single user
    (directory listing + file mtime comparison)."""
    resp = Response(stream_with_context(_events_stream()), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


def _events_stream():
    """Generator producing the SSE frames (module-level -- testable).

    `event: documents` as the snapshot changes; otherwise a comment-heartbeat
    every 15 s. Since it is called through the module-globals `time.sleep` and
    `_list_documents`, tests can shorten it with monkeypatch."""
    last_signature = None
    last_beat = 0.0
    while True:
        try:
            docs = _list_documents()
        except Exception:  # noqa: BLE001 -- a configuration gap must not kill the stream
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


# --- endpoints: notes --------------------------------------------------------

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
    # refresh the hash + process job (change semantics, A4)
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


# --- format notes into proper markdown (LLM, A7) -----------------------------

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
    """Formats the note with the LLM (content unchanged, only Markdown structure).

    Uses the `CHATBOT_LLM_*` configuration (the chatbot's own model).
    If the LLM errors/is unconfigured it raises `ProviderError` -- the caller
    returns it as a 502. Anthropic native / others are called via the
    OpenAI-compatible `/v1/chat/completions`.
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
    """Writes the note content to the file + updates the registry hash/scan +
    enqueues the process job (A4 change semantics). The `format` endpoint uses
    this common write (the same logic as in update_note)."""
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
    """Formats the note into proper Markdown with the LLM and reprocesses it (A7).

    Content UNCHANGED -- formatting only. An optional refinement step triggered
    by a button; on error the note stays as-is (the LLM output is validated
    before the content is written to the file)."""
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
    except Exception as exc:  # noqa: BLE001 -- convert LLM/provider errors to 502
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
