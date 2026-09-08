"""A simple, plain web chat UI: one page (vanilla HTML/CSS/JS) + API routes.
Conversation memory is RAM-only (`memory.ConversationMemory`) -- no
persistence/logging of chat state is kept; a server restart resets all
sessions, on purpose.

Background stages stream LIVE into the right panel: which intent was found,
which SQL was generated, the chunks returned from top_n -- `POST
/api/chat/stream` returns a Server-Sent Events stream
(`text/event-stream`); bridging Flask's WSGI (sync) server with the async
`Orchestrator.handle` uses a background thread + `queue.Queue` (see
`_stream_response`). This trace NEVER goes to the answering model (the
"flow internal steps don't leak to the model" rule in orchestrator.py holds
here too) -- it goes to the browser only.

Run:

    python -m medrag.api.webapp       # http://127.0.0.1:8507

Fill `.env` first (see .env.example): QDRANT_*/EMBEDDING_* (top_n),
CHATBOT_DB_QUERY_DB_PATH + LINKING_*/SQL_*/OPENAI_API_KEY (sql_topn),
CHATBOT_LLM_* (the answering model), LLM_* in retrieval/.env
(intent_classification + the model the reconciler shares by default;
separately if wanted via RECONCILER_*). If LLM_* is missing entirely, L0/L1
(reconciler) is skipped.

`GET /api/documents/<doc_id>` does NOT invent a SEPARATE env -- it reuses the
SAME `CHATBOT_DB_QUERY_DB_PATH` (+ committed bundled fallback) as the
`doc_download` flow's `document` table query, via
`factory.py::resolve_db_path_from_env` (see the `document_store.py` module
docstring).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    request,
    send_file,
    stream_with_context,
)

from medrag.api.auth import configure_auth
from medrag.api.conversation_log import CHANNEL_WEB, conversation_logger_from_config
from medrag.api.document_store import resolve_document_download
from medrag.api.image_store import resolve_image_path, resolve_storage_root
from medrag.api.library import bp as library_bp
from medrag.api.logging_setup import configure_logging_from_cfg, log_trace_step

logger = logging.getLogger("medrag.api.webapp")

_SESSION_COOKIE = "chatbot_session"

_TEMPLATES_DIR = Path(__file__).parent / "templates"

#: SPA derlemesi (`web/dist`). `MEDRAG_WEB_DIST` ile aşılabilir; bulunamazsa
#: eski şablon arayüzü sunulur (geliştirme ortamı için geri dönüş).
_DIST_ENV = "MEDRAG_WEB_DIST"


def _dist_dir() -> Path | None:
    raw = (os.environ.get(_DIST_ENV) or "").strip()
    candidate = Path(raw) if raw else Path(__file__).resolve().parents[3] / "web" / "dist"
    return candidate if (candidate / "index.html").is_file() else None


@lru_cache(maxsize=1)
def _load_index_html() -> str:
    """Single-page HTML/CSS/JS body of the chat UI -- read from the template
    file (so editing the template inside a container doesn't require rebuilding
    the image). The content is static (no formatting/interpolation), so it's
    cached on first call."""
    return (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


def _sse_event(kind: str, payload: dict) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app() -> Flask:
    from medrag.api import factory
    from medrag.api.config import load_config

    app = Flask(__name__, static_folder=None)
    cfg = load_config()
    # main() already sets this up; but if create_app is called directly from a
    # WSGI server (gunicorn/uwsgi), main never runs -> configure here too
    # (idempotent) so that path is logged as well.
    configure_logging_from_cfg(cfg)

    # Tek kullanıcı şifresi (K3) + kütüphane API'si (A1/A5/A7): SPA ve
    # /api/chat uçları aynı uygulamada, tek origin'de yaşar.
    configure_auth(app)
    app.register_blueprint(library_bp)
    # Detailed per-conversation logging -- a channel SEPARATE from [logging],
    # see conversation_log.py. `channel="web"`: records land under
    # logs/conversations/web/<session_id>/ (the WhatsApp worker calls the SAME
    # factory with `channel="whatsapp"`).
    conversation_logger = conversation_logger_from_config(cfg, channel=CHANNEL_WEB)
    logger.info(
        "create_app: routing=%s flow(top_n_k=%d, sql_k=%d, sql_topn_k=%d) sql_concurrency=%d",
        cfg.routing.default_flow, cfg.flow.top_n_k, cfg.flow.sql_k, cfg.flow.sql_topn_k,
        cfg.session.sql_concurrency,
    )
    # Report image serving AT STARTUP: a missing/wrong `CHATBOT_IMAGE_STORAGE_DIR`
    # was previously noticed only as a broken thumbnail in the browser -- now
    # it's visible in the first log lines.
    _img_dir = (os.getenv("CHATBOT_IMAGE_STORAGE_DIR") or "").strip()
    if not _img_dir:
        logger.warning(
            "görsel serve KAPALI: CHATBOT_IMAGE_STORAGE_DIR tanımsız -> "
            "/api/images/<image_id> 503 döner (cevaplardaki görseller görünmez)"
        )
    else:
        _img_root = resolve_storage_root(_img_dir)
        if _img_root.is_dir():
            logger.info("görsel serve açık: %s (%d dosya)", _img_root, len(list(_img_root.glob("*.*"))))
        else:
            logger.warning("görsel serve YAPILANDIRMASI HATALI: %s bir dizin değil", _img_root)

    # Orchestrator/memory/session_state wiring now lives in `factory.py`
    # (shared by the webapp and the WhatsApp worker -- no code duplication);
    # see the `factory.build_orchestrator_from_env` docstring.
    bundle = factory.build_orchestrator_from_env(cfg)
    orchestrator = bundle.orchestrator
    memory = bundle.memory
    session_state = bundle.session_state

    def _visible_images_for_session(session_id: str) -> list[dict]:
        """Extracts the visible image list from this turn's `last_results` in
        `session_state` (the Orchestrator overwrites it every turn, see
        session_state.py) -- filtered by `cfg.visual.exclude_types`."""
        from medrag.api.answering_model import extract_visible_images

        if not cfg.visual.enabled:
            return []
        state = session_state.get(session_id)
        return extract_visible_images(state.last_results, cfg.visual.exclude_types)

    def _evidence_chunks_for_session(session_id: str) -> list[dict]:
        """Extracts the list of evidence chunks the answer is BASED on from
        this turn's `last_results` in `session_state` -- same principle as
        `_visible_images_for_session`/`_visible_documents_for_session` (see the
        answering_model.py::extract_evidence_chunks docstring).

        [SQL] rows are ALSO enriched with an `evidence_chunks` field:
        specs.db keeps the source of a spec value only as a POINTER, the chunk
        TEXT lives in the chunker's output --
        `sql_evidence.attach_sql_evidence` joins the two and puts the text +
        `document.file_path` into the panel. Enrichment is an ADDITION: if the
        DB/corpus can't be read the field stays empty, the panel falls back to
        its old (column/value) rendering, the turn does NOT drop."""
        from medrag.api.answering_model import extract_evidence_chunks

        state = session_state.get(session_id)
        chunks = extract_evidence_chunks(state.last_results)
        if not cfg.evidence.sql_chunks:
            return chunks
        try:
            from medrag.api.sql_evidence import attach_sql_evidence

            return attach_sql_evidence(
                chunks,
                state.last_results,
                factory.resolve_db_path_from_env(),
                max_chunks=cfg.evidence.max_chunks_per_row,
                snippet_len=cfg.evidence.snippet_len,
            )
        except Exception:
            logger.warning("SQL kanıt chunk'ları çözülemedi", exc_info=True)
            return chunks

    def _visible_documents_for_session(session_id: str) -> list[dict]:
        """Extracts from this turn's `last_results` in `session_state` (the
        Orchestrator overwrites it every turn) the files the `doc_download`
        flow explicitly requested/sent for the user THIS TURN -- same principle
        as `_visible_images_for_session`; the `documents` counterpart of the
        `images` channel. `source_path` never enters this list (see
        answering_model.py::extract_visible_documents)."""
        from medrag.api.answering_model import extract_visible_documents

        state = session_state.get(session_id)
        return extract_visible_documents(state.last_results)

    @app.get("/")
    def index():
        dist = _dist_dir()
        if dist is not None:
            return send_file(dist / "index.html")
        return _load_index_html()

    @app.get("/assets/<path:filename>")
    def spa_assets(filename: str):
        dist = _dist_dir()
        if dist is None:
            return jsonify({"error": "SPA derlemesi yok"}), 404
        return send_file(dist / "assets" / filename)

    @app.get("/vite.svg")
    def spa_favicon():
        dist = _dist_dir()
        if dist is not None and (dist / "vite.svg").is_file():
            return send_file(dist / "vite.svg")
        return "", 404

    @app.get("/<path:subpath>")
    def spa_fallback(subpath: str):
        """SPA catch-all: bilinen dosya uzantıları 404 olur (yanlış asset
        yolu), diğerleri index.html'e düşer (istemci tarafı gezinme)."""
        dist = _dist_dir()
        if dist is None or "." in Path(subpath).name:
            return jsonify({"error": "bulunamadı"}), 404
        return send_file(dist / "index.html")

    @app.post("/api/chat")
    def chat():
        """One-shot (non-SSE) reply -- for programmatic/simple use (see README
        "test without the web UI"). The UI now uses `/api/chat/stream`.
        Background stages are NOT published to the CLIENT but they ARE
        RECORDED (conversation folder + chatbot.log) -- this path used to write
        `trace=[]`.

        `images` + `documents`: structured fields BESIDE `reply`, never mixed
        into the text (see
        answering_model.py::extract_visible_images/extract_visible_documents).
        This turn's `context.results` is read from `session_state` after the
        orchestrator returns (`SessionState.last_results`; the Orchestrator
        already overwrites it EVERY turn) -- without touching the Orchestrator/
        AnsweringModel `str` return contract at all. `documents` carries only
        `doc_id`/`model`/`doc_type`/`file_name` -- the real download always
        goes through `GET /api/documents/<doc_id>`."""
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message boş olamaz"}), 400

        session_id = request.cookies.get(_SESSION_COOKIE) or str(uuid.uuid4())
        # Same recording path as `/api/chat/stream`: background log lines that
        # flow during the turn also land in this session's own folder -- the
        # only difference is that the trace is NOT PUBLISHED to the browser,
        # not that it isn't recorded (`on_trace` is attached here too).
        req_tag = f"{session_id[:8]}: "
        with conversation_logger.turn(session_id, message=message) as turn_log:
            def on_trace(step: str, trace_data) -> None:
                log_trace_step(logger, step, trace_data, prefix=req_tag)
                turn_log.record_trace(step, trace_data)

            reply: str | None = None
            images: list[dict] = []
            documents: list[dict] = []
            chunks: list[dict] = []
            try:
                reply = asyncio.run(orchestrator.handle(session_id, message, on_trace))
                images = _visible_images_for_session(session_id)
                documents = _visible_documents_for_session(session_id)
                chunks = _evidence_chunks_for_session(session_id)
            finally:
                turn_log.reply = reply
                turn_log.images = images
                turn_log.documents = documents
                turn_log.chunks = chunks

        resp = make_response(jsonify({"reply": reply, "images": images, "documents": documents, "chunks": chunks}))
        resp.set_cookie(_SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return resp

    @app.post("/api/chat/stream")
    def chat_stream():
        """SSE endpoint that publishes background stages (intent, SQL, top_n
        chunks, ...) live -- the data source of the webapp's right panel.

        Flask's WSGI (sync) view function can't run the async
        `Orchestrator.handle` directly; a background thread starts its own
        event loop (`asyncio.run`) and writes events into a `queue.Queue` via
        `on_trace`, while this view (a sync generator) reads from the queue and
        publishes SSE lines. `queue.Queue` is thread-safe, so there's no
        synchronization problem.
        """
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message boş olamaz"}), 400

        session_id = request.cookies.get(_SESSION_COOKIE) or str(uuid.uuid4())
        events: queue.Queue = queue.Queue()
        # Short session suffix -> to grep the log lines of the same request
        # (the full session_id cookie shouldn't go into logs; first 8 chars
        # suffice).
        req_tag = f"{session_id[:8]}: "

        def worker() -> None:
            try:
                _run_turn()
            finally:
                # Sentinel LAST: the `turn()` block writes the record files on
                # exit; putting the sentinel INSIDE the block would open a race
                # where the file might still be unwritten after the stream
                # closed (tests read the file immediately after consuming the
                # response).
                events.put((None, None))

        def _run_turn() -> None:
            # The conversation record is opened ON THIS THREAD: `turn()` writes
            # the active conversation into a ContextVar and every log line
            # produced during the turn (including `asyncio.to_thread` LLM calls
            # running in the same context) lands in this session's own folder
            # -- had it been opened on the request thread, the worker thread
            # wouldn't see that context.
            with conversation_logger.turn(session_id, message=message) as turn_log:
                logger.info("istek [%s] mesaj=%r", session_id[:8], message)

                def on_trace(step: str, trace_data) -> None:
                    # Single choke point: EVERY background stage going to the
                    # browser panel (intent, SQL generation,
                    # sql_error+raw_text, chunks) is written to the log files
                    # at the same time -- the panel is ephemeral, the log is
                    # persistent.
                    log_trace_step(logger, step, trace_data, prefix=req_tag)
                    turn_log.record_trace(step, trace_data)
                    events.put(("step", {"step": step, "data": trace_data}))

                reply = None
                try:
                    reply = asyncio.run(orchestrator.handle(session_id, message, on_trace))
                    turn_log.reply = reply
                    turn_log.images = _visible_images_for_session(session_id)
                    turn_log.documents = _visible_documents_for_session(session_id)
                    turn_log.chunks = _evidence_chunks_for_session(session_id)
                    logger.info("istek [%s] tamam (cevap %d karakter)", session_id[:8], len(reply or ""))
                    events.put(("done", {
                        "reply": reply, "images": turn_log.images,
                        "documents": turn_log.documents, "chunks": turn_log.chunks,
                    }))
                except Exception as exc:
                    # `logger.exception` writes the FULL traceback to disk
                    # (both chatbot.log and this conversation's file); the
                    # client gets only the short message.
                    turn_log.error = str(exc)
                    turn_log.reply = reply
                    logger.exception("istek [%s] BAŞARISIZ: %s", session_id[:8], exc)  # noqa: TRY401 -- the traceback already goes to the file; the summary in the message is kept one-line on purpose for grep
                    events.put(("error", {"error": str(exc)}))

        threading.Thread(target=worker, daemon=True).start()

        def generate() -> Generator[str, None, None]:
            while True:
                kind, payload = events.get()
                if kind is None:
                    break
                yield _sse_event(kind, payload)

        resp = Response(stream_with_context(generate()), mimetype="text/event-stream")
        resp.headers["Cache-Control"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"  # disable reverse-proxy buffering
        resp.set_cookie(_SESSION_COOKIE, session_id, httponly=True, samesite="Lax")
        return resp

    @app.post("/api/reset")
    def reset():
        """"Yeni sohbet": RAM hafızasını ve oturum durumunu temizler ve
        oturum çerezini YENİLER -- böylece eski turlar kalıcı günlükte
        (conversation_log) denetim izi olarak DURUR ama yeni sohbet boş
        başlar (sayfa yenilense bile geçmiş geri gelmez)."""
        session_id = request.cookies.get(_SESSION_COOKIE)
        if session_id:
            memory.clear(session_id)
            session_state.clear(session_id)  # so the ongoing goal resets too
        resp = make_response(jsonify({"ok": True}))
        resp.set_cookie(_SESSION_COOKIE, str(uuid.uuid4()), httponly=True, samesite="Lax")
        return resp

    @app.get("/api/chat/history")
    def chat_history():
        """Sohbet geçmişi (B5): `chatbot_session` çerezinden oturum çözülür,
        `conversation_log` günlüğünden turlar okunur ve SPA'nın beklediği
        mesaj listesine çevrilir. Çerez yoksa / günlük yoksa boş liste --
        kayıt bozuksa o tur atlanır, diğerleri döner."""
        session_id = request.cookies.get(_SESSION_COOKIE)
        if not session_id:
            return jsonify({"messages": []})
        messages: list[dict] = []
        for turn in conversation_logger.read_history(session_id):
            messages.append({"role": "user", "text": turn.get("message") or ""})
            reply = turn.get("reply")
            error = turn.get("error")
            if reply is not None or error:
                messages.append({
                    "role": "assistant",
                    "text": reply if reply is not None else "",
                    "chunks": turn.get("chunks") or [],
                    "error": bool(error),
                })
        return jsonify({"messages": messages})

    @app.get("/api/images/<path:image_id>")
    def get_image(image_id: str):
        """Resolves the `image_id` (parser's `sha256:<hex>` format) from the
        `images` channel (the `images` field of `/api/chat` + `/api/chat/
        stream` responses) to the real crop file on disk.

        DELIBERATE ARCHITECTURAL EXCEPTION: `chatbot` reaches parser's storage
        DIRECTLY here via `CHATBOT_IMAGE_STORAGE_DIR` (see .env.example) -- no
        `parser`/`chunker`/`vectorize` CODE is imported, only the filesystem is
        shared (the resolution logic in `image_store.py` MIRRORs parser's
        naming rule, it doesn't import it).

        503 when `CHATBOT_IMAGE_STORAGE_DIR` is undefined (missing
        configuration; images aren't served in this deployment); 404 when the
        `image_id` can't be resolved / the file is missing.

        Every failure is ALSO logged -- a broken `<img>` in the browser
        couldn't distinguish 503/404/wrong-directory; now `chatbot.log` says
        which directory was searched for what.

        `CHATBOT_IMAGE_STORAGE_DIR` is not re-read here -- it uses `_img_dir`
        read once at the top of `create_app()` (via the closure)."""
        storage_dir = _img_dir
        if not storage_dir:
            logger.warning(
                "görsel isteği [%s] reddedildi: CHATBOT_IMAGE_STORAGE_DIR tanımsız "
                "(chatbot/.env'e parser'ın crop dizininin MUTLAK yolunu yaz)", image_id,
            )
            return jsonify({"error": "CHATBOT_IMAGE_STORAGE_DIR tanımsız"}), 503
        # Absolutization is for send_file: a relative value resolved against
        # CWD in the glob but against app.root_path in send_file (see
        # resolve_storage_root).
        root = resolve_storage_root(storage_dir)
        path = resolve_image_path(root, image_id)
        if path is None:
            if not root.is_dir():
                logger.warning(
                    "görsel isteği [%s]: CHATBOT_IMAGE_STORAGE_DIR bir dizin değil: %s",
                    image_id, root,
                )
            else:
                logger.warning("görsel isteği [%s]: %s altında dosya yok", image_id, root)
            return jsonify({"error": "görsel bulunamadı"}), 404
        return send_file(path)

    @app.get("/api/documents/<doc_id>")
    def get_document(doc_id: str):
        """Resolves the `doc_id` from the `documents` channel (the `documents`
        field of `/api/chat` + `/api/chat/stream` responses) to the real file
        on disk and downloads it -- `file_name` is used as the HTTP download
        name, and the real `source_path` (the single show-forbidden column of
        the `document` table, see facts/db/DATA_DICTIONARY.md) appears in no
        response body/header.

        DIFFERENT from the `/api/images/<image_id>` architectural exception:
        there, reaching parser's SEPARATE store (`CHATBOT_IMAGE_STORAGE_DIR`)
        required a new env; here `document.source_path` already stores the full
        path inside chatbot's OWN `specs.db` (the same
        `CHATBOT_DB_QUERY_DB_PATH` sql_topn uses) -- NO SECOND
        storage-access env INVENTED; `factory.py::resolve_db_path_from_env` is
        reused AS-IS (see the document_store.py module docstring). 404 when
        the row/file is missing (same behavior as the image endpoint -- there
        is no configuration-missing 503 case here because the committed
        facts/db/specs.db always falls back to a DB)."""
        from medrag.api.factory import resolve_db_path_from_env

        try:
            db_path = resolve_db_path_from_env()
        except Exception as exc:  # noqa: BLE001 - DbQueryError, missing config
            return jsonify({"error": f"veritabanı yapılandırması eksik: {exc}"}), 503
        download = resolve_document_download(db_path, doc_id)
        if download is None:
            return jsonify({"error": "belge bulunamadı"}), 404
        return send_file(download.path, download_name=download.file_name)

    return app


def main() -> int:
    import os

    from medrag.api.config import load_config
    from medrag.api.envfile import load_component_env
    from medrag.api.preflight import run_startup_preflight

    load_component_env()

    # Set up logging FIRST (so preflight + startup also reach the disk).
    # create_app will call it again idempotently; we set it up early here so
    # startup/preflight output isn't lost.
    log_file = configure_logging_from_cfg(load_config())
    logger.info("=== chatbot webapp başlıyor === log dosyası: %s", log_file)
    print(f"[log] Arka plan + SQL logları buraya yazılıyor: {log_file}")

    # Known-answer probe for every LLM role (answering/intent/linking/sql):
    # health + is thinking at the desired setting, printed to the terminal.
    # Default is warn+continue; if CHATBOT_PREFLIGHT_STRICT is truthy, a
    # failure aborts instead.
    ok = run_startup_preflight(os.environ)
    logger.info("preflight sonucu: %s", "OK" if ok else "BAŞARISIZ (uyarı)")
    if not ok and (os.getenv("CHATBOT_PREFLIGHT_STRICT") or "").strip().lower() in ("1", "true", "yes", "on"):
        logger.error("preflight başarısız + CHATBOT_PREFLIGHT_STRICT açık -> çıkılıyor")
        return 1

    app = create_app()
    # Port: the real deployment listens on 8507 (the old 5000 forward is
    # dead/unused). Since it's a connection setting it's overridable from env
    # (root CONFIG.md taxonomy: connection -> .env, same pattern as
    # `WA_BOT_PORT`), and the default now ALIGNS with the real port. `host`
    # deliberately stays 127.0.0.1 -- exposing it (0.0.0.0) is a separate
    # decision, never widened silently. Component isolation is preserved: this
    # doesn't move to a central core/config -- the bind address is read ONLY
    # here, at the process entry point.
    host = (os.getenv("CHATBOT_WEB_HOST") or "127.0.0.1").strip()
    port = int((os.getenv("CHATBOT_WEB_PORT") or "8507").strip())
    logger.info("webapp dinliyor: http://%s:%d", host, port)
    app.run(host=host, port=port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
