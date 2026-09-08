"""webapp.create_app + route tests. Heavy wiring (retrievers, intent
classifier, answering model) is faked via monkeypatch -- real Qdrant/SQL/LLM
are never touched."""

from __future__ import annotations

import json

import medrag.api.answering_model as answering_model_mod
import medrag.api.factory as factory_mod
import medrag.api.retrieval.modules.intent_classification as intent_mod
import medrag.api.webapp as webapp_mod
from medrag.api.config import Routing
from medrag.api.conversation_log import ConversationLogger
from medrag.api.flows.base import FlowContext
from medrag.api.retrieval.core import IntentLabel, IntentResult, RetrievalResult
from medrag.api.router import Router
from medrag.api.webapp import create_app


class _FakeFlow:
    """`results` is fixed so the image-channel tests (`images` field) can see
    the real metadata carried from `context.results` into
    `SessionState.last_results` (default: empty, old behavior)."""

    def __init__(self, results: list[RetrievalResult] | None = None) -> None:
        self._results = results if results is not None else []

    async def run(self, query: str, on_trace=None) -> FlowContext:
        if on_trace:
            on_trace("flow_ran", query)
        return FlowContext(results=self._results)


class _FakeClassifier:
    async def classify(self, query: str) -> IntentResult:
        return IntentResult(label=IntentLabel.OUT_OF_SCOPE)


class _FakeAnsweringModel:
    async def answer(self, query, context, strategy_prompt, history, **kwargs):
        return f"yanit ({len(history)} onceki mesaj): {query}"


def _patch_wiring(monkeypatch, *, flow_results: list[RetrievalResult] | None = None):
    router = Router(
        {"default_topn": _FakeFlow(flow_results)},
        Routing(default_flow="default_topn", intents={}),
    )
    monkeypatch.setattr(factory_mod, "build_router_from_env", lambda cfg=None: router)
    # Quality-analysis JSON conversation dump (conversation_log.py) is off here --
    # these tests don't exercise that feature and must not leave test debris in
    # the real `logs/conversations/` (see test_conversation_log.py, tested separately).
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=False),
    )
    # Pin the reconciler to None: create_app must not fall into real env/network
    # (L0/L1 skipped; webapp tests exercise the raw-query path).
    monkeypatch.setattr(factory_mod, "build_reconciler_from_env", lambda cfg=None: None)
    # `image_exclude_types` is passed as a kwarg from create_app --
    # the fake must accept it (to match the real call's signature).
    monkeypatch.setattr(
        answering_model_mod, "answering_model_from_env",
        lambda *a, **k: _FakeAnsweringModel(),
    )
    monkeypatch.setattr(intent_mod, "build_default_chat_model", lambda: object())
    monkeypatch.setattr(intent_mod, "LLMIntentClassifier", lambda *a, **k: _FakeClassifier())


def test_index_serves_html(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data


def test_chat_returns_reply_and_sets_session_cookie(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "merhaba"})
    assert resp.status_code == 200
    assert resp.json["reply"] == "yanit (0 onceki mesaj): merhaba"
    assert "chatbot_session" in resp.headers.get("Set-Cookie", "")


def test_chat_empty_message_rejected(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "  "})
    assert resp.status_code == 400


def test_second_message_in_same_session_sees_history(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    first = client.post("/api/chat", json={"message": "ilk"})
    assert first.json["reply"] == "yanit (0 onceki mesaj): ilk"
    second = client.post("/api/chat", json={"message": "ikinci"})
    assert second.json["reply"] == "yanit (2 onceki mesaj): ikinci"


def test_reset_clears_session_history(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    client.post("/api/chat", json={"message": "ilk"})
    client.post("/api/reset")
    second = client.post("/api/chat", json={"message": "ikinci"})
    assert second.json["reply"] == "yanit (0 onceki mesaj): ikinci"


def test_reset_rotates_session_cookie(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    first = client.post("/api/chat", json={"message": "ilk"})
    old = first.headers.get("Set-Cookie", "").split("chatbot_session=")[1].split(";")[0]
    resp = client.post("/api/reset")
    new = resp.headers.get("Set-Cookie", "").split("chatbot_session=")[1].split(";")[0]
    assert new and new != old


# --- /api/chat/history (B5: sohbet geçmişi) ----------------------------------


def test_chat_history_returns_empty_without_cookie(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/chat/history")
    assert resp.status_code == 200
    assert resp.json["messages"] == []


def test_chat_history_returns_stored_turns(monkeypatch, tmp_path):
    _patch_wiring(monkeypatch)
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=True, directory=tmp_path, channel="web"),
    )
    client = create_app().test_client()
    client.post("/api/chat", json={"message": "ilk soru"})
    client.post("/api/chat", json={"message": "ikinci soru"})

    resp = client.get("/api/chat/history")
    assert resp.status_code == 200
    messages = resp.json["messages"]
    assert messages[0] == {"role": "user", "text": "ilk soru"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["text"].startswith("yanit")
    assert messages[2] == {"role": "user", "text": "ikinci soru"}
    assert messages[3]["role"] == "assistant"


def test_chat_history_records_error_turn(monkeypatch, tmp_path):
    _patch_wiring(monkeypatch)
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=True, directory=tmp_path, channel="web"),
    )
    client = create_app().test_client()
    # Bir tur hatayla yazılır (record_turn üzerinden, gerçek akış olmadan).
    logger = ConversationLogger(enabled=True, directory=tmp_path, channel="web")
    from datetime import datetime

    logger.record_turn(
        "sess-x", started_at=datetime.now().astimezone(),
        ended_at=datetime.now().astimezone(), message="soru", trace=[],
        reply=None, error="model patladı",
    )
    # Çerezi elle bu oturuma bağla.
    client.set_cookie("chatbot_session", "sess-x")
    resp = client.get("/api/chat/history")
    messages = resp.json["messages"]
    assert messages == [
        {"role": "user", "text": "soru"},
        {"role": "assistant", "text": "", "chunks": [], "error": True},
    ]


# --- /api/chat/stream (live background panel) ----------------------------------


def test_chat_stream_emits_trace_events_and_final_reply(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "merhaba"})

    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)

    assert "event: step" in body
    assert '"step": "intent"' in body
    assert '"step": "flow_ran"' in body  # on_trace really forwarded to the Flow
    assert '"step": "strategy"' in body
    assert '"step": "answering"' in body
    assert "event: done" in body
    assert '"reply": "yanit (0 onceki mesaj): merhaba"' in body


def test_chat_stream_empty_message_rejected(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "  "})
    assert resp.status_code == 400


def test_chat_stream_sets_session_cookie(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "merhaba"})
    assert "chatbot_session" in resp.headers.get("Set-Cookie", "")


# --- quality analysis: conversation JSON dump (see conversation_log.py) --------


def test_chat_stream_writes_conversation_files_with_full_trace(monkeypatch, tmp_path):
    _patch_wiring(monkeypatch)
    # This test wants the REAL conversation_logger -- the general
    # `_patch_wiring` above disabled it; here we point it at tmp_path and re-enable.
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=True, directory=tmp_path, channel="web"),
    )
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "merhaba"})
    resp.get_data()  # fully consume the generator -- the worker() thread must run to its sentinel
    session_id = resp.headers.get("Set-Cookie", "").split("chatbot_session=")[1].split(";")[0]

    # One folder per channel/conversation.
    conv_dir = tmp_path / "web" / session_id
    assert conv_dir.is_dir()

    record = json.loads((conv_dir / "turns.jsonl").read_text(encoding="utf-8").strip())
    assert record["message"] == "merhaba"
    assert record["reply"] == "yanit (0 onceki mesaj): merhaba"
    assert record["channel"] == "web"
    assert record["turn"] == 1
    steps = [t["step"] for t in record["trace"]]
    assert "intent" in steps
    assert "flow_ran" in steps
    assert "answering" in steps
    assert record["error"] is None

    # Human-readable dump: message + background stages + reply in the same file.
    metin = (conv_dir / "conversation.log").read_text(encoding="utf-8")
    assert "KULLANICI MESAJI" in metin
    assert "merhaba" in metin
    assert "trace[" in metin  # log_trace_step lines landed in this conversation
    assert "TUR 1 BİTTİ" in metin

    meta = json.loads((conv_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["channel"] == "web"
    assert meta["conversation_id"] == session_id
    assert meta["turn_count"] == 1


def test_chat_writes_conversation_files_with_trace(monkeypatch, tmp_path):
    """`/api/chat` (the non-SSE path) also records intermediate stages --
    this path used to write `trace=[]`."""
    _patch_wiring(monkeypatch)
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=True, directory=tmp_path, channel="web"),
    )
    client = create_app().test_client()
    client.post("/api/chat", json={"message": "merhaba"})

    files = list(tmp_path.glob("web/*/turns.jsonl"))
    assert len(files) == 1

    record = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert record["message"] == "merhaba"
    assert [t["step"] for t in record["trace"]]  # no longer EMPTY


def test_second_turn_appends_to_same_conversation_folder(monkeypatch, tmp_path):
    _patch_wiring(monkeypatch)
    monkeypatch.setattr(
        webapp_mod, "conversation_logger_from_config",
        lambda cfg, **kw: ConversationLogger(enabled=True, directory=tmp_path, channel="web"),
    )
    client = create_app().test_client()
    client.post("/api/chat", json={"message": "ilk"})
    client.post("/api/chat", json={"message": "ikinci"})

    dirs = [p for p in (tmp_path / "web").iterdir() if p.is_dir()]
    assert len(dirs) == 1  # same cookie -> same folder
    lines = (dirs[0] / "turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["turn"] for x in lines] == [1, 2]


# --- image channel: `images` field ----------------------------------------------


def test_chat_includes_empty_images_field_by_default(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "merhaba"})
    assert resp.json["images"] == []


def test_chat_exposes_visible_images_alongside_reply_separate_from_text(monkeypatch):
    import medrag.api.config as config_mod

    # The default is now disabled -- this test verifies behavior while enabled;
    # it does NOT change the enablement decision.
    acik_cfg = config_mod.load_config().model_copy(
        update={"visual": config_mod.VisualTuning(enabled=True)}
    )
    monkeypatch.setattr(config_mod, "load_config", lambda override=None: acik_cfg)

    result = RetrievalResult(
        id="c1", score=0.9, text="metin",
        metadata={"images": [{"image_id": "sha256:aaa"}]},
    )
    _patch_wiring(monkeypatch, flow_results=[result])
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "bir görsel var mı"})
    # reply is plain text from the (fake) real model -- image_id never mixes
    # into the TEXT (separate, structured channel).
    assert "sha256:aaa" not in resp.json["reply"]
    assert resp.json["images"] == [{"image_id": "sha256:aaa", "source_node_id": "c1"}]


def test_chat_filters_images_by_configured_exclude_types(monkeypatch):
    import medrag.api.config as config_mod

    real_cfg = config_mod.load_config()
    filtered_cfg = real_cfg.model_copy(
        update={"visual": config_mod.VisualTuning(exclude_types=["chart"])}
    )
    monkeypatch.setattr(config_mod, "load_config", lambda override=None: filtered_cfg)

    results = [
        RetrievalResult(
            id="c1", score=0.9, text="a",
            metadata={"images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]},
        ),
        RetrievalResult(
            id="c2", score=0.8, text="b",
            metadata={"images": [{"image_id": "sha256:bbb", "visual_type": "product_photo"}]},
        ),
    ]
    _patch_wiring(monkeypatch, flow_results=results)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "gorseller"})
    assert resp.json["images"] == [{"image_id": "sha256:bbb", "visual_type": "product_photo", "source_node_id": "c2"}]


def test_chat_hides_images_with_default_config(monkeypatch):
    """The path where `load_config()` is NEVER overridden -- does the real
    default (`enabled=False`) work end to end, without leaning on a fake True?"""
    result = RetrievalResult(
        id="c1", score=0.9, text="metin",
        metadata={"images": [{"image_id": "sha256:aaa"}]},
    )
    _patch_wiring(monkeypatch, flow_results=[result])
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "bir görsel var mı"})
    assert resp.json["images"] == []


def test_chat_hides_all_images_when_visual_disabled(monkeypatch):
    """MASTER SWITCH: with `[visual] enabled = false` the image channel closes
    ENTIRELY. `exclude_types` cannot do this -- it is type-based and SHOWS
    type-less images under an "include when in doubt" policy; the second result
    below is exactly that case. Rationale: without a crop blob store the UI
    printed broken-image icons."""
    import medrag.api.config as config_mod

    real_cfg = config_mod.load_config()
    kapali = real_cfg.model_copy(update={"visual": config_mod.VisualTuning(enabled=False)})
    monkeypatch.setattr(config_mod, "load_config", lambda override=None: kapali)

    results = [
        RetrievalResult(
            id="c1", score=0.9, text="a",
            metadata={"images": [{"image_id": "sha256:aaa", "visual_type": "chart"}]},
        ),
        RetrievalResult(
            id="c2", score=0.8, text="b",
            metadata={"images": [{"image_id": "sha256:bbb"}]},  # tipi YOK
        ),
    ]
    _patch_wiring(monkeypatch, flow_results=results)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "gorseller"})
    assert resp.json["images"] == []


def test_visual_enabled_defaults_false(monkeypatch):
    """Default is DISABLED (the crop blob store is missing in this deployment,
    so the UI printed broken-image icons). Every environment that clones/pulls
    the repo inherits this behavior; to enable, set it explicitly to true in
    config."""
    import medrag.api.config as config_mod

    assert config_mod.load_config().visual.enabled is False


def test_chat_stream_done_event_includes_images(monkeypatch):
    import medrag.api.config as config_mod

    acik_cfg = config_mod.load_config().model_copy(
        update={"visual": config_mod.VisualTuning(enabled=True)}
    )
    monkeypatch.setattr(config_mod, "load_config", lambda override=None: acik_cfg)

    result = RetrievalResult(
        id="c1", score=0.9, text="metin",
        metadata={"images": [{"image_id": "sha256:aaa"}]},
    )
    _patch_wiring(monkeypatch, flow_results=[result])
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "merhaba"})
    body = resp.get_data(as_text=True)
    assert '"images": [{"image_id": "sha256:aaa", "source_node_id": "c1"}]' in body


# --- /api/images/<image_id> (architectural exception) ---------------------------


def test_get_image_503_when_storage_dir_unconfigured(monkeypatch):
    monkeypatch.delenv("CHATBOT_IMAGE_STORAGE_DIR", raising=False)
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/images/sha256:" + "a" * 64)
    assert resp.status_code == 503


def test_get_image_404_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATBOT_IMAGE_STORAGE_DIR", str(tmp_path))
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/images/sha256:" + "b" * 64)
    assert resp.status_code == 404


def test_get_image_404_on_malformed_image_id_path_traversal_guard(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATBOT_IMAGE_STORAGE_DIR", str(tmp_path))
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/images/../../etc/passwd")
    assert resp.status_code == 404


def test_get_image_200_serves_actual_crop_file(monkeypatch, tmp_path):
    image_hex = "c" * 64
    crop = tmp_path / f"{image_hex}.png"
    crop.write_bytes(b"\x89PNG\r\n\x1a\nfake-crop-bytes")
    monkeypatch.setenv("CHATBOT_IMAGE_STORAGE_DIR", str(tmp_path))
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get(f"/api/images/sha256:{image_hex}")
    assert resp.status_code == 200
    assert resp.data == b"\x89PNG\r\n\x1a\nfake-crop-bytes"


def test_get_image_serves_when_storage_dir_is_relative(monkeypatch, tmp_path):
    """A RELATIVE `CHATBOT_IMAGE_STORAGE_DIR`: glob resolved against CWD while
    `flask.send_file` resolved against `app.root_path`, so the file was FOUND
    but blew up while being served. `resolve_storage_root` pins both to CWD."""
    image_hex = "d" * 64
    crops = tmp_path / "crops"
    crops.mkdir()
    (crops / f"{image_hex}.png").write_bytes(b"\x89PNG\r\n\x1a\nrelative")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHATBOT_IMAGE_STORAGE_DIR", "crops")
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get(f"/api/images/sha256:{image_hex}")
    assert resp.status_code == 200
    assert resp.data == b"\x89PNG\r\n\x1a\nrelative"


# --- document channel: `documents` field -----------------------------------------


def test_chat_includes_empty_documents_field_by_default(monkeypatch):
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "merhaba"})
    assert resp.json["documents"] == []


def test_chat_exposes_requested_documents_alongside_reply_separate_from_text(monkeypatch):
    result = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
                   "file_name": "PN5106_Datasheet.pdf", "requested": True},
    )
    _patch_wiring(monkeypatch, flow_results=[result])
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "PN1309 datasheet gönder"})
    # reply is plain text from the (fake) real model -- doc_id never mixes
    # into the TEXT (separate structured channel; the real file path is NEVER visible).
    assert "d1" not in resp.json["reply"]
    assert resp.json["documents"] == [
        {"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET", "file_name": "PN5106_Datasheet.pdf"}
    ]


def test_chat_documents_field_excludes_not_requested_related_documents(monkeypatch):
    requested = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
                   "file_name": "a.pdf", "requested": True},
    )
    other = RetrievalResult(
        id="doc:d2:PN1309", score=0.5,
        metadata={"doc_id": "d2", "model": "PN1309", "doc_type": "BROCHURE",
                   "file_name": "b.pdf", "requested": False},
    )
    _patch_wiring(monkeypatch, flow_results=[requested, other])
    client = create_app().test_client()
    resp = client.post("/api/chat", json={"message": "PN1309 datasheet gönder"})
    assert [d["doc_id"] for d in resp.json["documents"]] == ["d1"]


def test_chat_stream_done_event_includes_documents(monkeypatch):
    result = RetrievalResult(
        id="doc:d1:PN1309", score=1.0,
        metadata={"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET",
                   "file_name": "a.pdf", "requested": True},
    )
    _patch_wiring(monkeypatch, flow_results=[result])
    client = create_app().test_client()
    resp = client.post("/api/chat/stream", json={"message": "merhaba"})
    body = resp.get_data(as_text=True)
    assert '"documents": [{"doc_id": "d1", "model": "PN1309", "doc_type": "DATASHEET", "file_name": "a.pdf"}]' in body


# --- /api/documents/<doc_id> -------------------------------------------------------


def test_get_document_503_when_no_db_configured_or_bundled(monkeypatch, tmp_path):
    monkeypatch.delenv("CHATBOT_DB_QUERY_DB_PATH", raising=False)
    monkeypatch.setattr(factory_mod, "_BUNDLED_DB_PATH", tmp_path / "yok.db")
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/documents/d1")
    assert resp.status_code == 503


def test_get_document_404_when_doc_id_unknown(monkeypatch, tmp_path):
    db_path = tmp_path / "specs.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE document (doc_id TEXT PRIMARY KEY, doc_type TEXT, "
        "file_name TEXT, file_path TEXT, is_active INTEGER)"
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", str(db_path))
    _patch_wiring(monkeypatch)
    client = create_app().test_client()
    resp = client.get("/api/documents/not-a-real-id")
    assert resp.status_code == 404


def test_get_document_200_serves_actual_file_and_never_leaks_source_path(monkeypatch, tmp_path):
    import sqlite3

    real_file = tmp_path / "PN5106_Datasheet.pdf"
    real_file.write_bytes(b"%PDF-1.4 fake-bytes")
    db_path = tmp_path / "specs.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE document (doc_id TEXT PRIMARY KEY, doc_type TEXT, "
        "file_name TEXT, file_path TEXT, is_active INTEGER)"
    )
    conn.execute(
        "INSERT INTO document VALUES (?, ?, ?, ?, ?)",
        ("d1", "DATASHEET", "PN5106_Datasheet.pdf", str(real_file), 1),
    )
    conn.commit()
    conn.close()
    monkeypatch.setenv("CHATBOT_DB_QUERY_DB_PATH", str(db_path))
    _patch_wiring(monkeypatch)
    client = create_app().test_client()

    resp = client.get("/api/documents/d1")

    assert resp.status_code == 200
    assert resp.data == b"%PDF-1.4 fake-bytes"
    # Content-Disposition contains only `file_name` -- the real disk path
    # (tmp_path itself) never appears in any header.
    disposition = resp.headers.get("Content-Disposition", "")
    assert "PN5106_Datasheet.pdf" in disposition
    assert str(tmp_path) not in disposition
