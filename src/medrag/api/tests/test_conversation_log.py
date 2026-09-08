"""Per-conversation detailed-logging tests (see conversation_log.py): one
folder per channel/conversation, one JSON line per turn + a text dump,
background log capture (ContextVar), disable/no-op, safe folder names. No
network; nothing is written to the real `logs/` directory (all
`tmp_path`)."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from medrag.api.conversation_log import (
    CHANNEL_WHATSAPP,
    ConversationLogger,
    conversation_logger_from_config,
)


def _turn(logger: ConversationLogger, conversation_id: str, **kwargs) -> None:
    started = datetime(2026, 7, 27, 12, 0, 0).astimezone()
    ended = datetime(2026, 7, 27, 12, 0, 1).astimezone()
    logger.record_turn(
        conversation_id, started_at=started, ended_at=ended,
        message=kwargs.pop("message", "merhaba"),
        trace=kwargs.pop("trace", []),
        reply=kwargs.pop("reply", "selam"),
        **kwargs,
    )


def _turns(directory: Path) -> list[dict]:
    text = (directory / "turns.jsonl").read_text(encoding="utf-8").strip()
    return [json.loads(line) for line in text.splitlines()]


# --- folder layout ------------------------------------------------------------


def test_creates_one_folder_per_conversation(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "session-a")
    _turn(logger, "session-b")
    assert (tmp_path / "web" / "session-a" / "turns.jsonl").exists()
    assert (tmp_path / "web" / "session-b" / "turns.jsonl").exists()


def test_whatsapp_channel_uses_phone_number_folder(tmp_path: Path):
    """Requirement: each new phone number opening a conversation gets its own
    folder, and its logs go inside it."""
    logger = ConversationLogger(enabled=True, directory=tmp_path, channel=CHANNEL_WHATSAPP)
    _turn(logger, "905551112233", message="fiyat?")
    conv_dir = tmp_path / "whatsapp" / "905551112233"
    assert conv_dir.is_dir()
    assert _turns(conv_dir)[0]["channel"] == "whatsapp"
    assert _turns(conv_dir)[0]["conversation_id"] == "905551112233"


def test_channels_are_isolated_from_each_other(tmp_path: Path):
    web = ConversationLogger(enabled=True, directory=tmp_path, channel="web")
    wa = ConversationLogger(enabled=True, directory=tmp_path, channel=CHANNEL_WHATSAPP)
    _turn(web, "123")
    _turn(wa, "123")
    assert (tmp_path / "web" / "123").is_dir()
    assert (tmp_path / "whatsapp" / "123").is_dir()


def test_meta_json_tracks_first_seen_and_turn_count(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path, channel=CHANNEL_WHATSAPP)
    _turn(logger, "905551112233", message="bir")
    _turn(logger, "905551112233", message="iki")
    meta = json.loads((tmp_path / "whatsapp" / "905551112233" / "meta.json").read_text(encoding="utf-8"))
    assert meta["channel"] == "whatsapp"
    assert meta["conversation_id"] == "905551112233"
    assert meta["turn_count"] == 2
    assert meta["first_seen_at"] <= meta["last_seen_at"]


# --- structured logging --------------------------------------------------------


def test_appends_one_json_line_per_turn(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "s1", message="ilk soru", reply="ilk cevap")
    _turn(logger, "s1", message="ikinci soru", reply="ikinci cevap")
    records = _turns(tmp_path / "web" / "s1")
    assert [r["message"] for r in records] == ["ilk soru", "ikinci soru"]
    assert [r["turn"] for r in records] == [1, 2]
    assert records[0]["reply"] == "ilk cevap"


def test_record_includes_trace_images_documents_and_timing(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(
        logger, "s1",
        trace=[{"step": "intent", "data": "product_fact", "ts": "2026-07-27T12:00:00+03:00"}],
        images=[{"image_id": "sha256:abc"}],
        documents=[{"doc_id": "d1", "file_name": "x.pdf"}],
    )
    record = _turns(tmp_path / "web" / "s1")[0]
    assert record["trace"][0]["step"] == "intent"
    assert record["images"] == [{"image_id": "sha256:abc"}]
    assert record["documents"] == [{"doc_id": "d1", "file_name": "x.pdf"}]
    assert record["duration_seconds"] == 1.0
    assert record["error"] is None
    # Local ISO-8601 with a UTC offset.
    assert "+" in record["started_at"] or record["started_at"].endswith("Z") is False


def test_error_is_recorded(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "s1", reply=None, error="boom")
    record = _turns(tmp_path / "web" / "s1")[0]
    assert record["error"] == "boom"
    assert record["reply"] is None


def test_session_id_is_sanitized_for_folder_name(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "../../etc/passwd")
    dirs = [p for p in (tmp_path / "web").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    assert ".." not in dirs[0].name
    assert "/" not in dirs[0].name


# --- turn(): text dump + background capture ------------------------------------


def test_turn_writes_message_reply_and_captures_background_logs(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path, channel=CHANNEL_WHATSAPP)
    log = logging.getLogger("medrag.api.test_conversation_log")
    log.setLevel(logging.INFO)

    with logger.turn("905551112233", message="PN1253 fiyatı?") as turn_log:
        log.info("arka planda bir sey oldu")
        turn_log.record_trace("intent", "product_fact")
        turn_log.note("gateway'e gönderildi")
        turn_log.reply = "cevap"

    conv_dir = tmp_path / "whatsapp" / "905551112233"
    metin = (conv_dir / "conversation.log").read_text(encoding="utf-8")
    assert "PN1253 fiyatı?" in metin
    assert "arka planda bir sey oldu" in metin  # captured from the root logger
    assert "gateway'e gönderildi" in metin
    assert "cevap" in metin

    record = _turns(conv_dir)[0]
    assert record["trace"][0]["step"] == "intent"
    assert record["notes"] == ["gateway'e gönderildi"]
    assert record["reply"] == "cevap"


def test_text2sql_line_is_written_once_even_when_propagating(tmp_path: Path):
    """The handler sits on BOTH the root and the `text2sql` logger; if
    `text2sql` propagates (default True before logging_setup is configured)
    the same record would appear twice -- the line must be written exactly
    ONCE."""
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    t2s = logging.getLogger("text2sql.linking")
    t2s.setLevel(logging.INFO)
    parent = logging.getLogger("text2sql")
    onceki = parent.propagate
    parent.propagate = True
    try:
        with logger.turn("s1", message="soru"):
            t2s.info("linking-satiri")
    finally:
        # Restore global logging state -- don't leak into other tests.
        parent.propagate = onceki

    metin = (tmp_path / "web" / "s1" / "conversation.log").read_text(encoding="utf-8")
    assert metin.count("linking-satiri") == 1


def test_background_logs_outside_a_turn_are_not_captured(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    log = logging.getLogger("medrag.api.test_conversation_log")
    log.setLevel(logging.INFO)

    with logger.turn("s1", message="soru"):
        log.info("tur icinde")
    log.info("tur DISINDA - hicbir konusmaya ait degil")

    metin = (tmp_path / "web" / "s1" / "conversation.log").read_text(encoding="utf-8")
    assert "tur icinde" in metin
    assert "tur DISINDA" not in metin


def test_concurrent_turns_do_not_mix_conversations(tmp_path: Path):
    """In WhatsApp, multiple phone numbers hit the same process CONCURRENTLY
    (wa_bot.py handles each request in its own thread) -- each thread's log
    line must land in its OWN conversation's file (ContextVar, see the module
    docstring)."""
    logger = ConversationLogger(enabled=True, directory=tmp_path, channel=CHANNEL_WHATSAPP)
    log = logging.getLogger("medrag.api.test_conversation_log")
    log.setLevel(logging.INFO)
    baslasin = threading.Event()

    def calis(numara: str) -> None:
        with logger.turn(numara, message=f"{numara} sorusu"):
            baslasin.wait(timeout=5)
            log.info("gizli-%s", numara)

    threads = [threading.Thread(target=calis, args=(n,)) for n in ("900001", "900002")]
    for t in threads:
        t.start()
    baslasin.set()
    for t in threads:
        t.join(timeout=5)

    bir = (tmp_path / "whatsapp" / "900001" / "conversation.log").read_text(encoding="utf-8")
    iki = (tmp_path / "whatsapp" / "900002" / "conversation.log").read_text(encoding="utf-8")
    assert "gizli-900001" in bir and "gizli-900002" not in bir
    assert "gizli-900002" in iki and "gizli-900001" not in iki


def test_turn_records_exception_and_reraises(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    try:
        with logger.turn("s1", message="soru"):
            raise RuntimeError("patladi")
    except RuntimeError:
        pass
    else:  # pragma: no cover - the exception should have propagated
        raise AssertionError("exception did not propagate")
    assert _turns(tmp_path / "web" / "s1")[0]["error"] == "patladi"


def test_capture_background_false_keeps_structured_record(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path, capture_background=False)
    log = logging.getLogger("medrag.api.test_conversation_log")
    log.setLevel(logging.INFO)
    with logger.turn("s1", message="soru") as turn_log:
        log.info("bu satir konusma dosyasina GIRMEMELI")
        turn_log.reply = "cevap"
    metin = (tmp_path / "web" / "s1" / "conversation.log").read_text(encoding="utf-8")
    assert "GIRMEMELI" not in metin
    assert _turns(tmp_path / "web" / "s1")[0]["reply"] == "cevap"


# --- disabled / config ----------------------------------------------------------


def test_disabled_is_noop_and_creates_no_directory(tmp_path: Path):
    directory = tmp_path / "conversations"
    logger = ConversationLogger(enabled=False, directory=directory)
    _turn(logger, "s1")
    with logger.turn("s1", message="soru") as turn_log:
        turn_log.reply = "cevap"
    assert not directory.exists()


def test_read_history_returns_turns_in_order(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "s1", message="ilk", reply="cevap1")
    _turn(logger, "s1", message="ikinci", reply="cevap2")
    turns = logger.read_history("s1")
    assert [t["message"] for t in turns] == ["ilk", "ikinci"]
    assert [t["reply"] for t in turns] == ["cevap1", "cevap2"]


def test_read_history_keeps_chunks_and_error(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "s1", message="soru", reply=None, chunks=[{"id": "c1"}], error="hata")
    turns = logger.read_history("s1")
    assert turns[0]["chunks"] == [{"id": "c1"}]
    assert turns[0]["error"] == "hata"
    assert turns[0]["reply"] is None


def test_read_history_missing_conversation_is_empty(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    assert logger.read_history("yok") == []


def test_read_history_disabled_is_empty(tmp_path: Path):
    logger = ConversationLogger(enabled=False, directory=tmp_path)
    assert logger.read_history("s1") == []


def test_read_history_skips_corrupt_line(tmp_path: Path):
    logger = ConversationLogger(enabled=True, directory=tmp_path)
    _turn(logger, "s1", message="iyi", reply="cevap")
    path = logger.conversation_dir("s1") / "turns.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{bozuk json\n", encoding="utf-8")
    turns = logger.read_history("s1")
    assert [t["message"] for t in turns] == ["iyi"]


def test_conversation_logger_from_config_respects_env_override(tmp_path: Path, monkeypatch):
    from medrag.api.config import load_config

    cfg = load_config()
    monkeypatch.setenv("CHATBOT_CONVERSATION_LOG_ENABLED", "false")
    monkeypatch.setenv("CHATBOT_CONVERSATION_LOG_DIR", str(tmp_path))
    logger = conversation_logger_from_config(cfg)
    assert logger.enabled is False
    assert logger.directory == tmp_path


def test_conversation_logger_from_config_default_enabled(monkeypatch):
    from medrag.api.config import load_config

    cfg = load_config()
    monkeypatch.delenv("CHATBOT_CONVERSATION_LOG_ENABLED", raising=False)
    monkeypatch.delenv("CHATBOT_CONVERSATION_LOG_DIR", raising=False)
    logger = conversation_logger_from_config(cfg)
    assert logger.enabled is True
    assert logger.directory.name == "conversations"
    assert logger.channel == "web"


def test_conversation_logger_from_config_channel_and_capture_override(tmp_path: Path, monkeypatch):
    from medrag.api.config import load_config

    cfg = load_config()
    monkeypatch.setenv("CHATBOT_CONVERSATION_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("CHATBOT_CONVERSATION_LOG_CAPTURE_BACKGROUND", "false")
    logger = conversation_logger_from_config(cfg, channel=CHANNEL_WHATSAPP)
    assert logger.channel == "whatsapp"
    assert logger.capture_background is False
