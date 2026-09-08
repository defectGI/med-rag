"""A SEPARATE, detailed log per conversation: messages + ALL background stages
belonging to that conversation.

Requirement: very detailed logging on both the WhatsApp bot and the webapp
side -- every conversation kept separately, with both messages and all
background stages inside the conversation logged; on the WhatsApp side logs
are keyed by the user's phone number, each new number opening its own folder.

## Directory layout

    logs/conversations/<channel>/<conversation_id>/
        meta.json          # channel, conversation id, first/last seen, turn count
        turns.jsonl        # ONE JSON line per turn (structured record)
        conversation.log   # HUMAN-READABLE text dump of the same turn + background

`<channel>` = `web` (webapp) | `whatsapp` (wa_bot). `<conversation_id>` is the
session cookie on webapp and the USER PHONE NUMBER on WhatsApp -- the first
message from a new number creates its folder automatically. The pre-relayout
layout was a flat `<dir>/<session_id>.jsonl` (no channel split); old files
stay where they are, new records go to the new layout.

## Why two files

- `turns.jsonl`: append-only structured record (quality-analysis scripts read
  it line by line). A turn = user message -> all `on_trace` steps ->
  answer/error.
- `conversation.log`: a readable dump of the SAME TURN PLUS every background
  line flowing through `logging` during that turn (`chatbot.*`, `text2sql`'s
  SQL linking/generation detail, `urllib`...). What makes this possible is
  `_ConversationFileHandler`: a handler attached to the root + `text2sql`
  loggers that reads its target file from a **ContextVar**. ContextVar (NOT
  thread-local) is deliberate: `asyncio.to_thread`
  (answering_model/reconciler/gpu_gate) copies the context but not thread
  locals -- so lines produced in the background threads of LLM calls still
  land in the right conversation.

`chatbot/logs/chatbot.log` (logging_setup.py -- ALL sessions in one file) is
NOT REMOVED: diagnosis still needs a single stream. The log here is an
ADDITIONAL channel feeding from the same `logging`/`on_trace` choke points.

Boundary: for a record to land here it must pass the `logging` level
(`config/default.toml [logging] level`, default INFO) -- DEBUG lines like chunk
lists enter the text dump only with `level = "DEBUG"`; the structured
`turns.jsonl` record already holds the full trace regardless of level.

Privacy: these files contain user messages and, on WhatsApp, the phone number
(as the folder name). `chatbot/logs/` is in the root `.gitignore`; logging can
be switched off entirely via `[conversation_log] enabled = false` (or
`CHATBOT_CONVERSATION_LOG_ENABLED=false`), and background capture separately
via `capture_background = false`.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("medrag.api.conversation_log")

# This file lives under src/medrag/api/ now; `logs/` is DATA, not CODE -- it
# wasn't moved (per the established pattern) and stayed in the old `chatbot/`
# directory at the repo root (same pattern as logging_setup.py).
# parents[3] = repo root.
_COMPONENT_ROOT = Path(__file__).resolve().parents[3] / "chatbot"
_DEFAULT_DIR = _COMPONENT_ROOT / "logs" / "conversations"

#: Channel names -- the first level of the directory layout. webapp uses
#: `web`, wa_bot uses `whatsapp`; any additional channel is defined here.
CHANNEL_WEB = "web"
CHANNEL_WHATSAPP = "whatsapp"

TURNS_FILE = "turns.jsonl"
BACKGROUND_FILE = "conversation.log"
META_FILE = "meta.json"

# session_id / phone number end up as a directory name -- defense-in-depth: an
# unexpected value containing path separators/double-dots (a client can set its
# own cookie by hand, a gateway may send a broken user id) must not escape the
# directory.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Background log file of the active turn. ContextVar, NOT thread-local: see
# the module docstring (`asyncio.to_thread` copies the context).
_ACTIVE_LOG_PATH: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "chatbot_conversation_log_path", default=None
)

# Same target as `logging_setup.py`: sqlretrieve sets up its own logger with
# `propagate=False`; attaching to the root wouldn't catch it.
_TEXT2SQL_LOGGER = "text2sql"

_capture_lock = threading.Lock()
_HANDLER: logging.Handler | None = None

# Serializes the read-modify-write of meta.json (multiple WhatsApp users hit
# the SAME process concurrently, see wa_bot.py).
_meta_lock = threading.Lock()


def _sanitize(name: str) -> str:
    return _SAFE_NAME.sub("_", str(name)) or "unknown"


def _now() -> datetime:
    """Local ISO-8601 with offset."""
    return datetime.now().astimezone()


def _append_text(path: Path, text: str) -> None:
    """A disk write NEVER breaks the request -- on error it's logged and the
    flow continues (this is a diagnostics/quality channel, not part of the
    product flow)."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        logger.exception("konuşma kaydı yazılamadı: %s", path)


class _ConversationFileHandler(logging.Handler):
    """Logging handler that writes into the active turn's conversation folder.

    The target file is read from the `_ACTIVE_LOG_PATH` ContextVar; if absent
    (records produced OUTSIDE a turn context -- startup, preflight, HTTP route
    noise) the record is silently SKIPPED. This way, despite being attached to
    the root logger, only lines BELONGING to one conversation land in that
    conversation's file.
    """

    def emit(self, record: logging.LogRecord) -> None:
        path = _ACTIVE_LOG_PATH.get()
        if path is None:
            return
        # The same handler is on BOTH the root and the `text2sql` logger; if
        # `text2sql` propagates (default True when logging_setup isn't
        # installed yet) the same LogRecord reaches us TWICE. We tag the
        # record object and drop the second pass -- so a line is written
        # exactly once INDEPENDENT of the `propagate` state.
        if getattr(record, "_conv_log_written", False):
            return
        record._conv_log_written = True  # type: ignore[attr-defined]
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 - a logging handler never breaks the flow
            self.handleError(record)


def install_background_capture() -> None:
    """Sets up per-conversation background capture (idempotent).

    The handler is attached to BOTH the root logger and the `text2sql` logger
    -- same reason as `logging_setup.py`: `sqlretrieve` configures its own
    logger with `propagate=False`, so attaching to the root does NOT capture
    its SQL linking/generation detail (and that detail is exactly what the
    conversation dump wants).

    "Is it installed" is checked by whether the handler is ACTUALLY attached,
    not by a flag: code that resets logging configuration (e.g.
    `logging.getLogger().handlers.clear()`) could interleave, and the flag
    would still say "installed" while capture silently died. That's why
    `turn()` calls this again every turn -- the cost is two list scans.
    """
    global _HANDLER
    with _capture_lock:
        if _HANDLER is None:
            _HANDLER = _ConversationFileHandler()
            _HANDLER.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            _HANDLER.setLevel(logging.NOTSET)
        for target in (logging.getLogger(), logging.getLogger(_TEXT2SQL_LOGGER)):
            if _HANDLER not in target.handlers:
                target.addHandler(_HANDLER)


@dataclass
class TurnLog:
    """Accumulator for an in-progress turn -- `ConversationLogger.turn()`
    yields it. Fields are assigned directly (`turn_log.reply = ...`); on
    `with`-block exit it is written as one JSON line + a text dump."""

    conversation_id: str
    channel: str
    turn_no: int
    message: str
    started_at: datetime
    trace: list[dict] = field(default_factory=list)
    reply: str | None = None
    images: list[dict] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)
    chunks: list[dict] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    # In a disabled (`enabled=False`) logger nothing is written to disk; field
    # assignments still work harmlessly (no-op object).
    _path: Path | None = None

    def record_trace(self, step: str, data: object) -> None:
        """Appends an `on_trace(step, data)` event to the turn's structured
        record. No separate write to the text dump is needed -- the caller
        already logs via `logging_setup.log_trace_step`, and since that line
        runs in the active-turn context the handler routes it to the same
        file."""
        self.trace.append(
            {"step": step, "data": data, "ts": _now().isoformat(timespec="seconds")}
        )

    def note(self, text: str) -> None:
        """Drops a free-text note into the turn's text dump (e.g. a WhatsApp
        session-reset warning, a gateway send result) and also records it in
        the structured log as `notes`."""
        self.notes.append(text)
        if self._path is not None:
            _append_text(self._path, f"    · {text}\n")


class ConversationLogger:
    """Detailed logger that opens a folder per channel + conversation.

    If `enabled=False`, every method is a no-op (switchable via env/config);
    if `capture_background=False`, only the structured `turns.jsonl` +
    message/answer dump are kept -- background log lines are not routed into
    the conversation folder.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        directory: Path | None = None,
        channel: str = CHANNEL_WEB,
        capture_background: bool = True,
    ) -> None:
        self.enabled = enabled
        self.directory = Path(directory) if directory is not None else _DEFAULT_DIR
        self.channel = channel
        self.capture_background = capture_background
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.capture_background:
                install_background_capture()

    # --- paths ----------------------------------------------------------

    def conversation_dir(self, conversation_id: str) -> Path:
        """`<dir>/<channel>/<conversation_id>/` -- on WhatsApp
        `<conversation_id>` is the user's phone number (each new number
        opens its own folder)."""
        return self.directory / _sanitize(self.channel) / _sanitize(conversation_id)

    # --- meta -----------------------------------------------------------

    def _touch_meta(self, directory: Path, conversation_id: str, now: datetime) -> int:
        """Creates/updates the conversation folder's `meta.json` and returns
        the NEW turn number. The read-modify-write is serialized by a lock
        (concurrent WhatsApp requests can touch the same file)."""
        path = directory / META_FILE
        with _meta_lock:
            meta: dict = {}
            if path.exists():
                try:
                    meta = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    meta = {}
            turn_no = int(meta.get("turn_count") or 0) + 1
            meta.update(
                {
                    "channel": self.channel,
                    "conversation_id": conversation_id,
                    "first_seen_at": meta.get("first_seen_at") or now.isoformat(timespec="seconds"),
                    "last_seen_at": now.isoformat(timespec="seconds"),
                    "turn_count": turn_no,
                }
            )
            try:
                path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError:
                logger.exception("konuşma meta'sı yazılamadı: %s", path)
            return turn_no

    # --- turn recording ---------------------------------------------------

    @contextlib.contextmanager
    def turn(self, conversation_id: str, *, message: str) -> Iterator[TurnLog]:
        """Records one turn from start to finish.

        EVERY `logging` line produced inside the `with` block (including
        `asyncio.to_thread` work running in the same context) is routed into
        this conversation's `conversation.log`; on block exit a single JSON
        line is written to `turns.jsonl`. If an exception escapes the block,
        `error` is auto-filled (if not already set) and the exception is
        re-raised VERBATIM -- recording never alters the flow.
        """
        started_at = _now()
        if not self.enabled:
            yield TurnLog(
                conversation_id=conversation_id, channel=self.channel, turn_no=0,
                message=message, started_at=started_at,
            )
            return

        directory = self.conversation_dir(conversation_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("konuşma klasörü açılamadı: %s", directory)
            yield TurnLog(
                conversation_id=conversation_id, channel=self.channel, turn_no=0,
                message=message, started_at=started_at,
            )
            return

        if self.capture_background:
            # Re-verify every turn: an interleaved `logging` reset may have
            # dropped the handler (see install_background_capture).
            install_background_capture()

        turn_no = self._touch_meta(directory, conversation_id, started_at)
        log_path = directory / BACKGROUND_FILE
        turn_log = TurnLog(
            conversation_id=conversation_id, channel=self.channel, turn_no=turn_no,
            message=message, started_at=started_at, _path=log_path,
        )

        _append_text(
            log_path,
            f"\n{'=' * 78}\n"
            f"[{started_at.isoformat(timespec='seconds')}] TUR {turn_no} BAŞLADI "
            f"(kanal={self.channel}, konuşma={conversation_id})\n"
            f"--- KULLANICI MESAJI ---\n{message}\n"
            f"--- ARKA PLAN ---\n",
        )

        token = _ACTIVE_LOG_PATH.set(log_path if self.capture_background else None)
        try:
            yield turn_log
        except BaseException as exc:
            if turn_log.error is None:
                turn_log.error = str(exc)
            raise
        finally:
            _ACTIVE_LOG_PATH.reset(token)
            ended_at = _now()
            self._write_turn(turn_log, ended_at)

    def _write_turn(self, turn_log: TurnLog, ended_at: datetime) -> None:
        duration = round((ended_at - turn_log.started_at).total_seconds(), 3)
        directory = self.conversation_dir(turn_log.conversation_id)
        log_path = directory / BACKGROUND_FILE

        durum = f"HATA: {turn_log.error}" if turn_log.error else "tamam"
        ekler = []
        if turn_log.images:
            ekler.append(f"görsel={len(turn_log.images)}")
        if turn_log.documents:
            ekler.append(f"belge={len(turn_log.documents)}")
        if turn_log.chunks:
            ekler.append(f"kanıt={len(turn_log.chunks)}")
        ek_metin = (" (" + ", ".join(ekler) + ")") if ekler else ""
        _append_text(
            log_path,
            f"--- CEVAP{ek_metin} ---\n{turn_log.reply if turn_log.reply is not None else '(cevap yok)'}\n"
            f"[{ended_at.isoformat(timespec='seconds')}] TUR {turn_log.turn_no} BİTTİ "
            f"(süre {duration} sn, {durum}, aşama={len(turn_log.trace)})\n",
        )

        record = {
            "channel": turn_log.channel,
            "conversation_id": turn_log.conversation_id,
            # Old readers (pre-relayout) expect `session_id` -- the field is
            # kept; the new name is `conversation_id`.
            "session_id": turn_log.conversation_id,
            "turn": turn_log.turn_no,
            "started_at": turn_log.started_at.isoformat(timespec="seconds"),
            "ended_at": ended_at.isoformat(timespec="seconds"),
            "duration_seconds": duration,
            "message": turn_log.message,
            "trace": turn_log.trace,
            "reply": turn_log.reply,
            "images": turn_log.images,
            "documents": turn_log.documents,
            "chunks": turn_log.chunks,
            "notes": turn_log.notes,
            "error": turn_log.error,
        }
        path = directory / TURNS_FILE
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("konuşma JSON kaydı yazılamadı: %s", path)

    def read_history(self, conversation_id: str) -> list[dict]:
        """Reads back the stored turns of a conversation (the SPA's history
        restore). Returns the raw `turns.jsonl` records in chronological
        order; a missing file, a disabled logger or corrupt lines degrade to
        `[]`/skipped -- never an exception (a half-written line must not hide
        the rest of the history)."""
        if not self.enabled:
            return []
        path = self.conversation_dir(conversation_id) / TURNS_FILE
        if not path.is_file():
            return []
        turns: list[dict] = []
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        turns.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            logger.exception("konuşma geçmişi okunamadı: %s", path)
            return []
        return turns

    def record_turn(
        self,
        conversation_id: str,
        *,
        started_at: datetime,
        ended_at: datetime,
        message: str,
        trace: list[dict],
        reply: str | None,
        images: list[dict] | None = None,
        documents: list[dict] | None = None,
        chunks: list[dict] | None = None,
        error: str | None = None,
        notes: list[str] | None = None,
    ) -> None:
        """Records a turn in ONE shot -- for callers that can't use the
        `turn()` context manager (they measure durations themselves). Background
        capture is ABSENT on this path (there's no block wrapping the context);
        only the structured record + text dump are written."""
        if not self.enabled:
            return
        directory = self.conversation_dir(conversation_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("konuşma klasörü açılamadı: %s", directory)
            return
        turn_no = self._touch_meta(directory, conversation_id, started_at)
        turn_log = TurnLog(
            conversation_id=conversation_id, channel=self.channel, turn_no=turn_no,
            message=message, started_at=started_at, trace=list(trace), reply=reply,
            images=list(images or []), documents=list(documents or []),
            chunks=list(chunks or []), error=error, notes=list(notes or []),
            _path=directory / BACKGROUND_FILE,
        )
        _append_text(
            directory / BACKGROUND_FILE,
            f"\n{'=' * 78}\n"
            f"[{started_at.isoformat(timespec='seconds')}] TUR {turn_no} BAŞLADI "
            f"(kanal={self.channel}, konuşma={conversation_id})\n"
            f"--- KULLANICI MESAJI ---\n{message}\n"
            f"--- ARKA PLAN ---\n",
        )
        self._write_turn(turn_log, ended_at)


def conversation_logger_from_config(cfg, *, channel: str = CHANNEL_WEB) -> ConversationLogger:
    """Builds a `ConversationLogger` from `cfg.conversation_log` + optional
    env overrides (`CHATBOT_CONVERSATION_LOG_ENABLED`/
    `CHATBOT_CONVERSATION_LOG_DIR`/
    `CHATBOT_CONVERSATION_LOG_CAPTURE_BACKGROUND`) -- the same pattern as
    `logging_setup.configure_logging_from_cfg` (webapp and wa_bot SHARE this
    single entry point; only `channel` differs)."""
    cl = cfg.conversation_log
    enabled = _env_bool("CHATBOT_CONVERSATION_LOG_ENABLED", cl.enabled)
    capture = _env_bool("CHATBOT_CONVERSATION_LOG_CAPTURE_BACKGROUND", cl.capture_background)
    dir_override = os.getenv("CHATBOT_CONVERSATION_LOG_DIR")
    directory = Path(dir_override) if dir_override else (_COMPONENT_ROOT / cl.dir)
    return ConversationLogger(
        enabled=enabled, directory=directory, channel=channel, capture_background=capture
    )


def _env_bool(name: str, fallback: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return fallback
    return raw.strip().lower() in ("1", "true", "yes", "on")


__all__ = [
    "CHANNEL_WEB",
    "CHANNEL_WHATSAPP",
    "ConversationLogger",
    "TurnLog",
    "conversation_logger_from_config",
    "install_background_capture",
]
