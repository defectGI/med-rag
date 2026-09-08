"""Conversation-log reader -- READS the `logs/conversations/` tree written by
the chatbot.

The code is NOT imported; only the file system is shared -- the fourth instance
of an established pattern (`facts`, `image_store`, `document_store`). The
chatbot is unaware of this module; here only the format it WRITES is read:

    <root>/<channel>/<conversation_id>/
        meta.json          # channel, id, first/last seen, turn count
        turns.jsonl        # ONE JSON line per turn (structured record)
        conversation.log   # human-readable dump of the same turn + background lines

The format is defined in the `src/medrag/api/conversation_log.py` module
docstring. This reader DEPENDS on it but does NOT import it -- if the format
changes, this module must be updated too (not the other way around; the chatbot
owes the panel nothing).

Privacy: on the WhatsApp channel `<conversation_id>` IS THE USER'S PHONE NUMBER
and `turns.jsonl` contains raw user messages. Anywhere this module is wired to
an HTTP surface, authentication is mandatory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Env var naming the conversation root the panel reads. If unset it falls back
#: to `chatbot/logs/conversations/` inside the repo (a single-machine dev
#: setup); when the panel runs on a separate machine this env is REQUIRED.
ENV_DIR = "PANEL_CONVERSATIONS_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DIR = _REPO_ROOT / "chatbot" / "logs" / "conversations"

META_FILE = "meta.json"
TURNS_FILE = "turns.jsonl"
BACKGROUND_FILE = "conversation.log"

#: Turn separator in `conversation.log` -- `conversation_log.py` writes a
#: 78-character `=` line plus a "TUR <n> BASLADI" line at the start of each turn.
_TURN_BANNER = "=" * 78


def conversations_root() -> Path:
    override = os.getenv(ENV_DIR)
    return Path(override) if override else _DEFAULT_DIR


def _safe_child(parent: Path, name: str) -> Path:
    """Resolves `parent/name` and verifies it REALLY stays under `parent`.

    The chatbot side already sanitizes folder names (`conversation_log._SAFE_NAME`),
    but here the name arrives as an HTTP query parameter -- the defense is
    rebuilt at this edge; the other side's discipline is not trusted.
    """
    if not name or name in (".", ".."):
        raise ValueError(f"gecersiz ad: {name!r}")
    if "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"gecersiz ad: {name!r}")
    candidate = (parent / name).resolve()
    if candidate != parent.resolve() and parent.resolve() not in candidate.parents:
        raise ValueError(f"kok disina tasan ad: {name!r}")
    return candidate


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_turns(root: Path, channel: str, conversation_id: str) -> list[dict]:
    """Reads ALL turns of a conversation. Corrupt lines are SKIPPED and reading
    continues -- a half-written last line must not hide the whole history."""
    directory = _safe_child(_safe_child(root, channel), conversation_id)
    path = directory / TURNS_FILE
    if not path.exists():
        return []
    turns = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return turns


def _summarize(turns: list[dict]) -> dict:
    """Enough of a summary for the list screen -- even if every turn must be
    read, the full raw messages are not sent to the browser."""
    errors = sum(1 for t in turns if t.get("error"))
    evidence = sum(len(t.get("chunks") or []) for t in turns)
    durations = [t.get("duration_seconds") for t in turns if isinstance(t.get("duration_seconds"), (int, float))]
    last_message = turns[-1].get("message") if turns else None
    if isinstance(last_message, str) and len(last_message) > 120:
        last_message = last_message[:120] + "…"
    return {
        "error_count": errors,
        "evidence_count": evidence,
        "max_duration_seconds": max(durations) if durations else None,
        "last_message": last_message,
    }


def list_conversations(root: Path | None = None) -> list[dict]:
    """Conversations across all channels, ordered by last-seen."""
    root = root or conversations_root()
    if not root.exists():
        return []
    out = []
    for channel_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for conv_dir in sorted(p for p in channel_dir.iterdir() if p.is_dir()):
            meta = _read_json(conv_dir / META_FILE)
            turns = read_turns(root, channel_dir.name, conv_dir.name)
            entry = {
                "channel": meta.get("channel") or channel_dir.name,
                "conversation_id": meta.get("conversation_id") or conv_dir.name,
                "dir_channel": channel_dir.name,
                "dir_id": conv_dir.name,
                "first_seen_at": meta.get("first_seen_at"),
                "last_seen_at": meta.get("last_seen_at"),
                # Still get the right turn count even if meta.json is missing/corrupt.
                "turn_count": meta.get("turn_count") or len(turns),
                "has_background_log": (conv_dir / BACKGROUND_FILE).exists(),
            }
            entry.update(_summarize(turns))
            out.append(entry)
    out.sort(key=lambda e: (e.get("last_seen_at") or "", e["conversation_id"]), reverse=True)
    return out


def read_conversation(root: Path | None, channel: str, conversation_id: str) -> dict:
    """A conversation's meta + all turns (structured record)."""
    root = root or conversations_root()
    directory = _safe_child(_safe_child(root, channel), conversation_id)
    meta = _read_json(directory / META_FILE)
    turns = read_turns(root, channel, conversation_id)
    return {
        "channel": meta.get("channel") or channel,
        "conversation_id": meta.get("conversation_id") or conversation_id,
        "dir_channel": channel,
        "dir_id": conversation_id,
        "meta": meta,
        "turns": turns,
        "exists": directory.exists(),
    }


def read_background_log(
    root: Path | None, channel: str, conversation_id: str, turn: int | None = None
) -> str:
    """The whole `conversation.log`, or -- if `turn` is given -- ONLY that turn's block.

    Slicing is done against the `TUR <n> BASLADI` banner; if no banner is found
    (an old file, a truncated log) the ENTIRE file is returned instead of an
    empty string -- claiming "nothing is there" would be misleading.
    """
    root = root or conversations_root()
    directory = _safe_child(_safe_child(root, channel), conversation_id)
    path = directory / BACKGROUND_FILE
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if turn is None:
        return text

    blocks = text.split(_TURN_BANNER)
    for block in blocks:
        # First line after the banner: "[<time>] TUR <n> BASLADI (...)"
        head = block.lstrip("\n")[:200]
        if f"TUR {turn} BAŞLADI" in head or f"TUR {turn} BASLADI" in head:
            return _TURN_BANNER + block
    return text


__all__ = [
    "ENV_DIR",
    "conversations_root",
    "list_conversations",
    "read_background_log",
    "read_conversation",
    "read_turns",
]
