"""Conversation memory -- RAM only, no persistence, no logging (deliberate
decision: conversation logs are not kept).

A session's messages live in memory only while this process is up; restarting
the server resets ALL sessions -- intentionally, nothing is written anywhere.
This is a DIFFERENT decision from flow intermediate steps never leaking into
the answering model (orchestrator.py): what is kept here are the real
user/assistant messages of previous turn(s), not a flow's internal steps.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str


class ConversationMemory:
    """A simple single-process dict -- a deployment with more than one worker
    needs a shared store; that was deliberately not set up here (persistence
    was not wanted, and it adds complexity). `_lock`: multiple WhatsApp users
    can issue concurrent `/mesaj` requests within the same process, so the
    public methods are guarded by a `threading.Lock`."""

    def __init__(self, *, max_turns: int = 20) -> None:
        self._sessions: dict[str, list[Message]] = {}
        self._max_turns = max_turns
        self._lock = threading.Lock()

    def get(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, role: Literal["user", "assistant"], content: str) -> None:
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.append(Message(role=role, content=content))
            # Keep only the last max_turns pairs (user+assistant).
            limit = self._max_turns * 2
            if len(messages) > limit:
                del messages[: len(messages) - limit]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


__all__ = ["ConversationMemory", "Message"]
