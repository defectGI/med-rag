"""Golden-set format + loaders.

The golden set is the ground truth the eval harness scores against. This repo
*defines the format and ships replaceable examples*; the real data is supplied
by the user under ``data/eval/``. The example files are placeholders — swap
their content, keep the format.

Both sets are JSONL (one JSON object per line). Blank lines and ``#`` comment
lines are ignored so a hand-authored file can carry section headers.

intent golden item::

    {"query": "arveles dozu kac mg", "expected_intent": "medical_fact", "lang": "tr"}

retrieval golden item::

    {"query": "de1000 calisma sicakligi", "relevant_ids": ["ACME_PN5001_Datasheet::c7"], "lang": "tr"}
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from medrag.api.retrieval.core.interfaces import IntentLabel


class IntentGoldenItem(BaseModel):
    """One labelled query for intent evaluation."""

    query: str
    expected_intent: IntentLabel
    lang: str | None = None
    id: str | None = None
    note: str | None = None


class RetrievalGoldenItem(BaseModel):
    """One query with the set of ids that count as relevant.

    ``relevant_ids`` are backend identifiers (e.g. ``ChunkNode.node_id`` or a
    product ``node_id``) — the same ``id`` a retriever puts on
    ``RetrievalResult``. ``k`` optionally overrides the default cutoff for this
    query only.
    """

    query: str
    relevant_ids: list[str] = Field(default_factory=list)
    lang: str | None = None
    k: int | None = None
    id: str | None = None


def _iter_jsonl(path: str | Path):
    """Yield (line_number, parsed_obj) for each non-blank, non-comment line."""
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON — {exc}") from exc


def load_intent_golden(path: str | Path) -> list[IntentGoldenItem]:
    """Load and validate an intent golden set (JSONL)."""
    items: list[IntentGoldenItem] = []
    for lineno, obj in _iter_jsonl(path):
        try:
            items.append(IntentGoldenItem.model_validate(obj))
        except Exception as exc:
            raise ValueError(f"{Path(path)}:{lineno}: {exc}") from exc
    return items


def load_retrieval_golden(path: str | Path) -> list[RetrievalGoldenItem]:
    """Load and validate a retrieval golden set (JSONL)."""
    items: list[RetrievalGoldenItem] = []
    for lineno, obj in _iter_jsonl(path):
        try:
            items.append(RetrievalGoldenItem.model_validate(obj))
        except Exception as exc:
            raise ValueError(f"{Path(path)}:{lineno}: {exc}") from exc
    return items


__all__ = [
    "IntentGoldenItem",
    "RetrievalGoldenItem",
    "load_intent_golden",
    "load_retrieval_golden",
]
