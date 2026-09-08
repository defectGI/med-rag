"""Dynamic clarification-question pool for the `recommendation` flow: instead
of a hardcoded 3-key `_ATTRIBUTE_SLOTS`, which attributes are worth asking
about comes from two CODE-LEVEL (LLM-free) sources:

  1. `load_attribute_glossary` -- the `attribute_glossary` field in
     `facts/db/schema.yaml` (`question` is no longer a DB column; `facts`
     embeds it structured into schema.yaml precisely for this kind of
     consumer). `chatbot` does NOT import `facts` CODE -- it only reads the
     generated schema.yaml BY PATH (same principle as
     `document_store.py`/`factory.py`).
  2. `discover_informative_slots` -- a DIRECT query (stdlib `sqlite3`, not
     the LLM-written SQL of text2sql -- this is a FIXED-SHAPE aggregation, no
     NEED to have an LLM write it) against the `spec_value` table of
     `facts/db/specs.db`: for the given family, which keys REALLY carry more
     than one distinct value with `status='present'` (i.e. worth asking as a
     recommendation question -- if every product gives the same answer,
     asking is pointless). The result feeds `slot_selection.py`'s LLM
     decision (which one to ASK); this module itself makes no LLM calls.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GlossaryEntry:
    block: str
    kind: str
    unit: str | None
    question: str


@dataclass(frozen=True)
class SlotCandidate:
    """A clarification-question candidate -- `key` is always a key that REALLY
    exists in `attribute_glossary` (not invented/hardcoded); `n_distinct` says
    how many different values were observed in this family (how DISCRIMINATIVE
    asking the question can be)."""

    key: str
    block: str
    kind: str
    unit: str | None
    question: str
    n_distinct: int


def load_attribute_glossary(schema_yaml_path: str | Path) -> dict[str, GlossaryEntry]:
    """`key -> GlossaryEntry` -- from the `attribute_glossary` field of
    `facts/db/schema.yaml`. The same `key` can live in multiple blocks
    (standing rule); this dict is ONLY a coarse surface for the recommendation
    flow's question-pool selection, so it keeps the first seen -- not a place
    that needs the full (block,key) split (`discover_informative_slots`
    already narrows by `family`, so collisions are rare in practice)."""
    data = yaml.safe_load(Path(schema_yaml_path).read_text(encoding="utf-8"))
    out: dict[str, GlossaryEntry] = {}
    for entry in data.get("attribute_glossary") or []:
        out.setdefault(
            entry["key"],
            GlossaryEntry(block=entry["block"], kind=entry["kind"], unit=entry.get("unit"), question=entry["question"]),
        )
    return out


def discover_informative_slots(
    db_path: str,
    *,
    family: str | None = None,
    glossary: dict[str, GlossaryEntry],
    exclude_keys: set[str] = frozenset(),
    limit: int = 8,
    product_codes: Sequence[str] | None = None,
) -> list[SlotCandidate]:
    """For the given SCOPE, keys that carry >=2 genuinely distinct values in
    `spec_value`, ordered by discriminative power (n_distinct DESC) -- without
    ever going through text2sql's LLM SQL generation, directly via `sqlite3`
    (this is a fixed-shape GROUP BY/HAVING; making an LLM write it every time
    has no benefit -- same "don't mix an LLM into a deterministic query"
    principle as `document_store.py`).

    The scope can be given two ways ("no thresholds; the user is steered by
    what REMAINS"):

    - `product_codes` -- the candidate products the filter so far has LEFT in
      hand. The PREFERRED path: the next clarification question discriminates
      the ACTUAL remaining candidates, not the whole family, and it works even
      when `family` was never resolved (a phrasing that doesn't fit the
      surface-form dictionary).
    - `family` -- the first-turn/legacy path: while no filter has run yet
      (no candidates in hand), the whole family is inspected.

    If neither is given, returns an empty list -- deliberately NOT scanning
    the ENTIRE database to produce a scope-less question (the caller then
    falls back to its own static emergency pool, see flows/recommendation.py)."""
    if product_codes:
        codes = list(dict.fromkeys(product_codes))
        placeholders = ",".join("?" * len(codes))
        where = f"product_code IN ({placeholders}) AND status = 'present'"
        params: tuple = tuple(codes)
    elif family:
        where = "family = ? AND status = 'present'"
        params = (family,)
    else:
        return []

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT key, block, COUNT(DISTINCT raw_text) AS n_distinct "
            f"FROM spec_value WHERE {where} "
            "GROUP BY key, block HAVING n_distinct >= 2 ORDER BY n_distinct DESC",
            params,
        ).fetchall()
    finally:
        con.close()

    candidates: list[SlotCandidate] = []
    for key, block, n_distinct in rows:
        if key in exclude_keys:
            continue
        entry = glossary.get(key)
        if entry is None:
            continue  # a key not in the glossary (drift) -- skip silently, no invention
        candidates.append(
            SlotCandidate(key=key, block=block, kind=entry.kind, unit=entry.unit,
                          question=entry.question, n_distinct=n_distinct)
        )
        if len(candidates) >= limit:
            break
    return candidates


__all__ = ["GlossaryEntry", "SlotCandidate", "discover_informative_slots", "load_attribute_glossary"]
