"""PHASE C -- WRITES the fact JSON files produced by `run_full.py` into
`facts/db/specs.db` (2026-08-04 simplified 4-table schema:
product/document/attribute/spec_value -- see `schema_rag.sql` header
comment).

`build_facts_db.py` (PHASE A) ONLY builds the skeleton and it is
valueless -- KARAR-016 makes it the SOLE writer of the skeleton. This
script is the SOLE writer of a different phase (PHASE C): it converts
bootstrap's `not_specified`/`not_applicable` rows into real
`present`/`absent` rows carrying values. Both write to specs.db but to
DIFFERENT column sets / phases -- not a KARAR-016 violation, but the
second phase anticipated by that decision.

**THIS IS A NEW CONTRACT DECISION (meant to be logged as KARAR-NNN in
PROTOCOL.md -- not yet logged).** The choices below are awaiting user
approval:

  1. **A new key (`is_new_key=true`) fact is NOT WRITTEN to the DB** --
     the dictionary (`attribute` table) is NOT modified by this script.
     Instead it is appended to `queue/new_key_proposals.jsonl`
     (append-only JSONL).
  2. **When MULTIPLE TRULY DIFFERENT values arrive for the same (product,
     key, condition)**: the one with the highest `document.trust_rank`
     becomes the LIVE (`present`) row, the others are written as
     additional `status='conflicting'` rows (`sv_unique_live_uix` is
     DELIBERATELY scoped only to present/not_specified/absent/
     not_applicable -- conflicting/superseded are OUTSIDE that
     constraint, allowing multiple rows). The SAME value repeated across
     multiple sources is NOT a CONFLICT -- it merges into a single row
     with multiple entries in `evidence[]` JSON (corroboration).
  3. **`condition` values not in the dictionary**: the fact is SKIPPED
     and logged -- not written to the DB (the INSERT trigger would
     reject it; it is pre-filtered HERE so the transaction does not blow
     up mid-stream).

Usage (first `--dry-run` to get a report without writing):
  cd facts/experiments/chunk_full_run
  python load_to_db.py --dry-run                 # results/*.json, no writes
  python load_to_db.py --models PN1071,PN1204    # subset, real writes
  python load_to_db.py                            # ALL products in results/
  python load_to_db.py --force                    # reset already-loaded products and reload

Idempotency: if a product is already written with `extractor='chunk_full_run_v1'`
rows, that product is SKIPPED (use --force to reset and reload, see
`_reset_product`). Processing is ONE transaction per product: if a product
errors mid-way, that product is ROLLED BACK; previous products stay committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from medrag.core.paths import resolve_specs_db_path

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/ (code, O-03)
_REPO_ROOT = HERE.parents[3]  # medrag/ (facts/pipeline/medrag/src/<repo>)
# DATA (db/, results/, queue/) was NOT moved with the code -- O-03 scope
# was code only; same principle as `discover.py::_FACTS_DATA_ROOT`: lives
# in `facts/` at the repo root.
_FACTS_DATA_ROOT = _REPO_ROOT / "facts"
DB_PATH = resolve_specs_db_path()
# Old location was facts/experiments/chunk_full_run/results/ -- data
# directory name (chunk_full_run/) is preserved, only the `experiments/`
# prefix on the code side dropped (see run_full.py's SAME RESULTS_DIR).
#
# N-22: `FACTS_RESULTS_DIR` -- SAME env var as `run_full.py::RESULTS_DIR`
# (default unchanged) -- if the two do not point to the SAME directory,
# run_full's written JSONs are NEVER FOUND by this script, hence ONE
# env var.
RESULTS_DIR = Path(os.environ.get(
    "FACTS_RESULTS_DIR", str(_FACTS_DATA_ROOT / "chunk_full_run" / "results")
))
# 2026-08-26: `FACTS_QUEUE_DIR` -- the directory where
# `new_key_proposals.jsonl` (the queue of keys NOT in the dictionary
# but actually read by the model from a document) is written. Default
# was `/app/facts/queue`, i.e. INSIDE THE CONTAINER: it evaporated on
# every redeploy and `.dockerignore` excludes `facts/queue/` so the
# directory never existed in the image -- it was created at runtime and
# erased on redeploy. This queue is the SOLE evidence source for
# dictionary growth (measurement of 2026-08-26 run: 41 of the first
# 1989 facts' keys were not in the dictionary, the vast majority were
# real specs like `k_factor`/`differential_gain_error`) -- if it is
# lost, that data vanishes silently. Same pattern as
# `FACTS_RESULTS_DIR`/`CHUNK_OWNERSHIP_DIR`.
QUEUE_DIR = Path(os.environ.get("FACTS_QUEUE_DIR", str(_FACTS_DATA_ROOT / "queue")))

from medrag.core.db.issues import connect_issues
from medrag.core.paths import resolve_issues_db_path
from medrag.pipeline.facts.db_integrity import SchemaIntegrityError, verify_db_integrity
from medrag.pipeline.facts.discover import (
    build_product_manifest,
    load_indexes,
    products_for_doc_id,
)
from medrag.pipeline.facts.issues_bridge import (
    record_new_key_proposal,
    record_skipped_facts,
)

EXTRACTOR = "chunk_full_run_v1"
_PROMPT_TEXT = (HERE / "prompts" / "chunk_strategy_system.md").read_text(encoding="utf-8")
PROMPT_VERSION = "chunk_v1:" + hashlib.sha256(_PROMPT_TEXT.encode("utf-8")).hexdigest()[:12]


@lru_cache(maxsize=1)
def extractor_version() -> str:
    """Value WRITTEN to the `spec_value.extractor_version` column --
    `"<facts_version>+<PROMPT_VERSION>"`, e.g.
    `"1.0.0+chunk_v1:a1b2c3d4e5f6"`.

    PREVIOUSLY only `PROMPT_VERSION` (`"chunk_v1:<sha12>"`) was written
    here. The reader (`staleness_audit.py` criterion (c) ->
    `nightly_report::_recorded_version_is_behind`) compared this to
    `[pipeline.versions] facts_version` as a SEMVER: `_version_tuple(
    "chunk_v1:<sha12>")` saw a single piece with no dots and treated it
    as `(0,)`, and `(0,) < (1,0,0)` was True for EVERY ROW on EVERY
    NIGHT. Result: every product written via `chunk_full_run_v1` was
    counted as stale every night, and
    `run_nightly.py::_resolve_incremental_codes`'s staleness gate
    effectively did nothing. This was the same "stale every night
    ~whole corpus" behaviour that K-97 closed, resurfacing through
    criterion (c) when K-98 added it back as a secondary reconciliation
    net.

    Both pieces are PRESERVED: the part being compared is `facts_version`
    (the single manual checkpoint when the extraction METHOD changes,
    K-96), the prompt hash after `+` is kept as forensic metadata --
    `_version_tuple` drops everything after `+` (see
    `nightly_report::_semver_part`), so a prompt hash change alone does
    NOT trigger staleness. This is DELIBERATE: every prompt-text edit
    is not a reason to re-extract the whole corpus; "method changed"
    is decided MANUALLY via `facts_version`.

    `lru_cache`: TOML is read once (this function is called per row
    written), but config is NOT read at module IMPORT -- tests must be
    able to import this module without standing up config.
    """
    from medrag.core.config.schema import load_core_config

    return f"{load_core_config().pipeline.versions.facts_version}+{PROMPT_VERSION}"

TRUST_RANK = {  # mirror of document.trust_rank in schema_rag.sql (STORED column, same values)
    "DATASHEET": 100, "USER_MANUAL": 90, "CE_DECLARATION": 85, "TECHNICAL_DRAWING": 80,
    "CATALOGUE": 60, "BROCHURE": 50, "QUICK_START_GUIDE": 40,
}
PAYLOAD_FIELDS = ("num_value", "text_value", "val_min", "val_typ", "val_max", "bool_value")

_NUMERIC_PAYLOAD_SLOTS = ("num_value", "val_min", "val_typ", "val_max")


def _normalize_numeric_payload(fact: dict) -> None:
    """2026-08-27 fix: the model sometimes returns `num_value`/
    `val_min`/`val_typ`/`val_max` as a numeric STRING instead of a number
    (e.g. `"20"` instead of `20.0`). `_value_signature` (CORROBORATION
    detection, see that function) compares these fields DIRECTLY as
    dict keys -- `"20" != 20.0` (different type, different hash) caused
    TWO observations carrying the SAME real value to drop WRONGLY into
    `conflicting` (observed symptom: 262 conflicting rows on the
    2026-08-26 run, higher than expected). `fact` is mutated IN PLACE --
    safe numeric strings ("20", " 20.5 ", Turkish decimal "20,5") are
    converted to float; non-numeric strings (genuinely non-numeric, e.g.
    "-40 C") are LEFT AS-IS -- the type guard in `_payload_ok` then
    filters such facts out (no silent fabrication as 0 or otherwise).
    """
    for slot in _NUMERIC_PAYLOAD_SLOTS:
        value = fact.get(slot)
        if not isinstance(value, str):
            continue
        text = value.strip()
        for candidate in (text, text.replace(",", ".")):
            try:
                fact[slot] = float(candidate)
                break
            except ValueError:
                continue


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class Dictionary:
    kind_by_key: dict[tuple[str, str], str]
    unit_by_key: dict[tuple[str, str], str | None]
    conditions_by_key: dict[tuple[str, str], set[str]]
    subfield_names_by_key: dict[tuple[str, str], set[str]]
    blocks_by_key: dict[str, set[str]]
    # `{key: {blocks that define it}}` -- to salvage new-key proposals
    # where the model wrote the wrong block (see _classify_fact).


def load_dictionary(con: sqlite3.Connection) -> Dictionary:
    kind_by_key: dict[tuple[str, str], str] = {}
    unit_by_key: dict[tuple[str, str], str | None] = {}
    conditions_by_key: dict[tuple[str, str], set[str]] = {}
    subfield_names_by_key: dict[tuple[str, str], set[str]] = {}
    blocks_by_key: dict[str, set[str]] = {}
    for block, key, kind, unit, conditions_json, subfields_json in con.execute(
        "SELECT block, key, kind, unit, conditions, subfields FROM attribute"
    ):
        k = (block, key)
        kind_by_key[k] = kind
        unit_by_key[k] = unit
        conditions_by_key[k] = set(json.loads(conditions_json) or [])
        subfield_names_by_key[k] = {s["name"] for s in (json.loads(subfields_json) or [])}
        blocks_by_key.setdefault(key, set()).add(block)
    return Dictionary(
        kind_by_key=kind_by_key, unit_by_key=unit_by_key,
        conditions_by_key=conditions_by_key, subfield_names_by_key=subfield_names_by_key,
        blocks_by_key=blocks_by_key,
    )


class LoadError(RuntimeError):
    pass


def _repair_payload_slot(fact: dict, dict_kind: str) -> str | None:
    """Repairs facts whose `kind` is correct but whose value sits in the
    WRONG slot.

    Pattern seen on the gemma4:31b run (2026-08-04): the model writes
    `"kind": "min_typ_max"` -- an exact dictionary match -- but puts
    the value in `num_value`. `_coerce_kind` sees the kinds are equal
    and does nothing, then `_payload_ok` rejects the fact with
    "requires at least one of". A slot error is a SMALLER error than a
    cardinality error; the same principle (lossless = fix) applies.

    Only for `present`, and only ONE WAY: never touches a slot that is
    already filled; never invents missing data."""
    if fact.get("status") != "present":
        return None
    num = fact.get("num_value")
    if num is None:
        return None
    if dict_kind == "min_typ_max" and all(
        fact.get(s) is None for s in ("val_min", "val_typ", "val_max")
    ):
        fact["val_typ"], fact["num_value"] = num, None
    elif dict_kind == "range" and fact.get("val_min") is None and fact.get("val_max") is None:
        fact["val_min"] = fact["val_max"] = num
        fact["num_value"] = None
    elif dict_kind == "list" and not fact.get("items"):
        fact["items"] = [{"ordinal": 1, "item_code": _slugify_item(num),
                          "item_text": fact.get("raw_text") or str(num)}]
        fact["num_value"] = None
    return None


def _coerce_kind(fact: dict, dict_kind: str) -> str | None:
    """Deterministicallyically converts the model's payload to the
    dictionary's `kind`; on failure returns the reason text (the fact
    is dropped).

    Why it exists (2026-08-04 measurement): 353 of the 901 facts dropped
    on the first full-corpus run were "kind mismatch", mostly the SAME
    MEANING in different cardinalities -- the document says "8 channels"
    while the model writes `single`, the dictionary wants `conditional`;
    the document lists a single connector while the model writes
    `single`, the dictionary wants `list`. Losing this information is
    pointless: as long as the conversion is lossless and one-way, the
    code's job is to apply it. Meaning-DESTROYING conversions (e.g.
    squashing a range into a single number) are DELIBERATELY NOT done
    -- those still drop.

    `fact` is mutated IN PLACE (payload slots are moved)."""
    fact_kind = fact.get("kind")
    if fact_kind == dict_kind:
        return _repair_payload_slot(fact, dict_kind)
    if fact.get("status") != "present":
        # For a payload-less observation (absent/not_specified) `kind` has
        # no meaning -- silently aligned to the dictionary's kind.
        fact["kind"] = dict_kind
        return None

    num, text = fact.get("num_value"), fact.get("text_value")
    vmin, vtyp, vmax = fact.get("val_min"), fact.get("val_typ"), fact.get("val_max")

    if dict_kind == "conditional":
        # conditional = a single number tied to a condition; same payload
        # shape as `single`. The distinguishing feature is that the
        # condition is REQUIRED, and that is checked in _classify_fact.
        # If the model says `single` it is a pure label error.
        if fact_kind == "single" and num is not None:
            fact["kind"] = dict_kind
            return None
        return f"kind mismatch: fact={fact_kind!r} dictionary='conditional' (conditional needs exactly one num_value)"

    if dict_kind == "single":
        if fact_kind == "conditional" and num is not None:
            fact["kind"] = dict_kind
            return None
        if fact_kind in ("range", "min_typ_max"):
            # A single value can only collapse to single if min==max or
            # ONLY ONE end is filled. Squashing "0-250 V" to 250 is an
            # information LOSS -- rejected (this is the dictionary's
            # fault; fix the dictionary there).
            known = [v for v in (vmin, vtyp, vmax) if v is not None]
            if vtyp is not None:
                fact["num_value"] = vtyp
            elif len(set(known)) == 1:
                fact["num_value"] = known[0]
            else:
                return (f"kind mismatch: fact={fact_kind!r} dictionary='single' "
                        f"(min={vmin} typ={vtyp} max={vmax} cannot be collapsed)")
            fact["val_min"] = fact["val_typ"] = fact["val_max"] = None
            fact["kind"] = dict_kind
            return None
        if fact_kind == "list":
            items = [i for i in (fact.get("items") or []) if isinstance(i, dict)]
            if len(items) == 1:
                fact["text_value"] = items[0].get("item_code") or items[0].get("item_text")
                fact["items"] = []
                fact["kind"] = dict_kind
                return None
            return (f"kind mismatch: fact='list' dictionary='single' "
                    f"(list of {len(items)} items cannot be collapsed)")
        return f"kind mismatch: fact={fact_kind!r} dictionary='single'"

    if dict_kind == "min_typ_max":
        if fact_kind == "single" and num is not None:
            # A single nominal value = typ. This is LOSSLESS: min/max stay unknown.
            fact["val_typ"], fact["num_value"] = num, None
            fact["kind"] = dict_kind
            return None
        if fact_kind == "range" and (vmin is not None or vmax is not None):
            fact["kind"] = dict_kind
            return None
        return f"kind mismatch: fact={fact_kind!r} dictionary='min_typ_max'"

    if dict_kind == "range":
        if fact_kind == "min_typ_max" and (vmin is not None or vmax is not None):
            # val_typ CANNOT be carried into range (no column, the guard rejects)
            # -- if min/max are already filled, dropping typ is not a
            # loss, it's a summary.
            fact["val_typ"] = None
            fact["kind"] = dict_kind
            return None
        if fact_kind == "min_typ_max" and vtyp is not None:
            fact["val_min"] = fact["val_max"] = vtyp
            fact["val_typ"] = None
            fact["kind"] = dict_kind
            return None
        if fact_kind in ("single", "conditional") and num is not None:
            # "24 VDC" -- a single nominal value: range has both ends equal.
            # No information is lost, no width is invented.
            fact["val_min"] = fact["val_max"] = num
            fact["num_value"] = None
            fact["kind"] = dict_kind
            return None
        return f"kind mismatch: fact={fact_kind!r} dictionary='range'"

    if dict_kind == "list":
        if fact_kind in ("single", "conditional") and (text is not None or num is not None):
            value = text if text is not None else num
            fact["items"] = [{"ordinal": 1, "item_code": _slugify_item(value),
                              "item_text": fact.get("raw_text") or str(value)}]
            fact["num_value"] = fact["text_value"] = None
            fact["kind"] = dict_kind
            return None
        if fact_kind in ("range", "min_typ_max"):
            # "±10 V" -- an interval, treated as a single-element list for
            # a key defined as a list (a selectable range list); raw_text
            # preserved verbatim, numeric ends normalised in item_code.
            ends = [v for v in (vmin, vtyp, vmax) if v is not None]
            if not ends:
                return f"kind mismatch: fact={fact_kind!r} dictionary='list' (no value to carry)"
            label = fact.get("raw_text") or "-".join(str(v) for v in ends)
            fact["items"] = [{"ordinal": 1, "item_code": _slugify_item(label), "item_text": label}]
            fact["val_min"] = fact["val_typ"] = fact["val_max"] = None
            fact["kind"] = dict_kind
            return None
        return f"kind mismatch: fact={fact_kind!r} dictionary='list'"

    if dict_kind == "boolean":
        return f"kind mismatch: fact={fact_kind!r} dictionary='boolean'"

    return f"kind mismatch: fact={fact_kind!r} dictionary={dict_kind!r}"


def _slugify_item(value) -> str:
    """`item_code` generation -- the code-side equivalent of the model's
    requested normalisation (lowercase, snake_case); item_codes that
    coercion invents follow the SAME shape."""
    text = str(value).strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch == ".":
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_") or "unknown"


def _classify_fact(fact: dict, dictionary: Dictionary) -> tuple[str | None, str | None, str | None, str | None]:
    """Returns `(outcome, block, key, condition_or_reason)`.
    `outcome`: "ok" | "new_key" | None (dropped -- 4th element is the REASON text)."""
    block, key = fact.get("block"), fact.get("key")
    if fact.get("is_new_key"):
        # The model's `is_new_key` declaration is NOT TRUSTED verbatim
        # (2026-08-04 measurement): 47 of 139 new-key proposals (34%)
        # were keys that ACTUALLY EXISTED in the dictionary (e.g.
        # core/supply_voltage) -- data was being sent to the queue
        # unnecessarily. The dictionary is now the AUTHORITY: if the key
        # is genuinely missing it goes to the queue, otherwise it is
        # processed normally.
        if (block, key) in dictionary.kind_by_key:
            fact["is_new_key"] = False
        else:
            resolved = dictionary.blocks_by_key.get(key)
            if resolved and len(resolved) == 1:
                # Key exists but the model wrote the wrong block -- one
                # candidate -> fix; multiple candidates -> ambiguous,
                # goes to the queue.
                fact["block"], fact["is_new_key"] = next(iter(resolved)), False
                block = fact["block"]
            else:
                return "new_key", None, None, None

    k = (block, key)
    dict_kind = dictionary.kind_by_key.get(k)
    if dict_kind is None:
        return None, None, None, f"unknown key (block={block!r}, key={key!r}) -- not in the dictionary"
    problem = _coerce_kind(fact, dict_kind)
    if problem:
        return None, None, None, problem

    cond_code = fact.get("condition")
    if isinstance(cond_code, dict):
        # Model sometimes sends "condition": "ethernet" as
        # "condition": {"code": "ethernet"} -- normalise first.
        cond_code = cond_code.get("code")
    if cond_code and cond_code not in dictionary.conditions_by_key.get(k, set()):
        return None, None, None, f"condition {cond_code!r} is not in this attribute's closed list"
    if dict_kind == "conditional" and not cond_code:
        return None, None, None, "kind=conditional but no valid condition"

    return "ok", block, key, (cond_code or None)


def _payload_ok(fact: dict, kind: str) -> str | None:
    """Python-side mirror of the (spec_value_kind_guard) trigger --
    filters BEFORE INSERT/UPDATE so the transaction does not blow up.
    None on success."""
    status = fact.get("status")
    if status not in ("present", "absent", "not_specified"):
        return f"extraction pass cannot write status={status!r}"
    if status != "present":
        # sv_empty_status_ck: not_specified/absent CARRY NO PAYLOAD.
        if any(fact.get(s) is not None for s in PAYLOAD_FIELDS) or fact.get("items"):
            return f"status={status!r} CANNOT CARRY A PAYLOAD (model violation)"
        if status == "absent" and not fact.get("source_chunk_id"):
            return "status='absent' requires source_chunk_id (sv_absent_needs_source_ck)"
        return None
    if kind == "single" and len([s for s in ("num_value", "text_value") if fact.get(s) is not None]) != 1:
        return "kind=single requires exactly one of num_value/text_value"
    if kind == "boolean" and fact.get("bool_value") is None:
        return "kind=boolean requires bool_value"
    if kind == "range" and fact.get("val_min") is None and fact.get("val_max") is None:
        return "kind=range requires val_min and/or val_max"
    if kind == "min_typ_max" and not any(fact.get(s) is not None for s in ("val_min", "val_typ", "val_max")):
        return "kind=min_typ_max requires at least one of min/typ/max"
    if kind == "list" and not fact.get("items"):
        return "kind=list requires items[]"
    vmin, vmax, vtyp = fact.get("val_min"), fact.get("val_max"), fact.get("val_typ")
    # 2026-08-27 fix: the PN1330 run stopped with TypeError -- the model
    # sometimes sends a string instead of a number in val_min/val_max/
    # val_typ (e.g. "12" or "-40 C"). `_coerce_kind` does NOT convert
    # these fields to numbers, only carries them for kind matching --
    # the `>`/`<` below raised on a str/float comparison, falling into
    # the `except Exception` block of `main()`'s per-product loop and
    # dropping the WHOLE product (a single bad fact was losing ALL of
    # a product's facts). A non-numeric end now filters THIS fact (not
    # crash, not affect other facts).
    for slot_name, slot_val in (("val_min", vmin), ("val_max", vmax), ("val_typ", vtyp)):
        if slot_val is not None and not isinstance(slot_val, (int, float)):
            return f"{slot_name} is not numeric: {slot_val!r}"
    if vmin is not None and vmax is not None and vmin > vmax:
        return f"val_min {vmin} > val_max {vmax}"
    if vtyp is not None and ((vmin is not None and vtyp < vmin) or (vmax is not None and vtyp > vmax)):
        return f"val_typ {vtyp} is outside the interval"
    return None


def _value_signature(fact: dict) -> tuple:
    """Normalised signature of the value a fact CARRIES -- two facts with
    MATCHING signatures carry the same real value repeated in different
    sentences/documents (CORROBORATION, not a CONFLICT)."""
    items = [(i.get("item_code"), i.get("item_text")) for i in (fact.get("items") or []) if isinstance(i, dict)]
    items_sig = tuple(sorted(items, key=lambda pair: (pair[0] or "", pair[1] or "")))
    return (
        fact.get("status"), fact.get("num_value"), fact.get("text_value"), fact.get("val_min"),
        fact.get("val_typ"), fact.get("val_max"), fact.get("bool_value"), items_sig,
    )


def _union_list_members(valued: list[tuple[dict, int]]) -> list[tuple[dict, int]]:
    """Collapses ALL observations of a `kind='list'` key into ONE group:
    items are merged on `item_code` (first-seen `item_text` preserved,
    ordinals renumbered), `raw_text` is merged across sources.

    The returned thing is ALL members of the group -- because the caller
    feeds `_merge_evidence` ALL of them, evidence also merges
    automatically. The representative fact (highest trust) is updated
    IN PLACE."""
    representative, _ = max(valued, key=lambda m: m[1])
    merged: dict[str, dict] = {}
    raw_parts: list[str] = []
    for fact, _trust in valued:
        for item in fact.get("items") or []:
            if not isinstance(item, dict):
                continue
            code = item.get("item_code") or item.get("item_text")
            if code is None or code in merged:
                continue
            merged[code] = {"item_code": code, "item_text": item.get("item_text")}
        raw = (fact.get("raw_text") or "").strip()
        if raw and raw not in raw_parts:
            raw_parts.append(raw)
    representative["items"] = [
        {"ordinal": i, "item_code": v["item_code"], "item_text": v["item_text"]}
        for i, v in enumerate(merged.values(), start=1)
    ]
    if raw_parts:
        representative["raw_text"] = "\n".join(raw_parts)
    return valued


_INTERVAL_SLOTS = ("val_min", "val_typ", "val_max")


def _intervals_complementary(a: dict, b: dict) -> bool:
    """Do two `range`/`min_typ_max` observations COMPLEMENT each other
    WITHOUT conflicting?

    Complementary = (0) both have `status='present'` (see below), (1) no
    filled slot differs between them, AND (2) at least one slot has
    one side filling the other's empty. All three conditions together
    mean "two halves of the same interval"; if any one is unmet the
    groups stay SEPARATE (real conflict, or already same value).

    (0) BUG (fuzzing finding, 2026-08-26): an `status='absent'`
    observation has its `val_min`/`val_typ`/`val_max` ALWAYS None per
    `_payload_ok` -- this is semantically DIFFERENT from a `present`
    observation's "this end was not reported" empty (absent = value is
    MISSING, present-partial = value is THERE but one end is unknown).
    Without the status check both would count as "complementary" (absent
    is empty in every slot -> `fills=True` ALWAYS). `_payload_ok`'s
    guard prevents the ACCUMULATOR itself from being corrupted (absent
    accumulator + present fact -> merge refused), but the
    representative-carrying step at the END of this function (below,
    `representative[slot] = accumulator.get(slot)`) does NOT check
    STATUS: when a present accumulator + absent fact enter the SAME
    group (harmlessly), and the group's highest-trust member HAPPENS to
    be that absent observation, the accumulator's filled interval slots
    are COPIED onto the absent object -- resulting in a `status='absent'`
    row WITH `val_min`/`val_max` filled, violating `sv_empty_status_ck`.
    Fuzzing: ~20% repro rate on conditioned range/r/min_typ_max keys
    (`operating_temperature`/`supply_voltage`/`switch_voltage`).

    ATTRIBUTION NOTE (2026-08-26): the upstream BUG fix above is a
    SEPARATE root cause from the per-product fact-loss incident in the
    same night's run -- that loss appeared 34 times in logs as "UNIQUE
    constraint failed: index 'sv_unique_live_uix'" (per-product
    transaction rollback). That cause was NOT FOUND; reproduction was
    ATTEMPTED and FAILED (30,000 fuzz runs failed to reproduce the UNIQUE
    violation even ONCE -- a valuable negative finding but not
    PROOF) and the fault may RESURFACE on the next real run. This note
    is written DELIBERATELY: if "34 products' cause was found and
    fixed" is assumed, the real fault would stop being investigated.
    """
    if a.get("status") != b.get("status"):
        return False
    fills = False
    for slot in _INTERVAL_SLOTS:
        x, y = a.get(slot), b.get(slot)
        if x is not None and y is not None:
            if x != y:
                return False
        elif x is None and y is not None or x is not None and y is None:
            fills = True
    return fills


def _merge_complementary(valued: list[tuple[dict, int]], kind: str) -> list[list[tuple[dict, int]]]:
    """Reduces `kind='range'`/`'min_typ_max'` observations into COMPLEMENTARY
    sets -- the range counterpart of `_union_list_members`
    (PROTOCOL KARAR-064 research; ROADMAP 20a).

    MEASURED LOSS: when a chunk carries only ONE end of an interval the
    model returns `val_min=null, status=present`, which is INDISTINGUISHABLE
    from a real "up to X". The old code treated this half as a RIVAL
    to the full value, picking the winner by `(trust, GROUP SIZE)` --
    two halves from the same document beat the full value 2-1 and
    dropped it into `conflicting`, with `v_product_spec` then removing
    it entirely from the answer surface (PN1260 `storage_temperature`:
    visible `..125`, hidden `-55..125`). Measurement: 415 conflicting
    rows, 67 cases where the visible value was MORE INCOMPLETE than the
    hidden one; 54 of those were complementary (this function), 13 had
    missing conditioned split (ROADMAP 20d, separate work -- this
    function DELIBERATELY does not merge them).

    Merging happens ONLY when conflict-free (`_intervals_complementary`)
    and the result MUST pass `_payload_ok` -- if it does not (e.g. a
    merge that would produce `val_min > val_max`), the merge is NOT
    done and groups stay separate. So silently writing a corrupted row
    is structurally impossible.

    The returned thing is a LIST of groups (each group `[(fact, trust),
    ...]`); because the caller feeds `_merge_evidence` ALL members of
    each group, evidence also merges automatically. Each group's
    representative (highest trust) is enriched IN PLACE.

    CORROBORATION COMES FIRST: the function FIRST collapses observations
    repeating the SAME value via `_value_signature` (old behaviour,
    unchanged), THEN looks for complementarity between those groups.
    This order is mandatory -- looking for complementarity directly
    would miss same-value corroboration as "no slot-filling" and turn
    every repetition into a separate `conflicting` row (measured: 415
    -> 676 in the corpus).

    Accumulation lives in the group's FIRST member (`accumulator`); it
    is moved to the representative only AFTER the group closes --
    holding accumulation "in the current highest-trust member" would
    silently drop earlier merges when a higher-trust member joined
    later (the third member's `val_typ` would disappear).
    """
    sig_groups: dict[tuple, list[tuple[dict, int]]] = {}
    for fact, trust in valued:
        sig_groups.setdefault(_value_signature(fact), []).append((fact, trust))

    groups: list[list[tuple[dict, int]]] = []
    for members in sig_groups.values():
        fact = members[0][0]  # same-signature members carry the same value
        for group in groups:
            accumulator = group[0][0]
            if not _intervals_complementary(accumulator, fact):
                continue
            merged = {slot: (accumulator.get(slot) if accumulator.get(slot) is not None
                             else fact.get(slot)) for slot in _INTERVAL_SLOTS}
            if _payload_ok({**accumulator, **merged}, kind) is not None:
                continue
            accumulator.update(merged)
            for member_fact, _trust in members:
                raw = (member_fact.get("raw_text") or "").strip()
                existing = accumulator.get("raw_text") or ""
                if raw and raw not in existing:
                    accumulator["raw_text"] = f"{existing}\n{raw}".strip()
            group.extend(members)
            break
        else:
            groups.append(list(members))

    # `load_product` picks the representative via `max(group, trust)` --
    # move accumulation to it so the written row carries the merged value.
    for group in groups:
        accumulator = group[0][0]
        representative = max(group, key=lambda m: m[1])[0]
        if representative is accumulator:
            continue
        for slot in _INTERVAL_SLOTS:
            representative[slot] = accumulator.get(slot)
        representative["raw_text"] = accumulator.get("raw_text")
    return groups


def _status_priority(status: str) -> int:
    # 'present'/'absent' are a real STATEMENT; 'not_specified' CARRIES NO INFORMATION.
    return 1 if status in ("present", "absent") else 0


def _evidence_entries(fact: dict, chunk_lookup: dict, doc_type_by_id: dict[str, str]) -> list[dict]:
    """One `evidence[]` entry per source document for the fact -- the same
    chunk may fan out to multiple products/documents, all recorded.

    `heading_anchor` (2026-08-06, KARAR-064): if the chunk was sent SLICED
    to the LLM, the section it came from is recorded as a NOTE -- the
    evidence envelope is STILL the full chunk (`chunk_sha256` unchanged,
    `differ` continues to work); this is only a navigation hint for the
    human auditing a 47-attribution-wide chunk. Field is OMITTED when no
    slicing happened."""
    sha = fact.get("source_chunk_id")
    chunk = chunk_lookup.get(sha)
    if not chunk:
        return []
    anchors = list(getattr(chunk, "anchors_applied", []) or [])
    entries = []
    for did in chunk.doc_ids:
        entry = {"doc_id": did, "file_name": None, "chunk_sha256": sha,
                 "trust_rank": TRUST_RANK.get(doc_type_by_id.get(did, ""), 0)}
        if anchors:
            entry["heading_anchor"] = anchors
        entries.append(entry)
    return entries


def _merge_evidence(group_members: list[tuple[dict, int]], chunk_lookup: dict, doc_type_by_id: dict[str, str],
                     file_name_by_doc: dict[str, str]) -> tuple[list[dict], str | None, str | None, str | None]:
    """Merged `evidence[]` (with one entry per source) + (primary doc_id,
    primary file_name, primary chunk_sha256) from ALL members of a
    corroboration group -- primary = highest trust_rank source."""
    seen: dict[tuple[str, str], dict] = {}
    for fact, _trust in group_members:
        for e in _evidence_entries(fact, chunk_lookup, doc_type_by_id):
            key = (e["doc_id"], e["chunk_sha256"])
            if key not in seen:
                e["file_name"] = file_name_by_doc.get(e["doc_id"])
                seen[key] = e
    entries = sorted(seen.values(), key=lambda e: e["trust_rank"], reverse=True)
    if not entries:
        return [], None, None, None
    primary = entries[0]
    return (
        [{"doc_id": e["doc_id"], "file_name": e["file_name"], "chunk_sha256": e["chunk_sha256"]} for e in entries],
        primary["doc_id"], primary["file_name"], primary["chunk_sha256"],
    )


def _reset_product(cur: sqlite3.Cursor, product_code: str) -> int:
    """`--force`: rolls back THIS product's previous `chunk_full_run_v1`
    write -- claimed bootstrap rows (condition IS NULL) are restored to
    the correct `not_specified`/`not_applicable` state, rows added later
    (extra condition/conflicting) are DELETED entirely."""
    rows = cur.execute(
        "SELECT value_id, block, key, condition FROM spec_value WHERE product_code = ? AND extractor = ?",
        (product_code, EXTRACTOR),
    ).fetchall()
    if not rows:
        return 0

    # For each (block,key) the smallest value_id row is restored to
    # bootstrap; the rest (later INSERTed condition/conflicting rows)
    # are DELETED.
    #
    # ATTENTION -- bootstrap candidate selection does NOT look at
    # `condition` (2026-08-06 bug fix, ROADMAP 20e). The old code looked
    # for bootstrap as "row with condition IS NULL"; but `load_product`
    # FILLS the condition when claiming bootstrap via
    # `_write_row(..., condition=requested_condition, ...)`. So for an
    # already-loaded (product,block,key) there is NO NULL-condition row
    # anymore -> the old reset would FAIL to find it and DELETE it ->
    # the second `--force` run would say "bootstrap row not found" and
    # drop the fact. Measured damage: a full `--force` run took
    # spec_value 24456 -> 23399 rows, `present` 2860 -> 1892, and
    # DELETED every `kind='conditional'` row (243 of them). Rows in this
    # state in the corpus: 1229.
    first_id_by_key: dict[tuple[str, str], int] = {}
    for value_id, block, key, _condition in rows:
        k = (block, key)
        if k not in first_id_by_key or value_id < first_id_by_key[k]:
            first_id_by_key[k] = value_id

    for value_id, block, key, condition in rows:
        if value_id != first_id_by_key.get((block, key)):
            cur.execute("DELETE FROM spec_value WHERE value_id = ?", (value_id,))
            continue
        # not_applicable/not_specified distinction comes from PHASE A's
        # own rule; resetting it is not needed -- the very next
        # `load_product` call will UPDATE it again. We just need a safe
        # default that does not violate DB CHECKs.
        cur.execute(
            # `condition=NULL` is REQUIRED (2026-08-06): it may have been
            # filled during the claim; if not cleared, the next run will
            # AGAIN fail to find it as bootstrap and the same loss repeats.
            "UPDATE spec_value SET status='not_specified', condition=NULL, num_value=NULL, text_value=NULL, "
            "val_min=NULL, val_typ=NULL, val_max=NULL, bool_value=NULL, items=NULL, subfields=NULL, "
            "unit_observed=NULL, raw_text=NULL, source_doc_id=NULL, source_file_name=NULL, source_chunk_id=NULL, "
            "evidence='[]', extractor=NULL, extractor_version=NULL, extracted_at=NULL WHERE value_id=?",
            (value_id,),
        )
    return len(rows)


def load_product(
    con: sqlite3.Connection, dictionary: Dictionary, result: dict, *,
    doc_type_by_id: dict[str, str], dry_run: bool, new_key_fh, force: bool = False,
    issues_con: sqlite3.Connection | None = None,
    record_new_key_issues: bool = False,
) -> dict:
    """`record_new_key_issues` (O-14 prep, DEFAULT OFF): until O-14's
    decision (1) is approved by the user, `NEW_KEY_PROPOSAL` is NOT
    WRITTEN to `issues` -- even when `issues_con` is provided, this flag
    must be True separately for `record_new_key_proposal` to be called
    (see issues_bridge.py docstring). `SKIPPED_FACT` is OUTSIDE this: by
    M-03, it activates the moment `issues_con` is provided, without
    user approval (see `record_skipped_facts` docstring for the
    rationale)."""
    cur = con.cursor()
    product_code = result["model_code"]
    row = cur.execute("SELECT product_code FROM product WHERE UPPER(product_code) = UPPER(?)", (product_code,)).fetchone()
    if row is None:
        raise LoadError(f"product_code={product_code!r} is not in specs.db")
    product_code = row[0]

    already = cur.execute(
        "SELECT COUNT(*) FROM spec_value WHERE product_code = ? AND extractor = ?", (product_code, EXTRACTOR)
    ).fetchone()[0]

    report = {
        "model_code": product_code, "already_loaded_rows": already,
        "skipped": [], "new_key_facts": 0, "written_present": 0, "written_absent": 0,
        "written_conflicting": 0,
    }
    if already and not force:
        report["skip_reason"] = "already loaded (use --force to reprocess)"
        return report
    if already and force:
        report["reset_rows"] = _reset_product(cur, product_code)

    # KARAR-064: `load_indexes()` 5th element (`chunk_anchor_index`) was
    # added 2026-08-06; this place USES `anchors_applied` (see
    # `heading_anchor` note in `_evidence_entries`) -- without passing
    # the index the manifest is built span-less and the evidence note
    # silently disappears.
    (all_chunks_index, atoms_bridge_index, exclude,
     chunk_ownership_index, chunk_anchor_index) = load_indexes()
    manifest = build_product_manifest(
        con, product_code, all_chunks_index=all_chunks_index, atoms_bridge_index=atoms_bridge_index,
        exclude_doc_types=exclude, chunk_ownership_index=chunk_ownership_index,
        chunk_anchor_index=chunk_anchor_index,
    )
    chunk_lookup = {c.chunk_sha256: c for c in manifest.chunks}
    file_name_by_doc = dict(cur.execute(
        f"SELECT doc_id, file_name FROM document WHERE doc_id IN ({','.join('?' * len(manifest.doc_ids))})",
        manifest.doc_ids,
    ).fetchall()) if manifest.doc_ids else {}

    facts = result.get("facts") or []
    groups: dict[tuple[str, str, str | None], list[tuple[dict, int]]] = {}
    for fact in facts:
        if not isinstance(fact, dict):
            report["skipped"].append({"key": None, "reason": f"facts[] contains a non-object entry: {str(fact)[:80]!r}"})
            continue
        _normalize_numeric_payload(fact)
        outcome, block, key, extra = _classify_fact(fact, dictionary)
        if outcome == "new_key":
            report["new_key_facts"] += 1
            if not dry_run:
                new_key_fh.write(json.dumps({"product_code": product_code, "fact": fact}, ensure_ascii=False) + "\n")
                if issues_con is not None and record_new_key_issues:
                    # O-14 prep (not yet a PROTOCOL decision, see
                    # issues_bridge.record_new_key_proposal docstring) --
                    # until `record_new_key_issues=True` is passed
                    # (default False), this branch NEVER runs and
                    # JSONL-only behaviour is UNCHANGED.
                    record_new_key_proposal(issues_con, product_code=product_code, fact=fact)
            continue
        if outcome is None:
            report["skipped"].append({"key": fact.get("key"), "reason": extra})
            continue
        condition = extra
        dict_kind = dictionary.kind_by_key[(block, key)]
        problem = _payload_ok(fact, dict_kind)
        if problem:
            report["skipped"].append({"key": fact.get("key"), "reason": problem})
            continue
        sha = fact.get("source_chunk_id")
        chunk = chunk_lookup.get(sha)
        if chunk is None:
            # 2026-08-27 fix (K-102): evidence that cannot be resolved is
            # rejected for EVERY status. The model sometimes mangles
            # chunk_sha256 (seen: "sha2_d024f42b60f50..." -- a string
            # whose middle is garbled, never in chunk_lookup) -- the
            # chunk_lookup holds chunks sent to THIS product; a sha
            # outside it is either a hallucination or leakage from
            # another product, both INVALID evidence.
            #
            # PREVIOUSLY this gate was CLOSED only for `status='absent'`
            # (rationale: sv_absent_needs_source_ck requires non-NULL
            # source_chunk_id). `present` facts were getting through, and
            # `_merge_evidence` returned `([], None, None, None)` for
            # them -- the row was written as `evidence='[]'` + `extractor`
            # SET. That is the class of bug `staleness_audit` criterion
            # (b) `empty_evidence` was MEANT to catch, but its "no normal
            # write path produces this" claim was FALSE (2026-08-27
            # measurement: 386 rows / 94 products).
            #
            # If the evidence is meaningless the FACT itself is
            # unreliable: instead of suppressing the trace, we DROP it
            # and record it (with reason) in `report["skipped"]`.
            report["skipped"].append({"key": fact.get("key"), "reason": f"source_chunk_id ({str(sha)[:24]!r}) cannot be resolved in chunk_lookup -- invalid evidence (status={fact.get('status')!r})"})
            continue
        trust = 0
        if chunk:
            for did in chunk.doc_ids:
                trust = max(trust, TRUST_RANK.get(doc_type_by_id.get(did, ""), 0))
        groups.setdefault((block, key, condition), []).append((fact, trust))

    claimed_keys: set[tuple[str, str]] = set()

    for (block, key, requested_condition), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        valued = [m for m in members if _status_priority(m[0]["status"])]
        empty = [m for m in members if not _status_priority(m[0]["status"])]

        if not valued:
            empty.sort(key=lambda m: m[1], reverse=True)
            winner_fact, _ = empty[0]
            for loser_fact, _ in empty[1:]:
                report["skipped"].append({"key": loser_fact.get("key"), "reason": "duplicate 'not_specified' observation dropped"})
            value_groups = [[(winner_fact, 0)]]
        else:
            for f, _t in empty:
                report["skipped"].append({"key": f.get("key"), "reason": "'not_specified' observation dropped (this key/condition already has a real value)"})
            if dictionary.kind_by_key[(block, key)] == "list":
                # `list` kind: DIFFERENT item sets are NOT a CONFLICT, they
                # are PARTS of the same list (2026-08-04 measurement): for
                # PN1071 `interfaces` 11 rows were written -- each chunk
                # saw a SUBSET of the interfaces and the loader treated
                # them as conflicting values. Correct behaviour is SET
                # UNION: all members' items collected in a single row,
                # evidence also merged from all of them.
                value_groups = [_union_list_members(valued)]
            elif dictionary.kind_by_key[(block, key)] in ("range", "min_typ_max"):
                # ROADMAP 20a: for interval kinds, two complementary
                # halves are NOT a CONFLICT, they are parts of the SAME
                # interval. First merge the conflict-free ones; the
                # REMAINING real conflicts compete by the old ordering
                # (trust, group size).
                merged = _merge_complementary(valued, dictionary.kind_by_key[(block, key)])
                value_groups = sorted(merged, key=lambda g: (max(t for _, t in g), len(g)), reverse=True)
            else:
                sig_groups: dict[tuple, list[tuple[dict, int]]] = {}
                for fact, trust in valued:
                    sig_groups.setdefault(_value_signature(fact), []).append((fact, trust))
                value_groups = sorted(sig_groups.values(), key=lambda g: (max(t for _, t in g), len(g)), reverse=True)

        dict_kind = dictionary.kind_by_key[(block, key)]
        unit = dictionary.unit_by_key.get((block, key))

        for i, group_members in enumerate(value_groups):
            rep_fact, _ = max(group_members, key=lambda m: m[1])
            target_status = rep_fact["status"]
            is_winner = (i == 0)
            evidence, src_doc, src_file, src_chunk = _merge_evidence(group_members, chunk_lookup, doc_type_by_id, file_name_by_doc)
            if not evidence:
                # K-102 second gate (belt + suspenders): after the
                # above entry gate every member's chunk has been resolved,
                # so a normal run should NEVER reach here with empty
                # evidence. Still, we cut the write here -- a row with
                # `evidence='[]'` + `extractor` SET is permanent residue
                # (`forget_source` only deletes rows whose evidence
                # FIELD empties) that no write path cleans up. If a new
                # group/merge path opens later, it must not silently
                # produce the same class.
                report["skipped"].append({"key": rep_fact.get("key"), "reason": "merged evidence[] is empty -- no evidenceless rows are written"})
                continue
            if target_status == "present":
                rep_fact["_items_json"], rep_fact["_subfields_json"] = _items_subfields_json(rep_fact, dictionary, block, key, report)
            else:
                rep_fact["_items_json"], rep_fact["_subfields_json"] = None, None

            try:
                if is_winner and (block, key) not in claimed_keys:
                    bootstrap = cur.execute(
                        "SELECT value_id, status FROM spec_value WHERE product_code=? AND block=? AND key=? AND condition IS NULL",
                        (product_code, block, key),
                    ).fetchone()
                    if bootstrap is None:
                        report["skipped"].append({"key": rep_fact.get("key"), "reason": "bootstrap row not found (was PHASE A run?)"})
                        continue
                    bootstrap_value_id, bootstrap_status = bootstrap
                    claimed_keys.add((block, key))
                    if bootstrap_status in ("not_specified", "not_applicable"):
                        value_id = bootstrap_value_id
                        _write_row(cur, value_id=value_id, condition=requested_condition, fact=rep_fact, status=target_status,
                                    unit=unit, evidence=evidence, src_doc=src_doc, src_file=src_file, src_chunk=src_chunk)
                    elif rep_fact["status"] != "present":
                        report["skipped"].append({"key": rep_fact.get("key"), "reason": f"bootstrap already filled, new {rep_fact['status']!r} observation (payload-less) cannot be written as a conflict -- dropped"})
                        continue
                    else:
                        _insert_row(cur, product_code=product_code, family=manifest.family, subfamily=manifest.subfamily,
                                    block=block, key=key, kind=dict_kind, unit=unit, condition=requested_condition,
                                    fact=rep_fact, status="conflicting", evidence=evidence, src_doc=src_doc, src_file=src_file, src_chunk=src_chunk)
                        report["written_conflicting"] += 1
                        target_status = "conflicting"
                elif not is_winner and rep_fact["status"] != "present":
                    report["skipped"].append({"key": rep_fact.get("key"), "reason": "winner 'present', loser 'absent' (payload-less) cannot be written as a conflict -- dropped"})
                    continue
                else:
                    insert_status = target_status if is_winner else "conflicting"
                    _insert_row(cur, product_code=product_code, family=manifest.family, subfamily=manifest.subfamily,
                                block=block, key=key, kind=dict_kind, unit=unit, condition=requested_condition,
                                fact=rep_fact, status=insert_status, evidence=evidence, src_doc=src_doc, src_file=src_file, src_chunk=src_chunk)
                    if not is_winner:
                        report["written_conflicting"] += 1
                    target_status = insert_status
            except sqlite3.IntegrityError as exc:
                # 2026-08-27 (root-cause hunt): `sv_unique_live_uix` violation
                # dropped 37 products ENTIRELY (transaction rollback) on
                # the 2026-08-26 run, cause not reproducible even with
                # 30,000 fuzz runs. The old error message only carried
                # "IntegrityError: UNIQUE constraint failed: ..." -- it
                # did NOT say WHICH (block,key,condition,status) write
                # was failing. So the next real occurrence should show
                # this detail (current row's status included) directly
                # in the error -- behaviour UNCHANGED (same exception
                # type, same ROLLBACK), only diagnostics enriched.
                existing = cur.execute(
                    "SELECT value_id, status, extractor, extracted_at FROM spec_value "
                    "WHERE product_code=? AND block=? AND key=? AND COALESCE(condition,'')=COALESCE(?,'')",
                    (product_code, block, key, requested_condition),
                ).fetchall()
                raise sqlite3.IntegrityError(
                    f"{exc} -- attempted: product={product_code} block={block!r} "
                    f"key={key!r} condition={requested_condition!r} new_status={target_status!r} "
                    f"is_winner={is_winner} "
                    f"-- existing rows for the same (block,key,condition): {existing!r}"
                ) from exc

            if target_status == "present":
                report["written_present"] += 1
            elif target_status == "absent":
                report["written_absent"] += 1

    if issues_con is not None and not dry_run and report["skipped"]:
        # M-03: all facts dropped for THIS product go to `issues` in a
        # single batched write (same pattern as O-07's
        # record_unresolved_docs/chunks -- one go at the end of the list,
        # not one per fact). HELD BACK for `dry_run` -- the "no writes"
        # promise for `--dry-run` extends to `issues.db` too (same spirit
        # as specs.db's SAVEPOINT/ROLLBACK, but issues.db is a SEPARATE
        # file so it does NOT see that rollback).
        record_skipped_facts(issues_con, report["skipped"], source_doc=product_code)

    return report


def _items_subfields_json(fact: dict, dictionary: Dictionary, block: str, key: str, report: dict) -> tuple[str | None, str | None]:
    items = []
    for fallback_ordinal, item in enumerate(fact.get("items") or [], start=1):
        if not isinstance(item, dict) or not item.get("item_code"):
            report["skipped"].append({"key": fact.get("key"), "reason": f"invalid list item: {str(item)[:80]!r}"})
            continue
        items.append({
            "ordinal": item.get("ordinal") or fallback_ordinal,
            "item_code": item.get("item_code"), "item_text": item.get("item_text"),
        })
    subfields = []
    known_names = dictionary.subfield_names_by_key.get((block, key), set())
    for sf in fact.get("subfields") or []:
        if not isinstance(sf, dict):
            report["skipped"].append({"key": fact.get("key"), "reason": f"invalid subfield: {str(sf)[:80]!r}"})
            continue
        if sf.get("name") not in known_names:
            report["skipped"].append({"key": fact.get("key"), "reason": f"unknown subfield {sf.get('name')!r}"})
            continue
        subfields.append(sf)
    return (
        json.dumps(items, ensure_ascii=False) if items else None,
        json.dumps(subfields, ensure_ascii=False) if subfields else None,
    )


def _write_row(cur: sqlite3.Cursor, *, value_id: int, condition: str | None, fact: dict, status: str,
               unit: str | None, evidence: list, src_doc: str | None, src_file: str | None, src_chunk: str | None) -> None:
    cur.execute(
        "UPDATE spec_value SET status=?, condition=?, num_value=?, text_value=?, val_min=?, val_typ=?, val_max=?, "
        "bool_value=?, items=?, subfields=?, unit_observed=?, raw_text=?, source_doc_id=?, source_file_name=?, "
        "source_chunk_id=?, evidence=?, extractor=?, extractor_version=?, extracted_at=? WHERE value_id=?",
        (
            status, condition, fact.get("num_value"), fact.get("text_value"), fact.get("val_min"),
            fact.get("val_typ"), fact.get("val_max"),
            (1 if fact.get("bool_value") else 0) if fact.get("bool_value") is not None else None,
            fact.get("_items_json"), fact.get("_subfields_json"),
            fact.get("unit_observed"), fact.get("raw_text"), src_doc, src_file, src_chunk,
            json.dumps(evidence, ensure_ascii=False), EXTRACTOR, extractor_version(), _now(), value_id,
        ),
    )


def _insert_row(cur: sqlite3.Cursor, *, product_code: str, family: str | None, subfamily: str | None,
                 block: str, key: str, kind: str, unit: str | None, condition: str | None, fact: dict, status: str,
                 evidence: list, src_doc: str | None, src_file: str | None, src_chunk: str | None) -> int:
    cur.execute(
        "INSERT INTO spec_value (product_code, family, subfamily, block, key, kind, unit, condition, status, "
        "num_value, text_value, val_min, val_typ, val_max, bool_value, items, subfields, unit_observed, raw_text, "
        "source_doc_id, source_file_name, source_chunk_id, evidence, extractor, extractor_version, extracted_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            product_code, family, subfamily, block, key, kind, unit, condition, status,
            fact.get("num_value"), fact.get("text_value"), fact.get("val_min"), fact.get("val_typ"), fact.get("val_max"),
            (1 if fact.get("bool_value") else 0) if fact.get("bool_value") is not None else None,
            fact.get("_items_json"), fact.get("_subfields_json"), fact.get("unit_observed"), fact.get("raw_text"),
            src_doc, src_file, src_chunk, json.dumps(evidence, ensure_ascii=False),
            EXTRACTOR, extractor_version(), _now(),
        ),
    )
    return cur.lastrowid


def load_product_incremental(
    con: sqlite3.Connection, dictionary: Dictionary, result: dict, *,
    doc_type_by_id: dict[str, str], new_key_fh,
    issues_con: sqlite3.Connection | None = None,
) -> dict:
    """O-12: entry point of the nightly incremental run -- "phase, when
    re-run, invalidates its own old derivatives" (replace semantics).

    Why `force=True` (and not `--force`'s existing `_reset_product` path)
    -- it is NOT O-11's `forget_source`: `forget_source` DELETES a
    `spec_value` row when its `evidence[]` fully empties (the correct
    behaviour per I-17 -- the source TRULY went away). But this
    product's `bootstrap` rows (PHASE A's `condition IS NULL` seed, ONE
    per `(product,block,key)`) are OFTEN fed from a single source --
    calling `forget_source(doc_ids=<this run's own documents>)` would
    DELETE such a row, and `_reset_product`'s bootstrap-safe reset path
    would FAIL to find it (the lookup is `condition IS NULL`, the row is
    gone -> "bootstrap row not found" silently drops that attribute --
    permanently, until PHASE A is rerun). `_reset_product`
    (`force=True`) does the OPPOSITE: it always restores bootstrap to
    `not_specified`, NEVER deletes -- so it is the safe way to
    invalidate one's own old derivatives for the same product
    (2026-08-06 ROADMAP 20e measurement; see `_reset_product` docstring).
    `forget_source` is NOT USED in this flow.

    Acceptance check (O-12): running the SAME `result` TWICE through
    `load_product_incremental` leaves `spec_value` row count AND
    `evidence[]` lengths UNCHANGED -- the second call first restores
    everything to bootstrap via `_reset_product`, then RE-WRITES the
    same fact set.
    """
    return load_product(
        con, dictionary, result, doc_type_by_id=doc_type_by_id,
        dry_run=False, new_key_fh=new_key_fh, force=True,
        issues_con=issues_con,
    )


def main(argv: list[str] | None = None) -> dict:
    """N-22: `argv=None` (default) -- SAME pattern as
    `run_parse_pipeline.main(argv=[])` (see `run_nightly.py` module
    docstring): `run_nightly.stage_load` calls this function with
    `argv=[...]` PROGRAMMATICALLY; it does NOT open a subprocess. The
    total counters computed below (`n_present`/`n_absent`/
    `n_conflict`/`n_new_key`/`n_skipped_facts`) are no longer only
    PRINTED but also RETURNED -- they are the only real source of
    `run_nightly.py`'s nightly report for this phase."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=None, help="Comma-separated product_code list (default: ALL in results/).")
    ap.add_argument("--doc-id", default=None,
                    help="O-12: process only products that USE this document (nightly incremental run). "
                         "Combined with --models, the INTERSECTION is taken. This flag uses REPLACE "
                         "semantics: `load_product_incremental` (equivalent to force=True) is used -- "
                         "no need to also pass --force.")
    ap.add_argument("--dry-run", action="store_true", help="No writes -- report only.")
    ap.add_argument("--force", action="store_true", help="Re-process already-loaded products (can INCREASE conflicting rows).")
    ap.add_argument("--results-dir", default=None,
                    help="Fact JSON source (default: results/). Useful for model comparison: apply the "
                         "SAME loader to a different model's output and put the loss rates side by side.")
    ap.add_argument("--issues-db", default=None,
                    help="M-03: path to the common issues.db where dropped facts (SKIPPED_FACT) are "
                         "written (default: ISSUES_DB_PATH env var, otherwise specs.db's SIBLING file, "
                         "K-75). Pass --no-issues to fully TURN OFF.")
    ap.add_argument("--no-issues", action="store_true",
                    help="Write NOTHING to issues.db (old behaviour -- only _load_report.json).")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    if not results_dir.is_dir():
        raise SystemExit(f"{results_dir} does not exist")

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")

    if args.doc_id:
        codes = products_for_doc_id(con, args.doc_id)
        if args.models:
            wanted = {c.strip() for c in args.models.split(",")}
            codes = [c for c in codes if c in wanted]
        paths = [results_dir / f"{c}.json" for c in codes]
        print(f"--doc-id={args.doc_id}: {len(codes)} products affected -> {codes}")
    elif args.models:
        codes = [c.strip() for c in args.models.split(",")]
        paths = [results_dir / f"{c}.json" for c in codes]
    else:
        paths = sorted(p for p in results_dir.glob("*.json") if not p.name.startswith("_"))

    # O-08 (I-18): pre-write integrity gate -- on failure it stops
    # WITHOUT writing a single row (the dictionary/loop below does not
    # run).
    try:
        verify_db_integrity(con)
    except SchemaIntegrityError as exc:
        con.close()
        print(f"INTEGRITY_CHECK_FAILED (pre-write): {exc}")
        raise SystemExit(1) from exc

    dictionary = load_dictionary(con)
    doc_type_by_id = dict(con.execute("SELECT doc_id, doc_type FROM document").fetchall())

    # M-03: `issues.db` is a SEPARATE file from specs.db (K-75) -- specs.db's
    # dry-run SAVEPOINT/ROLLBACK is independent of its own connection, so
    # the `dry_run` check in `load_product`/`load_product_incremental`
    # belongs THERE, not here.
    #
    # 2026-08-26 fix: `DEFAULT_ISSUES_DB_PATH` REPLACED with
    # `resolve_issues_db_path()` -- CALL-TIME reading of `ISSUES_DB_PATH`
    # env var (so the container can carry a persistent path, e.g.
    # `/corpus/facts/issues.db`).
    issues_con = None if args.no_issues else connect_issues(
        Path(args.issues_db) if args.issues_db else resolve_issues_db_path()
    )

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    new_key_path = QUEUE_DIR / "new_key_proposals.jsonl"
    reports = []
    with new_key_path.open("a", encoding="utf-8") as new_key_fh:
        for path in paths:
            if not path.is_file():
                reports.append({"model_code": path.stem, "error": f"{path} does not exist"})
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("skipped_reason") or not result.get("facts"):
                reports.append({"model_code": result.get("model_code", path.stem), "skip_reason": "no facts in result"})
                continue
            try:
                if args.dry_run:
                    con.execute("SAVEPOINT dry")
                if args.doc_id and not args.dry_run:
                    # O-12: dry-run + --doc-id's interaction with
                    # force=True's ROLLBACK TO savepoint was not tested
                    # -- when both are requested the normal load_product
                    # path is used (force=args.force, the else branch below).
                    report = load_product_incremental(
                        con, dictionary, result, doc_type_by_id=doc_type_by_id, new_key_fh=new_key_fh,
                        issues_con=issues_con,
                    )
                else:
                    report = load_product(
                        con, dictionary, result, doc_type_by_id=doc_type_by_id,
                        dry_run=args.dry_run, new_key_fh=new_key_fh, force=args.force,
                        issues_con=issues_con,
                    )
                if args.dry_run:
                    con.execute("ROLLBACK TO dry")
                else:
                    con.commit()
                reports.append(report)
            except Exception as exc:  # noqa: BLE001 -- an UNEXPECTED error per product
                if args.dry_run:
                    con.execute("ROLLBACK TO dry")
                else:
                    con.rollback()
                reports.append({"model_code": result.get("model_code", path.stem), "error": f"{type(exc).__name__}: {exc}"})

    # O-08 (I-18): post-write integrity gate -- if this run broke the DB
    # it is reported as *failed* (group N reads this).
    try:
        verify_db_integrity(con)
    except SchemaIntegrityError as exc:
        con.close()
        if issues_con is not None:
            issues_con.close()
        print(f"INTEGRITY_CHECK_FAILED (post-write): {exc}")
        raise SystemExit(1) from exc

    con.close()
    if issues_con is not None:
        issues_con.close()
    n_present = sum(r.get("written_present", 0) for r in reports)
    n_absent = sum(r.get("written_absent", 0) for r in reports)
    n_conflict = sum(r.get("written_conflicting", 0) for r in reports)
    n_new_key = sum(r.get("new_key_facts", 0) for r in reports)
    n_skipped_facts = sum(len(r.get("skipped", [])) for r in reports)
    print(f"{'[DRY RUN] ' if args.dry_run else ''}{len(reports)} products processed: "
          f"present={n_present} absent={n_absent} conflicting={n_conflict} "
          f"new_key(queued)={n_new_key} skipped-fact={n_skipped_facts}")
    for r in reports:
        if r.get("error"):
            print(f"  ERROR {r['model_code']}: {r['error']}")
        elif r.get("skip_reason"):
            print(f"  skipped {r['model_code']}: {r['skip_reason']}")
    (results_dir / "_load_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "n_products": len(reports), "n_present": n_present, "n_absent": n_absent,
        "n_conflicting": n_conflict, "n_new_key": n_new_key, "n_skipped_facts": n_skipped_facts,
        "reports": reports,
    }


if __name__ == "__main__":
    main()