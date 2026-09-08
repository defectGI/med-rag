"""Locks PHASE C's (chunk_full_run/load_to_db.py) loss-reduction layers
added on 2026-08-04: kind coercion, `list` set-union, dictionary-
authoritative new-key resolution, and run_full.py's `source_chunk_id`
stamping.

O-04 (2026-08-20): restored from the `facts/tests/test_chunk_full_run_load.py`
in the `facts-kodu-son-hali` tag. The old file loaded
`experiments/chunk_full_run/<module>.py` via package-outside means
(`importlib.util` + `sys.path.insert`); after O-03 those modules are
REAL package members (`medrag.pipeline.facts.discover` / `.load_to_db` /
`.run_full`, `medrag.pipeline.facts.catalog_chunk_ownership.build_all`),
so plain imports are used here; the hack is gone.

KNOWN BLOCK (same family as K-71, 02-SERVISLESTIRME-TASKLARI.md O-03
"DONE" note item 2): `load_to_db.py`, `run_full.py` and
`catalog_chunk_ownership/build_all.py` read `prompts/*.md` at module
level; those prompt files were NOT part of O-02's "9 files" set, so
these 3 modules currently throw `FileNotFoundError` at import. The
import block is preserved as-is so we don't grow the surface -- as long
as those modules can't be imported, the function-marked tests below
SKIP; tests depending only on `discover.py` (clean import chain) run
normally. Once the prompt files are moved into the package these skips
self-lift.
"""
from __future__ import annotations

import io
import json
import sqlite3

import pytest

from medrag.pipeline.facts import discover

try:
    from medrag.pipeline.facts import load_to_db

    _LOAD_TO_DB_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 -- deliberate broad catch, see module-level note
    load_to_db = None
    _LOAD_TO_DB_ERROR = exc

try:
    from medrag.pipeline.facts import run_full

    _RUN_FULL_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    run_full = None
    _RUN_FULL_ERROR = exc

try:
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    _BUILD_ALL_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    build_all = None
    _BUILD_ALL_ERROR = exc


_needs_load_to_db = pytest.mark.skipif(
    load_to_db is None,
    reason=f"medrag.pipeline.facts.load_to_db import blocked (prompts/*.md not moved into package): {_LOAD_TO_DB_ERROR}",
)
_needs_run_full = pytest.mark.skipif(
    run_full is None,
    reason=f"medrag.pipeline.facts.run_full import blocked (prompts/chunk_single_system.md not moved into package): {_RUN_FULL_ERROR}",
)
_needs_build_all = pytest.mark.skipif(
    build_all is None,
    reason=f"catalog_chunk_ownership.build_all import blocked (prompts/chunk_owner_system.md not moved into package): {_BUILD_ALL_ERROR}",
)


def _dictionary(**kinds):
    """Just enough dictionary for the test -- `kinds` {key: kind}, all 'core'."""
    return load_to_db.Dictionary(
        kind_by_key={("core", k): v for k, v in kinds.items()},
        unit_by_key={("core", k): None for k in kinds},
        conditions_by_key={("core", k): {"dc", "ac"} for k in kinds},
        subfield_names_by_key={("core", k): set() for k in kinds},
        blocks_by_key={k: {"core"} for k in kinds},
    )


def _fact(**kw):
    base = {"block": "core", "key": "x", "status": "present", "kind": "single",
            "num_value": None, "text_value": None, "val_min": None, "val_typ": None,
            "val_max": None, "items": [], "raw_text": "..."}
    base.update(kw)
    return base


# --- kind coercion: LOSSLESS conversions are accepted --------------------------

@_needs_load_to_db
def test_single_with_condition_becomes_conditional():
    """Largest single loss measured (49 facts): the document says
    "8 channels", the model finds the right condition but writes kind
    'single'."""
    fact = _fact(kind="single", num_value=8, condition="dc")
    assert load_to_db._coerce_kind(fact, "conditional") is None
    assert fact["kind"] == "conditional" and fact["num_value"] == 8


@_needs_load_to_db
def test_single_becomes_min_typ_max_as_typ():
    fact = _fact(kind="single", num_value=5.0)
    assert load_to_db._coerce_kind(fact, "min_typ_max") is None
    assert fact["val_typ"] == 5.0 and fact["num_value"] is None


@_needs_load_to_db
def test_single_becomes_range_with_equal_ends():
    """'24 VDC' -- a single nominal value; range has both ends equal,
    NO invented width."""
    fact = _fact(kind="single", num_value=24)
    assert load_to_db._coerce_kind(fact, "range") is None
    assert (fact["val_min"], fact["val_max"]) == (24, 24) and fact["num_value"] is None


@_needs_load_to_db
def test_min_typ_max_becomes_range_drops_typ():
    fact = _fact(kind="min_typ_max", val_min=18, val_typ=28, val_max=75)
    assert load_to_db._coerce_kind(fact, "range") is None
    assert (fact["val_min"], fact["val_max"]) == (18, 75) and fact["val_typ"] is None


@_needs_load_to_db
def test_range_becomes_min_typ_max_keeps_ends():
    fact = _fact(kind="range", val_min=4.75, val_max=5.25)
    assert load_to_db._coerce_kind(fact, "min_typ_max") is None
    assert (fact["val_min"], fact["val_max"]) == (4.75, 5.25)


@_needs_load_to_db
def test_single_becomes_single_item_list():
    fact = _fact(kind="single", text_value="DSUB-50", raw_text="Front Panel Connector: DSUB-50")
    assert load_to_db._coerce_kind(fact, "list") is None
    assert fact["items"] == [
        {"ordinal": 1, "item_code": "dsub_50", "item_text": "Front Panel Connector: DSUB-50"}
    ]
    assert fact["text_value"] is None


@_needs_load_to_db
def test_single_item_list_becomes_single():
    fact = _fact(kind="list", items=[{"ordinal": 1, "item_code": "rtrm", "item_text": "RTRM"}])
    assert load_to_db._coerce_kind(fact, "single") is None
    assert fact["text_value"] == "rtrm" and fact["items"] == []


@_needs_load_to_db
def test_range_becomes_single_item_list():
    fact = _fact(kind="range", val_min=-10, val_max=10, raw_text="±10 V")
    assert load_to_db._coerce_kind(fact, "list") is None
    assert len(fact["items"]) == 1 and fact["items"][0]["item_text"] == "±10 V"


@_needs_load_to_db
def test_payloadless_observation_aligns_silently():
    """For absent/not_specified `kind` is meaningless -- the old code
    dropped these with "kind mismatch" (measurement: 4 weight + 2
    rack_units)."""
    fact = _fact(kind="not_specified", status="not_specified", num_value=None)
    assert load_to_db._coerce_kind(fact, "single") is None
    assert fact["kind"] == "single"


# --- kind is correct but slot is wrong --------------------------------------

@_needs_load_to_db
def test_matching_kind_with_value_in_wrong_slot_is_repaired():
    """Pattern seen on the gemma4:31b run: `kind` is an exact dictionary
    match ('min_typ_max') but the value sits in `num_value`. Old code
    saw the kinds equal, did nothing, then `_payload_ok` dropped it."""
    fact = _fact(kind="min_typ_max", num_value=24)
    assert load_to_db._coerce_kind(fact, "min_typ_max") is None
    assert fact["val_typ"] == 24 and fact["num_value"] is None


@_needs_load_to_db
def test_range_with_value_in_num_slot_is_repaired():
    fact = _fact(kind="range", num_value=15)
    assert load_to_db._coerce_kind(fact, "range") is None
    assert (fact["val_min"], fact["val_max"]) == (15, 15)


@_needs_load_to_db
def test_repair_does_not_touch_a_ifilled_canonical_slot():
    fact = _fact(kind="min_typ_max", num_value=99, val_min=1, val_max=5)
    load_to_db._coerce_kind(fact, "min_typ_max")
    assert (fact["val_min"], fact["val_max"]) == (1, 5)
    assert fact["val_typ"] is None, "a filled slot must not be touched"


# --- kind coercion: meaning-DESTROYING conversions are rejected --------------

@_needs_load_to_db
def test_real_range_is_not_squashed_into_single():
    """Squashing '0-250 V' to 250 is an information LOSS. These cases
    indicate the dictionary is wrong (switch_voltage was fixed there);
    coercion does not silence them."""
    fact = _fact(kind="range", val_min=0, val_max=250)
    problem = load_to_db._coerce_kind(fact, "single")
    assert problem is not None and "cannot be collapsed" in problem


@_needs_load_to_db
def test_multi_item_list_is_not_squashed_into_single():
    fact = _fact(kind="list", items=[{"item_code": "a"}, {"item_code": "b"}])
    problem = load_to_db._coerce_kind(fact, "single")
    assert problem is not None and "cannot be collapsed" in problem


@_needs_load_to_db
def test_conditional_without_num_value_is_rejected():
    fact = _fact(kind="list", items=[{"item_code": "a"}, {"item_code": "b"}])
    assert load_to_db._coerce_kind(fact, "conditional") is not None


@_needs_load_to_db
def test_boolean_target_never_coerced():
    fact = _fact(kind="single", num_value=1)
    assert load_to_db._coerce_kind(fact, "boolean") is not None


# --- dictionary-authoritative new-key resolution ----------------------------

@_needs_load_to_db
def test_is_new_key_overridden_when_key_exists():
    """Measured loss: 47 of 139 proposals (34%) were keys that ALREADY
    EXISTED in the dictionary -- the model's declaration was unreliable,
    the dictionary is authoritative."""
    d = _dictionary(supply_voltage="min_typ_max")
    fact = _fact(key="supply_voltage", is_new_key=True, kind="min_typ_max", val_typ=3.3)
    outcome, block, key, _ = load_to_db._classify_fact(fact, d)
    assert outcome == "ok" and (block, key) == ("core", "supply_voltage")
    assert fact["is_new_key"] is False


@_needs_load_to_db
def test_wrong_block_repaired_when_unambiguous():
    d = _dictionary(input_impedance="single")
    fact = _fact(block="digital_io", key="input_impedance", is_new_key=True,
                 kind="single", num_value=50)
    outcome, block, _key, _ = load_to_db._classify_fact(fact, d)
    assert outcome == "ok" and block == "core"


@_needs_load_to_db
def test_genuinely_new_key_still_queued():
    d = _dictionary(supply_voltage="range")
    fact = _fact(key="stub_voltage", is_new_key=True, kind="single", num_value=1)
    outcome, _b, _k, _ = load_to_db._classify_fact(fact, d)
    assert outcome == "new_key"


# --- list set-union --------------------------------------------------------

@_needs_load_to_db
def test_list_members_union_instead_of_conflict():
    """PN1071 `interfaces`: each chunk sees a SUBSET of the interfaces;
    the old code treated them as conflicting values and wrote 11 rows."""
    a = _fact(kind="list", raw_text="HDMI, USB",
              items=[{"ordinal": 1, "item_code": "hdmi", "item_text": "HDMI"},
                     {"ordinal": 2, "item_code": "usb", "item_text": "USB"}])
    b = _fact(kind="list", raw_text="SFP, PCIe",
              items=[{"ordinal": 1, "item_code": "sfp", "item_text": "SFP"},
                     {"ordinal": 2, "item_code": "pcie", "item_text": "PCI-e"}])
    c = _fact(kind="list", raw_text="HDMI",
              items=[{"ordinal": 1, "item_code": "hdmi", "item_text": "HDMI (repeat)"}])

    members = load_to_db._union_list_members([(a, 100), (b, 60), (c, 50)])
    assert len(members) == 3, "evidence merge needs ALL members returned"
    representative = a  # highest trust
    codes = [i["item_code"] for i in representative["items"]]
    assert codes == ["hdmi", "usb", "sfp", "pcie"], "union, no duplicates"
    assert [i["ordinal"] for i in representative["items"]] == [1, 2, 3, 4]
    # first-seen item_text preserved ('HDMI', not 'HDMI (repeat)')
    assert representative["items"][0]["item_text"] == "HDMI"
    assert "SFP, PCIe" in representative["raw_text"]


@_needs_load_to_db
def test_union_of_of_one_member_is_noop_shaped():
    a = _fact(kind="list", raw_text="HDMI",
              items=[{"ordinal": 1, "item_code": "hdmi", "item_text": "HDMI"}])
    load_to_db._union_list_members([(a, 100)])
    assert a["items"] == [{"ordinal": 1, "item_code": "hdmi", "item_text": "HDMI"}]


@_needs_load_to_db
@pytest.mark.parametrize("value,expected", [
    ("RS-485", "rs_485"),
    ("Front Panel Connector: DSUB-50", "front_panel_connector_dsub_50"),
    ("  ", "unknown"),
    ("1.2 MΩ", "1.2_mω"),
])
def test_slugify_item(value, expected):
    """NOTE -- this function does NOT align with the corpus vocabulary:
    the model produces `rs485` while coercion produces `rs_485`. Reason:
    the corpus has no consistent rule (`rs485` but `dsub_50`,
    `mil_std_1553`); normalising `item_code` is a separate piece of
    work (2026-08-04 audit, item (d) -- 81 different spellings for
    `compliance_standards`). This test accepts that inconsistency and
    only locks down that the function is deterministic -- when the
    normalisation work is done, it must deliberately change this."""
    assert load_to_db._slugify_item(value) == expected


# --- run_full.py: source_chunk_id STAMPED by code ---------------------------

@_needs_run_full
def test_model_source_fields_are_stripped_and_replaced(monkeypatch):
    """The mangled-sha bug class (41.4% of facts) must be structurally
    impossible: every source field the model sent is dropped, the
    correct sha is written."""

    class _Chunk:
        chunk_sha256 = "sha256:" + "a" * 64
        text = "Operating Temperature: 0 °C to +55 °C"

    class _Manifest:
        product_code, family, subfamily, display_name = "DE1", "F", None, "t"

    reply = {"facts": [
        {"key": "operating_temperature", "source_chunk_id": "sha25_mangled", "val_min": 0, "val_max": 55},
        {"key": "weight", "chunk_sha256": "sha256:" + "b" * 64, "num_value": 1000},
        "non-object entry",
    ]}
    monkeypatch.setattr(run_full, "_chat", lambda *a, **k: (json.dumps(reply), {}))

    facts, reports = run_full._call_chunk(
        "ollama", "m", "http://x", None, _Manifest(), _Chunk(), "0", "sys"
    )
    dict_facts = [f for f in facts if isinstance(f, dict)]
    assert all(f["source_chunk_id"] == _Chunk.chunk_sha256 for f in dict_facts)
    assert not any("chunk_sha256" in f for f in dict_facts)
    assert reports[0]["model_sent_source_fields"] == 2
    assert "non-object entry" in facts, "non-object entry must be left for the loader's report"


@_needs_run_full
def test_user_message_carries_no_chunk_hash():
    class _Chunk:
        chunk_sha256 = "sha256:" + "c" * 64
        text = "body"

    class _Manifest:
        product_code, family, subfamily, display_name = "DE1", "F", None, "t"

    message = run_full.build_user_message(_Manifest(), _Chunk())
    assert "sha256" not in message and "body" in message


@_needs_run_full
def test_system_message_carries_the_dictionary():
    """Dictionary must live in the system message (for Ollama's KV
    prefix cache) -- NOT in the user message."""
    system = run_full.build_system_message()
    assert "Attribute dictionary" in system and "blocks" in system


# --- ROADMAP 20e: --force dropped data on the second run ------------------

def _skeleton_db():
    """Two bootstrap rows in a minimal spec_value -- one already
    CLAIMED (condition populated), the other untouched."""
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE spec_value (
        value_id INTEGER PRIMARY KEY, product_code TEXT, block TEXT, key TEXT,
        condition TEXT, status TEXT, num_value REAL, text_value TEXT,
        val_min REAL, val_typ REAL, val_max REAL, bool_value INTEGER,
        items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
        source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT,
        evidence TEXT, extractor TEXT, extractor_version TEXT, extracted_at TEXT)""")
    con.executemany(
        "INSERT INTO spec_value (value_id, product_code, block, key, condition, status, extractor) VALUES (?,?,?,?,?,?,?)",
        [(1, "DE1", "core", "supply_voltage", "dc", "present", load_to_db.EXTRACTOR),
         (2, "DE1", "core", "supply_voltage", "ac", "conflicting", load_to_db.EXTRACTOR),
         (3, "DE1", "core", "data_rate", None, "present", load_to_db.EXTRACTOR)],
    )
    return con


@_needs_load_to_db
def test_reset_restores_a_claimed_bootstrap_row():
    """MEASURED BUG (2026-08-06): `load_product` FILLS the condition
    when claiming bootstrap; the old `_reset_product` looked for
    bootstrap as "row with condition IS NULL" -- so it could NOT FIND it
    and DELETED it. Result: on the second `--force` run "bootstrap row
    not found" -> fact dropped. Full-run measured damage: spec_value
    24456 -> 23399 rows, `present` 2860 -> 1892, and DELETED every
    `kind='conditional'` row (243)."""
    con = _skeleton_db()
    cur = con.cursor()

    load_to_db._reset_product(cur, "DE1")

    rows = dict(cur.execute(
        "SELECT key, condition FROM spec_value WHERE product_code='DE1'").fetchall())
    assert set(rows) == {"supply_voltage", "data_rate"}, "one row per (block,key) must remain"
    assert rows["supply_voltage"] is None, "claimed bootstrap must be restored to condition=NULL"
    assert rows["data_rate"] is None


@_needs_load_to_db
def test_reset_clears_payload_and_provenance():
    con = _skeleton_db()
    cur = con.cursor()
    cur.execute("UPDATE spec_value SET num_value=12, raw_text='12 V', evidence='[{}]' WHERE value_id=1")

    load_to_db._reset_product(cur, "DE1")

    row = cur.execute(
        "SELECT status, num_value, raw_text, evidence, extractor FROM spec_value WHERE value_id=1").fetchone()
    assert row == ("not_specified", None, None, "[]", None)


# --- Pausable / resumable run (2026-08-06, user request) -------------------

@_needs_run_full
def test_skip_existing_reprocesses_a_partial_product(tmp_path):
    """If `--skip-existing` only checked file existence, a run killed
    at 3/11 chunks of a product would SKIP that product on resume and
    leave it silently incomplete. A `partial=true` file MUST be
    re-processed."""
    done = tmp_path / "done.json"
    done.write_text(json.dumps({"partial": False, "facts": []}), encoding="utf-8")
    half = tmp_path / "half.json"
    half.write_text(json.dumps({"partial": True, "facts": []}), encoding="utf-8")

    assert run_full._is_partial(done) is False, "finished product must be SKIPPED"
    assert run_full._is_partial(half) is True, "half product must be re-processed"


@_needs_run_full
def test_unreadable_result_counts_as_partial(tmp_path):
    """Rather than trust a half-written JSON, re-process it."""
    broken = tmp_path / "broken.json"
    broken.write_text('{"partial": fal', encoding="utf-8")

    assert run_full._is_partial(broken) is True
    assert run_full._is_partial(tmp_path / "missing.json") is True


# --- ROADMAP 20b / KARAR-064: span-based chunk ownership -------------------

_C16 = (
    "**Product Overview**\n\n"
    "## 2.6 RECOMENDED STUB PN1281\n\nTERMINATION: ...\n\n"
    "## 3. ENVIRONMENTAL SPECIFICATIONS\n\n"
    "## 3.1 HIGH TEMPERATURE ... +125C\n\nOPERATING:\n\n"
    "## 3.2 LOW TEMPERATURE ... -55C\n\nOPERATING:"
)


def test_slice_stops_at_the_next_heading():
    """PN1253 `c16`: a product-specific table and a family-wide
    environmental section in the SAME chunk. PN1281 must receive only
    its own section -- otherwise another product's numbers can be
    misattributed to it."""
    piece = discover.slice_by_anchor(_C16, "## 2.6 RECOMENDED STUB PN1281")

    assert piece.startswith("## 2.6 RECOMENDED STUB PN1281")
    assert "TERMINATION" in piece
    assert "ENVIRONMENTAL" not in piece, "must STOP at the next heading"


def test_slice_includes_subsections_of_the_anchored_section():
    """REGRESSION -- the real shape of PN1253 `c16`: family-wide section
    `# 3. ENVIRONMENTAL SPECIFICATIONS` (single `#`), with the temperature
    values under its `## 3.1`/`## 3.2` sub-headings. If the slice stopped
    at any heading the span would be EMPTY, recreating the SAME loss
    just fixed. A section includes its own sub-sections: the slice only
    stops at a heading of the SAME or HIGHER level."""
    text = (
        "## 2.6 RECOMENDED STUB PN1281\n\nTERMINATION\n\n"
        "# 3. ENVIRONMENTAL SPECIFICATIONS\n\n"
        "## 3.1 HIGH TEMPERATURE +125C\n\nOPERATING\n\n"
        "## 3.2 LOW TEMPERATURE -55C\n\nOPERATING\n\n"
        "# 4. ORDERING INFORMATION\n\nTYPE 1"
    )

    piece = discover.slice_by_anchor(text, "# 3. ENVIRONMENTAL SPECIFICATIONS")

    assert "+125C" in piece and "-55C" in piece, "sub-sections MUST be included in the span"
    assert "ORDERING" not in piece, "must STOP at the next same-level section"
    assert "PN1281" not in piece, "must NOT OVERFLOW into the previous section"


def test_slice_runs_to_the_end_when_no_following_heading():
    piece = discover.slice_by_anchor(_C16, "## 3.2 LOW TEMPERATURE ... -55C")

    assert piece.endswith("OPERATING:")


def test_slice_returns_none_for_an_unknown_anchor():
    """If the anchor is NOT IN the text, NO slicing happens (None) --
    the caller sends the full chunk and records the situation to
    `anchors_not_found`. No silent skipping."""
    assert discover.slice_by_anchor(_C16, "## 9.9 INVENTED HEADING") is None


def test_anchor_index_keeps_only_accepted_rows_with_an_anchor(tmp_path):
    """Accepted rows WITHOUT an anchor do NOT enter the index -- that
    means "whole chunk" (KARAR-064 clause 2), and must not trigger
    slicing."""
    path = tmp_path / "own.json"
    path.write_text(json.dumps({"rows": [
        {"doc_id": "d1", "chunk_id": "c16", "product_code": "PN1281",
         "accepted": True, "heading_anchor": "## 2.6 X"},
        {"doc_id": "d1", "chunk_id": "c16", "product_code": "PN1260",
         "accepted": True, "heading_anchor": None},          # no anchor -> whole chunk
        {"doc_id": "d1", "chunk_id": "c16", "product_code": "PN1267",
         "accepted": False, "heading_anchor": "## 3. Y"},    # NOT accepted
    ]}), encoding="utf-8")

    index = discover.load_chunk_anchor_index(path)

    assert index == {("d1", "c16", "PN1281"): ["## 2.6 X"]}


def test_anchor_index_collects_multiple_sections_for_one_product(tmp_path):
    """A product may own multiple sections in the same chunk (its own
    table + a family-wide section) -- BOTH must be carried."""
    path = tmp_path / "own.json"
    path.write_text(json.dumps({"rows": [
        {"doc_id": "d1", "chunk_id": "c16", "product_code": "PN1281",
         "accepted": True, "heading_anchor": "## 2.6 X"},
        {"doc_id": "d1", "chunk_id": "c16", "product_code": "PN1281",
         "accepted": True, "heading_anchor": "## 3. Y"},
    ]}), encoding="utf-8")

    assert discover.load_chunk_anchor_index(path)[("d1", "c16", "PN1281")] == ["## 2.6 X", "## 3. Y"]


def test_anchor_index_is_empty_when_the_table_is_missing(tmp_path):
    """When the table is missing the behaviour is UNCHANGED -- empty
    index = no slicing."""
    assert discover.load_chunk_anchor_index(tmp_path / "missing.json") == {}


@_needs_load_to_db
def test_evidence_records_the_heading_anchor_when_the_chunk_was_sliced():
    """KARAR-064: the evidence envelope stays the FULL chunk
    (`chunk_sha256` unchanged, differ keeps working) but when the chunk
    was sliced to a section, that section is recorded as a NOTE in
    `evidence[]` -- for the human auditing a 47-attribution-wide chunk."""
    chunk = discover.ChunkRow(
        chunk_sha256="sha256:" + "a" * 64, text="...", doc_ids=["d1"],
        page_start=5, page_end=5, anchors_applied=["# 3. ENVIRONMENTAL SPECIFICATIONS"],
    )
    fact = _fact(source_chunk_id=chunk.chunk_sha256)

    entries = load_to_db._evidence_entries(fact, {chunk.chunk_sha256: chunk}, {"d1": "CATALOGUE"})

    assert len(entries) == 1
    assert entries[0]["chunk_sha256"] == chunk.chunk_sha256, "envelope must remain the full chunk"
    assert entries[0]["heading_anchor"] == ["# 3. ENVIRONMENTAL SPECIFICATIONS"]


@_needs_load_to_db
def test_evidence_omits_the_anchor_field_when_no_slicing_happened():
    """When no slicing happened, the field is OMITTED -- same shape as
    older records."""
    chunk = discover.ChunkRow(
        chunk_sha256="sha256:" + "b" * 64, text="...", doc_ids=["d1"],
        page_start=1, page_end=1,
    )
    fact = _fact(source_chunk_id=chunk.chunk_sha256)

    entries = load_to_db._evidence_entries(fact, {chunk.chunk_sha256: chunk}, {"d1": "DATASHEET"})

    assert "heading_anchor" not in entries[0]


@_needs_build_all
def test_ownership_prompt_validates_the_anchor_against_the_chunk():
    """A model-invented heading cannot be used as an anchor: code
    validates the anchor against the closed set extracted from the
    body."""
    headings = build_all.headings_in(_C16)

    assert "## 3. ENVIRONMENTAL SPECIFICATIONS" in headings
    assert build_all.normalize_anchor({"heading_anchor": "## 3. ENVIRONMENTAL SPECIFICATIONS"},
                                      headings) == "## 3. ENVIRONMENTAL SPECIFICATIONS"
    assert build_all.normalize_anchor({"heading_anchor": "environmental section"}, headings) is None
    assert build_all.normalize_anchor({}, headings) is None


# --- ROADMAP 20a: complementary merging for interval kinds -----------------

def _rng(vmin=None, vtyp=None, vmax=None, raw="...", kind="range"):
    return _fact(kind=kind, val_min=vmin, val_typ=vtyp, val_max=vmax, raw_text=raw)


@_needs_load_to_db
def test_complementary_halves_merge_instead_of_conflicting():
    """PN1260 `storage_temperature`: three chunks of the same document
    -- one full (`-55..125`), two halves (`..125`). The old code let
    the halves beat the full value 2-1 and dropped it into
    `conflicting`."""
    full = _rng(vmin=-55, vmax=125, raw="-55°C to +125°C")
    half_a = _rng(vmax=125, raw="+125°C")
    half_b = _rng(vmax=125, raw="+125°C OPERATING")

    groups = load_to_db._merge_complementary([(full, 90), (half_a, 90), (half_b, 90)], "range")

    assert len(groups) == 1, "complementary halves MUST form ONE group"
    assert len(groups[0]) == 3, "evidence merge needs ALL members in the group"
    representative = max(groups[0], key=lambda m: m[1])[0]
    assert (representative["val_min"], representative["val_max"]) == (-55, 125)


@_needs_load_to_db
def test_half_first_still_yields_the_full_interval():
    """If the half observation comes FIRST the result is still the
    same -- merge order must not matter (in real runs fact order is
    chunk order, not guaranteed)."""
    half = _rng(vmax=125, raw="+125°C")
    full = _rng(vmin=-55, vmax=125, raw="-55°C to +125°C")

    groups = load_to_db._merge_complementary([(half, 90), (full, 90)], "range")

    assert len(groups) == 1
    representative = max(groups[0], key=lambda m: m[1])[0]
    assert (representative["val_min"], representative["val_max"]) == (-55, 125)


@_needs_load_to_db
def test_higher_trust_member_carries_the_merged_value():
    """The representative is the highest-trust member; the merged
    value is moved to IT -- otherwise `rep_fact` (selected by max trust)
    would write the old half value."""
    half_low = _rng(vmax=125, raw="+125°C")
    full_high = _rng(vmin=-55, vmax=125, raw="-55°C to +125°C")

    groups = load_to_db._merge_complementary([(half_low, 10), (full_high, 90)], "range")

    representative = max(groups[0], key=lambda m: m[1])[0]
    assert representative is full_high
    assert (representative["val_min"], representative["val_max"]) == (-55, 125)


@_needs_load_to_db
def test_genuine_conflict_is_not_merged():
    """PN1092 `switch_voltage`: '250 V AC' vs '-410..220 V DC' -- filled
    `val_max`es DIFFER, so not two halves of the same interval. They stay
    separate (real conditioned split is a separate job, ROADMAP 20d)."""
    ac = _rng(vmax=250, raw="up to 250 V AC")
    dc = _rng(vmin=-410, vmax=220, raw="Switch Voltage: -410 to 220 V DC")

    groups = load_to_db._merge_complementary([(ac, 90), (dc, 90)], "range")

    assert len(groups) == 2, "conflicting values DO NOT MERGE"


@_needs_load_to_db
def test_absent_observation_is_not_merged_as_the_missing_half():
    """MEASURED BUG (2026-08-26 nightly run): an `status='absent'`
    observation has its interval slots ALWAYS None per `_payload_ok`
    -- without the status check this was treated as "complementary"
    to a half `present` (e.g. only `val_max` filled). The merge
    would fill the absent accumulator's interval slots with the
    present's values but keep `status='absent'` -- a row violating
    `sv_empty_status_ck`, the actual cause of the per-product fact
    loss that night (transaction rollback); for conditioned
    range/r/min_typ_max keys (~20% repro rate -- fuzzing)."""
    present_half = _rng(vmax=125, raw="+125°C")
    absent = _fact(kind="range", status="absent", val_min=None, val_typ=None, val_max=None,
                   source_chunk_id="sha256:" + "a" * 64)

    groups = load_to_db._merge_complementary([(present_half, 90), (absent, 60)], "range")

    assert len(groups) == 2, "absent NEVER merges with an observation of another status"
    statuses = {g[0][0]["status"] for g in groups}
    assert statuses == {"present", "absent"}
    # The present half must STAY UNCORRUPTED (not overwritten by the absent's None slots).
    rep = next(g[0][0] for g in groups if g[0][0]["status"] == "present")
    assert (rep["val_min"], rep["val_max"]) == (None, 125)


@_needs_load_to_db
def test_numeric_string_payload_is_normalized_to_float():
    """2026-08-27 fix: the model sometimes returns `num_value`/
    `val_min`/`val_typ`/`val_max` as a numeric STRING -- `_value_signature`
    compares these fields as dict keys, so `"20" != 20.0` (different
    type) accidentally dropped TWO observations carrying the SAME real
    value into `conflicting`. `_normalize_numeric_payload` fixes the
    fact in place."""
    fact = {"num_value": "20", "val_min": " 0,5 ", "val_typ": None, "val_max": "-40"}
    load_to_db._normalize_numeric_payload(fact)
    assert fact["num_value"] == 20.0
    assert fact["val_min"] == 0.5
    assert fact["val_typ"] is None
    assert fact["val_max"] == -40.0


@_needs_load_to_db
def test_non_numeric_string_payload_is_left_alone():
    """A genuinely non-numeric string (e.g. with a unit appended) is NOT
    silently coerced to 0 -- left as-is, `_payload_ok`'s type guard
    then drops it (see that function's 2026-08-27 note)."""
    fact = {"num_value": "-40 C", "val_min": None, "val_typ": None, "val_max": None}
    load_to_db._normalize_numeric_payload(fact)
    assert fact["num_value"] == "-40 C"


@_needs_load_to_db
def test_numeric_string_and_float_of_same_value_corroborate():
    """End-to-end: `num_value` arriving as a string and as a float for
    the SAME value must, after normalisation, collapse into one group
    (corroboration) via `_value_signature` -- without normalisation they
    would become two `conflicting` rows."""
    a = {"status": "present", "num_value": "20", "text_value": None,
         "val_min": None, "val_typ": None, "val_max": None, "bool_value": None, "items": []}
    b = {"status": "present", "num_value": 20.0, "text_value": None,
         "val_min": None, "val_typ": None, "val_max": None, "bool_value": None, "items": []}
    load_to_db._normalize_numeric_payload(a)
    load_to_db._normalize_numeric_payload(b)
    assert load_to_db._value_signature(a) == load_to_db._value_signature(b)


def test_identical_values_still_corroborate_into_one_group():
    """REGRESSION (measured: corpus conflicting 415 -> 676): if
    complementarity is searched directly, a SAME value is not
    "slot-filling" so it is NOT counted as complementary and every
    repetition becomes a separate `conflicting` row. So the function
    FIRST collapses same-value observations via `_value_signature`."""
    a = _rng(vmin=0, vmax=55)
    b = _rng(vmin=0, vmax=55)

    groups = load_to_db._merge_complementary([(a, 90), (b, 90)], "range")

    assert len(groups) == 1, "same value is NOT a conflict, it is corroboration"
    assert len(groups[0]) == 2, "both members stay in the group for evidence merge"


@_needs_load_to_db
def test_merge_is_refused_when_result_would_be_invalid():
    """A merge that would produce `val_min > val_max` is NOT done --
    silently writing a corrupted row is avoided; groups stay
    separate (`_payload_ok` gate)."""
    lower_only = _rng(vmin=500)
    upper_only = _rng(vmax=100)

    groups = load_to_db._merge_complementary([(lower_only, 90), (upper_only, 90)], "range")

    assert len(groups) == 2


@_needs_load_to_db
def test_min_typ_max_merges_on_the_typ_slot():
    a = _rng(vmin=27.9, vmax=28.3, kind="min_typ_max")
    b = _rng(vtyp=28.0, kind="min_typ_max")

    groups = load_to_db._merge_complementary([(a, 90), (b, 90)], "min_typ_max")

    assert len(groups) == 1
    representative = max(groups[0], key=lambda m: m[1])[0]
    assert (representative["val_min"], representative["val_typ"], representative["val_max"]) == (27.9, 28.0, 28.3)


@_needs_load_to_db
def test_merged_raw_text_keeps_both_sources():
    full = _rng(vmin=-55, vmax=125, raw="-55°C to +125°C")
    half = _rng(vmax=125, raw="+125°C OPERATING")

    groups = load_to_db._merge_complementary([(full, 90), (half, 90)], "range")

    representative = max(groups[0], key=lambda m: m[1])[0]
    assert "-55°C to +125°C" in representative["raw_text"]
    assert "+125°C OPERATING" in representative["raw_text"]


@_needs_load_to_db
def test_duplicate_raw_text_is_not_repeated():
    full = _rng(vmin=-55, vmax=125, raw="+125°C")
    half = _rng(vmax=125, raw="+125°C")

    groups = load_to_db._merge_complementary([(full, 90), (half, 90)], "range")

    representative = max(groups[0], key=lambda m: m[1])[0]
    assert representative["raw_text"].count("+125°C") == 1


@_needs_load_to_db
def test_third_member_merge_survives_a_vitsh_trust_joiner():
    """Regression: if accumulation lived in the "current highest-trust
    member", a higher-trust member joining in between would SILENTLY
    DROP earlier merges (the first member's `val_max` or the third's
    `val_typ` would disappear). Accumulation lives in the group's
    FIRST member, only moved to the representative at the end."""
    low_max = _rng(vmax=125, raw="+125", kind="min_typ_max")
    high_min = _rng(vmin=-55, raw="-55", kind="min_typ_max")
    low_typ = _rng(vtyp=25, raw="25 typ", kind="min_typ_max")

    groups = load_to_db._merge_complementary(
        [(low_max, 10), (high_min, 90), (low_typ, 10)], "min_typ_max")

    assert len(groups) == 1
    representative = max(groups[0], key=lambda m: m[1])[0]
    assert representative is high_min
    assert (representative["val_min"], representative["val_typ"], representative["val_max"]) == (-55, 25, 125)


@_needs_load_to_db
def test_single_member_group_is_untouched():
    only = _rng(vmax=125, raw="+125°C")

    groups = load_to_db._merge_complementary([(only, 90)], "range")

    assert groups == [[(only, 90)]]
    assert (only["val_min"], only["val_max"]) == (None, 125)


# --- O-06: catalog_chunk_ownership pipeline acceptance (I-13) --------------
# Acceptance: zero model calls for single-owner documents, countable
# calls for multi-owner; confidence>=0.8 gate validated with a fake
# `_chat_ollama` WITHOUT touching real Ollama.

@_needs_build_all
def test_split_ownership_work_single_owner_doc_yields_zero_llm_jobs():
    docs = [{"doc_id": "d1", "file_name": "f1.pdf", "doc_type": "DATASHEET", "product_codes": ["DE1"]}]
    chunks_for = lambda doc_id: [(1, 1, "d1::c0", "body text")]

    rows, missing, llm_jobs, single_docs, multi_docs = build_all.split_ownership_work(
        docs, chunks_for, product_info={}, verbose=False)

    assert llm_jobs == [], "single-owner document must have EMPTY LLM job list (I-13)"
    assert missing == []
    assert len(single_docs) == 1 and multi_docs == []
    assert rows == [{
        "doc_id": "d1", "file_name": "f1.pdf", "chunk_id": "d1::c0",
        "page_start": 1, "page_end": 1, "product_code": "DE1",
        "confidence": 1.0, "accepted": True, "evidence": None, "source": "deterministic",
    }]


@_needs_build_all
def test_single_owner_corpus_calls_the_model_zero_times(monkeypatch):
    """I-13 acceptance criterion: `_chat_ollama` is NEVER CALLED -- every
    chunk of every single-owner document becomes a deterministic row."""
    calls = []
    monkeypatch.setattr(
        build_all, "_chat_ollama",
        lambda *a, **k: (calls.append((a, k)), (json.dumps({"assignments": []}), {}))[1],
    )
    docs = [
        {"doc_id": f"d{i}", "file_name": f"f{i}.pdf", "doc_type": "DATASHEET", "product_codes": [f"DE{i}"]}
        for i in range(5)
    ]
    chunks_for = lambda doc_id: [(1, 1, f"{doc_id}::c0", "body")]

    rows, _missing, llm_jobs, _single_docs, _multi_docs = build_all.split_ownership_work(
        docs, chunks_for, product_info={}, verbose=False)
    for job in llm_jobs:  # in a single-owner corpus this loop NEVER runs
        build_all.run_job(job)

    assert len(rows) == 5
    assert llm_jobs == []
    assert calls == [], "single-owner corpus must have ZERO model calls"


@_needs_build_all
def test_multi_owner_doc_schedules_one_llm_job_per_chunk_without_calling_yet(monkeypatch):
    calls = []
    monkeypatch.setattr(build_all, "_chat_ollama", lambda *a, **k: calls.append(1) or ("{}", {}))
    docs = [{"doc_id": "d2", "file_name": "f2.pdf", "doc_type": "CATALOGUE", "product_codes": ["DE1", "DE2"]}]
    chunks_for = lambda doc_id: [
        (1, 1, "d2::c0", "body 1"), (2, 2, "d2::c1", "body 2"), (3, 3, "d2::c2", "body 3"),
    ]
    product_info = {"DE1": {"product_code": "DE1"}, "DE2": {"product_code": "DE2"}}

    rows, _missing, llm_jobs, _single_docs, _multi_docs = build_all.split_ownership_work(
        docs, chunks_for, product_info, verbose=False)

    assert rows == [], "multi-owner document's rows are NOT yet produced at job-list stage"
    assert len(llm_jobs) == 3, "one job per chunk"
    assert calls == [], "no model call must have happened during job-list construction"


@_needs_build_all
def test_multi_owner_chunk_triggers_exactly_one_call_per_chunk(monkeypatch):
    """For multi-owner files call count is exactly proportional to
    chunk count (O-06: 'call count for multi-owner shows up in the report')."""
    calls = []

    def fake_chat(system, user):
        calls.append(user)
        return json.dumps({"assignments": [{"product_code": "DE1", "confidence": 0.95, "evidence": "x"}]}), {}

    monkeypatch.setattr(build_all, "_chat_ollama", fake_chat)
    d = {"doc_id": "d3", "file_name": "f3.pdf", "doc_type": "CATALOGUE", "product_codes": ["DE1", "DE2"]}
    jobs = [
        (d, (1, 1, "d3::c0", "chunk 0 text"), [{"product_code": "DE1"}, {"product_code": "DE2"}]),
        (d, (2, 2, "d3::c1", "chunk 1 text"), [{"product_code": "DE1"}, {"product_code": "DE2"}]),
    ]

    for job in jobs:
        build_all.run_job(job)

    assert len(calls) == 2, "two chunks -- two separate model calls"


@_needs_build_all
def test_confidence_gate_accepts_only_at_or_above_threshold(monkeypatch):
    """confidence>=0.8 gate: sub-threshold assignments are NOT DELETED
    but kept as `accepted=False` (see build_all.py docstring, lines ~16-22)."""
    d = {"doc_id": "d4", "file_name": "f4.pdf", "doc_type": "CATALOGUE", "product_codes": ["DE1", "DE2"]}
    assignments = [
        {"product_code": "DE1", "confidence": 0.8, "evidence": "exact threshold"},
        {"product_code": "DE2", "confidence": 0.79, "evidence": "below threshold"},
    ]

    new_rows, stats = build_all.absorb_chunk_result(
        d, 1, 1, "d4::c0", "body text", assignments, error=None)

    by_code = {r["product_code"]: r for r in new_rows}
    assert by_code["DE1"]["accepted"] is True, "threshold-EQUAL must be accepted"
    assert by_code["DE2"]["accepted"] is False, "below-threshold is REJECTED"
    assert by_code["DE2"] in new_rows, "rejected row is NOT DELETED, kept as accepted=False"
    assert stats["error"] == 0


@_needs_build_all
def test_confidence_gate_rejects_non_numeric_confidence():
    d = {"doc_id": "d5", "file_name": "f5.pdf", "doc_type": "CATALOGUE", "product_codes": ["DE1"]}
    assignments = [{"product_code": "DE1", "confidence": None, "evidence": "uncertain"}]

    new_rows, _stats = build_all.absorb_chunk_result(d, 1, 1, "d5::c0", "body", assignments, error=None)

    assert new_rows[0]["accepted"] is False


@_needs_build_all
def test_absorb_chunk_result_records_the_error_row(monkeypatch):
    d = {"doc_id": "d6", "file_name": "f6.pdf", "doc_type": "CATALOGUE", "product_codes": ["DE1"]}

    new_rows, stats = build_all.absorb_chunk_result(
        d, 1, 1, "d6::c0", "body", [], error="cannot parse assignments JSON")

    assert stats["error"] == 1
    assert new_rows == [{
        "doc_id": "d6", "file_name": "f6.pdf", "chunk_id": "d6::c0",
        "page_start": 1, "page_end": 1, "product_code": None,
        "confidence": None, "accepted": False, "evidence": None,
        "heading_anchor": None, "source": "llm", "error": "cannot parse assignments JSON",
    }]


# --- O-10 (I-16): two sources -- same value corroborates, DIFFERENT ---------
# values produce a conflict. `load_product()` is exercised END-TO-END,
# against a REAL sqlite connection, with `load_indexes()` monkey-patched
# to fake chunk indices, run TWICE -- reading the old production rows
# is NOT treated as evidence; this test RE-PRODUCES them.

def _evidence_test_db():
    """Minimal but REAL specs.db schema: every table/view `load_product()`
    touches (product, document + v_document_product, attribute,
    spec_value) -- the real `schema_rag.sql`'s minimum-reproduction
    equivalent. Two documents (DATASHEET/trust=100, CATALOGUE/trust=60),
    both SINGLE-owner (to stay out of fan-out)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT, subfamily TEXT, display_name TEXT)")
    con.execute("""CREATE TABLE document (
        doc_id TEXT PRIMARY KEY, file_name TEXT, doc_type TEXT,
        product_codes TEXT, product_code TEXT, is_active INTEGER)""")
    con.execute("CREATE VIEW v_document_product AS SELECT doc_id, doc_type, product_code, is_active FROM document")
    con.execute("CREATE TABLE attribute (block TEXT, key TEXT, kind TEXT, unit TEXT, conditions TEXT, subfields TEXT)")
    con.execute("""CREATE TABLE spec_value (
        value_id INTEGER PRIMARY KEY, product_code TEXT, family TEXT, subfamily TEXT,
        block TEXT, key TEXT, kind TEXT, unit TEXT, condition TEXT, status TEXT,
        num_value REAL, text_value TEXT, val_min REAL, val_typ REAL, val_max REAL,
        bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
        source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT, evidence TEXT,
        extractor TEXT, extractor_version TEXT, extracted_at TEXT)""")

    con.execute("INSERT INTO product (product_code, family, subfamily, display_name) VALUES ('DE1', 'F', NULL, 'DE1')")
    con.executemany(
        "INSERT INTO document (doc_id, file_name, doc_type, product_codes, product_code, is_active) VALUES (?,?,?,?,?,1)",
        [("doc1", "datasheet.pdf", "DATASHEET", '["DE1"]', "DE1"),
         ("doc2", "catalog.pdf", "CATALOGUE", '["DE1"]', "DE1")],
    )
    con.execute(
        "INSERT INTO attribute (block, key, kind, unit, conditions, subfields) VALUES ('core','supply_voltage','single',NULL,'[]','[]')"
    )
    con.execute(
        "INSERT INTO spec_value (value_id, product_code, block, key, condition, status, evidence) "
        "VALUES (1, 'DE1', 'core', 'supply_voltage', NULL, 'not_specified', '[]')"
    )
    con.commit()
    return con


_SHA_A = "sha256:" + "a" * 64
_SHA_B = "sha256:" + "b" * 64


def _evidence_all_chunks_index():
    """Two documents' chunks -- enough for `build_product_manifest`."""
    return {
        "doc1": [{"node_id": "c1", "content_sha256": _SHA_A, "text": "Supply Voltage: 24 VDC",
                  "page_start": 1, "page_end": 1}],
        "doc2": [{"node_id": "c2", "content_sha256": _SHA_B, "text": "Supply Voltage: 24 VDC",
                  "page_start": 3, "page_end": 3}],
    }


def _supply_voltage_fact(source_chunk_id, num_value=24):
    return _fact(block="core", key="supply_voltage", status="present", kind="single",
                 num_value=num_value, text_value=None, raw_text="24 VDC", source_chunk_id=source_chunk_id)


@_needs_load_to_db
def test_second_source_corroborates_instead_of_duplicating(monkeypatch):
    """I-16 acceptance test: `load_product()` is actually CALLED TWICE --
    once with one source (doc1), then with the SAME value arriving from
    a SECOND source (doc2). The table must keep ONE row for this
    property, but that row's `evidence[]` must GROW from 1 to 2
    (corroboration, not conflict)."""
    con = _evidence_test_db()
    monkeypatch.setattr(
        load_to_db, "load_indexes",
        lambda *a, **k: (_evidence_all_chunks_index(), {}, frozenset(), {}, {}),
    )
    dictionary = load_to_db.load_dictionary(con)
    doc_type_by_id = {"doc1": "DATASHEET", "doc2": "CATALOGUE"}

    # 1st run: evidence only from doc1.
    result_run1 = {"model_code": "DE1", "facts": [_supply_voltage_fact(_SHA_A)]}
    report1 = load_to_db.load_product(
        con, dictionary, result_run1, doc_type_by_id=doc_type_by_id,
        dry_run=False, new_key_fh=io.StringIO(),
    )
    con.commit()
    assert report1["written_present"] == 1

    rows_after_1 = con.execute(
        "SELECT status, evidence FROM spec_value WHERE product_code='DE1' AND block='core' AND key='supply_voltage'"
    ).fetchall()
    assert len(rows_after_1) == 1, "after the first run there must still be ONE row"
    evidence_1 = json.loads(rows_after_1[0][1])
    assert len(evidence_1) == 1 and evidence_1[0]["doc_id"] == "doc1"

    # 2nd run (TRULY a SEPARATE `load_product()` call, same product re-
    # processed with --force): this time the SAME value arrives from
    # BOTH doc1 AND doc2.
    result_run2 = {"model_code": "DE1", "facts": [
        _supply_voltage_fact(_SHA_A), _supply_voltage_fact(_SHA_B),
    ]}
    report2 = load_to_db.load_product(
        con, dictionary, result_run2, doc_type_by_id=doc_type_by_id,
        dry_run=False, new_key_fh=io.StringIO(), force=True,
    )
    con.commit()
    assert report2["written_present"] == 1
    assert report2["written_conflicting"] == 0, "same value MUST NOT create a conflict"

    rows_after_2 = con.execute(
        "SELECT value_id, status, evidence FROM spec_value WHERE product_code='DE1' AND block='core' AND key='supply_voltage'"
    ).fetchall()
    assert len(rows_after_2) == 1, "second source MUST NOT CREATE a second row"
    _value_id, status, evidence_json = rows_after_2[0]
    assert status == "present"
    evidence_2 = json.loads(evidence_json)
    assert len(evidence_2) == 2, "evidence[] must GROW from 1 to 2 (corroboration)"
    assert {e["doc_id"] for e in evidence_2} == {"doc1", "doc2"}


# --- O-12: incremental run (document/product-scoped, NOT corpus-scoped) --

def test_products_for_doc_id_reads_v_document_product():
    """discover.products_for_doc_id is INDEPENDENT of load_to_db/run_full
    -- it only queries `v_document_product`, so it is NOT AFFECTED by
    the prompts/*.md blocker (see module docstring)."""
    con = _evidence_test_db()
    assert discover.products_for_doc_id(con, "doc1") == ["DE1"]
    assert discover.products_for_doc_id(con, "doc2") == ["DE1"]
    assert discover.products_for_doc_id(con, "doc-missing") == []


@_needs_run_full
def test_resolve_codes_intersects_doc_id_with_models():
    # Fixture, same shape as the real schema_rag.sql's v_document_product:
    # `document` has doc_id PRIMARY KEY (one doc = one row), multi-product
    # carried via `product_codes` JSON array; the view unnests with
    # json_each (see facts/db/schema_rag.sql:323). First version of this
    # fixture wrongly assumed multiple rows per doc_id -- the
    # prompts/spec_keys.py blocker (K-71) prevented the test from EVER
    # RUNNING, so it was not caught.
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE product (product_code TEXT PRIMARY KEY, family TEXT, subfamily TEXT, display_name TEXT)")
    con.execute("""CREATE TABLE document (
        doc_id TEXT PRIMARY KEY, file_name TEXT, doc_type TEXT,
        product_codes TEXT NOT NULL DEFAULT '[]', is_active INTEGER)""")
    con.execute("""CREATE VIEW v_document_product AS
        SELECT d.doc_id, je.value AS product_code, d.doc_type, d.file_name, d.is_active
        FROM document d, json_each(d.product_codes) je""")
    con.executemany(
        "INSERT INTO product (product_code) VALUES (?)", [("DE1",), ("DE2",), ("DE3",)],
    )
    con.execute(
        "INSERT INTO document (doc_id, doc_type, product_codes, is_active) VALUES (?,?,?,1)",
        ("docX", "CATALOGUE", '["DE1","DE2"]'),
    )
    con.commit()

    # doc_id alone: every product that uses this document.
    assert run_full.resolve_codes(con, doc_id="docX") == ["DE1", "DE2"]
    # doc_id + models: INTERSECTION (DE3 doesn't use the document, DE2 not in --models).
    assert run_full.resolve_codes(con, doc_id="docX", models="DE2,DE3") == ["DE2"]
    # Neither doc_id nor models: ALL products (old bulk-run behaviour).
    assert run_full.resolve_codes(con) == ["DE1", "DE2", "DE3"]
    # only models: given subset.
    assert run_full.resolve_codes(con, models="DE3,DE1") == ["DE3", "DE1"]


@_needs_load_to_db
def test_incremental_reload_of_the_same_document_is_idempotent(monkeypatch):
    """O-12 acceptance test: the SAME fact set from the SAME document
    processed TWICE through `load_product_incremental` leaves
    `spec_value` row count AND `evidence[]` lengths UNCHANGED -- the
    phase invalidates its own old derivative and rewrites THE SAME
    thing (replace semantics), producing neither a LOSS nor an EXTRA
    (conflict/duplicate) row."""
    con = _evidence_test_db()
    monkeypatch.setattr(
        load_to_db, "load_indexes",
        lambda *a, **k: (_evidence_all_chunks_index(), {}, frozenset(), {}, {}),
    )
    dictionary = load_to_db.load_dictionary(con)
    doc_type_by_id = {"doc1": "DATASHEET", "doc2": "CATALOGUE"}
    result = {"model_code": "DE1", "doc_ids": ["doc1", "doc2"], "facts": [
        _supply_voltage_fact(_SHA_A), _supply_voltage_fact(_SHA_B),
    ]}

    report1 = load_to_db.load_product_incremental(
        con, dictionary, result, doc_type_by_id=doc_type_by_id, new_key_fh=io.StringIO(),
    )
    con.commit()
    rows1 = con.execute(
        "SELECT value_id, status, evidence FROM spec_value WHERE product_code='DE1'"
    ).fetchall()
    assert report1["written_present"] == 1

    # The SAME document/product processed a second time (nightly run
    # retriggered on the same day, or a previous-run repeat) -- fact
    # set UNCHANGED.
    report2 = load_to_db.load_product_incremental(
        con, dictionary, result, doc_type_by_id=doc_type_by_id, new_key_fh=io.StringIO(),
    )
    con.commit()
    rows2 = con.execute(
        "SELECT value_id, status, evidence FROM spec_value WHERE product_code='DE1'"
    ).fetchall()

    assert len(rows2) == len(rows1) == 1, "row count must not change (no loss, no conflict)"
    assert report2["written_present"] == 1
    assert report2["written_conflicting"] == 0, "the same fact MUST NOT create a conflict on repeat"
    evidence1 = json.loads(rows1[0][2])
    evidence2 = json.loads(rows2[0][2])
    assert len(evidence1) == len(evidence2) == 2, "evidence[] length must not change"
    assert {e["doc_id"] for e in evidence1} == {e["doc_id"] for e in evidence2} == {"doc1", "doc2"}
    assert rows2[0][1] == "present"


@_needs_load_to_db
def test_two_genuinely_different_sources_yield_one_present_and_one_conflicting_row(monkeypatch):
    """If the same (product,key,condition) gets TRULY DIFFERENT values from
    two sources: the source with the highest `document.trust_rank` stays
    `present`, the other is written as a separate `conflicting` row --
    I-16's contrast case (the row SPLITS, evidence[] does NOT GROW)."""
    con = _evidence_test_db()
    monkeypatch.setattr(
        load_to_db, "load_indexes",
        lambda *a, **k: (_evidence_all_chunks_index(), {}, frozenset(), {}, {}),
    )
    dictionary = load_to_db.load_dictionary(con)
    doc_type_by_id = {"doc1": "DATASHEET", "doc2": "CATALOGUE"}  # trust: 100 vs 60

    result = {"model_code": "DE1", "facts": [
        _supply_voltage_fact(_SHA_A, num_value=24),  # doc1, DATASHEET, trust=100
        _supply_voltage_fact(_SHA_B, num_value=48),  # doc2, CATALOGUE, trust=60 -- REAL conflict
    ]}
    report = load_to_db.load_product(
        con, dictionary, result, doc_type_by_id=doc_type_by_id,
        dry_run=False, new_key_fh=io.StringIO(),
    )
    con.commit()

    assert report["written_present"] == 1 and report["written_conflicting"] == 1
    rows = con.execute(
        "SELECT status, num_value, evidence FROM spec_value "
        "WHERE product_code='DE1' AND block='core' AND key='supply_voltage' ORDER BY status"
    ).fetchall()
    assert len(rows) == 2, "a real conflict must produce TWO separate rows"
    by_status = {r[0]: r for r in rows}
    assert by_status["present"][1] == 24
    assert json.loads(by_status["present"][2])[0]["doc_id"] == "doc1"
    assert by_status["conflicting"][1] == 48
    assert json.loads(by_status["conflicting"][2])[0]["doc_id"] == "doc2"


# --- MEASURED BUG (2026-08-26): conditioned range/min_typ_max key + absent --
# observation -- against the REAL schema (schema_rag.sql), tmp sqlite.
# `operating_temperature`/`supply_voltage`/`switch_voltage` etc. key
# shapes: kind='range'/'min_typ_max' BUT conditions[] filled
# (spec_keys.yaml).

_SCHEMA_RAG_SQL_PATH = None
if load_to_db is not None:
    _SCHEMA_RAG_SQL_PATH = load_to_db._FACTS_DATA_ROOT / "db" / "schema_rag.sql"

_needs_schema_rag_sql = pytest.mark.skipif(
    _SCHEMA_RAG_SQL_PATH is None or not _SCHEMA_RAG_SQL_PATH.is_file(),
    reason="facts/db/schema_rag.sql not found",
)


def _real_schema_db():
    """A specs.db built against the REAL schema_rag.sql, with a single
    attribute shaped like `operating_temperature` (kind=range,
    conditions=['forced_air', 'natural_convection']). Two documents
    (DATASHEET/trust=100, CATALOGUE/trust=60) -- winner determined by
    trust, NOT by order/tie-break."""
    con = sqlite3.connect(":memory:")
    con.executescript(_SCHEMA_RAG_SQL_PATH.read_text(encoding="utf-8"))
    con.execute("INSERT INTO product (product_code, family, subfamily, display_name) VALUES ('DE1','F',NULL,'DE1')")
    con.executemany(
        "INSERT INTO document (doc_id, file_name, extension, rel_path, content_hash, doc_type, product_codes, is_active) "
        "VALUES (?,?,?,?,?,?,?,1)",
        [("doc1", "d.pdf", ".pdf", "d.pdf", "sha256:x", "DATASHEET", '["DE1"]'),
         ("doc2", "c.pdf", ".pdf", "c.pdf", "sha256:y", "CATALOGUE", '["DE1"]')],
    )
    con.execute(
        "INSERT INTO attribute (block, key, kind, unit, conditions, subfields) VALUES "
        "('core','operating_temperature','range',NULL,'[\"forced_air\",\"natural_convection\"]','[]')"
    )
    con.execute(
        "INSERT INTO spec_value (product_code, block, key, kind, status, evidence) "
        "VALUES ('DE1','core','operating_temperature','range','not_specified','[]')"
    )
    con.commit()
    return con


@_needs_load_to_db
@_needs_schema_rag_sql
def test_conditioned_range_key_with_absent_and_present_writes_cleanly(monkeypatch):
    """REGRESSION: on the 2026-08-26 nightly run 34 of 218 products
    hit this exact pattern (conditioned range key with BOTH `absent`
    AND a half `present` observation for the SAME condition), which
    caused the product's ENTIRE fact set to be lost (transaction
    rollback -- `_merge_complementary` treated absent as the missing
    half and merged it, producing a row that violated
    `sv_empty_status_ck`). Before this fix this exact pattern would
    CRASH with `sqlite3.IntegrityError`; now the product's facts must
    be preserved entirely and the real conflict rule (highest trust
    wins) must apply."""
    con = _real_schema_db()
    sha_datasheet = "sha256:" + "a" * 64
    sha_catalog = "sha256:" + "b" * 64
    monkeypatch.setattr(
        load_to_db, "load_indexes",
        lambda *a, **k: (
            {"doc1": [{"node_id": "c1", "content_sha256": sha_datasheet, "text": "x", "page_start": 1, "page_end": 1}],
             "doc2": [{"node_id": "c2", "content_sha256": sha_catalog, "text": "x", "page_start": 1, "page_end": 1}]},
            {}, frozenset(), {}, {},
        ),
    )
    dictionary = load_to_db.load_dictionary(con)
    doc_type_by_id = {"doc1": "DATASHEET", "doc2": "CATALOGUE"}

    # DATASHEET (trust=100) reports a half `present` observation; the
    # low-trust CATALOGUE (trust=60) says 'absent' for the SAME condition
    # -- a real conflict (present vs absent), the exact pattern
    # `_merge_complementary` previously mistook for "complementary half".
    result = {"model_code": "DE1", "facts": [
        {"block": "core", "key": "operating_temperature", "status": "present", "kind": "range",
         "condition": "forced_air", "val_max": 55, "raw_text": "up to +55°C", "source_chunk_id": sha_datasheet,
         "num_value": None, "text_value": None, "val_min": None, "val_typ": None, "items": []},
        {"block": "core", "key": "operating_temperature", "status": "absent", "kind": "range",
         "condition": "forced_air", "raw_text": "...", "source_chunk_id": sha_catalog,
         "num_value": None, "text_value": None, "val_min": None, "val_typ": None, "val_max": None, "items": []},
    ]}

    report = load_to_db.load_product(
        con, dictionary, result, doc_type_by_id=doc_type_by_id, dry_run=False, new_key_fh=io.StringIO(),
    )
    con.commit()

    # Before the `_merge_complementary` fix this SCENARIO crashed with
    # `sqlite3.IntegrityError` (violating `sv_empty_status_ck`) and the
    # product's ENTIRE fact set (including this single key) was lost to
    # transaction rollback.
    assert report["written_present"] == 1
    assert report["skipped"] and "cannot be written as a conflict" in report["skipped"][0]["reason"], report["skipped"]
    rows = [
        (condition, status, val_max) for condition, status, val_max in con.execute(
            "SELECT condition, status, val_max FROM spec_value WHERE product_code='DE1' "
            "AND key='operating_temperature'"
        )
    ]
    assert rows == [("forced_air", "present", 55)], rows


# --- K-102: no row with evidence[] EMPTY must be written --------------------

def _load_product_db():
    """Minimum of the three tables `load_product` touches: product
    (existence check), document (file_name lookup), spec_value
    (bootstrap row)."""
    con = sqlite3.connect(":memory:")
    con.executescript("""
        CREATE TABLE product (product_code TEXT PRIMARY KEY);
        CREATE TABLE document (doc_id TEXT PRIMARY KEY, file_name TEXT);
        CREATE TABLE spec_value (
            value_id INTEGER PRIMARY KEY, product_code TEXT, family TEXT, subfamily TEXT,
            block TEXT, key TEXT, kind TEXT, unit TEXT, condition TEXT, status TEXT,
            num_value REAL, text_value TEXT, val_min REAL, val_typ REAL, val_max REAL,
            bool_value INTEGER, items TEXT, subfields TEXT, unit_observed TEXT, raw_text TEXT,
            source_doc_id TEXT, source_file_name TEXT, source_chunk_id TEXT,
            evidence TEXT NOT NULL DEFAULT '[]',
            extractor TEXT, extractor_version TEXT, extracted_at TEXT);
    """)
    con.execute("INSERT INTO product (product_code) VALUES ('DE1')")
    con.execute("INSERT INTO document (doc_id, file_name) VALUES ('doc-1', 'ds.pdf')")
    con.executemany(
        "INSERT INTO spec_value (value_id, product_code, block, key, condition, status) "
        "VALUES (?, 'DE1', 'core', ?, NULL, 'not_specified')",
        [(1, "good_key"), (2, "orphan_key")],
    )
    con.commit()
    return con


@_needs_load_to_db
def test_fact_with_unresolvable_chunk_is_dropped_not_written_evidenceless(monkeypatch, tmp_path):
    """MEASURED BUG (2026-08-27 nightly review): a `source_chunk_id`
    that cannot be resolved in `chunk_lookup` was only rejected for
    `status='absent'`. `present` facts slipped through, `_merge_evidence`
    returned `([], None, None, None)` for them, and the row was
    written as `evidence='[]'` + `extractor` SET -- a fact with NO
    provenance at all. Production measurement: 386 rows / 94 products
    (staleness_audit criterion (b) `empty_evidence`), and those rows
    were also flagged as stale every night and candidates for reprocess."""
    good_sha = "sha256:" + "a" * 64
    chunk = discover.ChunkRow(chunk_sha256=good_sha, text="x", doc_ids=["doc-1"],
                               page_start=1, page_end=1)
    manifest = discover.ProductManifest(
        product_code="DE1", family=None, subfamily=None, display_name=None,
        doc_ids=["doc-1"], docs_missing_text=[], chunks=[chunk],
    )
    monkeypatch.setattr(load_to_db, "load_indexes", lambda: ({}, {}, frozenset(), {}, {}))
    monkeypatch.setattr(load_to_db, "build_product_manifest",
                        lambda *a, **kw: manifest)

    con = _load_product_db()
    result = {"model_code": "DE1", "facts": [
        _fact(key="good_key", num_value=5, source_chunk_id=good_sha),
        _fact(key="orphan_key", num_value=7, source_chunk_id="sha2_mangled_hash"),
    ]}
    report = load_to_db.load_product(
        con, _dictionary(good_key="single", orphan_key="single"), result,
        doc_type_by_id={"doc-1": "DATASHEET"}, dry_run=False, new_key_fh=io.StringIO())

    rows = dict(con.execute(
        "SELECT key, evidence FROM spec_value WHERE extractor IS NOT NULL").fetchall())
    assert "good_key" in rows, "resolvable, evidenced fact MUST be written"
    assert "orphan_key" not in rows, "unresolvable fact MUST NOT be written"
    assert json.loads(rows["good_key"]), "the written row's evidence[] must be populated"
    assert any("cannot be resolved" in s["reason"] for s in report["skipped"]), \
        "dropped fact must be reported (not silently)"


@_needs_load_to_db
def test_no_row_is_ever_written_with_extractor_set_and_empty_evidence(monkeypatch):
    """K-102 invariant: NO write path may produce a row with `extractor`
    set + `evidence='[]'`. `forget_source` only deletes rows whose
    EVIDENCE FIELD empties, so once such a row lands in the DB nothing
    cleans it up."""
    good_sha = "sha256:" + "a" * 64
    chunk = discover.ChunkRow(chunk_sha256=good_sha, text="x", doc_ids=["doc-1"],
                               page_start=1, page_end=1)
    manifest = discover.ProductManifest(
        product_code="DE1", family=None, subfamily=None, display_name=None,
        doc_ids=["doc-1"], docs_missing_text=[], chunks=[chunk],
    )
    monkeypatch.setattr(load_to_db, "load_indexes", lambda: ({}, {}, frozenset(), {}, {}))
    monkeypatch.setattr(load_to_db, "build_product_manifest", lambda *a, **kw: manifest)

    con = _load_product_db()
    result = {"model_code": "DE1", "facts": [
        _fact(key="good_key", num_value=5, source_chunk_id=good_sha),
        _fact(key="orphan_key", status="absent", source_chunk_id="sha2_mangled"),
        _fact(key="orphan_key", num_value=7, source_chunk_id="sha2_mangled"),
    ]}
    load_to_db.load_product(
        con, _dictionary(good_key="single", orphan_key="single"), result,
        doc_type_by_id={"doc-1": "DATASHEET"}, dry_run=False, new_key_fh=io.StringIO())

    leaked = con.execute(
        "SELECT COUNT(*) FROM spec_value WHERE extractor IS NOT NULL AND evidence = '[]'"
    ).fetchone()[0]
    assert leaked == 0


@_needs_load_to_db
def test_written_extractor_version_is_semver_comparable():
    """K-101: the value written to the column must be COMPARABLE to
    `facts_version` -- `staleness_audit` criterion (c) reads it as
    semver."""
    from medrag.pipeline.nightly_report import _semver_part, current_pipeline_versions

    written = load_to_db.extractor_version()

    assert _semver_part(written) == current_pipeline_versions()["facts"]
    assert load_to_db.PROMPT_VERSION in written, "prompt hash must be preserved as metadata"