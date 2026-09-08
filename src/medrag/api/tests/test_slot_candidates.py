import sqlite3

import yaml

from medrag.api.slot_candidates import (
    GlossaryEntry,
    discover_informative_slots,
    load_attribute_glossary,
)


def test_load_attribute_glossary_reads_schema_yaml_field(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        yaml.safe_dump({
            "attribute_glossary": [
                {"block": "core", "key": "operating_temperature", "kind": "range", "unit": "C",
                 "question": "What temperature range?"},
                {"block": "analog_io", "key": "channel_count", "kind": "single", "unit": None,
                 "question": "How many channels?"},
            ]
        }),
        encoding="utf-8",
    )
    glossary = load_attribute_glossary(schema_path)
    assert glossary["operating_temperature"] == GlossaryEntry(
        block="core", kind="range", unit="C", question="What temperature range?"
    )
    assert glossary["channel_count"].question == "How many channels?"


def test_load_attribute_glossary_missing_field_returns_empty_dict(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(yaml.safe_dump({"name": "x", "tables": []}), encoding="utf-8")
    assert load_attribute_glossary(schema_path) == {}


def _build_spec_value_db(tmp_path, rows):
    db_path = tmp_path / "specs.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE spec_value (product_code TEXT, family TEXT, block TEXT, key TEXT, "
        "status TEXT, raw_text TEXT)"
    )
    con.executemany(
        "INSERT INTO spec_value (product_code, family, block, key, status, raw_text) VALUES (?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return str(db_path)


_GLOSSARY = {
    "operating_temperature": GlossaryEntry(block="core", kind="range", unit="C", question="Temp range?"),
    "channel_count": GlossaryEntry(block="analog_io", kind="single", unit=None, question="How many channels?"),
    "weight": GlossaryEntry(block="core", kind="single", unit="g", question="How heavy?"),
}


def test_discover_informative_slots_only_returns_keys_with_varying_present_values(tmp_path):
    # operating_temperature varies (2 distinct) -> informative.
    # weight is IDENTICAL across both products -> NOT informative (asking wouldn't narrow anything).
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "operating_temperature", "present", "0-50 C"),
        ("DE2", "DAQ SYSTEMS", "core", "operating_temperature", "present", "-40-85 C"),
        ("DE1", "DAQ SYSTEMS", "core", "weight", "present", "100 g"),
        ("DE2", "DAQ SYSTEMS", "core", "weight", "present", "100 g"),
    ])
    candidates = discover_informative_slots(db_path, family="DAQ SYSTEMS", glossary=_GLOSSARY)
    keys = {c.key for c in candidates}
    assert "operating_temperature" in keys
    assert "weight" not in keys


def test_discover_informative_slots_scopes_to_the_given_family(tmp_path):
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "operating_temperature", "present", "0-50 C"),
        ("DE2", "DAQ SYSTEMS", "core", "operating_temperature", "present", "-40-85 C"),
        ("DE3", "PXI EXPRESS SYSTEMS", "core", "operating_temperature", "present", "0-40 C"),
        ("DE4", "PXI EXPRESS SYSTEMS", "core", "operating_temperature", "present", "-20-70 C"),
    ])
    candidates = discover_informative_slots(db_path, family="PXI EXPRESS SYSTEMS", glossary=_GLOSSARY)
    assert {c.key for c in candidates} == {"operating_temperature"}


def test_discover_informative_slots_excludes_already_asked_keys(tmp_path):
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "operating_temperature", "present", "0-50 C"),
        ("DE2", "DAQ SYSTEMS", "core", "operating_temperature", "present", "-40-85 C"),
    ])
    candidates = discover_informative_slots(
        db_path, family="DAQ SYSTEMS", glossary=_GLOSSARY, exclude_keys={"operating_temperature"},
    )
    assert candidates == []


def test_discover_informative_slots_skips_keys_missing_from_glossary(tmp_path):
    """A key not in the glossary (drift/deleted) is silently skipped -- no fabrication."""
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "retired_key", "present", "a"),
        ("DE2", "DAQ SYSTEMS", "core", "retired_key", "present", "b"),
    ])
    candidates = discover_informative_slots(db_path, family="DAQ SYSTEMS", glossary=_GLOSSARY)
    assert candidates == []


def test_discover_informative_slots_orders_by_n_distinct_descending(tmp_path):
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "operating_temperature", "present", "a"),
        ("DE2", "DAQ SYSTEMS", "core", "operating_temperature", "present", "b"),
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
        ("DE3", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "16"),
    ])
    candidates = discover_informative_slots(db_path, family="DAQ SYSTEMS", glossary=_GLOSSARY)
    assert [c.key for c in candidates] == ["channel_count", "operating_temperature"]
    assert candidates[0].n_distinct == 3
    assert candidates[1].n_distinct == 2


def test_discover_informative_slots_scopes_to_given_product_codes(tmp_path):
    """The scope may be the REMAINING candidates -- a key that discriminates
    across the whole family, if it has collapsed to a single value among the
    remaining candidates, is no longer ASKED (asking would not narrow the set)."""
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "core", "operating_temperature", "present", "0-50 C"),
        ("DE2", "DAQ SYSTEMS", "core", "operating_temperature", "present", "0-50 C"),
        ("DE3", "DAQ SYSTEMS", "core", "operating_temperature", "present", "-40-85 C"),
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
        ("DE3", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    # Across the whole family, BOTH keys are discriminating...
    assert {c.key for c in discover_informative_slots(
        db_path, family="DAQ SYSTEMS", glossary=_GLOSSARY
    )} == {"operating_temperature", "channel_count"}
    # ...but if only DE1/DE2 remain, temperature no longer discriminates.
    candidates = discover_informative_slots(
        db_path, glossary=_GLOSSARY, product_codes=["DE1", "DE2"],
    )
    assert {c.key for c in candidates} == {"channel_count"}


def test_discover_informative_slots_product_codes_work_without_family(tmp_path):
    """Even when `family` could not be resolved at all (phrasing that does not
    match the surface-form glossary), discovery still works over the remaining candidates."""
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "PXI EXPRESS SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    candidates = discover_informative_slots(
        db_path, glossary=_GLOSSARY, product_codes=["DE1", "DE2"],
    )
    assert [c.key for c in candidates] == ["channel_count"]


def test_discover_informative_slots_without_any_scope_returns_empty(tmp_path):
    """Neither family nor product_codes -- scanning the whole DB to produce an
    unscoped question is deliberately NOT done."""
    db_path = _build_spec_value_db(tmp_path, [
        ("DE1", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "4"),
        ("DE2", "DAQ SYSTEMS", "analog_io", "channel_count", "present", "8"),
    ])
    assert discover_informative_slots(db_path, glossary=_GLOSSARY) == []


def test_discover_informative_slots_respects_limit(tmp_path):
    rows = []
    glossary = {}
    for i in range(5):
        key = f"key{i}"
        glossary[key] = GlossaryEntry(block="core", kind="single", unit=None, question=f"q{i}")
        rows.append(("DE1", "DAQ SYSTEMS", "core", key, "present", "a"))
        rows.append(("DE2", "DAQ SYSTEMS", "core", key, "present", "b"))
    db_path = _build_spec_value_db(tmp_path, rows)
    candidates = discover_informative_slots(db_path, family="DAQ SYSTEMS", glossary=glossary, limit=2)
    assert len(candidates) == 2
