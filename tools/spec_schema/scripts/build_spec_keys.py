"""Build `spec_keys.yaml` from `attribute_schema_v3.json`.

**Source of truth changed.** An older version of this script mined the
dictionary directly out of the corpus (`evidence.py` label-scanning) and
emitted 109 keys. The dictionary is now sourced from
`attribute_schema_v3.json` (v3.1) -- a canonical dictionary filtered
through rules R0-R3, with every key carrying its own `evidence`
measurement; 95 keys. This script flattens it into YAML in the exact
shape `schema.sql`'s dictionary layer expects (attribute_block /
attribute / attribute_condition / attribute_label / attribute_subfield
/ attribute_applicability).

`evidence.py` is still in the tree, but no longer part of this chain:
v3's `evidence` blocks were measured with it; re-measuring them is not
this script's job.

**Scope axis changed.** Old `applies_to` was `taxonomy_ids.yaml` ids.
The new scope is the family/subfamily NAME in `product_nodes.json`
(e.g. `"PXI EXPRESS SYSTEMS / PXI Express Systems Switches and
Multiplexers"`) and it lives at the BLOCK level -- every key in a block
shares the same scope. `core` has empty scope = applies to every
family.

**The "no invention" rule is preserved, location shifted.** This script
validates every produced scope name against `product_nodes.json`; if a
name doesn't match exactly, production STOPS. This is critical: a
mismatched scope name silently means "this block matches no product",
and every one of its keys would be born `not_applicable`. The v3
dictionary's 9 scope names had drifted in exactly this way.

Usage:  python scripts/build_spec_keys.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent  # .../tools/spec_schema/scripts
SCHEMA_DIR = HERE.parent  # .../tools/spec_schema
# spec_schema/ lives directly under tools/, while chatbot-corpus/ sits
# next to tools/ at the repo root, so we walk up one more level.
CORPUS_ROOT = SCHEMA_DIR.parent.parent / "chatbot-corpus"  # .../chatbot-corpus

SOURCE = SCHEMA_DIR / "attribute_schema_v3.json"
PRODUCT_NODES = CORPUS_ROOT / "product_info" / "product_nodes.json"
OUT = SCHEMA_DIR / "spec_keys.yaml"

# v3's rule-rationale field; it lives in YAML under `notes` (mirrors the
# `attribute.notes` column in schema.sql). The leading underscore is v3's
# "not machine-read" convention.
_NOTE_FIELD = "_note"


def load_scopes(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """The REAL family and `family / subfamily` names in product_nodes.json."""
    nodes = json.loads(path.read_text(encoding="utf-8"))["product_nodes"]
    families = {n["family"] for n in nodes}
    subfamilies = {f"{n['family']} / {n['subfamily']}" for n in nodes if n.get("subfamily")}
    return families, subfamilies


def iter_blocks(schema: dict):
    """(block name, is_core, scope list, attributes, description)."""
    core = schema["core"]
    yield "core", True, [], core["attributes"], core.get(_NOTE_FIELD)
    for block in schema["families"]:
        yield block["block"], False, list(block["applies_to"]), block["attributes"], block.get(_NOTE_FIELD)


def validate_scopes(schema: dict, families: set[str], subfamilies: set[str]) -> None:
    """Every scope name must appear verbatim in product_nodes.json, else STOP."""
    bad: list[tuple[str, str]] = []
    for name, _is_core, scopes, _attrs, _desc in iter_blocks(schema):
        for scope in scopes:
            if scope not in families and scope not in subfamilies:
                bad.append((name, scope))
    if bad:
        lines = "\n".join(f"  block {b!r}: {s!r}" for b, s in bad)
        raise SystemExit(
            f"{len(bad)} scope name(s) NOT in product_nodes.json:\n{lines}\n\n"
            "A mismatched scope name silently means that block matches NO "
            "product (every key in it is born not_applicable). Names must "
            "match verbatim -- normalize/fuzzy matching is deliberately NOT "
            "implemented."
        )


def validate_attributes(schema: dict) -> None:
    """Internal consistency of the dictionary -- the YAML-side mirror of
    schema.sql's CHECKs. We want to catch the same violations at build
    time, not at DB-load time."""
    kinds = set(schema["value_kinds"])
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()

    for block, _is_core, _scopes, attrs, _desc in iter_blocks(schema):
        for a in attrs:
            key = a["key"]
            # R2.5: one concept, one name within a BLOCK. Cross-block repetition
            # is legitimate (channel_count lives in 3 blocks) -- schema.sql's
            # UNIQUE(block_id, key) index says exactly that.
            if (block, key) in seen:
                problems.append(f"{block}/{key}: duplicate key within block")
            seen.add((block, key))

            if isinstance(a["kind"], list):
                problems.append(f"{block}/{key}: kind is a list -- a key has ONE kind")
            elif a["kind"] not in kinds:
                problems.append(f"{block}/{key}: unknown kind {a['kind']!r}")

            # R0.1: question is required and must be a real question (attr_r0_1_ck)
            if len((a.get("question") or "").strip()) < 10:
                problems.append(f"{block}/{key}: question missing or shorter than 10 chars")

            # R0.2: numeric range kinds require a unit (attr_r0_2_ck)
            if a["kind"] in ("min_typ_max", "range") and not a.get("unit"):
                problems.append(f"{block}/{key}: kind={a['kind']} requires a unit")

            # kind=conditional without a conditions list is meaningless;
            # schema.sql's attribute_value_kind_guard trigger would fail the
            # INSERT later -- catching it at build time is cheaper.
            if a["kind"] == "conditional" and not a.get("condition"):
                problems.append(f"{block}/{key}: kind=conditional but condition list is empty")

            ev = a.get("evidence") or {}
            # R0.3 spread / R0.4 discriminative power (attr_r0_3_ck / attr_r0_4_ck)
            if ev.get("ndocs") is not None and ev["ndocs"] < 2 and not a.get("is_family_defining"):
                problems.append(f"{block}/{key}: R0.3 ndocs={ev['ndocs']} < 2, and not family-defining either")
            if ev.get("distinct_values") is not None and ev["distinct_values"] < 2:
                problems.append(f"{block}/{key}: R0.4 distinct_values={ev['distinct_values']} < 2")

    if problems:
        raise SystemExit(f"{len(problems)} dictionary violation(s):\n" + "\n".join(f"  {p}" for p in problems))


def build_attribute(block: str, a: dict) -> dict:
    """Flatten one v3 attribute into a YAML row. Field order is for readability
    (preserved with `sort_keys=False`). Empty optional fields are OMITTED --
    in a 95-key file, `conditions: []` would be nothing but noise."""
    row: dict = {
        "block": block,
        "key": a["key"],
        "kind": a["kind"],
        "unit": a.get("unit"),
        "question": a["question"].strip(),
        "labels": list(a.get("labels") or []),
    }
    if a.get("condition"):
        row["conditions"] = list(a["condition"])
    if a.get("enum"):
        row["enum"] = list(a["enum"])
    if a.get("accepts_units"):
        row["accepts_units"] = list(a["accepts_units"])
    if a.get("subfields"):
        row["subfields"] = [
            {
                "name": s["name"],
                "unit": s.get("unit"),
                **({"labels": list(s["labels"])} if s.get("labels") else {}),
                **({"note": s["note"]} if s.get("note") else {}),
            }
            for s in a["subfields"]
        ]
    ev = a.get("evidence") or {}
    row["evidence"] = {
        "ndocs": ev.get("ndocs"),
        "distinct_values": ev.get("distinct_values"),
        **({"note": ev["note"]} if ev.get("note") else {}),
    }
    if a.get("is_family_defining"):
        row["is_family_defining"] = True
    # "structural": this key's value is generated deterministically from the
    # DB's own records (document/product node fan-out and friends), NOT from
    # chunk/atom LLM extraction -- the fill path is OUTSIDE this script's
    # scope (facts/experiments). For everything else (the majority), the
    # field is simply not written -- in 94 of 95 keys it would just be noise.
    if a.get("derivation") == "structural":
        row["derivation"] = "structural"
    if a.get(_NOTE_FIELD):
        row["notes"] = a[_NOTE_FIELD]
    return row


def build(schema: dict) -> dict:
    blocks = []
    attributes = []
    for name, is_core, scopes, attrs, desc in iter_blocks(schema):
        block: dict = {"name": name, "is_core": is_core, "applies_to": scopes}
        if desc:
            block["description"] = desc
        blocks.append(block)
        attributes += [build_attribute(name, a) for a in attrs]

    return {
        "version": "2.0.0",
        "source": f"attribute_schema_v3.json ({schema['schema_version']})",
        "generated_by": "scripts/build_spec_keys.py",
        "ruleset": schema.get("ruleset"),
        "ruleset_exceptions": schema.get("ruleset_exceptions") or [],
        "kinds": list(schema["value_kinds"]),
        "statuses": dict(schema["value_statuses"]),
        "facets": {
            name: spec["values"]
            for name, spec in schema["facets"].items()
            if isinstance(spec, dict) and "values" in spec
        },
        "relations": list(schema["relations"]),
        "blocks": blocks,
        "attributes": attributes,
    }


HEADER = """\
# ACME spec key dictionary -- GENERATED, DO NOT HAND-EDIT.
#
# Source: attribute_schema_v3.json   Generator: scripts/build_spec_keys.py
# Edits go there, not here; this file is overwritten on every build.
#
# This dictionary maps 1:1 onto layer 3 (ATTRIBUTE DICTIONARY) of schema.sql:
#   blocks[]                -> attribute_block
#   attributes[]            -> attribute (question/kind/unit/evidence)
#   attributes[].labels     -> attribute_label     (R1.5: label does NOT enter the name)
#   attributes[].conditions -> attribute_condition (R1.2/R1.3: condition does NOT enter the name)
#   attributes[].subfields  -> attribute_subfield  (R2.3: accuracy/resolution does NOT
#                              open their own key -- they hang off the parent quantity)
#   blocks[].applies_to     -> attribute_applicability
#
# SCOPE: `applies_to` lives at the BLOCK level and is the family / subfamily
# NAME from product_nodes.json (verbatim; the generator validates every build
# and aborts on drift). `core` has empty scope = applies to every family.
#
# `unit` is the canonical unit; each value row also carries its own
# `unit_observed`. `kind` is SINGLE: a document's presentation difference
# (4-column min|typ|max vs a single cell) is NOT a second kind -- the
# parser normalizes it.
"""


def main() -> int:
    schema = json.loads(SOURCE.read_text(encoding="utf-8"))
    families, subfamilies = load_scopes(PRODUCT_NODES)

    validate_scopes(schema, families, subfamilies)
    validate_attributes(schema)

    doc = build(schema)
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=88, default_flow_style=False)
    OUT.write_text(HEADER + "\n" + body, encoding="utf-8")

    print(f"{OUT.name}: {len(doc['attributes'])} key / {len(doc['blocks'])} block(s) "
          f"produced (source: {doc['source']})")
    for b in doc["blocks"]:
        n = sum(1 for a in doc["attributes"] if a["block"] == b["name"])
        scope = "every family" if b["is_core"] else f"{len(b['applies_to'])} scope(s)"
        print(f"  {b['name']:24s} {n:3d} key   {scope}")
    n_cond = sum(1 for a in doc["attributes"] if a.get("conditions"))
    n_sub = sum(1 for a in doc["attributes"] if a.get("subfields"))
    n_lab = sum(len(a["labels"]) for a in doc["attributes"])
    # NOTE: carrying `conditions` does NOT mean kind=conditional -- data_rate
    # is kind=single but carries 10 interface conditions (same quantity,
    # different contexts).
    print(f"labels: {n_lab}  key(s) with conditions: {n_cond}  key(s) with subfields: {n_sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
