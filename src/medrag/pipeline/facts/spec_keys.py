"""Reader for the `chatbot-corpus/spec_schema/spec_keys.yaml` dictionary.

**Source and scope axis changed (user decision).** The dictionary is now
generated not from corpus mining but from `attribute_schema_v3.json`
(95 keys / 10 blocks), and the DB's dictionary layer matches the shape
of `schema.sql`: instead of a single flat `spec_key` table, it carries
`attribute_block` + `attribute` + `attribute_condition` /
`attribute_label` / `attribute_subfield` / `attribute_applicability`.

The old `taxonomy_ids.yaml` LEFT this chain. Scope is no longer an
indirect id mapping but the family/subfamily NAME in `product_nodes.json`:

    "DAQ SYSTEMS"                                  -> whole family
    "PXI EXPRESS SYSTEMS / PXI Express Systems ..." -> only that subfamily

`facts` reads these files via PATH, does not import `chatbot-corpus`'s
CODE (root CLAUDE.md component independence).

**Validity rule (single place).** A (product, attribute) pair is valid
if the attribute's block covers the product's family. A block without
scope rows (`core`) applies to EVERY product. Inheritance goes ONLY
DOWNWARD: if `applies_to` lists a family, every subfamily under it is
covered; a key observed in one subfamily does NOT spread to siblings.

If the intersection is empty the row is `not_applicable` -- NOT `absent`.
`applies_to` is derived from observation and is deliberately narrow:
"never observed in this branch" does NOT mean "this feature does not
exist on this product". `absent` is only produced by an explicit
document declaration (KARAR-015) and the bootstrap never writes it.

The DB-side twin of this rule lives in `schema_rag.sql`'s
`attribute_value_scope_guard_*` triggers -- TWO-WAY checks, so if the
code here mislabels, the INSERT explodes (no silent drift).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/
_REPO_ROOT = HERE.parents[3]  # medrag/
CORPUS_ROOT = Path(os.environ.get("CHATBOT_CORPUS_DIR", _REPO_ROOT / "tools"))
SPEC_SCHEMA_DIR = CORPUS_ROOT / "spec_schema"
SPEC_KEYS_PATH = SPEC_SCHEMA_DIR / "spec_keys.yaml"

# Family/subfamily separator inside scope names. `product_nodes.json`
# names do not contain this string itself (verified by build_spec_keys.py),
# so parsing is one-way and lossless.
SCOPE_SEP = " / "


def load_spec_keys(path: Path = SPEC_KEYS_PATH) -> dict:
    """The full dictionary. Validation lives in `validate_dictionary`;
    this is the raw reader."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_scope(scope: str) -> tuple[str, str]:
    """`"FAMILY / Sub-Family"` -> `("FAMILY", "Sub-Family")`;
    `"FAMILY"` -> `("FAMILY", "")`.

    Empty subfamily means "whole family" and is stored AS-IS `''` in the
    `attribute_applicability.subfamily` column (not NULL -- it's part of
    the PK)."""
    if SCOPE_SEP in scope:
        family, subfamily = scope.split(SCOPE_SEP, 1)
        return family.strip(), subfamily.strip()
    return scope.strip(), ""


def key_applies(family: str, subfamily: str | None, scopes: list[tuple[str, str]]) -> bool:
    """The SINGLE application of the validity rule (see module docstring).

    If `scopes` is empty (core block), applies to every product."""
    if not scopes:
        return True
    for scope_family, scope_subfamily in scopes:
        if scope_family != family:
            continue
        if scope_subfamily == "" or scope_subfamily == (subfamily or ""):
            return True
    return False


def block_scopes(dictionary: dict) -> dict[str, list[tuple[str, str]]]:
    """block name -> parsed scope list."""
    return {b["name"]: [parse_scope(s) for s in (b.get("applies_to") or [])] for b in dictionary["blocks"]}


def validate_dictionary(dictionary: dict, families: set[str], subfamilies: set[str]) -> None:
    """Internal consistency of the dictionary + adherence to the taxonomy.
    No silent fix: an unknown kind/block/scope makes the build stop
    (root CLAUDE.md fail-loud).

    `build_spec_keys.py` already validates most of these WHEN generating;
    here it is checked again because `facts` reads the dictionary from
    disk and may encounter a hand-edited or stale file."""
    known_kinds = set(dictionary["kinds"])
    known_blocks = {b["name"] for b in dictionary["blocks"]}

    if not known_blocks:
        raise ValueError("spec_keys.yaml contains no blocks -- dictionary is empty or stale.")

    # Scope names must match product_nodes.json. A non-matching name means
    # this block will NEVER match any product (all keys born as
    # not_applicable) -- a silent data loss, so we fail loudly.
    for block in dictionary["blocks"]:
        for scope in block.get("applies_to") or []:
            scope_family, scope_subfamily = parse_scope(scope)
            full = scope if scope_subfamily else scope_family
            if scope_subfamily:
                if full not in subfamilies:
                    raise ValueError(
                        f"block {block['name']!r} references a subfamily that is not in "
                        f"product_nodes.json: {scope!r}"
                    )
            elif scope_family not in families:
                raise ValueError(
                    f"block {block['name']!r} references a family that is not in "
                    f"product_nodes.json: {scope!r}"
                )

    seen: set[tuple[str, str]] = set()
    for entry in dictionary["attributes"]:
        block, key = entry["block"], entry["key"]
        if block not in known_blocks:
            raise ValueError(f"key={key!r} belongs to unknown block: {block!r}")
        # R2.5: duplicate key within the same block. Cross-block duplicates
        # are legitimate -- channel_count lives separately in 3 blocks.
        if (block, key) in seen:
            raise ValueError(f"duplicate key in spec_keys.yaml: {block}/{key}")
        seen.add((block, key))

        if isinstance(entry["kind"], list):
            raise ValueError(f"key={block}/{key} kind is a list -- each key has exactly ONE kind.")  # noqa: TRY004 -- data validation; same ValueError family as the sibling raises
        if entry["kind"] not in known_kinds:
            raise ValueError(
                f"key={block}/{key} uses unknown kind: {entry['kind']!r} "
                f"(known: {sorted(known_kinds)})"
            )
        if entry["kind"] == "conditional" and not entry.get("conditions"):
            raise ValueError(
                f"key={block}/{key} kind=conditional but conditions dict is empty -- "
                "schema_rag.sql's kind guard requires a condition_id for every value."
            )


def validate_products(product_nodes: list[dict], families: set[str]) -> None:
    """Is every leaf product's family one the dictionary recognises?
    Otherwise it would get `not_applicable` in EVERY family block -- a
    silent loss; we fail instead."""
    orphans = [n["node_id"] for n in product_nodes if n["type"] == "product" and n["family"] not in families]
    if orphans:
        raise ValueError(
            f"{len(orphans)} product nodes have a family not in product_nodes.json's family set: "
            f"{orphans[:5]}{' ...' if len(orphans) > 5 else ''}"
        )


# ---------------------------------------------------------------------------
# Flattening into DB rows (2026-08-04 simplification: ONE `attribute` table,
# with JSON-embedded labels/conditions/subfields/applies_to -- replaces the
# 6-part block/attribute/condition/label/subfield/applicability schema).
# `question`/`notes` are DELIBERATELY not rows -- not "table-worthy"
# information (user decision, 2026-08-04): `build_schema.py` READS them from
# the dictionary and EMBEDS them in schema.yaml's description text; they
# are not transported as DB columns.
# Facets (the old facet_type/facet_value/product_facet) were DROPPED: no
# downstream consumer (chatbot included) was reading them -- they were
# gratuitous given the "3-4 table" target; unused code is removed entirely
# (root CLAUDE.md).
# ---------------------------------------------------------------------------


def ordered_attributes(dictionary: dict) -> list[dict]:
    """Dictionary rows -- block order, then key alphabetical (deterministic
    build order; no longer assigns an attribute_id, just iteration order)."""
    block_order = {b["name"]: i for i, b in enumerate(dictionary["blocks"])}
    return sorted(dictionary["attributes"], key=lambda a: (block_order[a["block"]], a["key"]))


def attribute_rows_flat(dictionary: dict) -> list[tuple]:
    """Rows for the `attribute` table (2026-08-04 single-table design):
    (block, key, kind, unit, enum_values, accepts_units, labels, conditions,
    subfields, applies_to, is_core, is_active) -- the last 4 are JSON text."""
    scopes = block_scopes(dictionary)
    is_core_by_block = {b["name"]: bool(b.get("is_core")) for b in dictionary["blocks"]}
    rows = []
    for a in ordered_attributes(dictionary):
        applies_to = [
            (f"{fam} / {sub}" if sub else fam) for fam, sub in scopes[a["block"]]
        ]
        rows.append((
            a["block"], a["key"], a["kind"], a.get("unit"),
            json.dumps(a["enum"], ensure_ascii=False) if a.get("enum") else None,
            json.dumps(a["accepts_units"], ensure_ascii=False) if a.get("accepts_units") else None,
            json.dumps(list(dict.fromkeys(a.get("labels") or [])), ensure_ascii=False),
            json.dumps(a.get("conditions") or [], ensure_ascii=False),
            json.dumps(a.get("subfields") or [], ensure_ascii=False),
            json.dumps(applies_to, ensure_ascii=False),
            1 if is_core_by_block.get(a["block"]) else 0,
            1,
        ))
    return rows