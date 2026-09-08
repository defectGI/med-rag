"""Validator for `spec_keys.yaml` -- a second pair of eyes, independent of the producer.

`build_spec_keys.py` already validates while producing; this script looks
at the FILE on disk. The two must stay separate: if a human has hand-
edited the YAML, if a stale version is lying around, or if the producer's
source has changed, the producer's own checks would never run -- this
catches that.

What it locks down:
  1. Every block's `applies_to` is a REAL family or `family / subfamily`
     name in `product_info/product_nodes.json`. (This is the test that
     guarantees the scope axis was not invented.)
  2. No block lists both a family and its own subfamily -- inheritance
     already covers downward; listing both is silent redundancy.
  3. Every attribute's `kind` is a single value from the enum;
     `question` / `labels` / `evidence` are all present -- a key that
     can't be asked, or that has no evidence, is not a key.
  4. Key names are snake_case and unique WITHIN a block (cross-block
     duplication is legitimate -- channel_count lives in 3 blocks).
  5. The naming rules from `schema.sql` (R1.1/R1.2/R1.4/R1.6) -- before
     anything hits the DB.
  6. Every `kind=conditional` key carries a non-empty `conditions` list.
  7. Every key carrying an `enum` has `kind=single` (a closed vocabulary
     yields one value).

Usage: python scripts/validate.py   (exit code 0 = clean)
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent  # .../tools/spec_schema/scripts
SCHEMA = HERE.parent  # .../tools/spec_schema
# spec_schema/ lives directly under tools/, but chatbot-corpus/ sits next
# to tools/ at the repo root, so we walk up one more level to find it.
CORPUS = SCHEMA.parent.parent / "chatbot-corpus"  # .../chatbot-corpus
SPEC_KEYS = SCHEMA / "spec_keys.yaml"
PRODUCT_NODES = CORPUS / "product_info" / "product_nodes.json"

SCOPE_SEP = " / "

# Mirrors the CHECK constraints on attribute in schema.sql. This list MUST
# be kept in sync with that file -- if it drifts, the DB raises
# IntegrityError when it loads (so a drift is not silent, just caught
# late; this script catches it early).
NAME_RULES = [
    ("R1.1", re.compile(r"^(max|min|typ|peak|nominal|rated|avg)_"), "ekstremum öneki"),
    ("R1.1", re.compile(r"_(max|min|typ|range)$"), "ekstremum/aralık soneki"),
    ("R1.2", re.compile(r"_(at_[0-9]|no_load|per_channel|front_panel|rear_panel)"), "koşul adda"),
    ("R1.4", re.compile(r"^(number_of_|total_number_of_|num_)"), "sayaç deseni"),
    ("R1.6", re.compile(r"_(v|mv|a|ma|w|hz|khz|mhz|mm|ohm|bit|bits|ms|us|db)$"), "birim adda"),
]


def corpus_scopes() -> tuple[set[str], set[str]]:
    nodes = json.loads(PRODUCT_NODES.read_text(encoding="utf-8"))["product_nodes"]
    families = {n["family"] for n in nodes}
    subfamilies = {f"{n['family']}{SCOPE_SEP}{n['subfamily']}" for n in nodes if n.get("subfamily")}
    return families, subfamilies


def main() -> int:
    doc = yaml.safe_load(SPEC_KEYS.read_text(encoding="utf-8"))
    errors: list[str] = []

    families, subfamilies = corpus_scopes()
    kinds = set(doc["kinds"])
    block_names = {b["name"] for b in doc["blocks"]}

    # --- 1/2) block scopes
    for b in doc["blocks"]:
        scopes = b.get("applies_to") or []
        if b.get("is_core") and scopes:
            errors.append(f"[blok] {b['name']}: core bloğun kapsamı BOŞ olmalı (her aile)")
        for s in scopes:
            if SCOPE_SEP in s:
                if s not in subfamilies:
                    errors.append(f"[blok] {b['name']}: product_nodes.json'da olmayan alt-aile {s!r}")
            elif s not in families:
                errors.append(f"[blok] {b['name']}: product_nodes.json'da olmayan aile {s!r}")
        for a in scopes:
            for c in scopes:
                if a != c and c.startswith(a + SCOPE_SEP):
                    errors.append(
                        f"[blok] {b['name']}: kapsamda hem {a!r} hem alt düğümü {c!r} var "
                        "(kalıtım zaten kapsar)"
                    )

    # --- 3..7) attributes
    seen: set[tuple[str, str]] = set()
    for a in doc["attributes"]:
        name = a.get("key")
        block = a.get("block")
        where = f"{block}/{name}"
        if not name:
            errors.append("[key] 'key' alanı yok")
            continue
        if block not in block_names:
            errors.append(f"[key] {where}: bilinmeyen blok")
        if (block, name) in seen:
            errors.append(f"[key] {where}: aynı blokta tekrar eden ad")
        seen.add((block, name))
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            errors.append(f"[key] {where}: snake_case değil")

        for rule, pattern, what in NAME_RULES:
            if pattern.search(name):
                errors.append(f"[key] {where}: {rule} ihlali ({what})")

        for field in ("kind", "question", "labels", "evidence"):
            if field not in a:
                errors.append(f"[key] {where}: '{field}' alanı yok")

        if isinstance(a.get("kind"), list):
            errors.append(f"[key] {where}: kind LİSTE olamaz, tek değer olmalı")
        elif a.get("kind") not in kinds:
            errors.append(f"[key] {where}: bilinmeyen kind {a.get('kind')!r}")

        if len((a.get("question") or "").strip()) < 10:
            errors.append(f"[key] {where}: question yok ya da 10 karakterden kısa")
        if not a.get("labels"):
            errors.append(f"[key] {where}: labels boş -- kanıtsız key olamaz")

        ev = a.get("evidence") or {}
        if not ev.get("ndocs"):
            errors.append(f"[key] {where}: evidence.ndocs yok/sıfır")
        if ev.get("distinct_values") is not None and ev["distinct_values"] < 2:
            errors.append(f"[key] {where}: R0.4 ihlali, distinct_values < 2")

        if a.get("kind") in ("min_typ_max", "range") and not a.get("unit"):
            errors.append(f"[key] {where}: R0.2 ihlali, kind={a['kind']} birim ister")
        if a.get("kind") == "conditional" and not a.get("conditions"):
            errors.append(f"[key] {where}: kind=conditional ama condition sözlüğü boş")
        if a.get("enum") and a.get("kind") != "single":
            errors.append(f"[key] {where}: enum taşıyor ama kind={a.get('kind')!r} (single olmalı)")

    # --- rapor
    print(f"blok            : {len(doc['blocks'])}")
    print(f"key             : {len(doc['attributes'])}  (hepsi tek kind, hepsi kanıtlı)")
    print(f"etiket          : {sum(len(a.get('labels') or []) for a in doc['attributes'])}")
    print(f"kapsam satırı   : {sum(len(b.get('applies_to') or []) for b in doc['blocks'])}")
    if errors:
        print(f"\nHATA: {len(errors)}")
        for e in errors:
            print("  " + e)
        return 1
    print("\ndoğrulama temiz.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
