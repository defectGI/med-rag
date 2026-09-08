"""Reduces the dictionary copy sent to the LLM -- not the FULL
`chatbot-corpus/spec_schema/spec_keys.yaml`, only the fields actually
needed for the extraction decision. The rest (audit/production
metadata) can be filled in by code; sending them to the LLM only
consumes tokens (user decision, 2026-08-03).

Fields removed and WHY the LLM doesn't need them:
  - `attributes[].evidence` (ndocs/distinct_values/note): provenance of
    HOW the dictionary was measured (R0.3/R0.4 CHECK inputs) -- it has
    NO impact on the extraction decision at runtime; the columns written
    to the DB by `build_facts_db.py` are already populated from the
    FROZEN spec_keys.yaml (NOT by this script).
  - `attributes[].is_family_defining`: same rationale, R0.3 exception's
    audit flag.
  - `facets` / `relations`: retrieval labelling + product-relation
    vocabulary -- this prompt only extracts attribute_value, neither
    is USED.
  - `ruleset_exceptions`: the dictionary's own record of which R-rule
    it exempts itself from while being built -- extraction carries its
    own "new key" rules in the system prompt (see
    chunk_strategy_system.md); the LLM has no need for that text.
  - `version`/`source`/`generated_by`/`ruleset` (pointer text): the
    dictionary's own production identity, does not enter the extraction
    decision.

Fields KEPT (the ones the LLM ACTUALLY uses -- see the prompt's
"Using the dictionary" section): `kinds`, `statuses`,
`blocks[].name/is_core/applies_to/description`,
`attributes[].block/key/kind/unit/question/labels/conditions/subfields/
enum/accepts_units/notes`.

Measurement (2026-08-03, tiktoken cl100k_base approximation): full
dictionary ~15160 tokens, trimmed ~11780 -- 22% reduction, fixed
overhead so it applies to EVERY call (ALL products/batches);
cumulative savings on a large run are significant.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_LLM_ATTRIBUTE_FIELDS = (
    "block", "key", "kind", "unit", "question", "labels", "conditions",
    "subfields", "enum", "accepts_units", "notes",
)
_LLM_BLOCK_FIELDS = ("name", "is_core", "applies_to", "description")
_LLM_TOP_FIELDS = ("kinds", "statuses", "blocks", "attributes")


def build_llm_schema_text(spec_keys_path: Path) -> str:
    """Reads `spec_keys_path`; returns the trimmed YAML text for the LLM."""
    full = yaml.safe_load(spec_keys_path.read_text(encoding="utf-8"))
    trimmed = {
        "kinds": full["kinds"],
        "statuses": full["statuses"],
        "blocks": [{k: b[k] for k in _LLM_BLOCK_FIELDS if k in b} for b in full["blocks"]],
        "attributes": [
            {k: a[k] for k in _LLM_ATTRIBUTE_FIELDS if k in a} for a in full["attributes"]
        ],
    }
    return yaml.dump(trimmed, allow_unicode=True, sort_keys=False)


if __name__ == "__main__":
    import sys

    # K-71 closure (2026-08-21): canonical file was moved under
    # `tools/spec_schema/` (the old `chatbot-corpus/spec_schema/` no
    # longer exists) -- path updated. `frozen` (atom_vs_chunk_bench/)
    # is OUT of K-59 scope and did not return; this line is only the
    # diagnostic __main__ block's fallback path, and `canonical`
    # always exists in practice, so it is effectively unused.
    HERE = Path(__file__).resolve().parent
    canonical = HERE.parents[3] / "tools" / "spec_schema" / "spec_keys.yaml"
    frozen = HERE.parents[3] / "facts" / "experiments" / "atom_vs_chunk_bench" / "schema_context" / "spec_keys.yaml"
    path = canonical if canonical.is_file() else frozen
    text = build_llm_schema_text(path)
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        print(f"source: {path}\ntrimmed token count (approximate, cl100k_base): {len(enc.encode(text))}", file=sys.stderr)
    except ImportError:
        print(f"source: {path}\ntrimmed character count: {len(text)}", file=sys.stderr)
    print(text)