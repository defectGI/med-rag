# spec_schema — the specs.db key dictionary

This directory holds the "what we know" layer of `facts/db/specs.db`
(the comparable/filterable information the chatbot stores) **and its
table DDL**. The canonical DDL is `schema.sql`; the dictionary source
is `attribute_schema_v3.json`; the flattened output is `spec_keys.yaml`.

## Where it sits in the pipeline

```
attribute_schema_v3.json  (this folder -- SOURCE: R0-R3 filtered dictionary)
         |  scripts/build_spec_keys.py
         v
   spec_keys.yaml          (this folder -- OUTPUT: flattened dictionary)
         |  (OUTSIDE this folder, in a separate hand-driven step)
         v
   facts/db/specs.db        (dictionary layer: attribute_block/attribute/...)
         |  read by path (not imported)
         v
   src/medrag/api/  (chatbot)  -- factory.py::_FACTS_DB_DIR, panel/ingest.py
```

This folder does **not** host the extractor code or the pipeline that
populates `specs.db` — it only defines the dictionary and owns
`schema.sql`. The Python that used to live under `facts/facts/` was
removed in a dead-code cleanup; `facts/db/specs.db` is currently a
**static, hand-built artifact** in this repo, consumed solely by
`src/medrag/api/` (the chatbot) — its path comes from the
`FACTS_DB_DIR` / `SPECS_DB_PATH` constants in
`src/medrag/core/paths.py`.

## The one rule: nothing in the dictionary is invented

`spec_keys.yaml` is **not hand-written; it is generated** (by
`scripts/build_spec_keys.py`). The rule's location shifted but it did
not disappear: the dictionary is no longer built by label-scanning the
corpus — `attribute_schema_v3.json` is the canonical dictionary,
filtered through rules R0-R3, with every key carrying its own
`evidence` measurement. Three checks run in parallel:

- **At build time** — the generator validates every scope name against
  `product_nodes.json` and aborts on drift (see "Scope axis" below).
- **At validation time** — `scripts/validate.py` reads the file from
  disk (a second pair of eyes, independent of the generator).
- **In the DB** — most of R0-R3 is enforced as CHECK constraints in
  `schema.sql`.

Why this matters: in the dictionary's early versions, the `kind` field
was never measured — it was written down from intuition. For
`operating_temperature` it read `[range, single]` — but the corpus has
**72/72 rows of `range`**; `single` never appears. Similarly, the
`priority` field claimed "customers ask for this often" without any
underlying customer-question data. A dictionary like that silently
breaks everything built on top of it (extractor, text2sql schema,
chatbot answer).

### Evidence threshold

A label must appear either in **at least 2 distinct documents** OR in
**at least 5 rows of a single document**. The second clause is for
catalog files: a MIL-STD-1553 coupler catalog gives specs for 103
products in one file, so document count alone wouldn't have the power
(`termination resistor value`: 52 rows / 1 document).

### Evidence chain

```
all_chunks.json (table rows + body "LABEL: value" prose lines)
  -> doc_id
  -> document_nodes.json  links.owner_ids
  -> product_nodes.json   node_id -> family/subfamily/subfamily_2
  -> taxonomy node id
```

Both table rows and body text are scanned (`scripts/evidence.py`,
distinguished by the `source` field). The second pass is necessary: in
coupler datasheets, `Characteristic Impedance | Fault Protection |
Termination Resistor Value` appear as a **column header** row, with
the values in body prose (`TERMINATION RESISTOR VALUE: 78.7 OHMS ±1%
2W (R7, R8)`). If only row labels are scanned, those three specs look
evidence-less and are dropped unfairly.

> `evidence.py` was used to measure v3's `evidence` fields; it is no
> longer in the build chain (`build_spec_keys.py` doesn't call it).
> See "Known issue" below.

## Second rule: every key has exactly ONE kind

A key's `kind` is **semantic** and belongs to the key: `operate_time`
is an upper bound, `operating_temperature` is a two-ended interval,
`number_of_channels` is a count. How a document presents it (a 4-column
`min | typ | max` table, or a single cell) is **a presentation
difference**, not a kind difference — the parser normalizes it.

Example: `switch operate time` is `4ms` in an LXI datasheet and
`— | — | 10 ms | —` in a PXIe datasheet. Both are saying "max". One
kind: `min_typ_max`; the single-cell value lands in the max column.

When a measurement shows a second shape for a key, one of three things
is going on, and none of them require a per-row kind:

1. **Presentation difference** -> the parser normalizes it (above).
2. **Conditional variant** -> one kind + a `condition` field.
   `number_of_channels`: `16` vs
   `32 single-ended or 16 fully differential (software-selectable)` —
   both are counts; the second's configuration goes to `condition`.
3. **The key conflates two different things** -> split it. That is
   why PDU outlet type (`outlet_type`) and digital-output driver type
   (`digital_output_type`) are separate keys: both appear under the
   `output type` label in documents, the node splits them.

`scripts/validate.py` treats a `kind` being a list as an **error**.

## Scope axis: `family` / `subfamily` / `subfamily_2`

No invented product classification is used. Every block carries an
`applies_to` list whose names come verbatim from the category nodes in
`product_nodes.json` (a family name, or `family / subfamily`); the
generator validates this against `product_nodes.json` on every build.
`core`'s scope is empty = every family.

`applies_to` **derives from measurement**: a label is in a block
because it was observed in documents at that node — not by judgement.
Inheritance runs downward: listing a node also covers its descendants.

> Because `applies_to` is observation-based, it is **not exhaustive —
> it is biased toward the narrow side**: a spec may be perfectly
> meaningful at a node whose documents just don't use that label, and
> it will not show up there. This is deliberate — widening scope by
> guesswork would be invention.

## Value format

- `kind`: `single` / `min_typ_max` / `range` / `list` / `conditional` / `boolean`
- `raw` is always stored; the typed columns are derived from raw.
- `status` (the `attribute_value_status` enum in `schema.sql`):

| status | meaning | chatbot answer |
|---|---|---|
| `present` | value is explicitly stated in the document | "it is ..." |
| `not_specified` | key applies to this product, document gives no value | "not stated in the datasheet" |
| `absent` | document explicitly states the feature is ABSENT | "no, this product does not have it" |
| `not_applicable` | key was never observed for this product's family/subfamily (out of scope) | question not asked |
| `conflicting` | two documents give different values for the same product | separate status until resolved |
| `superseded` | a newer document invalidated the value (PROPOSED) | old value drops out of retrieval |

`absent` is produced only on an explicit statement — observed
example: `Termination Resistor Value: Not terminated`. It is NEVER
generated by peer comparison or inference. `not_applicable` is also
**NOT `absent`**: turning "I didn't observe this on that branch" into
"this product doesn't have it" would be invention.

## Dictionary contents (spec_keys.yaml v2.0.0, source: attribute_schema_v3.json v3.1)

**95 keys, 10 blocks**, every one evidenced, every one single-kind.

| kind | count | | block | key count | scope |
|---|---|---|---|---|---|
| `single` | 59 | | `core` | 19 | all families |
| `range` | 15 | | `analog_io` | 15 | DAQ, signal conditioning, PXIe DAQ/DMM, LXI DAQ |
| `list` | 8 | | `embedded_compute` | 10 | embedded board, FMC, PXIe chassis/controller, AI/robotics |
| `conditional` | 7 | | `chassis_controller` | 8 | PXIe chassis/controller, LXI mainframe |
| `min_typ_max` | 4 | | `switching_relay` | 13 | PXIe/LXI switch-mux, RF switch-mux |
| `boolean` | 2 | | `digital_io` | 7 | PXIe digital I/O-FPGA, DAQ, signal conditioning |
| | | | `mil_std_1553_coupler` | 7 | MIL-STD-1553 bus coupler |
| | | | `power_distribution` | 6 | PDU products |
| | | | `battery_simulator` | 5 | battery simulators |
| | | | `avionics_bus` | 5 | avionics interfaces, navigation |

`spec_keys.yaml` also carries `facets` (function/application/
form_factor labels), `relations` (`compatible_with` /
`accessory_of` / `variant_of` / `requires`) and `ruleset_exceptions`
blocks — these were added in `attribute_schema_v3.json` v3.1 and have
not yet been carried over into `schema.sql`/the DB; see the files
themselves for their content.

> Note: the 95-key count in this dictionary differs from the older
> "109 key" count carried in earlier drafts of this README. The 109-key
> figure belonged to a now-removed corpus-mined dictionary; this
> section was re-computed against the current `spec_keys.yaml`.

## Out of scope (dropped from the dictionary)

- **Pinout rows, signal descriptions, register maps, document
  metadata** (`rev. no`, `sheet`, `scale`), **figure captions**,
  software UI screen text. `scripts/evidence.py::NOISE` filters these
  out. For reference, the great majority of the 430 keys in the
  earlier `specs.db` had leaked in from exactly this kind of noise
  (`bj770`, `gth_dp0_c2m_p`, `hpc11_p`, `io_header`, `mio68`…).
- **CE / RoHS / EMC standards.** Present in the corpus, but they
  don't fit the key/label model: the standard **name is the label**,
  the year/class is **value** (`en iec 61326-1` -> `2021, Class B`).
  That needs a separate extraction mechanism; rather than invent an
  `emc_standards` key, it was left out of scope. There is no
  `compliance` category for this reason.
- **Concepts without values.** E.g. `compatible_products` (no
  document carries a "compatible ACME product codes" field),
  `rf_insertion_loss` / `rf_vswr` (the single RF product has a
  datasheet that doesn't give these values), `mtbf` /
  `ingress_protection` / `warranty_period` (appear in prose, not in
  label-value form). Dropped from the dictionary.
- **`ce_declared`.** Its source is NOT a datasheet — it's the
  `owner_ids` of a `doc_type=CE_DECLARATION` entry in
  `document_nodes.json`. This dictionary only carries keys measurable
  from document content; CE information comes from a different (non-LLM
  deterministic) pipeline and is written to the DB there.

## Files

The folder root is **output**, `scripts/` is **the generator**:

```
spec_schema/
  schema.sql                <- CANONICAL DDL (PostgreSQL 15+/pgvector). The schema's source of truth.
  attribute_schema_v3.json  <- SOURCE: the dictionary itself (v3.1, 95 keys / 10 blocks)
  spec_keys.yaml             <- OUTPUT: the flattened dictionary (v2.0.0). Not hand-edited.
  README.md
  scripts/
    build_spec_keys.py        generator: v3 JSON -> spec_keys.yaml
    evidence.py                corpus evidence collector (used to measure v3; NO LONGER in
                                 the build chain -- its path is broken, see "Known issue")
    validate.py                validator (second pair of eyes, independent of the generator)
```

- To add a new key / change a kind -> edit `attribute_schema_v3.json`
  and re-run the generator.
- To change scope -> the block's `applies_to` in v3. The name must
  match a family/subfamily name in `product_nodes.json` **verbatim**;
  the generator validates and aborts on drift.

```
python scripts/build_spec_keys.py    # aborts on a drifted scope name
python scripts/validate.py           # 0 = clean
```

Hand-editing `spec_keys.yaml` is pointless — the next build will wipe it.

## How the dictionary sits in `schema.sql`

The dictionary fans out to layer 3 of `schema.sql` (the dictionary
layer), not into one table:

```
blocks[]                -> attribute_block
attributes[]            -> attribute (question/kind/unit/evidence)
attributes[].labels     -> attribute_label      (R1.5: label does NOT enter the name)
attributes[].conditions -> attribute_condition  (R1.2/R1.3: condition does NOT enter the name)
attributes[].subfields  -> attribute_subfield   (R2.3: accuracy does NOT open its own key)
blocks[].applies_to     -> attribute_applicability
```

Rules R0-R3 are DB constraints (`attr_r0_1_ck` … `attr_r1_6_ck`):
short-question, unit-less range, `max_`-prefixed, or single-valued
keys cannot enter the table.

### Scope is a hint, not a hard limit

`applies_to` decides which questions get asked and which `(product x
key)` matrix row gets born `not_applicable`. **It does not reject a
value.** If the extractor model writes a value outside the declared
scope, the DB accepts it — because the raw text of an out-of-scope
value was measured to be CORRECT (e.g. a USB adapter has logic levels
too, even if `applies_to` doesn't list it). A hard limit would lose
genuine correct values.

The boundary does not vanish, it becomes **visible**: the
`v_out_of_scope_value` view reports every out-of-scope written value.
When a key shows up there repeatedly, that's the evidence to widen
`applies_to`.

## Known issue: `product_nodes.json` path

The `PRODUCT_NODES` path inside `build_spec_keys.py`, `validate.py`,
and `evidence.py` walks three levels up from `__file__` to find
`chatbot-corpus/product_info/product_nodes.json`. After the move of
this folder to `tools/spec_schema/`, that walk needs to land at the
repo root and then descend into `chatbot-corpus/` — without a manual
fix, all three scripts will look for
`tools/product_info/product_nodes.json` (which does not exist) and
crash with `FileNotFoundError`. The path expressions have been
updated to walk two levels up (correct for the current layout), but
verify the resolved path is `chatbot-corpus/product_info/
product_nodes.json` before running the scripts.