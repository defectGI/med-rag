# ACME Spec Store — Table Reference Catalogue (Data Dictionary)

> **Purpose:** Reference document for the text-to-SQL pipeline's Schema
> Linking stage (natural language -> table/column matching) and for
> error-free JOIN/WHERE generation.
> **Source schema:** `schema_pg.sql` (PostgreSQL) and `specs.db` (SQLite
> copy) -- both in `facts/db/`.
> **Schema status:** 249 products (12 families), 95 canonical attribute
> keys, 0 `spec_value` rows (PHASE A writes the skeleton only), 1512
> `document` rows (772 unique files, covering 231 products).

---

## ⚠ READ FIRST: this is an EAV (Entity-Attribute-Value) database

This schema is NOT the classic "every attribute is a column" design.
Product attributes (voltage, temperature, channel count, ...) are NOT
separate columns -- they are ROWS in the `spec_value` table.

- **Entity** = `product` (one product model)
- **Attribute** = `attribute` (one attribute definition, e.g.
  `operating_temperature`)
- **Value** = `spec_value` (that product's actual value for that
  attribute)

**Golden rule for LLMs:** "What's PN1309's operating temperature?" ->
NOT a SELECT against a column, but a ROW in `spec_value` with
`WHERE product_code='PN1309' AND key='operating_temperature'`.

```sql
-- WRONG (no such column): SELECT operating_temperature FROM product WHERE product_code='PN1309';
-- RIGHT:
SELECT raw_text FROM spec_value WHERE product_code='PN1309' AND key='operating_temperature' AND status='present';
```

**Golden rule #2:** add `AND status='present'` to almost every real
query (otherwise the 16k+ "this product doesn't have this" rows will
be returned too).

**Golden rule #3:** `raw_text` is always authoritative and is the
verbatim cell from the source. Use `raw_text` for human display /
text answers; use the typed `value_*` columns for numeric comparison
and filtering.

---

## Table map (quick reference)

| Table | Role | Rows | Natural-language equivalent |
|---|---|---|---|
| `product` | Entity -- product catalogue | 249 | "product", "model", "device", "board" |
| `document` | Entity -- product-linked files (brochure/datasheet/CE/drawing/image...) | 1512 (772 unique files) | "document", "file", "brochure", "datasheet", "CE declaration", "technical drawing", "user manual", "any other file" |
| `attribute` | Attribute dictionary (metadata) | 95 | "attribute type", "parameter", "spec definition" |
| `spec_value` | Value -- the actual technical data (fact table) | 0 (PHASE A writes the skeleton only; see schema_rag.sql header) | "value", "spec", "technical attribute" |

**Typical JOIN chain:** `product ──product_code──> spec_value
<──block, key── attribute`; `product ──product_code──> document`
(one product can have many files).

---

## Table: product

* **Purpose:** Product catalogue. Each row is an ACME product model.
  Entity table of the EAV -- all specs join into this via `product_code`.
  `family` is the most important filter axis on this table.
* **Natural-language matches (synonyms/keywords):** `product`,
  `model`, `device`, `board`, `module`, `unit`, `DE...` (model code,
  e.g. "PN1309"), `family`, `product family`, `series`, `category`.

### Column details
| Column | Type | Purpose & meaning | Possible values / enum examples |
| :--- | :--- | :--- | :--- |
| `product_code` | TEXT (PK) | Unique model code. The JOIN key for all relations. | `PN1309`, `PN1225`, `PN1106`, `PN1190`, `PN1169` |
| `family` | TEXT (NOT NULL) | Product family (top-level category). Main filter axis. | `MIL-STD-1553 BUS COUPLERS` (103), `TEST SYSTEM COMPONENTS` (31), `PXI EXPRESS SYSTEMS` (30), `AVIONICS INTERFACES` (29), `SIGNAL CONDITIONING & SIMULATION SYSTEMS` (24), `EMBEDDED BOARDS & MODULES` (13), `ETHERNET CONTROLLED INSTRUMENTS (LXI)` (9), `DAQ SYSTEMS` (4), `SOFTWARE PRODUCTS` (2), `PCI Express Systems` (2), `NAVIGATION SYSTEM PRODUCTS` (1), `FMC MODULES` (1) |
| `display_name` | TEXT (NULL permitted) | Human-readable name (`display_name` from `product_nodes.json`). For most products this is filled; if `display_name` is empty it stays NULL. Use this for product NAME; use `product_code` always for WHERE matching. | `SLSC Pass Through Module`, `PXIe Jetson AGX Xavier Module` |
| `subfamily` | TEXT (NULL) | Product subfamily (finer grouping, from the product tree). | `PXI Express Systems AI and Robotics Modules`, `Aviolinks (R) ETH/USB`, `NULL` |
| `acme_code` | TEXT (NULL) | Internal ACME order code. Rarely user-facing. | |
| `list_price` | NUMERIC (NULL) | Catalogue price. NULL means no price recorded in the registry. | |
| `price_on_request` | INTEGER (NOT NULL, 0/1) | 1 = price is deliberately 'on request'. | `0`, `1` |
| `valid_until` | TEXT (NULL) | ISO date the price is valid until. | |
| `price_break_10_99` | NUMERIC (NULL) | Unit price for orders of 10-99 units. | |
| `price_break_100_499` | NUMERIC (NULL) | Unit price for orders of 100-499 units. | |
| `price_break_500_999` | NUMERIC (NULL) | Unit price for orders of 500-999 units. | |
| `is_variant` | INTEGER (NOT NULL, 0/1) | 1 = this product is a variant of another. | `0`, `1` |
| `variant_base` | TEXT (NULL) | product_code of the base product when `is_variant=1`. | |

### Joins
* **spec_value:** `spec_value.product_code` -> `product.product_code`.
  *When asked for all/some specs of a product.* e.g. "PN1309's specs",
  "power consumption of PXI family products".
* **document:** `document.product_code` -> `product.product_code`.
  *When asked for files of a product.* e.g. "Does PN1309 have a
  datasheet?", "any other document for this product?".
* Note: family/category-based queries need three-table join: `product`
  (for family) + `spec_value` (for facts) + `spec_key` (for category).

### Business rules
* `family` values are UPPERCASE and SPACE-BEARING fixed strings.
  Prefer loose matching via `ILIKE '%pxi%'` over equality (users type
  "PXI", not the full name).
* Model codes are case-sensitive; use
  `UPPER(product_code)=UPPER('de9001')` or `product_code ILIKE 'de9001'`
  for user input.
* `display_name` may still be NULL for some products; use it when
  filled, fall back to `product_code` (= the product's identifier)
  when NULL. Never fabricate.
* In-Line bus coupler models have cable-length placeholders in the
  code (e.g. `PN1239.X-X.X`, `PN1246.X-X.X` = Space variant); PCB
  (`DE81xx`), box (`DE82xx`) and In-Line Space (`DE81xxx-...`)
  subfamilies are distinguished by `subfamily`. `DE85xx`–`DE88xx`
  box-type couplers are accessories (only `title`, no measurable
  specs).

---

## Table: document

* **Purpose:** Catalogue of files linked to a product (brochure,
  datasheet, product image, catalogue, CE declaration, technical
  drawing, STP/CAD, user manual, quick-start guide). Multi-document
  model that replaces `product.doc_id`/`source_path`: a product can
  have MORE THAN ONE file. Answers the "any other documents for this
  product/file?" question in the `doc_download` flow.
* **Natural-language matches:** `document`, `file`, `brochure`,
  `datasheet`, `catalogue`, `CE declaration`, `technical drawing`,
  `STP`, `CAD`, `user manual`, `quick start`, `product image`,
  `download`.

### Column details
| Column | Type | Purpose & meaning | Possible values / enum examples |
| :--- | :--- | :--- | :--- |
| `doc_id` | TEXT (part of composite PK) | Stable UUID of the physical file. **NOT unique on its own** -- a file (especially subfamily-level BROCHURE/CATALOGUE, sometimes a single DATASHEET) may belong to multiple products, so the same `doc_id` repeats across rows for different `product_code` values (fan-out, not duplicates). | `89d81777-dc7d-4f41-9a72-079e471cb515` |
| `product_code` | TEXT (part of composite PK, FK->product) | The product the file belongs to. A model can have many `document` rows. | `PN1309`, `PN1127` |
| `doc_type` | TEXT (NOT NULL) | File kind. | `BROCHURE`, `DATASHEET`, `PRODUCT_IMAGE`, `CATALOGUE`, `CE_DECLARATION`, `TECHNICAL_DRAWING`, `STP`, `USER_MANUAL`, `QUICK_START_GUIDE` |
| `file_name` | TEXT (NOT NULL) | Original file name. **MUST BE SHOWN to the user** -- use as the file name / link label in answers. | `ACME_PN5029_Datasheet.pdf`, `PN5106_1.png` |
| `source_path` | TEXT (NOT NULL) | Real filesystem path. **NEVER shown to the user** -- only a separate download-endpoint uses it to resolve the file via `doc_id`. Never write this into answers. | (internal path -- do not show) |
| `is_active` | INTEGER (NOT NULL, 0/1) | 1 = current/active, 0 = superseded but kept for history. Filter on `is_active=1` unless asked otherwise. | `1`, `0` |
| `content_hash` | TEXT (NOT NULL) | `sha256:<hex>` of the file bytes (mirror of `document_nodes.json`'s `scan.content_hash`). Proves which file VERSION produced this row. Never user-facing. | `sha256:1306a4fd5ee90956e365b21a9fa44725bcee7b2b5ed055c50c03cfdc6e1a9ca9` |

### Joins
* **product:** `document.product_code` -> `product.product_code`.
  *When listing a product's files or doing family/category-based
  document queries.* e.g. "Technical drawings of PXI family products".

### Business rules
* **PK is `(doc_id, product_code)`, NOT `doc_id` alone.** A subfamily
  BROCHURE/CATALOGUE file belongs to ALL the products under that
  subfamily and is expanded as a separate row for each. This is not
  limited to BROCHURE/CATALOGUE -- rarely DATASHEET/CE_DECLARATION/
  TECHNICAL_DRAWING/PRODUCT_IMAGE files also fan out the same way;
  "DATASHEET always belongs to a single product" is WRONG.
* **Never show `source_path` to the user, only `file_name`.** The
  download-endpoint resolves the file by `doc_id` (and optionally
  `model`). This is enforced at the schema level -- the model leaking
  this column into answers is an information disclosure.
* All rows are currently `is_active=1` (first scan); once
  `is_active=0` starts appearing do not forget the default filter.
* `product` and `document` are built IN THE SAME RUN (`build_facts_db.py`)
  from the SAME node corpus (`product_nodes.json`/`document_nodes.json`)
  -- there is NO LONGER a category like "not yet ingested into specs.db"
  (KARAR-025: EVERY product in the corpus enters `product`, even if it
  has no documents/facts). When the corpus changes (new product/
  document), rerun `cd facts && python -m facts.build_facts_db`.

---

## Table: spec_key  ⚠ (NOTE: legacy PostgreSQL-only table -- see schema_rag.sql)

> The PostgreSQL schema (`schema_pg.sql`) carries `spec_key` with the
> 9 fixed category enum; the active SQLite schema (`schema_rag.sql`)
> replaced this with a SINGLE `attribute` table (95 keys / 10 blocks)
> with JSON-embedded `labels`/`conditions`/`subfields`/`applies_to`.
> This section documents the PostgreSQL-only `spec_key` table for
> historical/PostgreSQL-deployment reference only.

* **Purpose:** Canonical attribute dictionary (the data dictionary
  itself). Holds the definitions of all 95 distinct technical
  attributes. THIS TABLE DOES NOT HOLD VALUES -- only "which
  attributes exist and what they mean". Source of the `category`
  filter. (`build_facts_db.py` builds this table from the canonical
  `tools/spec_schema/spec_keys.yaml` source.)
* **Natural-language matches:** `attribute`, `parameter`,
  `spec definition`, `field`, `power`, `environmental`, `measurement`,
  `I/O interface`, `switching`, `mechanical`.

### Column details
| Column | Type | Purpose & meaning | Possible values / enum examples |
| :--- | :--- | :--- | :--- |
| `key` | TEXT (PK) | Canonical machine name of the attribute. Target of `spec_value.key`. | `operating_temperature`, `input_voltage`, `number_of_channels`, `bandwidth`, `dimensions`, `power_consumption`, `relay_type` |
| `category` | TEXT (NOT NULL, CHECK enum) | One of the fixed categories. Stronger filter axis than `family` for RAG. | `io_interface` (138), `power` (133), `measurement` (73), `switching` (22), `operational` (18), `navigation` (12), `protection` (12), `environmental` (10), `compute` (7), `mechanical` (5), `identity` (3) |
| `kind` | TEXT (NOT NULL) | Expected value shape (hint for `spec_value.kind`). Can be multiple separated by `/`. | `single`, `list`, `boolean`, `range`, `min_typ_max`, `single / list`, `single / min_typ_max`, `single / range`, `conditional` |
| `unit` | TEXT (NULL) | Canonical unit of the attribute (when applicable). | `°C`, `V`, `Hz`, `ppm`, `bit`, `mm`, `A`, `W`, `Ω`, `%` |
| `scope` | TEXT (NOT NULL, CHECK enum) | Universal or family-specific. | `family` (359: family-specific), `shared` (74: shared across all families -- e.g. operating_temperature, input_voltage) |
| `description` | TEXT (NOT NULL) | Detailed description + Turkish/English aliases (gold mine for Schema Linking). | e.g. accuracy: "...aliases 'doğruluk', 'hassasiyet', 'accuracy', 'tolerance'..." |

### Joins
* **spec_value:** `spec_value.key` -> `spec_key.key`.
  *When a category/key meaning, unit, or type is needed.* e.g.
  "environmental conditions" (`category='environmental'`),
  "all power specs" (`category='power'`).

### Business rules
* **`description` is critical for Schema Linking** -- Turkish user
  words (`doğruluk`, `çalışma sıcaklığı`, `güç tüketimi`) are mapped to
  which `key` here. For ambiguous natural-language requests start with
  `description ILIKE '%kelime%'`.
* Users often speak in category language ("power requirements",
  "environmental conditions", "interfaces"). Map these directly to
  `category`; key search may not be necessary.
* `expected_kind` is GUIDANCE, not binding; the real shape is in
  `spec_value.kind` (one product may give "single", another "list").
* `unit` here is the canonical unit; the actual recorded unit lives in
  `spec_value.unit` (may differ).

---

## Table: spec_value  ⭐ (main fact table)

* **Purpose:** THE technical data. Each row = one fact for one
  product's one attribute. The datasheet cell is stored verbatim in
  `raw`; the parsed numeric/logical values live in the `value_*` slots
  for filtering. Nearly every "how many", "which products", "how
  much" question lands here.
* **Natural-language matches:** `value`, `spec`, `technical
  attribute`, `how many`, `how much`, `which products`, `voltage`,
  `temperature`, `channel count`, `frequency`, `dimension`, `weight`.

### Column details
| Column | Type | Purpose & meaning | Possible values / enum examples |
| :--- | :--- | :--- | :--- |
| `id` | BIGINT (PK, IDENTITY) | Auto row id. | `1, 2, 3...` |
| `model` | TEXT (NOT NULL, FK->product) | Which product. | `PN1309` |
| `key` | TEXT (NOT NULL, FK->spec_key) | Which attribute. | `operating_temperature` |
| `status` | TEXT (NOT NULL, CHECK enum) | Presence status. **CRITICAL for filtering.** | `present` (value exists), `not_specified` (row exists, cell empty), `absent` (product declares this attribute missing), `not_applicable` (attribute is NOT in scope for this product's family -- NOT an answer; "we don't know", NEVER "it has none") |
| `kind` | TEXT (CHECK enum, NULL permitted) | Value shape -- which `value_*` slot is filled. NULL for `absent`/`not_specified`. | `single`, `min_typ_max`, `range`, `list`, `conditional`, `boolean` |
| `value_num` | DOUBLE PRECISION | `single` and `conditional`: single numeric value. | `20.0`, `0.6` |
| `value_min` | DOUBLE PRECISION | `min_typ_max`: minimum. | `50.0` |
| `value_typ` | DOUBLE PRECISION | `min_typ_max`: typical. | `44.0` |
| `value_max` | DOUBLE PRECISION | `min_typ_max`: maximum. | `150.0`, `20.0` |
| `value_from` | DOUBLE PRECISION | `range`: interval start. | `0.0`, `10.0` |
| `value_to` | DOUBLE PRECISION | `range`: interval end. | `40.0`, `90.0` |
| `value_bool` | BOOLEAN | `boolean`: yes/no. | `TRUE` ("Yes"), `FALSE` ("No") |
| `value_text` | TEXT | Non-numeric single text value (`single`). | `482.6 x 43.5 x 206 mm`, `50/60 Hz` |
| `value_list` | JSONB (PostgreSQL only) | `list`: array of values. SQLite has no such column; the list lives in `value_text` as a JSON string. | `["Electromechanical", "latching"]`, `["18 bit", "SAR ..."]` |
| `unit` | TEXT | Recorded unit (per row). | `°C`, `MHz`, `mm`, `Hz`, `%`, `A` |
| `condition` | TEXT | Qualifier the value holds under (from datasheet). | `Relative, non-condensing`, `Forced-air cooling from chassis`, `per channel`, `3.3 V or 2 W` |
| `raw` | TEXT (NOT NULL) | VERBATIM copy of the datasheet cell. This is the authority. | `10% - 90%`, `max 20MHz`, `0°C - 40°C`, `min 50mΩ / max 150mΩ` |
| `raw_tsv` | tsvector (GENERATED, PG only) | Full-text search vector over `raw` (GIN index). For keyword retrieval. | (derived) |
| `review_status` | TEXT (NOT NULL, CHECK enum) | INTERNAL provenance/QA status; NOT a product property. `auto` = extractor produced, no human review; `human_verified` = verified by human; `rejected`/`needs_review`/`conflict` = flagged, be cautious. DO NOT use as a filter unless the user explicitly asks about data quality/verification. | `auto`, `human_verified`, `needs_review`, `conflict`, `rejected` |
| `conflict` | INTEGER (0/1) | 1 = two pieces of evidence with the same specificity level gave DIFFERENT values (a real conflict; do not confuse with `fact_evidence.overridden_by` -- that is a SILENT overwrite, `conflict` is NOT SET). Internal; not a product attribute. | `0`, `1` |
| `extraction_run_id` | TEXT (FK->extraction_run, NULL permitted) | Filled ONLY for `status IN ('absent','not_specified')` rows -- which extraction run concluded "missing/not declared". NULL for `present` (its evidence is in `fact_evidence`). Pass 3 (13d-4) is not implemented yet, so today always NULL. | `NULL` |

### `kind` -> populated slot map (critical LLM reference)
| kind | Populated `value_*` column(s) | How to write a numeric filter |
| :--- | :--- | :--- |
| `single` (numeric) | `value_num` (+ `unit`) | `value_num >= X` |
| `single` (text) | `value_text` (+ `unit`) | `value_text ILIKE '%...%'` |
| `min_typ_max` | `value_min` / `value_typ` / `value_max` | `value_max <= X`, `value_typ = X` |
| `range` | `value_from` / `value_to` | `value_to >= X` (upper bound), `value_from <= X` (lower bound) |
| `conditional` | `value_num` + `condition` | `value_num`; condition is in `condition` |
| `boolean` | `value_bool` | `value_bool = TRUE` |
| `list` | `value_list` (PG JSONB) / `value_text` (SQLite) | PG: `value_list @> '["X"]'`; or `raw ILIKE '%X%'` |

### Joins
* **product:** `spec_value.model` -> `product.model`.
  *When family filtering (`product.family`) or product metadata is
  needed.* e.g. "Products in the PXI family that operate above 70°C".
* **spec_key:** `spec_value.key` -> `spec_key.key`.
  *When a category filter (`spec_key.category`) or a key's
  description/unit is needed.* e.g. "PN1309's power specs".
* Single product + single key queries (e.g. "PN1309's operating
  temperature") need NO JOIN — direct `WHERE model=.. AND key=..`.

### Business rules
* **`UNIQUE(model, key)`** — at most ONE row per (product, key). "PN1309's
  operating temperature" returns one row; no `GROUP BY`/`DISTINCT`
  needed.
* **`status='present'` filter is needed almost always.** `absent` rows
  mean "this product doesn't have this attribute" and carry no value;
  counting them gives wrong answers. "How many products support X"
  type questions = `WHERE key=.. AND status='present'`.
* **For numeric comparison use `value_*`, for display/text answers
  use `raw`.** Complex cells may have only `raw` reliable. When using
  numeric filters also implicitly require `value_x IS NOT NULL`.
* **SQLite vs PostgreSQL difference:** `value_list` (JSONB) and
  `raw_tsv` (tsvector) columns exist ONLY in PostgreSQL. In the SQLite
  copy list values live in `value_text` as a JSON string. Write the
  query according to the target engine.
* Unit (`unit`) is per-row and may differ from `spec_key.unit`; verify
  unit consistency in `unit` before numeric comparison.

---

## Table: fact_evidence  (INTERNAL provenance, not normally JOINed in text-to-SQL user queries)

* **Purpose:** Evidence chain for `spec_value`'s `status='present'`
  rows (PROTOCOL KARAR-015/017/023/024). One fact can carry MULTIPLE
  pieces of evidence (N:1) -- `spec_value.id` remains a single row
  (`UNIQUE(model,key)` is preserved); the MANY side is the evidence.
  Today: 0 rows (spec_value is empty).
* **Natural-language matches:** none normally -- this is internal
  provenance; it does not directly answer user queries. JOIN only for
  meta-questions like "what's the source of this info / which document
  did it come from".

### Column details
| Column | Type | Purpose & meaning |
| :--- | :--- | :--- |
| `evidence_id` | TEXT (PK) | Opaque id (same pattern as KARAR-001). |
| `fact_id` | INTEGER/BIGINT (FK->spec_value.id) | Which fact the evidence is for. |
| `doc_id` | TEXT | Evidence document (FK sense -> `document.doc_id`, but no FK enforced at schema level -- evidence may point at a document). |
| `chunk_sha256` | TEXT | Evidence chunk hash (`sha256:<hex>`); the text snapshot lives in `facts/snapshots/` under that hash, NOT in the DB. |
| `page_start` / `page_end` | INTEGER (NULL permitted) | Evidence page range. |
| `heading_path` | TEXT (JSON array) | Heading path inside the document. |
| `chunk_node_id` | TEXT (NULL permitted) | For DIAGNOSTICS, NOT BINDING (KARAR-003/004/015) -- no query should rely on it. |
| `specificity` | TEXT (CHECK enum) | Specificity of the evidence document in the product tree. | `family`, `subfamily`, `subfamily_2`, `product` (most specific) |
| `derivation` | TEXT (CHECK enum) | `direct` (document linked directly to this product) / `inherited` (inherited from a category document, KARAR-014 fan-out). |
| `overridden_by` | TEXT (FK->fact_evidence.evidence_id, NULL permitted) | If this evidence was SILENTLY OVERWRITTEN by a more specific evidence, that evidence's id (KARAR-023 "overwrite" -- NOT a conflict, `spec_value.conflict` is NOT SET). |
| `stale` | INTEGER (0/1) | 1 = this evidence's `doc_id` has `scan_status IN {MODIFIED,MOVED,DELETED}` in the registry OR its `chunk_sha256` no longer exists in the current chunk set (KARAR-018 signals 1/2). No automatic re-anchoring -- the record (with its snapshot) is KEPT, only flagged. A fact whose ALL evidence rows are `stale=1` turns into `needs_review`. | `0`, `1` |
| `added_at` | TEXT | KARAR-008 offset local ISO-8601. |

### Business rules
* Overwrite (`overridden_by` populated) vs real conflict
  (`spec_value.conflict=1`) are DIFFERENT states -- do not confuse.
  An overwritten evidence record is NOT deleted, only flagged
  (traceability).
* `stale=1` and `overridden_by` populated are also DIFFERENT states:
  `stale` says "this evidence is no longer current, needs review"
  (13c differ); `overridden_by` says "this evidence was OVERWRITTEN
  by a more specific evidence" (13d-2 reduce, KARAR-023); they work
  independently.

---

## Table: extraction_run  (INTERNAL provenance, 0 rows today)

* **Purpose:** Evidence for `spec_value`'s `absent`/`not_specified`
  rows (PROTOCOL KARAR-015) -- not "a chunk of evidence", but "we
  LOOKED at this document/chunk set and FOUND NOTHING". Distinguishing
  "looked, none" from "never looked" is built ONLY by this. Producer:
  Pass 3 (ROADMAP 13d-4) -- not yet implemented; today the table is
  EMPTY.
* **Natural-language matches:** none, internal provenance.

### Column details
| Column | Type | Purpose & meaning |
| :--- | :--- | :--- |
| `run_id` | TEXT (PK) | Run id. |
| `model` | TEXT (FK->product.model) | Product the run covered. |
| `doc_ids` | TEXT/JSONB (JSON array) | Document set examined by this run. |
| `chunk_shas` | TEXT/JSONB (JSON array) | Chunk set examined (`chunk_sha256` list). |
| `extractor_version` / `prompt_version` | TEXT | Extractor/prompt version used (KARAR-007 pattern). |
| `generated_at` | TEXT | KARAR-008 offset local ISO-8601. |

---

## VIEW: spec_chunk

* **Purpose:** RAG (embedding/semantic search) — ready-made sentence
  producer view. Contains ONLY `status='present'`; pre-JOINed with
  `product` and `spec_key`. Each row is reduced to ONE human-readable
  sentence (`chunk_text`). NOT normally used in text-to-SQL numeric
  queries — that work happens in `spec_value`. This view is for
  semantic retrieval / context fetching.
* **Natural-language matches:** `context`, `description sentence`,
  `RAG chunk`, `chunk`, `context`, `embedding text`.

### Column details
| Column | Source | Purpose & meaning |
| :--- | :--- | :--- |
| `id` | `spec_value.id` | Source value row id |
| `model` | `spec_value.model` | Product model |
| `family` | `product.family` | Product family (JOIN) |
| `category` | `spec_key.category` | Attribute category (JOIN) |
| `key`, `status`, `unit`, `condition`, `raw` | `spec_value.*` | Relevant value columns |
| `chunk_text` | derived | `family / / / model — key: raw` single-sentence form. e.g. `DAQ SYSTEMS / PN1015 — input_voltage: 20 V` |

### Business rules
* View cannot be written to; `WHERE status='present'` is already
  baked in (no need to add it).
* For open-ended questions about a product/category (e.g. "what do you
  know about PN1309"), this view is ideal; for precise numeric
  filtering use `spec_value`.

---

## Text-to-SQL ready-made query templates

```sql
-- 1) Single product, single attribute (no JOIN) -- "PN1309's operating temperature"
SELECT raw FROM spec_value
WHERE model = 'PN1309' AND key = 'operating_temperature' AND status = 'present';

-- 2) Cross-family numeric filter -- "products operating at 70°C or above"
SELECT model, raw FROM spec_value
WHERE key = 'operating_temperature' AND status = 'present' AND value_to >= 70;

-- 3) Category-based, single product -- "PN1309's power specs" (spec_key JOIN)
SELECT v.key, v.raw FROM spec_value v
JOIN spec_key k ON k.key = v.key
WHERE v.model = 'PN1309' AND k.category = 'power' AND v.status = 'present';

-- 4) Family-based -- "channel count in PXI family products" (product JOIN)
SELECT p.model, v.raw FROM spec_value v
JOIN product p ON p.model = v.model
WHERE p.family ILIKE '%PXI%' AND v.key = 'number_of_channels' AND v.status = 'present';

-- 5) Count products supporting a feature -- "how many products support power_down_mode"
SELECT COUNT(*) FROM spec_value
WHERE key = 'power_down_mode' AND status = 'present' AND value_bool = TRUE;

-- 6) Ambiguous natural language -> find the right key (Schema Linking)
SELECT key, category, description FROM spec_key
WHERE description ILIKE '%doğruluk%' OR key ILIKE '%accuracy%';

-- 7) List value (PostgreSQL) -- "products with latching relay type"
SELECT model, raw FROM spec_value
WHERE key = 'relay_type' AND status = 'present' AND value_list @> '["latching"]';
```

### LLM checklist
1. A question that "looks like a column" is still EAV -> think `spec_value` ROW.
2. Add `status = 'present'` to almost every query.
3. Product name/model -> `model` (case-insensitive via `ILIKE`); family -> `product.family ILIKE '%..%'`.
4. Category language ("power", "environmental") -> `spec_key.category` enum.
5. Numeric comparison -> the right `value_*` slot per `kind` (map above).
6. Text answer -> **`raw`** (authority); numeric filter -> `value_*`.
7. If the target engine is not PostgreSQL, `value_list`/`raw_tsv` don't exist -- use `value_text`/`raw ILIKE`.