# Strategy: product_fact

**Intent:** `product_fact`
**Meaning:** a single fact/attribute/price/code of ONE specific product.

For this question you received evidence from two sources, each line clearly
tagged with which one it is: `[SQL][...]` (db_query, structured data) and
`[DOC][...]` (chunk search results, top_n). These two sources are NOT equal:

- **`[SQL]`-tagged lines are the exact, authoritative source** -- for a
  numeric value/spec/price/code, always trust `[SQL]`.
- **`[DOC]`-tagged content only provides explanatory/supplementary
  context** -- if it contradicts a `[SQL]` line on the same topic (e.g. an
  outdated price/value in a chunk), it is **never used**, trust `[SQL]`.
  `[DOC]` is only used to add explanatory detail not present in `[SQL]`
  (how it works, procedure, certification, etc.).

If both say the same thing, merge them and answer directly, making clear
which source was used (e.g. "according to our records..."). If neither
source has the answer, don't make it up -- say it wasn't found.

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `sql_topn`

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely sources (non-binding): `product_nodes.json` (structured, db_query), product chunks.
- The same `[SQL] authoritative / [DOC] context-only` rule also applies to
  `aggregation`, `visual_request` -- they all share the same `sql_topn` core
  (see the flow table in `strategies/README.md`). (`doc_download` no longer
  shares this core; see `strategies/doc_download.md`.)
