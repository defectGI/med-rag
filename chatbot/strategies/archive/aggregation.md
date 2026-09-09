# Strategy: aggregation

**Intent:** `aggregation`
**Meaning:** counting, filtering, listing, or an extreme value of ANY attribute (price, weight, temperature range, power, speed, ...) across MANY products.

This question asks for an aggregation over MULTIPLE products/records
(counting, filtering, listing, an extreme value/range of an attribute). The
evidence you receive may consist of multiple parts -- each gathered
separately for its own category/filter dimension, containing chunk search
results (top_n) and SQL rows (db_query). Merge all the parts into ONE
answer; don't expose the internal split to the user as "part 1", "part 2" --
give the aggregated result directly.

Each line is clearly tagged with which source it came from: `[SQL][...]`
(db_query, structured data) and `[DOC][...]` (chunk search results, top_n).
These two are NOT equal: **`[SQL]`-tagged lines are the exact, authoritative
source** -- for a numeric value/extreme value/range, always trust `[SQL]`.
**`[DOC]`-tagged content only provides explanatory/supplementary context** --
if it contradicts a `[SQL]` line (e.g. an outdated price/value in a chunk vs.
a current `[SQL]` row), it is **never used**, trust `[SQL]`. Don't silently
resolve the conflict and give the user only the `[SQL]`-based answer; make
clear which source was used (e.g. "according to our records..."). This rule
applies to EACH part separately -- a conflict in one part doesn't affect the
evaluation of the other parts.

The aggregation may not be complete/exact (e.g. the query was split into N
categories but you received no evidence at all for one category) -- don't
silently skip a missing dimension, state "no data found for X"; give the
result you have for the remaining dimensions. If none of the parts/sources
have the answer, don't make it up -- say it wasn't found.

If the user phrased the question in the singular (asking for "the" product
with an extreme value, a count, etc.) but the evidence shows the answer is
actually plural -- several records tie for that same extreme value, or
otherwise all equally match the criteria -- don't arbitrarily pick just one
of them to answer with. If the matching set is small enough to name
individually, list all of them and build the answer around that full set
instead of a single item. If the set is too large to usefully enumerate,
say how many records match instead of naming one arbitrarily, and ask a
short follow-up question that would help narrow the filter down, rather
than guessing which one the user meant.

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely sources (non-binding): `product_nodes.json` (structured, db_query), product chunks.
- The `aggregation` flow (`AggregationFlow`, see `chatbot/flows/aggregation.py`)
  runs `sql_topn`'s single-sided retrieval core (`run_pass`) MULTIPLE times
  -- once per category/filter dimension -- and merges the results; each
  part's evidence arrives distinguishable via an `aggregation_part`
  (0, 1, 2, ...) metadata tag (for trace/diagnostics -- this is NOT a
  "which source is more trustworthy" signal).
- The same `[SQL] authoritative / [DOC] context-only` rule also applies to
  `product_fact`, `visual_request` (see `strategies/product_fact.md` +
  the flow table in `strategies/README.md`) -- it's the SAME rule across all
  `sql_topn`-derived flows (single pass or multi-part). (`doc_download` no
  longer shares the `sql_topn` core; it has its own deterministic flow,
  see `strategies/doc_download.md`.)
