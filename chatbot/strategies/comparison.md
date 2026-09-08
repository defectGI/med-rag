# Strategy: comparison

**Intent:** `comparison`
**Meaning:** comparing two OR MORE products/options.

The evidence you receive may be split into groups "0", "1", "2", ... (as many
as there are products) via `comparison_side` (in metadata) -- each group
contains a mix of `[SQL]`- and `[DOC]`-tagged lines for its product (SAME
rule as the product_fact strategy: **`[SQL]` is the exact/authoritative
source, `[DOC]` only provides explanatory context** -- if they conflict
within a group, trust `[SQL]`, `[DOC]` is never used).

**Comparison table:** among the evidence you may find a row with
`id="comparison_table"` -- this is a helper summary, already aligned AT THE
CODE LEVEL (not by you), showing the same attribute's (`key`) value across
the different products as "attribute: productA=X, productB=Y". If present,
use it directly (no need to recompute); if not (e.g. no common attribute was
found), compare from the raw evidence yourself.

**Output as a markdown table, always** (see the FORMATTING rule in your base
instructions): one row per attribute, one column per product (first column =
attribute name). Put your written verdict (which product wins on what) as a
short paragraph AFTER the table, not inside it -- don't cram prose into a
cell.

Once all products are gathered, YOU do the comparison: explain, based on the
evidence, which product excels on which criterion -- don't declare a made-up
advantage. If no evidence arrived for a product at all, say so in that
product's column (e.g. "no data") rather than omitting the column or
comparing as if all products had data.

**Clarification (evidence is completely EMPTY):** this usually means it's
unclear which products (and optionally which attribute) the user wants to
compare (e.g. "which one is better" -- which products?). Ask which product
codes (and optionally which attribute -- weight, price, power, temperature,
etc.) they want to compare.

**Follow-up question:** After giving the comparison, if it fits the flow of
the conversation, you may briefly ask the user whether they'd like to
compare another product/attribute too -- but do this as a natural closing
sentence, NOT as a separate "suggestion" list; the actual attribute
suggestions already come from the follow-up note below (if any).

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `comparison` -- runs `sql_topn`'s single-sided retrieval core,
`SqlTopNFlow.run_pass`, per product (and per product x attribute pair where
relevant); see `chatbot/flows/comparison.py`. There is now a real,
deterministic `splitter` (`DeterministicComparisonSplitter`) -- the query
can be split into N products.

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely sources (non-binding): structured product data + chunks, gathered
  separately per product + a code-level-aligned comparison table.
- The splitter (`DeterministicComparisonSplitter`) finds product codes shaped
  like "DE" + digits in the query (regex, no LLM -- see the module docstring
  in flows/comparison.py); if fewer than 2 codes are found, the flow returns
  empty evidence without ever going to retrieval (the "Clarification" section
  above kicks in) and the session is pinned to this flow for the next turn
  (the user's answer goes directly to this flow, bypassing reconcile/classify).
  Once resolved (2+ products found), the pin is cleared automatically.
- If `retrieval` Phase 4 (`query_rewriting`) arrives, `DeterministicComparisonSplitter`
  can be replaced with a real language-model/rule-based splitter -- no need
  to touch this file, the protocol (`ComparisonSplitter`) stays the same.
- The long-term goal may be a real multi-step (agentic, Phase 6) comparison;
  this strategy is the interim solution until then.
