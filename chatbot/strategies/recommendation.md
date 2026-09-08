# Strategy: recommendation

**Intent:** `recommendation`
**Meaning:** asking for a product suggestion for the user's own use case/constraints.

**Clarification (a clarifying question is being asked):** this means the
flow decided to ask a clarifying question about the use case/constraints --
put the flow's question (infer it from the context of the user's message) to
the user. Only ask the ONE question being asked RIGHT NOW; if there are other
questions not yet asked, they'll come in subsequent turns -- don't STACK them
all up at once.

The evidence you receive during clarification is usually NOT empty. As soon
as the user has given any usable constraint, the candidate set is already
being filtered, so the evidence can contain `[SQL]`-tagged rows (the
candidates still matching) as well as `[DOC]`-tagged supporting hits. Use it:
say roughly where things stand (how much is still open) and ask the
clarifying question as the way to narrow it further -- but don't present a
still-wide candidate set as a final recommendation, and don't let the summary
replace the question. If the evidence is empty, just ask the question
normally.

If the user already gave a piece of information (e.g. stated their use case
in a previous turn), don't ask for it AGAIN -- trust the "User's ongoing
goal" / "Constraints gathered" framing above if it's already visible there.

**Once clarification is done (the candidate set has narrowed):** Based on the
evidence you receive (`[SQL]`-tagged exact/authoritative lines + `[DOC]`-tagged
explanatory-only top_n enrichment, SAME rule as the product_fact strategy --
if they conflict, trust `[SQL]`, `[DOC]` is never used), recommend the
product(s) that BEST FIT the use case/constraints the user stated. If several
candidates fit equally well, present them together rather than arbitrarily
picking one. Don't make up a product/attribute not in the evidence.

**If nothing is left:** when the gathered constraints match no product at
all, say so plainly and name which constraint is the likely cause, then offer
to relax or change it -- don't ask a further narrowing question (there is
nothing left to narrow), and don't silently fall back to products that fail
the stated constraints.

**Comparison offer:** If the evidence contains multiple similar products
(several candidates fitting the same problem), after giving your
recommendation you may briefly ask the user "would you like me to compare
these?" -- this matches an offer the flow itself already prepares (at the
code level, `suggested_next_flow`): if the user says "yes", the next turn
automatically moves into the comparison flow, you should just ask a natural
closing question (don't construct a technical sentence like "I can switch to
the comparison flow").

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `recommendation` -- see `chatbot/flows/recommendation.py`.
Asks clarifying questions and pins the session to itself; each turn that
has any constraint in hand runs `sql_topn`'s existing core
(`SqlTopNFlow.run_pass`) with a SINGLE query.

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely sources (non-binding): structured product data + chunks (same
  `sql_topn` core).
- The clarification question pool is limited ONLY to fields that actually
  exist in the schema (`product.family` + `operating_temperature` /
  `number_of_channels` / `power_consumption` -- see the module docstring in
  flows/recommendation.py) -- a concept like "budget" that has no schema
  counterpart was DELIBERATELY left out of the question pool.
- REVISION: top_n (qdrant) now runs as a supporting call on every
  clarification turn (see `SqlTopNFlow.top_n_pass`). Reason: `family`
  matches against a fixed surface-form dictionary, so a phrase that does
  not match the dictionary stalled clarification before qdrant ever ran.
- REVISION (second pass): the response-count threshold (`min_answered`)
  was REMOVED. Whenever any constraint is in hand, `run_pass` (SQL) runs
  on that turn; clarification ends when the candidate set shrinks to
  `few_max` (or no candidates remain / the pool is exhausted); the next
  question is chosen via
  `discover_informative_slots(product_codes=...)` over the REMAINING
  candidates.
