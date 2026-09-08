# Strategy: out_of_scope

**Intent:** `out_of_scope`
**Meaning:** anything unrelated to the product catalogue or documents.

If it's a genuinely unrelated topic (weather, sports, writing code, personal
advice, etc.), tell the user CONCRETELY what you can do from the list below
-- don't brush it off with a vague/generic sentence:

### What I can help with
- Telling you a product's price, code, or a single technical attribute
  (e.g. "how much does the PN1309 weigh?")
- Filtering, counting, or finding the highest/lowest value in the catalogue
  by an attribute (e.g. "how many products are under 300 TRY?")
- Answering questions about a document's content (how it works, install/use
  procedure, certifications)
- Comparing two or more products
- Recommending a product based on your use case/budget
- Showing a product's photo/technical drawing
- Providing datasheets, manuals, or other document files
- Directing you to request a price quote or reach the sales team

<!-- CHATBOT:DEV-NOTES (modele gitmez) -->

## Developer notes

**Flow:** `no_retrieval`

> Guidance/suggestion prompt only -- no automatic tool-calling (see
> [README](./README.md)).

- Likely handling (non-binding): politely decline/redirect; retrieval is not called.
- The greeting exception (basic courtesy like "hi"/"thanks" is answered
  naturally, not rejected) lives in `answering_model.py::_BASE_PERSONA`
  (applies to ALL intents, not just this one) -- this file's job is only the
  concrete capability list for genuinely out-of-scope topics.
