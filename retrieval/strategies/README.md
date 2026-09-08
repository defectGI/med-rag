# Strategies (per-intent prompt drafts)

Once a query's intent is detected (by `intent_classification`, which returns
**only a label**), the consuming orchestration layer decides what to do. This
directory holds one Markdown file per intent — a **draft of the guidance prompt
the model receives when that intent is detected**.

These are intentionally mostly empty for now; they get filled in as each
intent's strategy is worked on separately.

## Design principles

- **Advisory only — no auto tool-calling.** In every strategy we give the model
  *suggestions / guidance* (which sources to prefer, how to answer, what to ask
  back). We do **not** wire a tool-calling flow that executes actions directly.
  The prompt shapes the model's answer; it does not hand it live tools.
- **Not part of the classifier.** `intent_classification` does **not** import,
  read, or reference these files (ARCHITECTURE.md #7). The label → strategy
  mapping and the use of these prompts belong to the orchestration layer /
  consuming project, kept separate from the label-only module (ARCHITECTURE.md
  #6).
- **One intent, one file.** Each intent's strategy is developed independently;
  changing one must not require touching another.

## Files (9 intents, v1)

| Intent | File |
|--------|------|
| product_fact | [product_fact.md](./product_fact.md) |
| aggregation | [aggregation.md](./aggregation.md) |
| doc_question | [doc_question.md](./doc_question.md) |
| comparison | [comparison.md](./comparison.md) |
| recommendation | [recommendation.md](./recommendation.md) |
| visual_request | [visual_request.md](./visual_request.md) |
| doc_download | [doc_download.md](./doc_download.md) |
| quote_or_contact | [quote_or_contact.md](./quote_or_contact.md) |
| out_of_scope | [out_of_scope.md](./out_of_scope.md) |
