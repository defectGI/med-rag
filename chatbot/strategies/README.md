# Strategies (per-intent prompt draft)

Once a query's intent has been determined
(`retrieval.modules.intent_classification` returns **only a label**),
this component decides which guidance prompt is given to the main
conversational model for that label. Each file in this folder maps to
one intent.

## Why here, not `retrieval/strategies/`

`retrieval/strategies/*.md` carries draft files for the same 9 intents,
but is not imported/read from there for two reasons:

1. That directory is not included in the PyPI wheel (absent from
   `retrieval/MANIFEST.in`) — these files are not part of `mkd-retriever`
   installed via pip.
2. The label → strategy mapping and the strategy contents are already
   **the consuming project's responsibility** (retrieval ARCHITECTURE.md
   #6/#7) — retrieval deliberately keeps this out of its own scope.

So this is not a copy of the retrieval drafts: this component has its own
strategy files — the retrieval versions were used only as starting points.

## Model-facing vs. developer-notes split

Not ALL of these files reach the model:
`chatbot/orchestrator.py::load_strategy` reads each file up to the
`<!-- CHATBOT:DEV-NOTES (modele gitmez) -->` marker — everything AFTER
the marker (decision references, `chatbot/flows/X.py` pointers,
speculative '## Notes' items) is for developers only and never reaches
the model. A file that lacks the marker (backward compatibility) goes
to the model ENTIRELY — forgetting the marker in a new strategy file
won't break the file, it just falls back to the old (undifferentiated)
behavior. Tone/style instructions (natural conversational tone, no
technical jargon) live in ONE place:
`chatbot/chatbot/answering_model.py::_BASE_PERSONA` — individual
strategy files do NOT duplicate them.

## Principles

- **Guidance/suggestion only — no automatic tool-calling.** Each strategy
  *suggests* to the model (which source to prefer, how to answer, what
  to ask back). No flow is set up that triggers a direct action via
  tool-calling.
- **`chatbot/router.py` does NOT import/read these files** — only
  `chatbot/orchestrator.py::load_strategy` reads them, using the
  classified `IntentLabel` itself directly. The Router only maps
  intent -> flow; flows know nothing about strategies.
- **One intent, one file.** Each strategy runs independently; changing
  one should not require touching another.

## Files (6 medical intents, med-rag)

| Intent | File | Current flow |
|--------|-------|--------------|
| medical_fact | [medical_fact.md](./medical_fact.md) | `default_topn` |
| clinical_decision | [clinical_decision.md](./clinical_decision.md) | `default_topn` |
| comparison | [comparison.md](./comparison.md) | `default_topn` |
| interaction | [interaction.md](./interaction.md) | `default_topn` |
| library | [library.md](./library.md) | `default_topn` |
| out_of_scope | [out_of_scope.md](./out_of_scope.md) | `no_retrieval` |

The mapping is defined in `config/default.toml` `[routing]`; this table
is only a summary of the current state, not the source of truth.

The industrial v1 strategies (product_fact, aggregation, doc_question,
quote_or_contact, recommendation, visual_request, doc_download) were
archived to [archive/](./archive/) — kept, not deleted (PLAN.md C5).

