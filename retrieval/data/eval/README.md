# Eval data (`data/eval/`)

This directory holds the **golden sets** the eval harness scores against
(`retrieval.eval`). The repo defines the format and ships `*.example.jsonl`
placeholders; **the real data is supplied by the user**.

## Files

| File | Purpose | Tracked? |
|------|---------|----------|
| `intent_golden.example.jsonl` | 50-item placeholder for intent eval | yes (example) |
| `retrieval_golden.example.jsonl` | small placeholder for retrieval eval | yes (example) |
| `intent_golden.jsonl` | **your real** intent golden set | no (git-ignored) |
| `retrieval_golden.jsonl` | **your real** retrieval golden set | no (git-ignored) |

The examples are meant to be replaced. Either edit them in place, or (preferred)
copy to the non-`.example` names and keep your real data out of git.

## Format (JSONL — one JSON object per line)

Blank lines and lines starting with `#` are ignored, so you can add section
headers/comments.

### Intent golden

```json
{"query": "de1000 fiyati ne kadar", "expected_intent": "product_fact", "lang": "tr"}
```

- `query` (required) — the user query, written the way real users type
  (informal, lowercase, missing diacritics, sloppy). Don't normalise it.
- `expected_intent` (required) — one of the 9 v1 labels: `product_fact`,
  `aggregation`, `doc_question`, `comparison`, `recommendation`,
  `visual_request`, `doc_download`, `quote_or_contact`, `out_of_scope`.
- `lang` (optional) — `"tr"` / `"en"`; useful for per-language breakdowns.
- `id`, `note` (optional) — free-form.

### Retrieval golden

```json
{"query": "de1000 calisma sicakligi", "relevant_ids": ["ACME_PN5001_Datasheet::c7"], "lang": "tr"}
```

- `query` (required).
- `relevant_ids` (required) — ids that count as a correct hit. These are the
  same `id` a retriever puts on `RetrievalResult` (a `ChunkNode.node_id` like
  `"<doc_id>::c<N>"`, or a product `node_id`). List every relevant chunk.
- `lang`, `k`, `id` (optional). `k` overrides the recall/MRR cutoff for that one
  query.

## Loading in code

```python
from retrieval.eval import load_intent_golden, load_retrieval_golden

intent = load_intent_golden("data/eval/intent_golden.jsonl")
retr   = load_retrieval_golden("data/eval/retrieval_golden.jsonl")
```

Loaders validate every line and raise `path:lineno: ...` on bad JSON or an
unknown intent label, so a malformed golden set fails loudly.
