# chunker (package)

The main Python package. Module map and dependency direction:

```
adapters/  ──►  core/  ◄──  tokenization/
                  │
                  ▼
             enrichment/
```

- `adapters/` — parser-specific input transformers. The ONLY place that
  knows parser types/schemas.
- `core/` — parser-agnostic inner document model + chunking engine (atomic
  rules, token budget, flex allowance, hierarchical heading injection).
- `tokenization/` — tokenizer abstraction; `core/` counts tokens only
  through this interface.
- `enrichment/` — post-chunk enrichment (cross-reference resolution).

Rule: `core/`, `tokenization/`, and `enrichment/` must not import any
adapter or parser schema; dependencies always flow inward (toward `core`).