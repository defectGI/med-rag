# core

Parser-agnostic chunking engine. This module knows no parser; it works
only over its own inner document model and counts tokens through the
`tokenization/` interface.

Responsibilities:

- **Inner document model** (`document.py` — present) — the structure
  adapters target (heading, paragraph, table, list, code, image;
  heading path, page range, source block ids). pydantic v2; design
  decisions in the file's docstring.
- **Chunk output schema** (`chunk.py` — present) — the full field list
  lives in the root README "Chunk schema" section. Core produces leaf
  nodes (`tree_level=0`): content + metadata (source block ids, page
  range, heading path, structured `images` list, flex allowance
  diagnostic, prev/next neighbors, provenance summary). Enrichment
  fields (`summary`, `keywords`, `parent_id`,
  `cross_refs.target_chunk_id`) stay None/empty in core output.
- **Atomic rules** — tables and lists are not split. A table that hits
  the hard upper limit is split with k-row overlap without breaking
  structure, and the table description (the `description` the adapter
  carries — tables and images share the same field/meaning, see
  `document.py` module docstring "A table is a flat text matrix") is
  injected into every part.
- **Flex allowance (this chunker's distinguishing feature)** — the
  token limit can flex up by a formula-determined amount to avoid
  splitting a valuable structure. The formula's coefficients come from
  `config/`; nothing is hardcoded.
- **Hierarchical chunking** — when a section has to be split, the
  heading chain it belongs to is prepended to the chunk.

Nothing here needs an LLM (that's `enrichment/`'s job); core runs offline.