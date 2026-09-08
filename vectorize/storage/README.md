# storage/

Local artifact home for vectorize. A data folder, not code — its content is
generated at run time; only this README is tracked in git (same pattern as
`chunker/storage/`).

**vectorize's "real" output is NOT this folder but the collection in Qdrant**
(`config/default.toml` `[qdrant].collection_name`). This folder keeps only
local bookkeeping files:

| File | Contents | When it changes |
|---|---|---|
| `state.json` | `scope_id -> signature` (see `vectorize/core.py` `ChunkSet.signature`) — the staleness gate. | Updated after every successfully processed scope. If deleted, the worst case is the next run re-embedding everything unnecessarily — no wrong/missing data risk. |

Can be moved elsewhere via `VECTORIZE_OUTPUT_DIR`; the layout itself is
defined in one place: `src/medrag/pipeline/vectorize/layout.py`.
