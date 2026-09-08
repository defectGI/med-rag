# storage/

The chunker output directory. A data folder, not code — its contents are
generated at run time; only this README is tracked in git (same pattern as
`parser/storage/`).

The root can be moved elsewhere via `CHUNKER_OUTPUT_DIR`; the layout itself
(file names) is defined in exactly one place: `chunker/layout.py`.

RAPTOR (summary tree / corpus-profile clustering, formerly stored in
`all_raptor.json`) was removed entirely.

| File/folder | What it holds | When it changes |
|---|---|---|
| `all_chunks.json` | `documents` dictionary: `doc_id -> ChunkSet` — chunk sets for ALL documents (leaves only) in a single file. The registry's `chunk.chunks_path` points here (every SUCCESS record shares the same path). | Rewritten every run — there is no per-file staleness gate; the entire corpus is reprocessed each run even when input has not changed. |
| `all_combined.json` | Wrapped file carrying the same `documents` field as `all_chunks.json`. | Rewritten every run alongside `all_chunks.json`. |
| `viz/` | `{doc_id}.tree.html` + `index.html` — offline interactive tree viewer. | Overwritten every run (entirely derivative; does not affect chunk production; disable with `CHUNKER_VIZ=off`). |

Accepted trade-off: because there is no per-document file, the question
"this document did not change, skip it" can no longer be asked through the
file system — every run reprocesses the full corpus.

Note: this folder is not a database; each chunk set's identity/history lives
in its own `provenance` block (which IR it came from, which version, when).