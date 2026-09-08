# enrichment

Post-chunk enrichment steps. Core chunking completes without this module;
each step here is optional and best-effort.

Currently implemented:

- **Cross-reference resolution** — DOCUMENT-ONLY: links in the body (with
  the target info the adapter carries) are mapped to chunks; if the target
  is another chunk in the same document, the target chunk id is written
  into `cross_refs`; out-of-document links stay raw. This step requires no
  LLM but must run AFTER the full chunk set is produced (ids are only
  known then).

Tests stub the LLM client and embedding provider; no real model calls.