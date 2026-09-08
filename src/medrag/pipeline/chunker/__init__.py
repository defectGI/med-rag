"""chunker — parser-independent chunking library.

Sub-packages: `core` (inner document model + splitting engine), `adapters`
(parser-specific transformers), `tokenization` (token-counting abstraction),
`enrichment` (cross-reference resolution). See the root README for
architecture.
"""

# Single behavior version of the chunker — chunk output changes that ship in
# any fix bump this: the CLI's staleness gate, whose provenance carries an
# old version, regenerates that chunk set; an un-bumped fix would have the
# corpus sitting on old output while the repo looks "fixed". Schema breaks
# go to CHUNK_SCHEMA_VERSION instead (core/chunk.py) — that one versions
# "how the file is read", this one versions "which code produced the
# content".
CHUNKER_VERSION = "1.0.1"