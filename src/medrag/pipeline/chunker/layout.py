"""Output layout: which artifact goes where.

One place, because the output path is a CONTRACT: the chunker writes, the
`pipeline` registry records, `run_chunk_test` validates, the consumer
reads. If the path logic spreads across these files, one of them silently
misses.

Under the root (`CHUNKER_OUTPUT_DIR`, default `./storage`) two JSON files
(earlier per-document `chunks/{doc_id}.chunks.json` were consolidated into
two big files for simplicity):

    storage/
      all_chunks.json     chunk sets for ALL documents (scope="document"),
                          doc_id -> ChunkSet dictionary
      all_combined.json   wrapped form of the same dictionary
      viz/                {doc_id}.tree.html + index.html (separate — these
                          are derivative visualizations, not chunk output)

RAPTOR (summary tree / corpus-profile clustering, formerly stored in
`all_raptor.json`) was removed entirely.

Deliberate trade-off: with the staleness gate gone (single file, no
per-document filesystem presence to query), every run reprocesses the
entire corpus. File simplicity was chosen over recompute cost.
"""

from __future__ import annotations

from pathlib import Path

#: Root used when `CHUNKER_OUTPUT_DIR` is not set (relative to the chunker/ package).
DEFAULT_OUTPUT_ROOT = "./storage"

ALL_CHUNKS_FILE = "all_chunks.json"
ALL_COMBINED_FILE = "all_combined.json"
VIZ_SUBDIR = "viz"


class OutputLayout:
    """An output root + the two fixed files + viz folder for one run.

    `ensure()` opens the viz folder (idempotent); the JSON files are not
    folders — they're written in one batch at the end of the run
    (`write_outputs`, cli.py).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def viz(self) -> Path:
        return self.root / VIZ_SUBDIR

    @property
    def all_chunks_file(self) -> Path:
        return self.root / ALL_CHUNKS_FILE

    @property
    def all_combined_file(self) -> Path:
        return self.root / ALL_COMBINED_FILE

    def ensure(self) -> OutputLayout:
        self.root.mkdir(parents=True, exist_ok=True)
        self.viz.mkdir(parents=True, exist_ok=True)
        return self

    def __repr__(self) -> str:  # show the root in logs
        return f"OutputLayout({self.root})"