"""Output layout: where local artifacts are written (root PROTOCOL.md
KARAR-012 principle -- every artifact has a documented home, under the
component that produced it). `vectorize`'s "real" output is the Qdrant
collection; this folder only holds the local trace-tracking artifacts
(see `storage/README.md`).
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_OUTPUT_ROOT = "./storage"

STATE_FILE = "state.json"


class OutputLayout:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_file(self) -> Path:
        return self.root / STATE_FILE
