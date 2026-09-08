"""Çıktı yerleşimi: yerel artefaktlar nereye yazılır (kök PROTOCOL.md
KARAR-012 ilkesi — her artefaktın belgeli bir evi, ev üreten bileşenin
altında). `vectorize`in "gerçek" çıktısı Qdrant'taki koleksiyondur; bu klasör
yalnız yerel iz sürme artefaktlarını tutar (bkz. `storage/README.md`).
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
