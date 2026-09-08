"""Yapılandırma yükleme: `config/default.toml` → `VectorizeConfig`.

Desen `chunker/chunker/config.py`den aynalanır (kök CONFIG.md "Yükleyici
deseni"): pydantic alanlarında Python default'u yok, `extra="forbid"` (kendi
tuning dosyamız için — girdi sözleşmesindeki `core.py`nin `extra="ignore"`si
farklı bir kural, dışarıdan gelen veri içindir), eksik/bilinmeyen anahtar
yüksek sesle patlar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from medrag.core.config.loader import deep_merge as _deep_merge
from medrag.core.config.loader import read_toml as _read_toml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "default.toml"


class Embedding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    include_summary_nodes: bool
    prefix_heading_path: bool
    heading_separator: str


class Qdrant(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection_name: str
    distance: Literal["cosine", "dot", "euclid"]
    upsert_batch_size: int = Field(ge=1)
    on_dim_mismatch: Literal["error", "recreate"]


class Staleness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skip_unchanged: bool


class VectorizeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: Embedding
    qdrant: Qdrant
    staleness: Staleness

    def embed_text(self, *, heading_path: list[str], text: str) -> str:
        """Embed edilecek nihai metin: `prefix_heading_path` açıkken
        `core.ChunkSet.signature`den bağımsız, salt render mantığı."""
        if not self.embedding.prefix_heading_path or not heading_path:
            return text
        return f"{self.embedding.heading_separator.join(heading_path)}\n\n{text}"


def load_config(override: str | Path | None = None) -> VectorizeConfig:
    data = _read_toml(DEFAULT_CONFIG_PATH)
    if override is not None:
        data = _deep_merge(data, _read_toml(Path(override)))
    return VectorizeConfig.model_validate(data)
