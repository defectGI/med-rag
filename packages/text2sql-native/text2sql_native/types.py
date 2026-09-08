"""Dataclasses returned by the pipeline.

Plain dataclasses, no external deps. Easy to inspect and serialize.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TokenUsage:
    """Token counts for one LLM call. Zero when the provider reports nothing."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    """One provider call, normalized.

    Same shape for every provider, so the pipeline never touches an SDK type.
    ``raw`` keeps the native response for advanced callers.
    """

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    raw: Any = None


@dataclass
class LinkedColumn:
    """A column picked by Stage 1."""

    table: str
    column: str
    reason: str = ""


@dataclass
class LinkedJoin:
    """A candidate join between two tables, found during linking."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str


@dataclass
class LinkedSchema:
    """Stage 1 output: the reduced schema.

    This is what Stage 2 receives. It is also returned on the final result so
    callers can see what the pipeline picked as relevant.
    """

    tables: list[str] = field(default_factory=list)
    columns: list[LinkedColumn] = field(default_factory=list)
    joins: list[LinkedJoin] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    rationale: str = ""

    def is_empty(self) -> bool:
        """True when linking found no usable table or column."""
        return not self.tables and not self.columns


@dataclass
class StageMetadata:
    """Metadata for one stage: which model, how many tokens, how long."""

    provider: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    attempts: int = 1


@dataclass
class RunMetadata:
    """Metadata for a full run (both stages)."""

    linking: StageMetadata
    generation: StageMetadata
    total_latency_seconds: float = 0.0

    @property
    def total_usage(self) -> TokenUsage:
        """Token usage of both stages combined."""
        return self.linking.usage + self.generation.usage


@dataclass
class Text2SQLResult:
    """Final result of :meth:`text2sql.Text2SQL.run`.

    Attributes:
        sql: The generated SQL string. This is the deliverable.
        explanation: Optional short explanation from Stage 2.
        linked_schema: Stage 1 output (the reduced schema).
        metadata: Models, tokens, and latency for the run.
        request: The original request.
    """

    sql: str
    linked_schema: LinkedSchema
    metadata: RunMetadata
    request: str
    explanation: str = ""


class _Timer:
    """Small monotonic timer for stage latency. Internal only."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
