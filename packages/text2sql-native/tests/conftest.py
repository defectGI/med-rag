"""Shared test fixtures: fake LLM providers and hand-built settings.

No network anywhere. LLM calls are replaced by deterministic fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from text2sql_native.config import (
    GeneralSettings,
    ProviderSelection,
    RetrySettings,
    Settings,
    StageSettings,
)
from text2sql_native.prompts import PromptTemplates
from text2sql_native.providers.base import LLMProvider
from text2sql_native.schema import FileSchemaProvider
from text2sql_native.types import LLMResponse, TokenUsage

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SCHEMA = REPO_ROOT / "examples" / "schema.yaml"
PROMPT_DIR = REPO_ROOT / "text2sql_native" / "prompts"


class FakeProvider(LLMProvider):
    """A provider that returns scripted JSON and records every call.

    Args:
        payloads: One entry is consumed per call. Each entry is a dict (dumped
            to JSON) or a raw string.
        fail_times: How many leading calls return junk before a good one. Used
            to test the retry path.
    """

    name = "fake"

    def __init__(
        self,
        payloads: list[Any] | None = None,
        *,
        fail_times: int = 0,
        model: str = "fake-model",
    ) -> None:
        super().__init__(model=model)
        self._payloads = payloads or []
        self._fail_times = fail_times
        self.calls: list[dict[str, Any]] = []

    def _next_payload(self) -> str:
        payload = self._payloads.pop(0) if self._payloads else {}
        return payload if isinstance(payload, str) else json.dumps(payload)

    def complete(
        self, *, system: str, user: str, temperature: float, top_p: float, max_tokens: int
    ) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        if self._fail_times > 0:
            self._fail_times -= 1
            return LLMResponse(text="not valid json", usage=TokenUsage(1, 1, 2), model=self.model)
        return LLMResponse(
            text=self._next_payload(), usage=TokenUsage(10, 5, 15), model=self.model
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.calls.append(
            {"system": system, "user": user, "schema_name": schema_name}
        )
        if self._fail_times > 0:
            self._fail_times -= 1
            return LLMResponse(text="oops not json", usage=TokenUsage(1, 1, 2), model=self.model)
        return LLMResponse(
            text=self._next_payload(), usage=TokenUsage(10, 5, 15), model=self.model
        )


def make_settings(**overrides: Any) -> Settings:
    """Build Settings that point at the example schema and prompts."""
    base = Settings(
        schema_path=EXAMPLE_SCHEMA,
        prompt_dir=PROMPT_DIR,
        linking_selection=ProviderSelection(provider="fake", model="fake-link"),
        sql_selection=ProviderSelection(provider="fake", model="fake-sql"),
        general=GeneralSettings(sql_dialect="postgres", log_level="WARNING", strict_json=False),
        linking=StageSettings(prompt_template="linking.txt", max_tables=8),
        generation=StageSettings(prompt_template="generation.txt"),
        retry=RetrySettings(max_attempts=3, backoff_seconds=0.0, backoff_factor=1.0),
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def schema_provider() -> FileSchemaProvider:
    return FileSchemaProvider(EXAMPLE_SCHEMA)


@pytest.fixture
def prompts() -> PromptTemplates:
    return PromptTemplates(PROMPT_DIR)


@pytest.fixture
def linked_payload() -> dict[str, Any]:
    """A valid Stage 1 output that uses real tables/columns."""
    return {
        "tables": ["products", "order_items", "orders"],
        "columns": [
            {"table": "products", "column": "name", "reason": "product identity"},
            {"table": "order_items", "column": "quantity", "reason": "units sold"},
            {"table": "orders", "column": "created_at", "reason": "time filter"},
        ],
        "joins": [
            {
                "left_table": "order_items",
                "left_column": "product_id",
                "right_table": "products",
                "right_column": "id",
            },
            {
                "left_table": "order_items",
                "left_column": "order_id",
                "right_table": "orders",
                "right_column": "id",
            },
        ],
        "entities": ["best-selling products"],
        "filters": ["orders.created_at within last month"],
        "rationale": "Join order_items to products and orders to rank by units sold.",
    }


@pytest.fixture
def generation_payload() -> dict[str, Any]:
    return {
        "sql": (
            "SELECT p.name, SUM(oi.quantity) AS units_sold "
            "FROM order_items oi "
            "JOIN products p ON oi.product_id = p.id "
            "JOIN orders o ON oi.order_id = o.id "
            "WHERE o.created_at >= date_trunc('month', CURRENT_DATE) - INTERVAL '1 month' "
            "AND o.created_at < date_trunc('month', CURRENT_DATE) "
            "GROUP BY p.name ORDER BY units_sold DESC LIMIT 5;"
        ),
        "explanation": "Sums quantities per product for last month and returns the top 5.",
    }


ProviderFactory = Callable[..., FakeProvider]
