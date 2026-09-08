"""Stage 2 — SQL generation.

Takes the reduced schema from Stage 1 and produces a SQL string, plus an
optional short explanation, in the configured dialect.

Two output modes (config: [generation] output_mode):

* "structured" (default) — force JSON with "sql" and "explanation" fields.
  Reliable parsing. Best for general chat/instruct models.
* "raw" — ask for plain SQL text. No explanation. Better for specialized
  text-to-SQL models that are trained to emit SQL only.
"""

from __future__ import annotations

import re
from typing import Any

from ..config import Settings
from ..errors import StructuredOutputError
from ..logging_utils import get_logger
from ..prompts import PromptTemplates
from ..providers.base import LLMProvider
from ..schema import Schema, schema_to_prompt
from ..types import LinkedSchema, LLMResponse, StageMetadata, _Timer
from .json_utils import extract_json, with_retry

# Matches a ```sql ... ``` or ``` ... ``` fenced block.
_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _strip_sql(text: str) -> str:
    """Return plain SQL from a raw response. Drops markdown fences if present."""
    stripped = text.strip()
    fence = _FENCE_RE.search(stripped)
    if fence:
        return fence.group(1).strip()
    return stripped

logger = get_logger("pipeline.generation")

GENERATION_SCHEMA_NAME = "sql_generation"

GENERATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sql": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": ["sql", "explanation"],
}


class GenerationStage:
    """Runs SQL generation (Stage 2)."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        prompts: PromptTemplates,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._prompts = prompts

    def run(
        self,
        request: str,
        linked: LinkedSchema,
        full_schema: Schema,
    ) -> tuple[str, str, StageMetadata]:
        """Generate SQL for ``request`` using the linked subset.

        Returns ``(sql, explanation, metadata)``.
        """
        stage_cfg = self._settings.generation

        # Render the reduced schema from the real metadata (types, keys, FKs),
        # limited to the linked tables. Richer than just table names.
        reduced_schema = full_schema.subset(linked.tables)
        reduced_text = schema_to_prompt(reduced_schema) if reduced_schema.tables else "(none)"

        # Forward Stage 1's reasoning to Stage 2. The linking model already
        # worked out the logic (ranges, joins, which column means what); the SQL
        # model should follow it instead of guessing.
        columns_text = "\n".join(
            f"- {c.table}.{c.column}" + (f" — {c.reason}" if c.reason else "")
            for c in linked.columns
        ) or "(none)"
        joins_text = "\n".join(
            f"- {j.left_table}.{j.left_column} = {j.right_table}.{j.right_column}"
            for j in linked.joins
        ) or "(none)"
        rationale_text = linked.rationale.strip() or "(none)"

        template = self._prompts.load(stage_cfg.prompt_template)
        system, user = template.render(
            linked_schema=reduced_text,
            request=request,
            dialect=self._settings.general.sql_dialect,
            entities=", ".join(linked.entities) or "(none)",
            filters="; ".join(linked.filters) or "(none)",
            columns=columns_text,
            joins=joins_text,
            rationale=rationale_text,
        )

        attempts = {"n": 0}

        def attempt_structured() -> tuple[str, str, LLMResponse]:
            attempts["n"] += 1
            response = self._provider.complete_json(
                system=system,
                user=user,
                json_schema=GENERATION_JSON_SCHEMA,
                schema_name=GENERATION_SCHEMA_NAME,
                temperature=stage_cfg.temperature,
                top_p=stage_cfg.top_p,
                max_tokens=stage_cfg.max_tokens,
            )
            data = extract_json(response.text, strict=self._settings.general.strict_json)
            sql = str(data.get("sql", "")).strip()
            explanation = str(data.get("explanation", "")).strip()
            if not sql:
                raise StructuredOutputError("Generation returned an empty 'sql' field.")
            return sql, explanation, response

        def attempt_raw() -> tuple[str, str, LLMResponse]:
            attempts["n"] += 1
            response = self._provider.complete(
                system=system,
                user=user,
                temperature=stage_cfg.temperature,
                top_p=stage_cfg.top_p,
                max_tokens=stage_cfg.max_tokens,
            )
            sql = _strip_sql(response.text)
            if not sql:
                raise StructuredOutputError("Generation returned empty SQL.")
            # Raw mode does not produce an explanation.
            return sql, "", response

        attempt = attempt_raw if stage_cfg.output_mode == "raw" else attempt_structured

        with _Timer() as timer:
            sql, explanation, response = with_retry(
                attempt,
                retry=self._settings.retry,
                description="SQL generation",
            )

        metadata = StageMetadata(
            provider=self._provider.name,
            model=self._provider.model,
            usage=response.usage,
            latency_seconds=timer.elapsed,
            attempts=attempts["n"],
        )
        logger.info("Generated SQL (%d chars).", len(sql))
        return sql, explanation, metadata
