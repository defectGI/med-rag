"""Stage 1 — schema linking.

Reduces the full schema to the part relevant to the request. Uses an
intermediate LLM with forced structured (JSON) output.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings
from ..logging_utils import get_logger
from ..prompts import PromptTemplates
from ..providers.base import LLMProvider
from ..schema import Schema, schema_to_prompt
from ..types import (
    LinkedColumn,
    LinkedJoin,
    LinkedSchema,
    LLMResponse,
    StageMetadata,
    _Timer,
)
from .json_utils import extract_json, with_retry

logger = get_logger("pipeline.linking")

#: Name for the structured-output tool / response_format schema.
LINKING_SCHEMA_NAME = "linked_schema"

#: JSON schema forced on the Stage 1 output.
LINKING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tables": {"type": "array", "items": {"type": "string"}},
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "table": {"type": "string"},
                    "column": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["table", "column", "reason"],
            },
        },
        "joins": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "left_table": {"type": "string"},
                    "left_column": {"type": "string"},
                    "right_table": {"type": "string"},
                    "right_column": {"type": "string"},
                },
                "required": [
                    "left_table",
                    "left_column",
                    "right_table",
                    "right_column",
                ],
            },
        },
        "entities": {"type": "array", "items": {"type": "string"}},
        "filters": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["tables", "columns", "joins", "entities", "filters", "rationale"],
}


class LinkingStage:
    """Runs schema linking (Stage 1)."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings,
        prompts: PromptTemplates,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._prompts = prompts

    def run(self, request: str, schema: Schema) -> tuple[LinkedSchema, StageMetadata]:
        """Link the request against ``schema``.

        Returns the reduced schema and the stage metadata.
        """
        stage_cfg = self._settings.linking
        template = self._prompts.load(stage_cfg.prompt_template)
        system, user = template.render(
            schema=schema_to_prompt(schema),
            request=request,
            dialect=self._settings.general.sql_dialect,
            max_tables=stage_cfg.max_tables,
        )

        attempts = {"n": 0}

        def attempt() -> tuple[LinkedSchema, LLMResponse]:
            attempts["n"] += 1
            response = self._provider.complete_json(
                system=system,
                user=user,
                json_schema=LINKING_JSON_SCHEMA,
                schema_name=LINKING_SCHEMA_NAME,
                temperature=stage_cfg.temperature,
                top_p=stage_cfg.top_p,
                max_tokens=stage_cfg.max_tokens,
            )
            data = extract_json(response.text, strict=self._settings.general.strict_json)
            linked = self._to_linked_schema(data, schema, stage_cfg.max_tables)
            return linked, response

        with _Timer() as timer:
            linked, response = with_retry(
                attempt,
                retry=self._settings.retry,
                description="Schema linking",
            )

        metadata = StageMetadata(
            provider=self._provider.name,
            model=self._provider.model,
            usage=response.usage,
            latency_seconds=timer.elapsed,
            attempts=attempts["n"],
        )
        logger.info(
            "Linking picked %d table(s): %s",
            len(linked.tables),
            ", ".join(linked.tables) or "(none)",
        )
        return linked, metadata

    @staticmethod
    def _to_linked_schema(
        data: dict[str, Any],
        schema: Schema,
        max_tables: int,
    ) -> LinkedSchema:
        """Check the parsed JSON against the real schema and build a LinkedSchema.

        Tables/columns not in the source schema are dropped. This guards against
        hallucinated names. Tables are capped at ``max_tables``.
        """
        valid_tables: list[str] = []
        for name in data.get("tables", []):
            tbl = schema.table(str(name))
            if tbl is not None and tbl.name not in valid_tables:
                valid_tables.append(tbl.name)
            elif tbl is None:
                logger.debug("Dropping hallucinated table '%s'.", name)
        valid_tables = valid_tables[:max_tables]
        valid_set = {t.lower() for t in valid_tables}

        columns: list[LinkedColumn] = []
        for col in data.get("columns", []):
            if not isinstance(col, dict):
                continue
            table_name = str(col.get("table", ""))
            column_name = str(col.get("column", ""))
            tbl = schema.table(table_name)
            if tbl is None or tbl.name.lower() not in valid_set:
                continue
            if tbl.column(column_name) is None:
                logger.debug("Dropping hallucinated column '%s.%s'.", table_name, column_name)
                continue
            columns.append(
                LinkedColumn(
                    table=tbl.name,
                    column=column_name,
                    reason=str(col.get("reason", "")),
                )
            )

        joins: list[LinkedJoin] = []
        for j in data.get("joins", []):
            if not isinstance(j, dict):
                continue
            joins.append(
                LinkedJoin(
                    left_table=str(j.get("left_table", "")),
                    left_column=str(j.get("left_column", "")),
                    right_table=str(j.get("right_table", "")),
                    right_column=str(j.get("right_column", "")),
                )
            )

        entities = [str(e) for e in data.get("entities", []) if str(e).strip()]
        filters = [str(f) for f in data.get("filters", []) if str(f).strip()]

        return LinkedSchema(
            tables=valid_tables,
            columns=columns,
            joins=joins,
            entities=entities,
            filters=filters,
            rationale=str(data.get("rationale", "")),
        )
