"""Public API: the :class:`Text2SQL` orchestrator.

Combines Stage 1 (linking) and Stage 2 (generation) in one ``run`` call.
Everything a caller may want to swap — settings, schema provider, prompt
templates, and the per-stage LLM providers — is injectable.
"""

from __future__ import annotations

from ..config import Settings, load_settings
from ..errors import EmptyLinkingError, SchemaError
from ..logging_utils import configure_logging, get_logger
from ..prompts import PromptTemplates
from ..providers.base import LLMProvider
from ..providers.registry import build_provider
from ..schema import FileSchemaProvider, Schema
from ..schema.provider import SchemaProvider
from ..types import RunMetadata, Text2SQLResult, _Timer
from .generation import GenerationStage
from .linking import LinkingStage

logger = get_logger("pipeline")


class Text2SQL:
    """Two-stage, schema-agnostic Text-to-SQL engine.

    Use :meth:`from_config` for the config-driven path, or the constructor
    directly to inject custom parts.

    Args:
        settings: Full settings.
        schema_provider: Gives the schema metadata.
        prompts: Prompt template loader.
        linking_provider: LLM provider for Stage 1.
        generation_provider: LLM provider for Stage 2.
    """

    def __init__(
        self,
        settings: Settings,
        schema_provider: SchemaProvider,
        prompts: PromptTemplates,
        linking_provider: LLMProvider,
        generation_provider: LLMProvider,
    ) -> None:
        self._settings = settings
        self._schema_provider = schema_provider
        self._prompts = prompts
        self._linking = LinkingStage(linking_provider, settings, prompts)
        self._generation = GenerationStage(generation_provider, settings, prompts)

        # Load the schema once, at construction, so problems show up early.
        self._schema: Schema = schema_provider.load()
        if not self._schema.tables:
            raise SchemaError("Loaded schema has no tables.")

    # Construction
    @classmethod
    def from_config(
        cls,
        *,
        env_file: str | None = None,
        config_path: str | None = None,
        schema_provider: SchemaProvider | None = None,
        prompts: PromptTemplates | None = None,
        linking_provider: LLMProvider | None = None,
        generation_provider: LLMProvider | None = None,
    ) -> "Text2SQL":
        """Build an engine from ``.env`` + ``config.toml``.

        Any part can be overridden with a keyword argument to inject a custom
        one. Anything left as ``None`` is built from config.

        Raises:
            ConfigError: If config is missing or invalid.
            SchemaError: If the schema cannot be loaded.
        """
        settings = load_settings(env_file=env_file, config_path=config_path)
        configure_logging(settings.general.log_level)

        schema_provider = schema_provider or FileSchemaProvider(settings.schema_path)
        # settings.prompt_dir may be None -> PromptTemplates uses packaged files.
        prompts = prompts or PromptTemplates(settings.prompt_dir)

        linking_provider = linking_provider or build_provider(
            settings.linking_selection, timeout=settings.linking.timeout_seconds,
            num_ctx=settings.linking.num_ctx,
        )
        generation_provider = generation_provider or build_provider(
            settings.sql_selection, timeout=settings.generation.timeout_seconds,
            num_ctx=settings.generation.num_ctx,
        )

        logger.debug(
            "Text2SQL ready: linking=%s/%s generation=%s/%s dialect=%s",
            linking_provider.name,
            linking_provider.model,
            generation_provider.name,
            generation_provider.model,
            settings.general.sql_dialect,
        )
        return cls(
            settings=settings,
            schema_provider=schema_provider,
            prompts=prompts,
            linking_provider=linking_provider,
            generation_provider=generation_provider,
        )

    # Execution
    def run(self, request: str) -> Text2SQLResult:
        """Run the full pipeline for a natural-language ``request``.

        Returns:
            A :class:`~text2sql.types.Text2SQLResult` with the SQL string, the
            Stage 1 linked schema, and run metadata.

        Raises:
            EmptyLinkingError: If Stage 1 links no tables/columns.
            StructuredOutputError: If a stage's output cannot be parsed after
                all retries.
            ProviderError: If an LLM call fails.
        """
        if not request or not request.strip():
            raise ValueError("request must be a non-empty string.")

        logger.info("Running Text2SQL for request: %r", request)
        with _Timer() as total_timer:
            linked, linking_meta = self._linking.run(request, self._schema)

            if linked.is_empty():
                raise EmptyLinkingError(
                    "Schema linking found no relevant table or column for the "
                    "request. The request may not match the loaded schema."
                )

            sql, explanation, generation_meta = self._generation.run(
                request, linked, self._schema
            )

        metadata = RunMetadata(
            linking=linking_meta,
            generation=generation_meta,
            total_latency_seconds=total_timer.elapsed,
        )
        return Text2SQLResult(
            sql=sql,
            explanation=explanation,
            linked_schema=linked,
            metadata=metadata,
            request=request,
        )

    @property
    def schema(self) -> Schema:
        """The loaded schema. Read-only access for callers."""
        return self._schema
