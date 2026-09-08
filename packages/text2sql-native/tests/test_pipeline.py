"""End-to-end pipeline tests. LLM calls are mocked. Runs on the example schema."""

from __future__ import annotations

import pytest

from text2sql_native import Text2SQL
from text2sql_native.errors import EmptyLinkingError, StructuredOutputError

from .conftest import FakeProvider


def _engine(settings, schema_provider, prompts, linking_provider, generation_provider):
    return Text2SQL(
        settings=settings,
        schema_provider=schema_provider,
        prompts=prompts,
        linking_provider=linking_provider,
        generation_provider=generation_provider,
    )


def test_full_run_produces_sql(
    settings, schema_provider, prompts, linked_payload, generation_payload
):
    linking = FakeProvider([linked_payload], model="fake-link")
    generation = FakeProvider([generation_payload], model="fake-sql")
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    result = engine.run("get me the top 5 best-selling products last month")

    # SQL string produced.
    assert result.sql.startswith("SELECT")
    assert "order_items" in result.sql
    assert result.explanation

    # Stage 1 intermediate surfaced.
    assert "products" in result.linked_schema.tables
    assert result.linked_schema.entities == ["best-selling products"]

    # Metadata populated for both stages.
    assert result.metadata.linking.model == "fake-link"
    assert result.metadata.generation.model == "fake-sql"
    assert result.metadata.total_usage.total_tokens == 30
    assert result.metadata.total_latency_seconds >= 0.0


def test_stage2_receives_only_linked_tables(
    settings, schema_provider, prompts, linked_payload, generation_payload
):
    linking = FakeProvider([linked_payload])
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)
    engine.run("top products")

    stage2_user_prompt = generation.calls[0]["user"]
    # Linked tables appear; unlinked "customers" table does not.
    assert "Table: products" in stage2_user_prompt
    assert "Table: customers" not in stage2_user_prompt


def test_stage2_prompt_carries_linking_reasoning(
    settings, schema_provider, prompts, linked_payload, generation_payload
):
    # Stage 1's rationale and joins must reach the Stage 2 prompt so the SQL
    # model follows the linking logic instead of guessing.
    linking = FakeProvider([linked_payload])
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)
    engine.run("top products")

    stage2_user_prompt = generation.calls[0]["user"]
    assert "rank by units sold" in stage2_user_prompt  # from the rationale
    assert "order_items.product_id = products.id" in stage2_user_prompt  # a join


def test_hallucinated_names_are_dropped(
    settings, schema_provider, prompts, generation_payload
):
    payload = {
        "tables": ["products", "not_a_real_table"],
        "columns": [
            {"table": "products", "column": "name", "reason": "ok"},
            {"table": "products", "column": "ghost_col", "reason": "hallucinated"},
        ],
        "joins": [],
        "entities": [],
        "filters": [],
        "rationale": "x",
    }
    linking = FakeProvider([payload])
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)
    result = engine.run("list products")

    assert result.linked_schema.tables == ["products"]
    cols = [(c.table, c.column) for c in result.linked_schema.columns]
    assert ("products", "name") in cols
    assert ("products", "ghost_col") not in cols


def test_retry_on_bad_json_then_success(
    settings, schema_provider, prompts, linked_payload, generation_payload
):
    # First linking call returns junk, second returns valid payload.
    linking = FakeProvider([linked_payload], fail_times=1)
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    result = engine.run("top products")
    assert result.sql.startswith("SELECT")
    assert result.metadata.linking.attempts == 2


def test_exhausted_retries_raise(
    settings, schema_provider, prompts, generation_payload
):
    # Always fails (more failures than attempts).
    linking = FakeProvider([], fail_times=99)
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    with pytest.raises(StructuredOutputError):
        engine.run("top products")


def test_empty_linking_raises(
    settings, schema_provider, prompts, generation_payload
):
    empty_payload = {
        "tables": [],
        "columns": [],
        "joins": [],
        "entities": [],
        "filters": [],
        "rationale": "nothing matched",
    }
    linking = FakeProvider([empty_payload])
    generation = FakeProvider([generation_payload])
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    with pytest.raises(EmptyLinkingError):
        engine.run("what's the weather today")


def test_empty_request_rejected(
    settings, schema_provider, prompts, linked_payload, generation_payload
):
    engine = _engine(
        settings,
        schema_provider,
        prompts,
        FakeProvider([linked_payload]),
        FakeProvider([generation_payload]),
    )
    with pytest.raises(ValueError):
        engine.run("   ")


def test_raw_mode_returns_plain_sql(settings, schema_provider, prompts, linked_payload):
    # Raw mode: Stage 2 returns plain SQL, no explanation.
    settings.generation.output_mode = "raw"
    settings.generation.prompt_template = "generation_raw.txt"
    linking = FakeProvider([linked_payload])
    generation = FakeProvider(["SELECT p.name FROM products p;"])
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    result = engine.run("list products")
    assert result.sql == "SELECT p.name FROM products p;"
    assert result.explanation == ""
    # complete() was used, not complete_json() -> no schema_name recorded.
    assert "schema_name" not in generation.calls[0]


def test_raw_mode_strips_markdown_fences(settings, schema_provider, prompts, linked_payload):
    settings.generation.output_mode = "raw"
    settings.generation.prompt_template = "generation_raw.txt"
    fenced = "```sql\nSELECT 1;\n```"
    linking = FakeProvider([linked_payload])
    generation = FakeProvider([fenced])
    engine = _engine(settings, schema_provider, prompts, linking, generation)

    result = engine.run("anything")
    assert result.sql == "SELECT 1;"
