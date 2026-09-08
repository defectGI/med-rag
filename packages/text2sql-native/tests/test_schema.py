"""Schema loading and model tests."""

from __future__ import annotations

import json

import pytest

from text2sql_native.errors import SchemaError
from text2sql_native.schema import FileSchemaProvider, schema_to_prompt

from .conftest import EXAMPLE_SCHEMA, REPO_ROOT

EXAMPLE_SCHEMA_JSON = REPO_ROOT / "examples" / "schema.json"


def test_load_yaml_example():
    schema = FileSchemaProvider(EXAMPLE_SCHEMA).load()
    assert schema.name == "ecommerce"
    assert set(schema.table_names) == {
        "customers",
        "categories",
        "products",
        "orders",
        "order_items",
    }
    orders = schema.table("orders")
    assert orders is not None
    assert orders.primary_keys == ["id"]
    assert any(fk.references_table == "customers" for fk in orders.foreign_keys)


def test_yaml_and_json_are_equivalent():
    yaml_schema = FileSchemaProvider(EXAMPLE_SCHEMA).load()
    json_schema = FileSchemaProvider(EXAMPLE_SCHEMA_JSON).load()
    assert yaml_schema.table_names == json_schema.table_names
    for name in yaml_schema.table_names:
        y = yaml_schema.table(name)
        j = json_schema.table(name)
        assert [c.name for c in y.columns] == [c.name for c in j.columns]


def test_subset_and_serialize():
    schema = FileSchemaProvider(EXAMPLE_SCHEMA).load()
    subset = schema.subset(["products", "unknown_table"])
    assert subset.table_names == ["products"]
    text = schema_to_prompt(subset)
    assert "Table: products" in text
    assert "Primary key: id" in text
    assert "customers" not in text


def test_missing_file_raises():
    with pytest.raises(SchemaError):
        FileSchemaProvider("does/not/exist.yaml").load()


def test_empty_tables_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"tables": []}), encoding="utf-8")
    with pytest.raises(SchemaError):
        FileSchemaProvider(bad).load()


def test_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "schema.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(SchemaError):
        FileSchemaProvider(bad).load()
