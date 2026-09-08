"""File-backed SchemaProvider for JSON and YAML.

The file format (documented in ``examples/schema.yaml``) is::

    name: my_database            # optional
    tables:
      - name: customers
        description: "..."
        columns:
          - name: id
            type: integer
            description: "..."
            primary_key: true
            sample_values: [1, 2, 3]
        foreign_keys:
          - column: category_id
            references_table: categories
            references_column: id

JSON uses the same structure. Format is picked from the file extension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import SchemaError
from .models import Column, ForeignKey, Schema, Table
from .provider import SchemaProvider


class FileSchemaProvider(SchemaProvider):
    """Load schema metadata from a JSON or YAML file.

    Format comes from the extension (``.json``, ``.yaml``, ``.yml``).

    Args:
        path: Path to the metadata file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()

    def load(self) -> Schema:
        if not self._path.is_file():
            raise SchemaError(f"Schema file not found: '{self._path}'.")

        raw_text = self._path.read_text(encoding="utf-8")
        suffix = self._path.suffix.lower()

        try:
            if suffix in (".yaml", ".yml"):
                data = self._parse_yaml(raw_text)
            elif suffix == ".json":
                data = json.loads(raw_text)
            else:
                raise SchemaError(
                    f"Unsupported schema file extension '{suffix}'. "
                    f"Use .json, .yaml, or .yml."
                )
        except SchemaError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize parser errors
            raise SchemaError(f"Failed to parse schema file '{self._path}': {exc}") from exc

        return self._build_schema(data)

    @staticmethod
    def _parse_yaml(text: str) -> Any:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise SchemaError(
                "PyYAML is required to load YAML schema files. "
                "Install it with 'pip install PyYAML'."
            ) from exc
        return yaml.safe_load(text)

    @staticmethod
    def _build_schema(data: Any) -> Schema:
        if not isinstance(data, dict):
            raise SchemaError("Schema root must be a mapping/object.")

        raw_tables = data.get("tables")
        if not isinstance(raw_tables, list) or not raw_tables:
            raise SchemaError("Schema must have a non-empty 'tables' list.")

        tables: list[Table] = []
        for entry in raw_tables:
            if not isinstance(entry, dict):
                raise SchemaError("Each table entry must be a mapping/object.")
            name = entry.get("name")
            if not name:
                raise SchemaError("Every table needs a 'name'.")

            columns: list[Column] = []
            for col in entry.get("columns", []) or []:
                if not isinstance(col, dict) or not col.get("name"):
                    raise SchemaError(
                        f"Invalid column in table '{name}': every column needs a 'name'."
                    )
                columns.append(
                    Column(
                        name=str(col["name"]),
                        type=str(col.get("type", "")),
                        description=str(col.get("description", "")),
                        primary_key=bool(col.get("primary_key", False)),
                        sample_values=list(col.get("sample_values", []) or []),
                    )
                )

            foreign_keys: list[ForeignKey] = []
            for fk in entry.get("foreign_keys", []) or []:
                if not isinstance(fk, dict):
                    raise SchemaError(f"Invalid foreign_key in table '{name}'.")
                try:
                    foreign_keys.append(
                        ForeignKey(
                            column=str(fk["column"]),
                            references_table=str(fk["references_table"]),
                            references_column=str(fk["references_column"]),
                        )
                    )
                except KeyError as exc:
                    raise SchemaError(
                        f"Foreign key in table '{name}' is missing field {exc}."
                    ) from exc

            tables.append(
                Table(
                    name=str(name),
                    description=str(entry.get("description", "")),
                    columns=columns,
                    foreign_keys=foreign_keys,
                )
            )

        return Schema(name=str(data.get("name", "")), tables=tables)
