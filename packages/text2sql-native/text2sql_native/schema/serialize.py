"""Render a :class:`Schema` to text for LLM prompts.

Kept apart from the models so prompt formatting can change without touching the
data classes.
"""

from __future__ import annotations

from .models import Schema


def schema_to_prompt(schema: Schema, *, max_sample_values: int = 5) -> str:
    """Render a schema as text for humans and LLMs.

    Includes table descriptions, columns (type + description), primary keys,
    foreign keys, and a capped list of sample values.

    Args:
        schema: The schema to render.
        max_sample_values: Max sample values shown per column.
    """
    lines: list[str] = []
    if schema.name:
        lines.append(f"Database: {schema.name}")
        lines.append("")

    for table in schema.tables:
        lines.append(f"Table: {table.name}")
        if table.description:
            lines.append(f"  Description: {table.description}")
        pks = table.primary_keys
        if pks:
            lines.append(f"  Primary key: {', '.join(pks)}")

        lines.append("  Columns:")
        for col in table.columns:
            parts = [f"    - {col.name} ({col.type})"]
            if col.description:
                parts.append(f": {col.description}")
            lines.append("".join(parts))
            if col.sample_values:
                sample = col.sample_values[:max_sample_values]
                rendered = ", ".join(repr(v) for v in sample)
                lines.append(f"        samples: {rendered}")

        if table.foreign_keys:
            lines.append("  Foreign keys:")
            for fk in table.foreign_keys:
                lines.append(
                    f"    - {table.name}.{fk.column} -> "
                    f"{fk.references_table}.{fk.references_column}"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
