"""Schema metadata models.

Schema-agnostic containers. They describe any relational schema, loaded from a
file or from live introspection. Nothing here is tied to one database product.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Column:
    """A column in a table."""

    name: str
    type: str
    description: str = ""
    primary_key: bool = False
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class ForeignKey:
    """A foreign key: a column pointing to another table's column."""

    column: str
    references_table: str
    references_column: str


@dataclass
class Table:
    """A table with its columns, primary keys, and foreign keys."""

    name: str
    description: str = ""
    columns: list[Column] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    @property
    def primary_keys(self) -> list[str]:
        """Names of the primary-key columns."""
        return [c.name for c in self.columns if c.primary_key]

    def column(self, name: str) -> Column | None:
        """Return the named column, or ``None``. Case-insensitive."""
        lowered = name.lower()
        for col in self.columns:
            if col.name.lower() == lowered:
                return col
        return None


@dataclass
class Schema:
    """A full schema: a list of tables."""

    tables: list[Table] = field(default_factory=list)
    name: str = ""

    def table(self, name: str) -> Table | None:
        """Return the named table, or ``None``. Case-insensitive."""
        lowered = name.lower()
        for tbl in self.tables:
            if tbl.name.lower() == lowered:
                return tbl
        return None

    @property
    def table_names(self) -> list[str]:
        return [t.name for t in self.tables]

    def subset(self, table_names: list[str]) -> "Schema":
        """Return a new Schema with only the named tables.

        Matches names case-insensitively. Unknown names are skipped.
        """
        wanted = {n.lower() for n in table_names}
        return Schema(
            name=self.name,
            tables=[t for t in self.tables if t.name.lower() in wanted],
        )
