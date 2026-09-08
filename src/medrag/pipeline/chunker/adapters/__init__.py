"""Parser-specific input transformers.

Each adapter converts a parser's output into the shared inner document
model in `chunker.core.document.Document`. Every parser-specific fact
(schema, field names, version migration, format quirks) stays in this
package — it does not leak into the engine.
"""

"""Parser-specific input transformers.

Each adapter converts a parser's output into `chunker.core.document.Document`.
Parser-specific knowledge (schema, field names, version migration, format
quirks) stays inside this package and does not leak into the engine.
"""