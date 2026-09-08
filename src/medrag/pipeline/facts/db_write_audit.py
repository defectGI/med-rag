"""O-09 (I-15, K-17): AST-based inventory of the VALUE-writing flows into
`specs.db`.

NOT grep-based: every `.py` file is parsed via `ast.parse`, and the
FIRST ARGUMENT (literal string, or f-string's static prefix) of every
`execute` / `executemany` / `executescript` call is checked to see if
it starts with INSERT/UPDATE/DELETE/REPLACE. This avoids mis-tagging a
COMMENT inside a string that happens to contain "INSERT" (a trap
naive grep would fall into).

Known limitation: when the SQL text comes from a VARIABLE (e.g.
`cur.execute(q, ...)`), this function does NOT SEE that call
(dynamic SQL is outside static analysis's scope) -- there is NO SUCH
VAL-writing call in the current code base (they are all SELECT), see
the `test_no_dynamic_sql_write_calls_exist` test note.

**KARAR-016 context, expected result:** there are two modules that
write VALUES to `specs.db` -- `build_facts_db.py` (PHASE A, builds/
resets the skeleton, different column set) and `load_to_db.py` (PHASE
C, writes fact values). These TWO WRITERS are NOT a violation (the two
phases KARAR-016 anticipated).

**K-78 (O-11) exception:** `forget_source.py` was added as the THIRD
legitimate writer -- a deliberate exception to KARAR-016's "two
writers" decision for the DELETE-only case (DELETEs/UPDATEs that
REMOVE evidence when its source is gone). I-08 (chunk/vector side
deletion) and N-06 (cleanup of removed-file derivatives) ALSO use this
SAME function; no separate deletion code paths are opened. What this
module locks down is: under `facts/` there are NO WRITERS other than
THESE THREE FILES.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/

# KARAR-016's two anticipated legitimate writers + K-78/O-11's third
# ("forget the source" deletion flow) -- see module docstring.
ALLOWED_WRITER_FILENAMES = frozenset({"load_to_db.py", "build_facts_db.py", "forget_source.py"})

_WRITE_SQL_RE = re.compile(r"^\s*(insert|update|delete|replace)\b", re.IGNORECASE)
_ANY_WRITE_KEYWORD_RE = re.compile(r"\b(insert|update|delete|replace)\b", re.IGNORECASE)
_EXECUTE_METHOD_NAMES = ("execute", "executemany", "executescript")


def _leading_sql_text(node: ast.expr) -> str | None:
    """Extracts the SQL text from the first argument of a `.execute(...)`
    call -- a literal string, or the CONSTANT prefix of a JoinedStr
    (f-string). Returns None if the SQL is dynamic / from a variable."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr) and node.values:
        first = node.values[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first.value
    return None


def _sql_write_snippets_in_file(path: Path) -> list[str]:
    """Returns the first line of every value-writing (INSERT/UPDATE/
    DELETE/REPLACE) `.execute*` call in `path`. Empty list = this file
    does NOT WRITE to specs.db (or any other DB) within the limits of
    this analysis (see module docstring's known limitation)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _EXECUTE_METHOD_NAMES):
            continue
        if not node.args:
            continue
        text = _leading_sql_text(node.args[0])
        if text is None:
            continue
        is_write = _WRITE_SQL_RE.match(text) or (
            func.attr == "executescript" and _ANY_WRITE_KEYWORD_RE.search(text)
        )
        if is_write:
            first_line = text.strip().splitlines()[0] if text.strip() else text
            hits.append(first_line)
    return hits


def find_sql_write_files(root: Path | None = None) -> dict[str, list[str]]:
    """From `root` (default: `src/medrag/pipeline/facts/`) finds the `.py`
    files that contain value-writing SQL -- `tests/` and `__pycache__`
    are EXCLUDED. Returns a dict `{filename: [snippet, ...]}`; if the
    same filename appears in multiple places (e.g. different sub-
    packages) the FULL path (`str(path)`) is used as the key (to avoid
    collisions, see the caller's key normalisation)."""
    base = root or HERE
    writers: dict[str, list[str]] = {}
    for path in sorted(base.rglob("*.py")):
        rel_parts = path.relative_to(base).parts
        if "tests" in rel_parts or "__pycache__" in rel_parts:
            continue
        try:
            hits = _sql_write_snippets_in_file(path)
        except SyntaxError:
            # Unparseable file is outside this analysis's scope --
            # silently skipped (no such file in the repo; defensive).
            continue
        if hits:
            writers[str(path.relative_to(base))] = hits
    return writers