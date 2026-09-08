"""Audit the `medrag` package's layer contracts with import-linter.

The ruff TID251 banned-api rule only catches DIRECT imports (a file-based
static rule). Here `lint-imports` scans the real module dependency graph
(including transitive chains) and verifies the same intent more strongly -- the
contracts are defined under `[tool.importlinter]` in the root `pyproject.toml`:

1. "independence": medrag.api <-> medrag.pipeline never import each other.
2. "layers": medrag.core is at the bottom -- neither api nor pipeline may violate
   it top-down, core imports no upper layer.

Scope is deliberately only the `medrag` package (everything under src/medrag);
tools/ was packaged as a top layer but is not part of this layer contract
(root_package = "medrag", tools/ is a separate top-level package).
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

from importlinter import cli as importlinter_cli

# src/medrag/tests/test_import_contracts.py -> repo root (next to pyproject.toml,
# where [tool.importlinter] is defined).
_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_import_linter_contracts_hold() -> None:
    # importlinter has no `python -m importlinter` entry point (the package has
    # no __main__.py), and calling the `lint-imports` console script from a
    # subprocess would mean hunting for a platform-dependent .exe/shim -- so we
    # call the library's own Python API (importlinter.cli.lint_imports)
    # directly. This function resolves its filesystem scan relative to the
    # directory holding `pyproject.toml`, so cwd is briefly moved to the root.
    output = io.StringIO()
    original_cwd = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        with contextlib.redirect_stdout(output):
            exit_code = importlinter_cli.lint_imports(
                config_filename=str(_REPO_ROOT / "pyproject.toml"),
                no_cache=True,
            )
    finally:
        os.chdir(original_cwd)

    assert exit_code == 0, (
        "import-linter katman sözleşmesi ihlal edildi "
        f"(medrag.api <-> medrag.pipeline bağımsızlığı / medrag.core en altta):\n"
        f"{output.getvalue()}"
    )
