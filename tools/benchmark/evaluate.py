"""Module entry point, for parity with parser/scripts/*.py.

    python -m tools.benchmark.evaluate -f <folder> [-s scope.md] [--model ...]

Equivalent to `python -m tools.benchmark ...`. `tools/` is a real installed
package now (D-46/D-62), so no sys.path bootstrap is needed to import
`tools.benchmark.cli` -- but that also means this module must be run with
`-m` (bare `python tools/benchmark/evaluate.py` is no longer supported, since
that puts only this file's own directory on sys.path, not the repo root).
"""

from __future__ import annotations

from tools.benchmark.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
