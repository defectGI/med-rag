"""The `medrag.core.config` package.

`default.toml` is validated against `schema.CoreConfig` as soon as this
package is first imported -- "validate the config at startup". Every component
that already imports `medrag.core.config.loader` (today: the chatbot's
`src/medrag/api/config.py`, the vectorize
`src/medrag/pipeline/vectorize/config.py`) gets this check the moment it imports
this package, BEFORE it calls its own loader -- no separate "startup hook"
needed.

An invalid `default.toml` (wrong type, out of range, unknown/missing key) stops
the process here with a CLEAR `pydantic.ValidationError`; nothing silently falls
back to a default.
"""

from __future__ import annotations

from medrag.core.config.schema import load_core_config as _load_core_config

# Module-level side effect: runs as soon as the package is imported (see the
# docstring above). We don't keep the result -- the point is only validation;
# this package does NOT yet DISTRIBUTE `default.toml`'s values (the
# read-through wiring is a later step).
_load_core_config()
