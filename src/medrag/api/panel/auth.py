"""Read/write authorization split for the panel.

The panel should be **open to the internal network, closed to the outside** --
real network isolation is set up outside this code, at the deploy/infrastructure
layer (see the `app.py` module docstring: default bind `127.0.0.1`, the
`PANEL_HOST`/firewall decision is the operator's). Only the one thing that CAN
be enforced at the code level is implemented here: two separate authorization
levels.

- **Read**: everyone -- every `GET` endpoint inside the panel requires NO auth
  (assumption: read and write authority are separate; read is free).
- **Write**: authorized -- the panel has no trigger authority; its single
  write surface is `core.db.issues.resolve_issue` (only the `resolved_*`
  fields -- see that module's docstring). This function is guarded by the
  `PANEL_WRITE_TOKEN` environment variable -- the same pattern as
  `wa_bot.py::create_app`'s `SERVIS_ANAHTARI`: the secret is never written into
  code, it is read from `.env`, and the value is read at call time (NOT at
  import time).

If the token is NOT configured, write stays CLOSED (fail-closed) -- "allow
everyone when the token is empty" would be the WRONG default, silently leaving
an open write surface.
"""

from __future__ import annotations

import hmac
import os

#: Secret defined in `.env` -- its value is NEVER written here or into any
#: file, only its name is referenced (see the [panel] section of
#: `src/medrag/api/.env.example`).
ENV_WRITE_TOKEN = "PANEL_WRITE_TOKEN"

#: HTTP header the client carries the token in -- same naming pattern as
#: `wa_bot.py`'s `X-Servis-Anahtari` header.
WRITE_TOKEN_HEADER = "X-Panel-Write-Token"


def write_token_configured() -> bool:
    """Is a write token defined in `.env` (`False` if empty/absent)."""
    return bool((os.environ.get(ENV_WRITE_TOKEN) or "").strip())


def check_write_token(provided: str | None) -> bool:
    """Does `provided` (the header value from the request) match the configured
    token. If no token is configured at all, this ALWAYS returns `False`
    (fail-closed) -- `hmac.compare_digest` is used against the timing side
    channel (stricter than `wa_bot.py`'s plain `==` comparison; extra care here
    because a write triggers a DB update)."""
    expected = (os.environ.get(ENV_WRITE_TOKEN) or "").strip()
    if not expected:
        return False
    return hmac.compare_digest(expected, (provided or "").strip())


__all__ = [
    "ENV_WRITE_TOKEN",
    "WRITE_TOKEN_HEADER",
    "check_write_token",
    "write_token_configured",
]
