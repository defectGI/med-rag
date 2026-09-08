"""Deployable Flask entry point for the panel.

Wraps the `panel_bp` blueprint from `blueprint.py` into a standalone runnable
Flask app -- same image, same deploy, no extra service: no separate
container/service is added; it comes up inside the SAME image as the `medrag`
package via `python -m medrag.api.panel.app` (or a WSGI server calling
`create_panel_app()`).

**Network access:** the panel must be unreachable from the internet -- open to
the internal network, closed to the outside. The ONLY thing this module can do
about that (real network isolation -- firewall/security group/internal network
-- is enforced at the deploy layer, not here) is **keep the default bind
address at `127.0.0.1`** (the same pattern as `wa_bot.py::main`/`webapp.py`
with `WA_BOT_HOST`/`CHATBOT_WEB_HOST`): switching `PANEL_HOST` to `0.0.0.0` or
a public interface OPENS the panel to the network, which must be a deliberate
operator decision, never the default. It also listens on a **different port**
from `wa_bot`/`webapp` (`PANEL_PORT`, default 8010 -- the same as
`serve.py::PORT`) so it does NOT share a listener with the internet-facing
`wa_bot` webhook: sharing the same Flask app/port would couple the panel's
network isolation to the internet-facing service's isolation.
"""

from __future__ import annotations

import logging
import os

from flask import Flask

from medrag.api.panel.blueprint import panel_bp

logger = logging.getLogger("medrag.api.panel")


def create_panel_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(panel_bp)
    return app


def main() -> int:
    app = create_panel_app()
    # Same pattern as elsewhere: the bind address is read only at the entry
    # point, at call time (NOT at import time).
    host = os.getenv("PANEL_HOST", "127.0.0.1")
    port = int(os.getenv("PANEL_PORT", "8010"))
    if host not in ("127.0.0.1", "localhost"):
        logger.warning(
            "PANEL_HOST=%s -- panel varsayılan localhost DIŞINA bağlanıyor; "
            "M-14 (I-31) 'internetten erişilemez' kabulü bir ağ/firewall "
            "katmanıyla YENİDEN doğrulanmalı, bu süreç bunu kendisi garanti ETMEZ.",
            host,
        )
    logger.info("panel dinliyor: http://%s:%d", host, port)
    app.run(host=host, port=port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
