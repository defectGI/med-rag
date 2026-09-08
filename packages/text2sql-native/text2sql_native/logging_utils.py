"""Logging helpers.

The library uses stdlib :mod:`logging` only, never ``print``, and never logs
secrets. Call :func:`configure_logging` once with the level from config. Use
:func:`get_logger` everywhere else.
"""

from __future__ import annotations

import logging

_LIBRARY_LOGGER_NAME = "text2sql"
_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Set up the library logger.

    Attaches one stream handler to the ``text2sql`` logger, not the root logger,
    so the host application's logging is left alone. Idempotent: later calls
    only change the level.

    Args:
        level: A standard level name, e.g. ``"DEBUG"``, ``"INFO"``.
    """
    global _configured
    logger = logging.getLogger(_LIBRARY_LOGGER_NAME)
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logger.setLevel(numeric_level)

    if not _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
        _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the ``text2sql`` namespace.

    Args:
        name: Optional dotted suffix, e.g. ``"pipeline.linking"``.
    """
    if name:
        return logging.getLogger(f"{_LIBRARY_LOGGER_NAME}.{name}")
    return logging.getLogger(_LIBRARY_LOGGER_NAME)
