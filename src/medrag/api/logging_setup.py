"""Persistent file logging: configured once when the webapp starts;
everything in the background (especially the SQL chain) is written to disk --
no more blind error guessing.

Why a separate module and why it attaches to two loggers:

- `chatbot`'s own code uses stdlib `logging` (until this module is configured
  it had no handlers -> everything evaporated). Attaching a file handler to
  the root logger collects `chatbot.*`, Flask/`werkzeug` and `urllib` logs in
  one file (they all propagate to root).
- The `text2sql` (sqlretrieve) package sets up its OWN logger with
  `propagate=False` -- so configuring the root does NOT capture its logs.
  SQL linking/generation detail is exactly what needs to be visible, so the
  same file (and console) handler is attached to the `text2sql` logger as
  well, with its level tuned independently (`text2sql_level`) -- to pull SQL
  to DEBUG while keeping the rest at INFO.

Privacy note: these logs contain user query text (resolved_query,
sql_rewrite, incoming messages). This is a SEPARATE, deliberate choice from
"ConversationMemory is RAM-only, no persistence" -- diagnostic logs were
explicitly requested. If you don't want query text on disk, set `level` to
WARNING (queries/messages are logged at INFO, SQL ERRORS at ERROR -- i.e. at
WARNING level queries are hidden while errors stay visible).
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

# This file lives under src/medrag/api/ now; `logs/` as DATA stayed in the old
# `chatbot/` directory at the repo root (same pattern as
# conversation_log.py). parents[3] = repo root.
_COMPONENT_ROOT = Path(__file__).resolve().parents[3] / "chatbot"
_DEFAULT_LOG_DIR = _COMPONENT_ROOT / "logs"

_TEXT2SQL_LOGGER = "text2sql"

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"

# Don't double the handlers on a second configure_logging call (both
# create_app and main call it).
_configured = False


def _coerce_level(level: str | int, fallback: int = logging.INFO) -> int:
    if isinstance(level, int):
        return level
    numeric = logging.getLevelName(str(level).upper())
    return numeric if isinstance(numeric, int) else fallback


def configure_logging(
    *,
    level: str | int = "INFO",
    text2sql_level: str | int = "DEBUG",
    log_dir: str | Path | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
    console: bool = True,
) -> Path:
    """Sets up file (+optional console) logging. Idempotent: the second call
    adds no handlers, only updates levels.

    Args:
        level: root + `chatbot.*` level (includes Flask/werkzeug/urllib).
        text2sql_level: SEPARATE level for `text2sql` (SQL
            linking/generation) -- to pull SQL to DEBUG while keeping the
            rest at INFO. SQL is the priority, hence the DEBUG default.
        log_dir: directory for log files. None -> chatbot/logs/.
        max_bytes / backup_count: rotating-file (RotatingFileHandler) limits.
        console: if True also writes to stderr (to watch the webapp live in
            the terminal).

    Returns:
        Full path of the log file being written (printed to the user at
        startup).
    """
    global _configured

    root_level = _coerce_level(level)
    sql_level = _coerce_level(text2sql_level)

    directory = Path(log_dir) if log_dir else _DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    log_file = directory / "medrag.api.log"

    root = logging.getLogger()
    text2sql = logging.getLogger(_TEXT2SQL_LOGGER)

    if _configured:
        # Update levels only (e.g. an env override arrives on the second call).
        root.setLevel(root_level)
        text2sql.setLevel(sql_level)
        return log_file

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    handlers: list[logging.Handler] = [file_handler]
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    root.setLevel(root_level)
    for handler in handlers:
        root.addHandler(handler)

    # text2sql (sqlretrieve) adds its own StreamHandler in its OWN
    # configure_logging and sets `propagate=False`; that call runs in
    # create_app AFTER ours (when the SQL engine is built). To make the
    # interleaving deterministic:
    #  - ONLY the FILE handler is added to text2sql (console is left to
    #    sqlretrieve; a second stream handler would print every SQL line
    #    TWICE on the console),
    #  - `propagate=False` is also guaranteed here (if sqlretrieve hasn't run
    #    yet, the default True would leak to the root stream and double again),
    #  - we set the level, but sqlretrieve may set it again from
    #    `log_level` in `config.toml`; both are kept aligned at DEBUG (see
    #    chatbot/text2sql/config.toml [general] log_level + this component's
    #    config text2sql_level).
    text2sql.setLevel(sql_level)
    text2sql.propagate = False
    text2sql.addHandler(file_handler)

    _configured = True
    return log_file


def configure_logging_from_cfg(cfg) -> Path:
    """Sets up `configure_logging` from `cfg.logging` + operator env overrides
    (`CHATBOT_LOG_LEVEL`/`CHATBOT_LOG_DIR`/`CHATBOT_TEXT2SQL_LOG_LEVEL`) --
    the SINGLE entry point shared by webapp.py (`create_app`/`main`) AND
    `wa_bot.py`; both use the same env names / same idempotent setup (no code
    duplication). Returns the path of the log file being written."""
    lg = cfg.logging
    return configure_logging(
        level=os.getenv("CHATBOT_LOG_LEVEL") or lg.level,
        text2sql_level=os.getenv("CHATBOT_TEXT2SQL_LOG_LEVEL") or lg.text2sql_level,
        log_dir=os.getenv("CHATBOT_LOG_DIR") or None,
        max_bytes=lg.max_bytes,
        backup_count=lg.backup_count,
        console=lg.console,
    )


def log_trace_step(logger: logging.Logger, step: str, data: object, *, prefix: str = "") -> None:
    """Writes an `on_trace(step, data)` event to the log -- called from the
    webapp's trace bridge, so every background stage that flashes by in the
    browser panel is also kept on disk.

    Level mapping is deliberate:
    - `sql_error` -> ERROR + FULL `raw_text` (what the intermediate model
      returned; the primary diagnostic data, never hidden at any level).
    - `sql_generated` / `sql_rewrite` / `resolved_query` / linking/generation
      metrics -> INFO.
    - chunk result lists (`top_n_results`/`sql_rows`) -> DEBUG (voluminous).
    """
    tag = f"{prefix}{step}"
    if step == "sql_error":
        logger.error("trace[%s]: %s", tag, _format_trace_data(data))
    elif isinstance(data, list):
        logger.debug("trace[%s]: %d sonuç %s", tag, len(data), _format_trace_data(data))
    else:
        logger.info("trace[%s]: %s", tag, _format_trace_data(data))


def _format_trace_data(data: object) -> str:
    if isinstance(data, dict):
        return " | ".join(f"{k}={v!r}" for k, v in data.items())
    if isinstance(data, list):
        return "; ".join(
            f"[{d.get('id')}]({d.get('score')}) {str(d.get('text', ''))[:120]}"
            if isinstance(d, dict) else str(d)
            for d in data[:20]
        )
    return str(data)


__all__ = ["configure_logging", "configure_logging_from_cfg", "log_trace_step"]
