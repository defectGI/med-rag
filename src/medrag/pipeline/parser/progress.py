"""Thread-safe terminal progress bars for the model-bound parsing stages.

The expensive stages all know their unit counts up front -- a PDF's page
count, a document's unresolved-image count, its table count -- so each stage
renders one bar per document while it works, e.g.:

    datasheet.pdf: pages      45%|####      | 9/20 [00:41<00:50, page]

Several documents run at once (DOC_CONCURRENCY in run_parse_pipeline.py), so
bars are slot-based: an opening bar takes the lowest free terminal line
(tqdm `position`) and frees it on close. Up to DOC_CONCURRENCY bars sit on
stable lines instead of every bar fighting over the same one.

Bars render only when tqdm is importable AND stderr is a real terminal;
otherwise (redirected logs, CI, missing dependency) `bar()` returns a no-op
and `write()` degrades to plain print -- log files never see \\r control
characters. PROGRESS_BAR=0 forces them off (read per call, so it can live in
either .env like every other flag).

Exception to "no TTY, no bar": PROGRESS_IPC=1 (set by a parent process that
launched this one with piped output, e.g. pipeline/run_e2e_test.py's
run_stage()). Then `bar()` emits machine-readable "@@BAR ..." lines on
stdout instead of rendering, and the parent -- whose stderr IS a real
terminal -- parses them and drives the visible bar itself. Protocol, one
event per line, flushed immediately:

    @@BAR OPEN <id> <total> <unit> <desc may contain spaces>
    @@BAR UPD <id> <n>
    @@BAR CLOSE <id>

Besides the per-stage bars there is one global gauge: how many model
requests (VLM/LLM HTTP calls) are in flight right now. The llm clients
report request start/finish (see llm/openai_compat.py / anthropic_client.py)
and the count shows as a pinned status line under the bars -- or, under
PROGRESS_IPC, as `@@REQ <count>` events the parent sums across its children
(concurrent stages / IMAGE_CONCURRENCY / DESCRIBE_CONCURRENCY all land in the
same number).

Anything printed while bars may be on screen must go through `write()`
(tqdm interleaves it cleanly above the bars); a bare print() would garble
them.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from typing import Self

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # optional dependency -- bars just turn off
    _tqdm = None


def _off() -> bool:
    # PROGRESS_BAR now comes from config/default.toml [progress].bar (env can
    # override it; the inverted logic lives in _cast_barbool: 0/false/no/off turn
    # the bar OFF). Lazy import: progress is a light util, so it loads config only
    # when actually asked.
    from medrag.pipeline.parser.config import get_config
    return not get_config().progress.bar


def _on() -> bool:
    if _off():
        return False
    try:
        return _tqdm is not None and sys.stderr.isatty()
    except Exception:  # noqa: BLE001 -- a broken/replaced stderr means no TTY
        return False


def _ipc_on() -> bool:
    return not _off() and os.getenv("PROGRESS_IPC", "").strip() == "1"


class _Noop:
    """Context-manager/bar stand-in used whenever rendering is off."""

    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        return False


_NOOP = _Noop()

_slot_lock = threading.Lock()
_slots_taken: set[int] = set()


def _take_slot() -> int:
    with _slot_lock:
        pos = 0
        while pos in _slots_taken:
            pos += 1
        _slots_taken.add(pos)
        return pos


class _Bar:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self._pos = _take_slot()
        self._t = _tqdm(total=total, desc=desc, unit=unit, position=self._pos,
                        leave=False, dynamic_ncols=True)
        self._closed = False

    def update(self, n: int = 1) -> None:
        self._t.update(n)

    def close(self) -> None:
        # Idempotent so `with` + an explicit close can coexist; the slot must
        # be released exactly once or a later bar lands on a stale line.
        if self._closed:
            return
        self._closed = True
        self._t.close()
        with _slot_lock:
            _slots_taken.discard(self._pos)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


_ipc_id_lock = threading.Lock()
_ipc_next_id = 0


class _IpcBar:
    """Emits @@BAR protocol lines (see module docstring) instead of rendering.

    Newlines in desc would break the line-oriented protocol, so they're
    flattened; ids are process-unique so a parent multiplexing several
    concurrent stages never confuses two bars."""

    def __init__(self, total: int, desc: str, unit: str) -> None:
        global _ipc_next_id
        with _ipc_id_lock:
            _ipc_next_id += 1
            self._id = _ipc_next_id
        self._closed = False
        desc = " ".join(str(desc).split()) or "-"
        unit = str(unit).split()[0] if str(unit).split() else "it"
        print(f"@@BAR OPEN {self._id} {total} {unit} {desc}", flush=True)

    def update(self, n: int = 1) -> None:
        print(f"@@BAR UPD {self._id} {n}", flush=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        print(f"@@BAR CLOSE {self._id}", flush=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def bar(total: int, desc: str, unit: str = "it") -> _Bar | _IpcBar | _Noop:
    """One progress bar over `total` known units of work.

    Usable as a context manager (preferred -- an exception mid-stage still
    frees the terminal line) or via an explicit close(). `total <= 0` returns
    the no-op directly: a zero-length bar would flash an empty line."""
    if total <= 0:
        return _NOOP
    if _on():
        return _Bar(total, desc, unit)
    if _ipc_on():
        return _IpcBar(total, desc, unit)
    return _NOOP


_req_lock = threading.Lock()
_req_count = 0

_status_lock = threading.Lock()
_status_bar = None  # lazily created pinned line; lives until process exit


def _status(text: str) -> None:
    """Update the pinned status line (created on first use; it keeps its
    terminal slot for the whole run so the gauge doesn't jump around)."""
    global _status_bar
    with _status_lock:
        if _status_bar is None:
            _status_bar = _tqdm(total=0, position=_take_slot(), leave=False,
                                bar_format="{desc}", dynamic_ncols=True)
            atexit.register(_status_bar.close)
        _status_bar.set_description_str(text)


def set_inflight(n: int) -> None:
    """Show `n` model requests currently in flight. Called by the llm
    clients around every request, and by a parent runner summing its
    children's @@REQ events (pipeline/run_e2e_test.py)."""
    if _on():
        _status(f"model requests in flight: {n}")
    elif _ipc_on():
        print(f"@@REQ {n}", flush=True)


def request_started() -> None:
    global _req_count
    with _req_lock:
        _req_count += 1
        n = _req_count
    set_inflight(n)


def request_finished() -> None:
    global _req_count
    with _req_lock:
        _req_count -= 1
        n = _req_count
    set_inflight(n)


def write(msg: str) -> None:
    """print() replacement that doesn't garble bars currently on screen."""
    if _on():
        _tqdm.write(msg)
    else:
        print(msg, flush=True)
