"""N-05 (I-40): the VERY BEGINNING of `medrag-nightly` -- no writes, this module
does not start without a successful backup.

K-63 connection: the nightly run OVERWRITES manually-corrected data (KARAR-KAYDI).
The only protection is this backup -- so if `run_backup()` FAILS here,
`run_nightly.main()` must NOT call any `stage_*`, and the night must be reported
as *failed* (falls into N-08/N-10's IntegritySection as `backup_completed=False`
-- see the TODO note in nightly_report.py).

Backed-up items (the task text's list, same as I-42):
  1. `specs.db`      -- via sqlite3's `.backup()` API running on a LIVE
                        connection (taking a DB running in WAL mode with a plain
                        `shutil.copy`/file copy would miss the uncommitted pages
                        (specs.db-wal) and could produce a CORRUPT .db --
                        `sqlite3.Connection.backup()` reads page by page from the
                        live connection, guaranteeing a consistent image).
  2. `document_nodes.json`
  3. `all_chunks.json`
  4. the parse output dir (PARSED_OUTPUT_DIR, the whole tree -- `shutil.copytree`)
  5. the Qdrant collection -- via the snapshot API (K-63/I-42): Qdrant does NOT
     RUN on THIS MACHINE (repo-wide rule, see AGENTS.md/CLAUDE.md), so this step
     was TESTED against a real server -- it was only written as a function
     (`backup_qdrant_snapshot`, the client is injectable from outside, mirroring
     `qdrant_store.py`'s same DI pattern). Real validation must happen on a
     machine where Qdrant actually runs (known limitation). If `QDRANT_URL` is
     undefined (or the connection/snapshot call dies) this ITEM is NOT counted
     as a backup FAILURE -- it is noted with `skipped=True`, because the task
     text says "if there is one" (accepting that Qdrant may be an absent
     dependency in this environment); the others (1-4) are MANDATORY -- if any
     one is missing/fails, `BackupError` is raised.

Target dir (deliberately NOT added to TOML to reduce the COLLISION risk of
writing parallel to N-13/N-15 config/default.toml -- the task text's explicit
preference): fixed `DEFAULT_BACKUP_ROOT`, overridable via the `NIGHTLY_BACKUP_ROOT`
env var (for tests and the deployment environment) -- same pattern as
`nightly_lock.py`'s `NIGHTLY_LOCK_PATH`.

N-25: automatic restore. The previous design only took backups -- if a stage
crashed halfway or the process DIED via `kill -9`/crash, the data written so far
(partial/corrupt) STAYED on disk and restore was manual. Automatic repair was
added for two scenarios:

  1. If a stage CRASHES MIDWAY in the full chain (process still alive, the
     exception can be caught): `run_nightly._run_full_chain_with_report` calls
     `restore_backup()` on this run's own backup (`backup_manifest.root`), so
     EVERYTHING this run wrote (including successfully finished stages) returns
     to the pre-run state -- to not leave it in a "halfway but partially
     progressed" state the WHOLE run is rolled back; per-stage PARTIAL rollback
     is NOT done (to keep it simple and predictable).
  2. If the process DIES for a reason like `kill -9`/power loss (no
     except/finally runs): a "in progress" marker is kept via
     `mark_in_progress()`/`clear_in_progress()` (see below). The next
     `medrag-nightly` call checks this marker BEFORE taking a NEW backup; if the
     marker is still there (previous run did not exit cleanly) it returns to that
     run's backup, THEN starts its own work.

Deliberate limit to NOT collide with K-63: this automatic rollback ONLY restores
what `medrag-nightly` itself wrote (this run's / the previous half-run's backup)
-- it never automatically OVERWRITES a user's manual correction, because restore
ALWAYS returns to a backup taken BEFORE this run started, not to a time after/
outside the run. On the `--stage`/`--from` (manual intervention, K-60) paths a
failure does NOT trigger an automatic restore (scope deliberately narrow, see
`_run_full_chain_with_report`'s docstring in `run_nightly.py`) -- only the
`kill -9` recovery (scenario 2) also covers those paths, because that scenario
is "the process could not exit cleanly", handling an INTERRUPTION, not the
outcome of a manual intervention.

Qdrant OUT OF SCOPE: like the backup (server-side snapshot, no Qdrant on this
machine) the restore SKIPS the qdrant item, only noting it -- real restore must
be done manually via the snapshot API on the machine where Qdrant runs (known
limitation, the SAME restriction as the backup).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# src/medrag/pipeline/cli/nightly_backup.py -> repo root (4 levels up, the SAME
# convention as run_nightly.py's _REPO_ROOT).
_REPO_ROOT = Path(__file__).resolve().parents[4]

NIGHTLY_BACKUP_ROOT_ENV = "NIGHTLY_BACKUP_ROOT"
DEFAULT_BACKUP_ROOT = _REPO_ROOT / "nightly_backups"

# N-25: the "in progress" marker -- the same env-override pattern as
# `nightly_lock.py`'s `NIGHTLY_LOCK_PATH` (for tests and the deployment
# environment); fixed default in the system temp dir.
NIGHTLY_INPROGRESS_PATH_ENV = "NIGHTLY_INPROGRESS_PATH"
DEFAULT_INPROGRESS_PATH = Path(tempfile.gettempdir()) / "medrag-nightly-inprogress.json"


class BackupError(RuntimeError):
    """One of the MANDATORY backup items failed -- the caller (`run_nightly.main`)
    must catch this and NOT call any `stage_*`; the night must be reported as
    *failed* (N-05 rule)."""


class RestoreError(RuntimeError):
    """One of the MANDATORY restore items failed (N-25). This is a MORE serious
    situation than a backup FAILURE -- the data may now be in neither a pre-run
    nor a post-run CONSISTENT state; the caller must NOT swallow it, must stop
    loudly (with a traceback) so manual intervention can happen."""


@dataclass
class BackupEntry:
    name: str
    source: str
    destination: str
    size_bytes: int = 0
    skipped: bool = False
    note: str = ""


@dataclass
class BackupManifest:
    root: Path
    started_at: str
    finished_at: str
    duration_seconds: float
    entries: list[BackupEntry] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries if not e.skipped)

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "total_size_bytes": self.total_size_bytes,
            "entries": [
                {
                    "name": e.name, "source": e.source, "destination": e.destination,
                    "size_bytes": e.size_bytes, "skipped": e.skipped, "note": e.note,
                }
                for e in self.entries
            ],
        }


def _backup_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root)
    raw = os.environ.get(NIGHTLY_BACKUP_ROOT_ENV)
    return Path(raw) if raw else DEFAULT_BACKUP_ROOT


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def _backup_sqlite(src: Path, dest: Path) -> int:
    """see the module docstring -- a WAL-safe SQLite backup."""
    if not src.is_file():
        raise BackupError(f"specs.db bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_con = sqlite3.connect(str(src))
    try:
        dest_con = sqlite3.connect(str(dest))
        try:
            src_con.backup(dest_con)
        finally:
            dest_con.close()
    finally:
        src_con.close()
    return dest.stat().st_size


def _backup_file(src: Path, dest: Path) -> int:
    if not src.is_file():
        raise BackupError(f"yedeklenecek dosya bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.stat().st_size


def _backup_dir(src: Path, dest: Path) -> int:
    if not src.is_dir():
        raise BackupError(f"yedeklenecek dizin bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def _restore_sqlite(backup_src: Path, live_dest: Path) -> None:
    """The INVERSE of `_backup_sqlite`. The backup file is already a
    CONSISTENT/plain sqlite file taken via the `.backup()` API (no WAL) -- but the
    live target may still have leftover `-wal`/`-shm` side files from a previous
    run; these are DELETED (otherwise an old wal/shm won't MATCH the newly written
    main file and an inconsistent image can be read); then the SAME `.backup()`
    pattern is used in the REVERSE direction (not a plain file copy -- so N-25
    carries the SAME WAL-safe guarantee as N-05)."""
    if not backup_src.is_file():
        raise RestoreError(f"yedekte specs.db bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        sidecar = live_dest.with_name(live_dest.name + suffix)
        sidecar.unlink(missing_ok=True)
    src_con = sqlite3.connect(str(backup_src))
    try:
        dest_con = sqlite3.connect(str(live_dest))
        try:
            src_con.backup(dest_con)
        finally:
            dest_con.close()
    finally:
        src_con.close()


def _restore_file(backup_src: Path, live_dest: Path) -> None:
    if not backup_src.is_file():
        raise RestoreError(f"yedekte dosya bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_src, live_dest)


def _restore_dir(backup_src: Path, live_dest: Path) -> None:
    if not backup_src.is_dir():
        raise RestoreError(f"yedekte dizin bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    if live_dest.exists():
        shutil.rmtree(live_dest)
    shutil.copytree(backup_src, live_dest)


def restore_backup(root: Path | str) -> list[BackupEntry]:
    """N-25: writes a backup under `root` (each item's `destination` in
    manifest.json) BACK to its `source` (live) location -- exactly the REVERSE
    direction of `run_backup`, reading the same manifest.json (basing it on what
    was actually backed up rather than INVENTING items).

    The `qdrant` item is ALWAYS skipped (see the module docstring -- server-side
    snapshot, cannot be restored from this machine). If one of the MANDATORY items
    (specs.db/document_nodes.json/all_chunks.json/parsed_output_dir) is marked
    `skipped=True` in the manifest (normally it should have failed during backup
    -- but such a backup should not have remained on disk after raising
    BackupError), or if it fails during restore, `RestoreError` is raised: the
    caller must NOT swallow it, the data may now be in an UNDEFINED state."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RestoreError(f"manifest okunamadi ({manifest_path}): {exc}") from exc

    restored: list[BackupEntry] = []
    for item in raw.get("entries", []):
        name = item["name"]
        source = Path(item["source"])
        destination = Path(item["destination"])
        if name == "qdrant":
            restored.append(BackupEntry(name, str(source), str(destination), skipped=True,
                                         note="qdrant restore desteklenmiyor -- Qdrant'in "
                                              "calistigi makinede snapshot API'siyle elle "
                                              "geri yuklenmeli (bilinen sinir)"))
            continue
        if item.get("skipped"):
            raise RestoreError(f"yedekte '{name}' kalemi zaten basarisiz/eksik isaretli "
                                f"({item.get('note')}) -- geri yukleme guvenli degil")
        if name == "specs.db":
            _restore_sqlite(destination, source)
        elif name == "parsed_output_dir":
            _restore_dir(destination, source)
        else:
            _restore_file(destination, source)
        restored.append(BackupEntry(name, str(source), str(destination),
                                     size_bytes=item.get("size_bytes", 0)))
    return restored


def _inprogress_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    raw = os.environ.get(NIGHTLY_INPROGRESS_PATH_ENV)
    return Path(raw) if raw else DEFAULT_INPROGRESS_PATH


def mark_in_progress(backup_root: Path | str, *, path: Path | str | None = None) -> None:
    """N-25: a run writes this RIGHT BEFORE it takes its own backup and starts
    calling the `stage_*`s. If the process exits cleanly (successfully OR with a
    caught error -- both call `clear_in_progress()` inside the `finally` of
    `_run_full_chain_with_report`/`main()`) this file is DELETED. A run that does
    NOT exit cleanly (like `kill -9`/crash) leaves this file on disk -- the next
    call reads it as the 'previous run was left halfway, return to that backup'
    signal."""
    p = _inprogress_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"backup_root": str(backup_root)}), encoding="utf-8")


def read_in_progress(*, path: Path | str | None = None) -> dict | None:
    p = _inprogress_path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def clear_in_progress(*, path: Path | str | None = None) -> None:
    p = _inprogress_path(path)
    p.unlink(missing_ok=True)


def backup_qdrant_snapshot(
    *, dest_dir: Path, collection: str, url: str | None, api_key: str | None,
    client_factory: Callable[[str, str | None], Any] | None = None,
) -> BackupEntry:
    """Backup of the Qdrant collection via the snapshot API (I-42). `client_factory`
    is injectable for tests (the same DI pattern as `qdrant_store.py`'s `from_env`
    classmethod) -- if not given, the real `QdrantClient` is constructed. In NO
    case does it RAISE `BackupError`: if the url is undefined or the
    connection/snapshot call dies it returns with `skipped=True` (the task text
    says "if there is one" -- this single exceptional/optional item)."""
    if not url:
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            skipped=True, note="QDRANT_URL tanimsiz")
    try:
        if client_factory is not None:
            client = client_factory(url, api_key)
        else:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url, api_key=api_key or None)
        result = client.create_snapshot(collection_name=collection)
        snapshot_name = getattr(result, "name", None) or str(result)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "snapshot_name.txt").write_text(snapshot_name, encoding="utf-8")
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            note=f"snapshot={snapshot_name} (server-side, sunucuda kaldi)")
    except Exception as exc:  # noqa: BLE001 -- best-effort, bkz. modul docstring'i
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            skipped=True, note=f"snapshot alinamadi: {exc}")


def run_backup(
    *, backup_root: Path | str | None = None,
    qdrant_url: str | None = None, qdrant_api_key: str | None = None,
    qdrant_collection: str | None = None,
    qdrant_client_factory: Callable[[str, str | None], Any] | None = None,
) -> BackupManifest:
    """The FIRST step of `run_nightly.main()` (N-05). If any MANDATORY item
    (specs.db/document_nodes.json/all_chunks.json/parse output dir) fails it
    raises `BackupError`; the half-done target dir is CLEANED UP (so a
    failed/incomplete backup does not remain on disk as if it were valid) -- the
    caller must catch it and run NO stage."""
    from medrag.pipeline.cli import run_parse_pipeline
    from medrag.pipeline.facts import discover

    started_monotonic = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    root = _backup_root(backup_root) / _timestamp()
    entries: list[BackupEntry] = []

    try:
        run_parse_pipeline._bootstrap()

        dest = root / "specs.db"
        size = _backup_sqlite(discover.DB_PATH, dest)
        entries.append(BackupEntry("specs.db", str(discover.DB_PATH), str(dest), size))

        dest = root / "document_nodes.json"
        src = Path(run_parse_pipeline.DOCUMENT_NODES_PATH)
        size = _backup_file(src, dest)
        entries.append(BackupEntry("document_nodes.json", str(src), str(dest), size))

        dest = root / "all_chunks.json"
        size = _backup_file(discover.ALL_CHUNKS_PATH, dest)
        entries.append(BackupEntry("all_chunks.json", str(discover.ALL_CHUNKS_PATH),
                                    str(dest), size))

        dest = root / "parsed_output"
        src = Path(run_parse_pipeline.PARSED_OUTPUT_DIR)
        size = _backup_dir(src, dest)
        entries.append(BackupEntry("parsed_output_dir", str(src), str(dest), size))

        if qdrant_collection is None:
            from medrag.pipeline.vectorize.config import load_config
            qdrant_collection = load_config().qdrant.collection_name
        entries.append(backup_qdrant_snapshot(
            dest_dir=root / "qdrant", collection=qdrant_collection,
            url=qdrant_url if qdrant_url is not None else os.environ.get("QDRANT_URL"),
            api_key=qdrant_api_key if qdrant_api_key is not None
                    else os.environ.get("QDRANT_API_KEY"),
            client_factory=qdrant_client_factory))
    except BackupError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise BackupError(f"yedekleme basarisiz: {exc}") from exc

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    duration = time.monotonic() - started_monotonic
    manifest = BackupManifest(root=root, started_at=started_at, finished_at=finished_at,
                               duration_seconds=duration, entries=entries)
    try:
        (root / "manifest.json").write_text(
            json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass  # even if the manifest write fails, the backup ITSELF is already done
    return manifest
