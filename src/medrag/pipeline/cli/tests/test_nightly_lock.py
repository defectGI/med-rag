"""N-02 (I-11) testleri: `NightlyLock` -- ayni anda iki gece kosusu
calisamaz (REDDET, bekleme kuyrugu YOK); kilidi tutan surec olse bile
(stale PID) bir sonraki cagri kilidi devralir.

Calistirma: `python -m pytest src/medrag/pipeline/cli/tests/test_nightly_lock.py`
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time

import pytest

from medrag.pipeline.cli.nightly_lock import NightlyLock, NightlyLockHeld, _read_lock


def test_acquire_then_release_removes_lock_file(tmp_path):
    lock_path = tmp_path / "nightly.lock"
    lock = NightlyLock(lock_path)

    lock.acquire()
    assert lock_path.exists()
    payload = _read_lock(lock_path)
    assert payload["pid"] == os.getpid()

    lock.release()
    assert not lock_path.exists()


def test_context_manager_releases_on_exception(tmp_path):
    lock_path = tmp_path / "nightly.lock"

    with pytest.raises(ValueError), NightlyLock(lock_path):
        assert lock_path.exists()
        raise ValueError("kosu ortasinda patladi")

    assert not lock_path.exists(), "kilit istisna sonrasi da SERBEST kalmali"


def test_second_run_is_rejected_while_first_holds_lock(tmp_path):
    """Kilit tutulurken IKINCI bir cagri REDDEDILIR -- bekleme kuyrugu YOK."""
    lock_path = tmp_path / "nightly.lock"
    first = NightlyLock(lock_path)
    first.acquire()
    try:
        second = NightlyLock(lock_path)
        with pytest.raises(NightlyLockHeld, match=str(os.getpid())):
            second.acquire()
        # ikinci basarisiz denemenin kilidi ETKILEMEDIGI dogrulanir --
        # dosya hala BIRINCI surecin PID'ini tasiyor.
        assert _read_lock(lock_path)["pid"] == os.getpid()
    finally:
        first.release()


def test_stale_lock_from_dead_pid_is_reclaimed(tmp_path):
    """Kilidi tutan surec `kill -9` ile olse bile (burada gercek bir alt
    surec baslatilip oldurulerek taklit edilir) bir sonraki cagri kilidi
    SERBEST sayip devralir (TTL degil, PID canliligi kontrolu) -- AYNI
    host/konteynerde (`hostname` bizimkiyle AYNI, bkz. `_is_stale`)."""
    lock_path = tmp_path / "nightly.lock"

    # Kisa omurlu bir alt surec baslat, PID'ini kilit dosyasina biz yazalim
    # (gercek NightlyLock.acquire() ile ayni format), sonra sureci OLDUR.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    lock_path.write_text(json.dumps({
        "pid": proc.pid, "hostname": socket.gethostname(), "started_at": "x",
    }), encoding="utf-8")

    # Surec hala CANLI iken: reddedilmeli.
    lock = NightlyLock(lock_path)
    with pytest.raises(NightlyLockHeld):
        lock.acquire()

    proc.kill()
    proc.wait(timeout=10)
    # Isletim sisteminin PID'i gercekten geri almasi icin kucuk bir bekleme
    # payi (zombie/temizlik gecikmesi -- CI'da flaky olmasin diye).
    for _ in range(50):
        import psutil
        if not psutil.pid_exists(proc.pid):
            break
        time.sleep(0.1)

    # Surec artik OLU -- kilit STALE sayilip devralinmali (reddedilmemeli).
    lock.acquire()
    try:
        assert _read_lock(lock_path)["pid"] == os.getpid()
    finally:
        lock.release()


def test_lock_from_different_hostname_is_always_stale_even_if_pid_alive(tmp_path):
    """2026-08-27 fix: konteynerli dagitimda kilit kalici bir yola tasindiginda
    (`NIGHTLY_LOCK_PATH=/corpus/...`) `pid` bir ONCEKI konteyner neslinin PID
    namespace'ine ait olabilir -- redeploy sonrasi YENI konteynerde AYNI PID
    numarasi TAMAMEN ALAKASIZ (ama gercekten canli) baska bir surece denk
    gelebilir. `hostname` (konteyner kimligi) bizimkiyle AYNI DEGILSE -- PID
    yasiyor OLSA BILE -- kilit STALE sayilmali (bu dagitimda `pipeline` TEK
    REPLIKA ve stack birlikte redeploy edildigi icin bu KESIN bir sinyal)."""
    lock_path = tmp_path / "nightly.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        lock_path.write_text(json.dumps({
            "pid": proc.pid, "hostname": "eski-konteyner-id", "started_at": "x",
        }), encoding="utf-8")

        # PID GERCEKTEN canli olmasina ragmen -- hostname FARKLI oldugu icin
        # STALE sayilip devralinmali, REDDEDILMEMELI.
        lock = NightlyLock(lock_path)
        lock.acquire()
        try:
            assert _read_lock(lock_path)["pid"] == os.getpid()
            assert _read_lock(lock_path)["hostname"] == socket.gethostname()
        finally:
            lock.release()
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_default_lock_path_overridden_by_env(monkeypatch, tmp_path):
    custom = tmp_path / "custom.lock"
    monkeypatch.setenv("NIGHTLY_LOCK_PATH", str(custom))

    lock = NightlyLock()
    assert lock.path == custom
    lock.acquire()
    try:
        assert custom.exists()
    finally:
        lock.release()
