"""N-02 (I-11): `medrag-nightly`'nin kosu kilidi.

Ayni anda IKI gece kosusu calisamaz -- ikinci tetiklenen REDDEDILIR (bekleme
kuyrugu YOK, basit tutulur -- gorev metninin acik tercihi). Kilit dosya
tabanlidir (tek makine, Redis'e gerek yok): PID + baslama zamani tasir.
Kilidi tutan surec `kill -9` ile olse bile bir sonraki cagri kilidi kalici
SANMAZ -- `psutil.pid_exists()` ile PID'in hala yasadigini dogrular; PID
artik yoksa kilit STALE sayilir ve devralinir (TTL YOK, tek kaynak PID
canliligidir -- bu, "sure dolunca serbest kalsin" yerine "sahibi gercekten
olu mu" sorusuna dogrudan cevap verdigi icin tercih edildi).

`os.kill(pid, 0)` KULLANILMAZ: Windows'ta CPython'in `os.kill()` uygulamasi
sinyal 0'i da `TerminateProcess`e cevirir (bkz. CPython `nt_kill`) -- yani
"yasiyor mu" diye sormak yanlislikla surecin KENDISINI oldurebilir. Bunun
yerine `psutil.pid_exists()` kullanilir: hicbir sinyal GONDERMEZ, salt PID
tablosunu okur, her platformda guvenlidir.

2026-08-27 fix (`_is_stale`): kilit dosyasi konteynerli dagitimda kalici bir
yola (`NIGHTLY_LOCK_PATH=/corpus/...`) tasindiginda "tek makine" varsayimi
artik tam dogru degil -- redeploy AYNI host'ta YENI bir konteyner (YENI PID
namespace'i) yaratir, eski kilitteki `pid` sayisi yeni konteynerde TAMAMEN
ALAKASIZ (ama gercekten canli) bir surece denk gelebilir, `psutil.pid_exists()`
yanlis-pozitif doner. Coz: kilidi yazan `hostname` bizimkiyle FARKLIYSA
(`socket.gethostname()`, Docker'da konteyner kimligi) PID'e HIC bakilmadan
STALE sayilir -- bu dagitimda `pipeline` TEK REPLIKA ve butun stack birlikte
redeploy edildigi icin bu KESIN bir sinyal (TTL tahmini DEGIL). Ayni konteyner
icinde (testler, tek-host senaryo) hostname hep ayni kalir, davranis eskisiyle
BIREBIR ayni.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import Self

import psutil

DEFAULT_LOCK_PATH = Path(tempfile.gettempdir()) / "medrag-nightly.lock"


def _lock_path() -> Path:
    """`NIGHTLY_LOCK_PATH` env degiskeni ile override edilebilir (testler ve
    dagitim ortami icin) -- yoksa sistem gecici dizininde sabit bir dosya."""
    raw = os.environ.get("NIGHTLY_LOCK_PATH")
    return Path(raw) if raw else DEFAULT_LOCK_PATH


class NightlyLockHeld(RuntimeError):
    """Baska bir `medrag-nightly` kosusu zaten calisiyor -- REDDET (I-11'in
    karari; bekleme kuyrugu yok)."""


def _read_lock(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _is_alive(pid: object) -> bool:
    return isinstance(pid, int) and psutil.pid_exists(pid)


def _is_stale(existing: dict) -> bool:
    """2026-08-27 fix: konteynerli dagitimda kilit dosyasi kalici bir yola
    (`/corpus`) tasindiginda, `pid` KENDI konteynerimizin PID namespace'ine
    ait olmayabilir -- redeploy sonrasi YENI bir konteyner sifirdan PID
    sayar, eski kilitteki numara (ornegin 7) yeni konteynerde TAMAMEN
    ALAKASIZ bir surece denk gelip "hala calisiyor" yanilgisina yol acabilir
    (canli testle risk dogrulandi degil, kod okumasiyla tespit edildi).

    Duzeltme: kilidi yazan `hostname` (zaten kayitli) bizim su anki
    `socket.gethostname()`imizle AYNI DEGILSE, farkli bir konteyner
    nesline ait demektir -- bu dagitimda `pipeline` TEK REPLIKA ve butun
    stack birlikte redeploy edildigi icin bu KESIN bir sinyal (TTL tahmini
    degil): eski konteyner artik yok, PID'e hic bakmaya gerek yok. Ayni
    konteyner icinde (testler, tek-host senaryo) hostname hep ayni kalir --
    davranis eskisiyle BIREBIR ayni (`_is_alive(pid)`), geriye donuk uyumlu."""
    if existing.get("hostname") != socket.gethostname():
        return True
    return not _is_alive(existing.get("pid"))


class NightlyLock:
    """Context manager: `with NightlyLock():` -- kilit alinamiyorsa
    `NightlyLockHeld` firlatir. `path` verilmezse `_lock_path()` (env
    degiskeni ya da sistem gecici dizini) kullanilir."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else _lock_path()
        self._acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_lock(self.path)
        if existing is not None:
            pid = existing.get("pid")
            if not _is_stale(existing):
                raise NightlyLockHeld(
                    f"baska bir medrag-nightly kosusu calisiyor (pid={pid}, "
                    f"host={existing.get('hostname')}, "
                    f"baslangic={existing.get('started_at')}) -- "
                    "REDDEDILDI (bekleme kuyrugu yok, I-11)")
            print(f"medrag-nightly: eski kilit STALE (pid={pid}, "
                  f"host={existing.get('hostname')}) -- devraliniyor")
        # write-then-replace: bir yaris durumunda (iki surec ayni anda
        # acquire cagirirsa) sonuncu yazan kazanir -- tek makine icin kabul
        # edilebilir basitlik (gorev metninin "basit tut" tercihi); gercek
        # coklu-makine coordinasyonu Redis kilidi gerektirirdi, kapsam disi.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=".tmp_lock_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }, f)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._acquired = True

    def release(self) -> None:
        if not self._acquired:
            return
        current = _read_lock(self.path)
        # Yalniz KENDI kilidimizi sileriz -- release() cagrilana kadar baska
        # bir surec kilidi STALE sayip devralmis olabilir, o zaman onun
        # kilidini silmek yanlis olur.
        if current is not None and current.get("pid") == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._acquired = False

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False
