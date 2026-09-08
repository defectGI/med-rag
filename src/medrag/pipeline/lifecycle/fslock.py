"""Tek dosyalık danışman kilit: `fcntl.flock` ile registry kilidi.

Registry'yi (document_nodes.json) İKİ yazar günceller: api (upload/silme)
ve worker (parse/chunk blokları). Atomik replace tek başına yeterli değil --
iki yazar aynı an okuyup farklı kopyalar yazarsa birinin kaydı KAYBOLUR.
`flock` Linux'ta süreçler arası karşılıklı dışlama sağlar; kilit dosyası
registry'nin kardeşi `<ad>.lock`tur.

api tarafı AYNI kilidi AYNI yolda alır (aynı sözleşme); kilit dosyası
adı `<registry adı>.lock`.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def lock_path_for(target: Path) -> Path:
    return target.with_name(target.name + ".lock")


@contextmanager
def file_lock(target: Path) -> Iterator[Path]:
    """`target` (korunacak dosyanın KENDİSİ) için süreç-arası münhasır kilit."""
    lp = lock_path_for(target)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield lp
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


__all__ = ["file_lock", "lock_path_for"]
