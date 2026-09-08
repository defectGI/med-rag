"""med-rag gecelik yedeği (E3/K17): `python -m medrag.pipeline.lifecycle.backup`.

İki katman:
  1. `nightly_backup.run_backup` -- registry + all_chunks + parse çıktısı +
     Qdrant snapshot'ı (mevcut altyapı, yeniden yazılmaz).
  2. med-rag eki -- KAYNAK korpus (`BELGELER/`: belgeler + notlar) ve durum
     dosyaları (`durum/`). Kaynak dosyalar yedeğin can damarıdır: türevler
     yeniden üretilebilir ama kaynak 500+ belge yeniden taranamaz.

Kullanım: Coolify scheduled task / cron:
    python -m medrag.pipeline.lifecycle.backup
Yedek kökü: `NIGHTLY_BACKUP_ROOT` (varsayılan `<repo>/nightly_backups`).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("medrag.pipeline.lifecycle.backup")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from medrag.pipeline.cli import nightly_backup
    from medrag.pipeline.lifecycle.paths import belgeler_dir, durum_dir

    manifest = nightly_backup.run_backup(
        qdrant_url=(sys.argv[1] if len(sys.argv) > 1 else None)
        or os.environ.get("QDRANT_URL"),
    )
    logger.info("altyapı yedeği: %s", manifest.as_dict())

    # med-rag eki: kaynak korpus + notlar + durum
    backup_root = nightly_backup._backup_root(os.environ.get("NIGHTLY_BACKUP_ROOT"))
    hedef = backup_root / nightly_backup._timestamp()
    belgeler = belgeler_dir()
    if belgeler.is_dir():
        boyut = nightly_backup._backup_dir(belgeler, hedef / "BELGELER")
        logger.info("korpus+notlar yedeklendi: %s (%d dosya baytı)", belgeler, boyut)
    d = durum_dir()
    if d.is_dir():
        boyut = nightly_backup._backup_dir(d, hedef / "durum")
        logger.info("durum dosyaları yedeklendi (%d bayt)", boyut)
    print(f"yedek tamam: {hedef}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
