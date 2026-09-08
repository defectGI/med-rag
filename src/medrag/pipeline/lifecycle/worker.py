"""İş kuyruğu worker'ı (A6): `python -m medrag.pipeline.lifecycle.worker`.

`ISLER_DIR`deki iş dosyalarını sırayla alır (jobs.claim_next) ve
`runner.process_document`/`runner.delete_document` çalıştırır. Durum
geçişleri `/durum/<doc_id>.json`a yazılır; api bu dosyaları okur.

Eşzamanlılık: TEK iş parçacığı, sıralı koşum. VLM/embedding çağrıları
zaten ağır; 500+ belge ölçeğinde tek sıra öngörülebilir maliyet
verir (PARSED+EMBED eşzamanlılığı karmakarışık hata modları üretir).
Gerekirse ileride kuyruk başına worker sayısı eklenir.

Kesinti toparlama: yarıda kalan `.claim` dosyaları yeniden başlangıçta
yeniden işlenir (action'lar idempotent). İşlem başarısızsa job düşer,
durum `error`a döner -- yeniden deneme kullanıcı eylemidir (dosyayı
yeniden yüklemek A4 semantiğiyle zaten "yeniden işle" demektir).
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from medrag.pipeline.lifecycle import jobs as job_queue
from medrag.pipeline.lifecycle import runner
from medrag.pipeline.lifecycle.durum import write_status
from medrag.pipeline.lifecycle.paths import durum_dir, isler_dir

logger = logging.getLogger("medrag.pipeline.lifecycle.worker")

#: Kuyruk yokluğundaki uyku süresi (sn). Kuyrukta iş varken bekleme yok.
POLL_INTERVAL_SECONDS = 1.0


class _Stopper:
    """SIGTERM/SIGINT -> nazik duruş: koşan iş bitince çık."""

    def __init__(self) -> None:
        self.stop = False

    def request(self, *_args) -> None:
        self.stop = True


def process_job(job: job_queue.Job, *, runner_mod=None) -> None:
    """Tek işi koşturur; istisnayı yakayıp `error` durumuna çevirir
    (worker'ın kendisi ASLA patlamaz -- kuyruk canlı kalır). `runner_mod`
    testler için enjekte edilebilir; verilmezse modül-düzeyindeki `runner`
    kullanılır (çağrı anında çözülür -- monkeypatch'i kıran default-arg
    yok)."""
    runner_mod = runner_mod or runner
    ddir = durum_dir()
    try:
        if job.action == "process":
            write_status(ddir, job.doc_id, "queued", detail="işleniyor",
                         rel_path=job.rel_path)
            runner_mod.process_document(
                job.doc_id,
                status=lambda state, detail=None, **kw:
                    write_status(ddir, job.doc_id, state, detail=detail,
                                 rel_path=job.rel_path)
                    if not kw.get("silent") else None,
            )
        elif job.action == "delete":
            runner_mod.delete_document(
                job.doc_id,
                status=lambda state, detail=None, **kw:
                    write_status(ddir, job.doc_id, state, detail=detail,
                                 rel_path=job.rel_path)
                    if not kw.get("silent") else None,
            )
        else:
            logger.warning("bilinmeyen eylem atlandı: %r (%s)", job.action, job.job_id)
    except Exception as exc:
        logger.exception("iş başarısız: %s %s", job.action, job.doc_id)
        write_status(ddir, job.doc_id, "error",
                     detail=f"{type(exc).__name__}: {exc}",
                     rel_path=job.rel_path)
    finally:
        job_queue.finish_job(isler_dir(), job)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info("=== lifecycle worker başlıyor ===")

    # .env'ler (cli + parser) erken yüklenir: parse aşaması aynı env ile
    # çalışır; eksik zorunlu değişken burada yüksek sesle düşer.
    runner._bootstrap_env()

    stopper = _Stopper()
    signal.signal(signal.SIGTERM, stopper.request)
    signal.signal(signal.SIGINT, stopper.request)

    jdir = isler_dir()
    # Kesinti toparlama: önceki çalışmadan kalan .claim işleri kuyruğa geri
    # döner (action'lar idempotent). Ayrı bir adım -- claim_next'ın
    # tek-kazanan garantisini bozmaz.
    kalan = job_queue.recover(jdir)
    if kalan:
        logger.info("kesinti toparlama: %d iş kuyruğa geri alındı", kalan)
    logger.info("kuyruk: %s | durum: %s", jdir, durum_dir())

    while not stopper.stop:
        job = job_queue.claim_next(jdir)
        if job is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        logger.info("iş alındı: %s %s (%s)", job.action, job.doc_id, job.job_id)
        process_job(job)

    logger.info("worker durdu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
