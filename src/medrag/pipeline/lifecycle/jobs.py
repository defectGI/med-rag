"""Dosya-tabanlı iş kuyruğu (A6): redis'siz, tek worker varsayımıyla.

Kuyruk = bir dizindeki `{job_id}.json` dosyaları. İddia (claim) atomiktir:
dosyanın `.json` -> `.claim` yeniden adlandırmasıyla alınır (aynı işi iki
kez almayı OS düzeyinde imkânsız kılar). İş bitince `.claim` silinir.

Kesinti toparlama: worker ölürse `.claim` dosyası diskte kalır; yeniden
başlayınca `claim_next` önce `.claim` dosyalarını döner -- işler kaybolmaz
(action'lar idempotent olduğu için iki kez koşmak güvenlidir: process
yeniden parse eder, delete yeniden siler ve sessizce no-op olur).

Sıra: job_id zaman damgalıdır (`YYYYmmdd-HHMMSS-<uuid8>`), dosya adına
göre alfabetik sıralama = kuyruğa girme sırası (aynı saniyede uuid
karışabilir; tek kullanıcı için yeterli).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("medrag.pipeline.lifecycle.jobs")

#: Kuyrukta işaretli olabilecek eylemler. api yazar, worker yorumlar.
ACTIONS = ("process", "delete")


@dataclass(frozen=True)
class Job:
    job_id: str
    action: str
    doc_id: str
    rel_path: str
    enqueued_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "action": self.action,
            "doc_id": self.doc_id,
            "rel_path": self.rel_path,
            "enqueued_at": self.enqueued_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        return cls(
            job_id=str(data["job_id"]),
            action=str(data["action"]),
            doc_id=str(data["doc_id"]),
            rel_path=str(data.get("rel_path") or ""),
            enqueued_at=str(data.get("enqueued_at") or ""),
        )


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


def enqueue(jobs_dir: Path, *, action: str, doc_id: str, rel_path: str) -> Job:
    """Yeni iş dosyası yazar; `isler/{job_id}.json`."""
    if action not in ACTIONS:
        raise ValueError(f"bilinmeyen eylem: {action!r} (geçerli: {ACTIONS})")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    job = Job(
        job_id=now.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8],
        action=action,
        doc_id=doc_id,
        rel_path=rel_path,
        enqueued_at=now.isoformat(timespec="seconds"),
    )
    _atomic_write_json(jobs_dir / f"{job.job_id}.json", job.to_dict())
    logger.info("iş kuyruğa girildi: %s %s (%s)", action, doc_id, job.job_id)
    return job


def pending_count(jobs_dir: Path) -> int:
    if not jobs_dir.is_dir():
        return 0
    return sum(1 for p in jobs_dir.iterdir() if p.suffix in (".json", ".claim"))


def _parse_job_file(path: Path) -> Job | None:
    try:
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.warning("okunamayan iş dosyası atlandı: %s", path, exc_info=True)
        return None


def recover(jobs_dir: Path) -> int:
    """Kesinti toparlama: kalan `.claim` dosyalarını `.json`'a geri çevirir.

    Worker BAŞLANGICINDA bir kez çağrılır -- önceki çalışmada yarıda kalan
    işler kuyruğa geri döner (action'lar idempotent). `claim_next` içine
    YERLEŞTİRİLMEMİŞTİR: orada olsaydı `.claim` döndürmek, aynı işin İKİ
    kez iddia edilebilmesine (atomik tek-kazanan garantisinin kırılmasına)
    yol açardı. Dönen değer: geri alınan iş sayısı."""
    if not jobs_dir.is_dir():
        return 0
    n = 0
    for claim in jobs_dir.glob("*.claim"):
        try:
            claim.rename(claim.with_suffix(".json"))
            n += 1
        except OSError:
            continue
    return n


def claim_next(jobs_dir: Path) -> Job | None:
    """Sıradaki işi atomik olarak iddia eder (`.json` -> `.claim`).

    Yalnızca `.json` dosyaları denenir (en eski önce); rename OS düzeyinde
    atomiktir -- aynı işi iki süreç birden alamaz. Okunamayan dosyalar
    silinir (tek bozuk dosya kuyruğu sonsuza dek kilitlemesin). Kalan
    `.claim` dosyaları İDDİA EDİLMEZ -- bunlar `recover()`ın işidir."""
    if not jobs_dir.is_dir():
        return None

    for path in sorted(jobs_dir.glob("*.json")):
        claim = path.with_suffix(".claim")
        try:
            os.rename(path, claim)  # atomik iddia
        except OSError:  # başka bir süreç aldı
            continue
        job = _parse_job_file(claim)
        if job is None:
            claim.unlink(missing_ok=True)
            continue
        return job
    return None


def finish_job(jobs_dir: Path, job: Job) -> None:
    """İşin `.claim` dosyasını siler (idempotent)."""
    (jobs_dir / f"{job.job_id}.claim").unlink(missing_ok=True)


__all__ = ["ACTIONS", "Job", "claim_next", "enqueue", "finish_job", "pending_count", "recover"]
