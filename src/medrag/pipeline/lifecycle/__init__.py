"""Belge yaşam döngüsü (med-rag): upload → parse → chunk → vectorize,
silme → türev temizliği, değişiklik → eski türevler + yeniden işleme.

Bu paket `medrag.api` tarafından ÇAĞRILMAZ (katman kuralı: api ↔ pipeline
bağımsızlığı; aralarındaki tek bağ dosya sistemidir):

  - api, işleri `/corpus/isler/` altına job dosyası olarak yazar
    (`jobs.py` sözleşmesi),
  - `worker.py` (`python -m medrag.pipeline.lifecycle.worker`) bu klasörü
    izleyip işleri sırayla koşar,
  - durum bilgisi `/corpus/durum/<doc_id>.json` dosyalarına yazılır
    (`durum.py` sözleşmesi) -- api bu dosyaları okuyup SSE ile tarayıcıya
    akıtır.

Job dosyası sözleşmesi (api yazar, worker okur):
    {"job_id": "...", "action": "process"|"delete",
     "doc_id": "...", "rel_path": "a/b.pdf", "enqueued_at": "ISO-8601"}

Durum dosyası sözleşmesi (worker yazar, api okur):
    {"doc_id": "...", "state": "queued"|"parsing"|"chunking"|"vectorizing"
     |"ready"|"error", "detail": str|null, "updated_at": ISO-8601, ...}
"""

from medrag.pipeline.lifecycle.durum import (
    read_all,
    read_status,
    remove_status,
    write_status,
)
from medrag.pipeline.lifecycle.jobs import (
    Job,
    claim_next,
    enqueue,
    finish_job,
    pending_count,
)

__all__ = [
    "Job",
    "claim_next",
    "enqueue",
    "finish_job",
    "pending_count",
    "read_all",
    "read_status",
    "remove_status",
    "write_status",
]
