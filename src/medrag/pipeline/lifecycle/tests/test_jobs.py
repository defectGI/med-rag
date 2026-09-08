"""jobs.py sözleşmesi: kuyruk sırası, atomik iddia, kesinti toparlama."""

from medrag.pipeline.lifecycle.jobs import (
    Job,
    claim_next,
    enqueue,
    finish_job,
    pending_count,
    recover,
)


def test_enqueue_claim_order_covers_all(tmp_path):
    for ad in ("b", "a"):
        enqueue(tmp_path, action="process", doc_id=ad, rel_path=f"{ad}.pdf")
    # claim -> finish döngüsü TÜM işleri bir kez çıkarır; üçüncüde kuyruk boş.
    alinan = set()
    for _ in range(2):
        job = claim_next(tmp_path)
        assert job is not None
        alinan.add(job.doc_id)
        finish_job(tmp_path, job)
    assert alinan == {"a", "b"}
    assert claim_next(tmp_path) is None


def test_claim_holds_until_finish_then_queue_empty(tmp_path):
    enqueue(tmp_path, action="process", doc_id="d1", rel_path="d1.pdf")
    job = claim_next(tmp_path)
    assert isinstance(job, Job) and job.doc_id == "d1"
    # Tek-kazanan garantisi: iddia edilen iş yeniden ALINAMAZ (.claim
    # dosyası claim_next'ta görülmez; toparlama recover()'un işi).
    assert claim_next(tmp_path) is None
    assert pending_count(tmp_path) == 1
    finish_job(tmp_path, job)
    assert pending_count(tmp_path) == 0
    assert claim_next(tmp_path) is None


def test_crash_recovery_reclaims_leftover_claim(tmp_path):
    job = enqueue(tmp_path, action="delete", doc_id="d2", rel_path="x.pdf")
    # Worker ".claim" aldıktan sonra öldü -> dosya .claim olarak kaldı.
    (tmp_path / f"{job.job_id}.json").rename(tmp_path / f"{job.job_id}.claim")
    # claim_next kalan .claim'i DÖNMEZ (tek-kazanan); recover geri alır.
    assert claim_next(tmp_path) is None
    assert recover(tmp_path) == 1
    yeniden = claim_next(tmp_path)
    assert yeniden is not None and yeniden.doc_id == "d2"
    assert yeniden.action == "delete"


def test_corrupt_job_file_is_skipped_not_fatal(tmp_path):
    (tmp_path / "20260101-000000-bozuk.json").write_text("{bozuk", encoding="utf-8")
    enqueue(tmp_path, action="process", doc_id="d3", rel_path="d3.pdf")
    job = claim_next(tmp_path)
    assert job is not None and job.doc_id == "d3"
    # Bozuk dosya düştü.
    assert not (tmp_path / "20260101-000000-bozuk.claim").exists()


def test_unknown_action_rejected(tmp_path):
    try:
        enqueue(tmp_path, action="patla", doc_id="d", rel_path="x")
    except ValueError:
        return
    raise AssertionError("bilinmeyen eylem ValueError fırlatmalı")


def test_claim_skips_corrupt_claim_file(tmp_path):
    # .claim olarak kalan BOZUK dosya: recover -> .json'a çevirir, claim_next
    # parse edemez -> düşürür ve kuyruğu kilitlemeden sonraki işe geçer.
    enqueue(tmp_path, action="process", doc_id="iyi", rel_path="iyi.pdf")
    (tmp_path / "20260101-000000-bozuk.claim").write_text("{bozuk", encoding="utf-8")
    assert recover(tmp_path) == 1  # bozuk .claim geri alındı
    job = claim_next(tmp_path)
    assert job is not None and job.doc_id == "iyi"
    assert not (tmp_path / "20260101-000000-bozuk.json").exists()
    assert not (tmp_path / "20260101-000000-bozuk.claim").exists()


def test_claim_concurrent_single_winner(tmp_path):
    """İki iş parçacığı aynı işi iddia etmeye çalışırsa OS düzeyinde YALNIZ
    biri kazanır (rename atomikliği)."""
    import threading

    enqueue(tmp_path, action="process", doc_id="tek", rel_path="tek.pdf")
    sonuc = []

    def _claim():
        j = claim_next(tmp_path)
        sonuc.append(j is not None)

    threads = [threading.Thread(target=_claim) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(sonuc) == 1, "atomik iddia yalnız tek kazanan üretmeli"


def test_pending_count_missing_dir_zero(tmp_path):
    assert pending_count(tmp_path / "yok") == 0
