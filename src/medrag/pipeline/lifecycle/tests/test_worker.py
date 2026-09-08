"""worker.py: iş dağıtımı + durum geçişleri + kesinti toparlama.

`process_job` ASLA fırlatmamalı -- istisna `error` durumuna çevrilir, iş
`.claim` dosyası düşürülür (kuyruk canlı kalır). Bu sözleşme tek tek
sahtelenmiş koşucularla test edilir.
"""

import json

import pytest

from medrag.pipeline.lifecycle import jobs as job_queue
from medrag.pipeline.lifecycle import worker
from medrag.pipeline.lifecycle.durum import read_status
from medrag.pipeline.lifecycle.paths import durum_dir


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ISLER_DIR", str(tmp_path / "isler"))
    monkeypatch.setenv("DURUM_DIR", str(tmp_path / "durum"))
    return tmp_path


class _FakeRunner:
    """`worker.process_job`'un içinden çağrılan koşucu arayüzü (A2/A3)."""

    def __init__(self, monkeypatch):
        self.calls: list[tuple[str, str]] = []
        self.process_error: Exception | None = None
        monkeypatch.setattr(worker, "runner", self)

    def _bootstrap_env(self):
        return None

    def process_document(self, doc_id, *, status):
        self.calls.append(("process", doc_id))
        if self.process_error:
            raise self.process_error
        status("ready", detail="bitti")

    def delete_document(self, doc_id, *, status):
        self.calls.append(("delete", doc_id))
        status("parsing", "siliniyor")


def test_process_job_dispatch_process(env, monkeypatch):
    runner = _FakeRunner(monkeypatch)
    job = job_queue.Job(job_id="j1", action="process", doc_id="d1",
                        rel_path="a.pdf", enqueued_at="t")
    worker.process_job(job)
    assert runner.calls == [("process", "d1")]
    assert read_status(durum_dir(), "d1")["state"] == "ready"


def test_process_job_dispatch_delete(env, monkeypatch):
    runner = _FakeRunner(monkeypatch)
    job = job_queue.Job(job_id="j2", action="delete", doc_id="d2",
                        rel_path="b.pdf", enqueued_at="t")
    worker.process_job(job)
    assert runner.calls == [("delete", "d2")]


def test_process_job_error_becomes_status_and_finishes(env, monkeypatch):
    runner = _FakeRunner(monkeypatch)
    runner.process_error = RuntimeError("VLM çöktü")
    # işin .claim dosyası gerçekten kuyrukta olsun ki finish doğrulansın
    (env / "isler").mkdir(parents=True, exist_ok=True)
    (env / "isler" / "j1.claim").write_text(json.dumps(
        {"job_id": "j1", "action": "process", "doc_id": "d1", "rel_path": "a.pdf"}),
        encoding="utf-8")

    job = job_queue.Job(job_id="j1", action="process", doc_id="d1",
                        rel_path="a.pdf", enqueued_at="t")
    # process_job istisnayı yutar, fırlatmaz
    worker.process_job(job)

    durum = read_status(durum_dir(), "d1")
    assert durum["state"] == "error"
    assert "RuntimeError" in durum["detail"]
    assert "VLM çöktü" in durum["detail"]
    # İş düşürüldü -- kuyruk kilitlenmesin
    assert not (env / "isler" / "j1.claim").exists()


def test_process_job_unknown_action_no_crash(env, monkeypatch):
    _FakeRunner(monkeypatch)
    job = job_queue.Job(job_id="j3", action="patlat", doc_id="d3",
                        rel_path="c.pdf", enqueued_at="t")
    worker.process_job(job)  # patlamamalı
    assert read_status(durum_dir(), "d3") is None


def test_main_loop_claims_until_stopped(env, monkeypatch, capsys):
    """main()'in kuyruk döngüsü: iş geldikçe işlenir, bittiğinde durur.

    claim_next iki iş döndürür, üçüncüde SystemExit fırlatarak "dur" sinyali
    verir (SIGTERM handler'ının `stopper.stop=True` etkisini taklit eder)."""
    runner = _FakeRunner(monkeypatch)  # fake `_bootstrap_env` + koşucu
    isler = env / "isler"
    isler.mkdir(parents=True, exist_ok=True)
    jobs = [
        job_queue.enqueue(isler, action="process", doc_id="d1", rel_path="a.pdf"),
        job_queue.enqueue(isler, action="delete", doc_id="d2", rel_path="b.pdf"),
    ]
    kaynak = iter([jobs[0], jobs[1]])

    def _claim(jdir):
        try:
            return next(kaynak)
        except StopIteration:
            raise SystemExit(0)

    monkeypatch.setattr(job_queue, "claim_next", _claim)
    with pytest.raises(SystemExit):
        worker.main()

    assert runner.calls == [("process", "d1"), ("delete", "d2")]


def test_stopper_requests_stop(env):
    stopper = worker._Stopper()
    assert stopper.stop is False
    stopper.request()
    assert stopper.stop is True
