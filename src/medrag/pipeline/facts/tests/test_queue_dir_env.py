"""`FACTS_QUEUE_DIR` -- `new_key_proposals.jsonl`in dizini env'den gelir.

Gercek olay (2026-08-26): kuyruk `/app/facts/queue`a, yani KONTEYNER ICINE
yaziliyordu ve `.dockerignore` `facts/queue/`yu disliyor -- dizin imajda hic
yok, kosuda yaratiliyor, redeploy'da siliniyordu. O dosya sozlukte OLMAYAN
ama modelin dokumandan gercekten okudugu alanlarin (olcum: bir kosunun ilk
1989 fact'inin 41'i) TEK kanit kaydi; kaybolursa sozlugu buyutecek veri
sessizce yok olur.
"""

from __future__ import annotations

import importlib

from medrag.pipeline.facts import load_to_db


def test_queue_dir_env_ile_override_edilir(monkeypatch, tmp_path):
    hedef = tmp_path / "kalici" / "queue"
    monkeypatch.setenv("FACTS_QUEUE_DIR", str(hedef))
    try:
        yeniden = importlib.reload(load_to_db)
        assert yeniden.QUEUE_DIR == hedef
    finally:
        monkeypatch.delenv("FACTS_QUEUE_DIR", raising=False)
        importlib.reload(load_to_db)


def test_env_yoksa_varsayilan_degismez(monkeypatch):
    """Env verilmeyen kurulumlarda (gelistirme makinesi) davranis AYNI."""
    monkeypatch.delenv("FACTS_QUEUE_DIR", raising=False)
    yeniden = importlib.reload(load_to_db)
    assert yeniden.QUEUE_DIR.name == "queue"
    assert yeniden.QUEUE_DIR.parent.name == "facts"
