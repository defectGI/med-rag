"""durum.py sözleşmesi: yaz/oku/kaldır + geçersiz durum reddi."""

import pytest

from medrag.pipeline.lifecycle.durum import (
    read_all,
    read_status,
    remove_status,
    write_status,
)


def test_write_read_roundtrip(tmp_path):
    write_status(tmp_path, "d1", "parsing", detail="okuma", rel_path="a.pdf")
    data = read_status(tmp_path, "d1")
    assert data["state"] == "parsing"
    assert data["detail"] == "okuma"
    assert data["rel_path"] == "a.pdf"
    assert data["updated_at"]


def test_overwrite_and_read_all(tmp_path):
    write_status(tmp_path, "d1", "queued")
    write_status(tmp_path, "d1", "ready", detail="5 parça")
    write_status(tmp_path, "d2", "error", detail="patladı")
    hepsi = read_all(tmp_path)
    assert hepsi["d1"]["state"] == "ready"
    assert hepsi["d2"]["state"] == "error"


def test_remove_status(tmp_path):
    write_status(tmp_path, "d1", "ready")
    assert read_status(tmp_path, "d1") is not None
    remove_status(tmp_path, "d1")
    assert read_status(tmp_path, "d1") is None
    # Idempotent: ikinci silme patlamaz.
    remove_status(tmp_path, "d1")


def test_invalid_state_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_status(tmp_path, "d1", "uçtu")


def test_read_all_skips_corrupt_and_tmp_files(tmp_path):
    write_status(tmp_path, "d1", "ready")
    (tmp_path / "bozuk.json").write_text("{bozuk", encoding="utf-8")
    (tmp_path / ".tmp_yari.json").write_text("{}", encoding="utf-8")
    hepsi = read_all(tmp_path)
    assert list(hepsi) == ["d1"], "bozuk ve geçici dosyalar sessizce atlanmalı"


def test_read_missing_and_corrupt_returns_none(tmp_path):
    assert read_status(tmp_path, "yok") is None
    (tmp_path / "d1.json").write_text("{bozuk", encoding="utf-8")
    assert read_status(tmp_path, "d1") is None
