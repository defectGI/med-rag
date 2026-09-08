"""backup_cli: med-rag gecelik yedeği -- altyapı yedeği + korpus eki."""

from types import SimpleNamespace

import pytest

from medrag.pipeline.lifecycle import backup_cli, paths


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("NIGHTLY_BACKUP_ROOT", str(tmp_path / "yedek"))
    monkeypatch.setenv("BELGELER_DIR", str(tmp_path / "BELGELER"))
    monkeypatch.setenv("DURUM_DIR", str(tmp_path / "durum"))
    (tmp_path / "BELGELER").mkdir()
    (tmp_path / "durum").mkdir()
    (tmp_path / "durum" / "d1.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_main_runs_base_backup_plus_corpus(env, monkeypatch, capsys):
    import medrag.pipeline.cli.nightly_backup as nb

    calls = {}

    monkeypatch.setattr(nb, "run_backup", lambda **kw: calls.update(kw=kw)
                        or SimpleNamespace(as_dict=lambda: {"a": 1}))
    monkeypatch.setattr(nb, "_backup_root", lambda root: env / "yedek")
    monkeypatch.setattr(nb, "_timestamp", lambda: "20260908")
    monkeypatch.setattr(nb, "_backup_dir",
                        lambda src, dest: 42 if src.name == "BELGELER" else 7)

    rc = backup_cli.main()
    assert rc == 0
    # QDRANT_URL ortamdan alınıp run_backup'a geçirildi
    assert "qdrant_url" in calls["kw"]
    out = capsys.readouterr().out
    assert "yedek tamam" in out


def test_main_skips_missing_dirs(env, monkeypatch):
    import medrag.pipeline.cli.nightly_backup as nb

    monkeypatch.setattr(nb, "run_backup", lambda **kw: SimpleNamespace(as_dict=dict))
    monkeypatch.setattr(nb, "_backup_root", lambda root: env / "yedek")
    monkeypatch.setattr(nb, "_timestamp", lambda: "t")
    monkeypatch.setattr(nb, "_backup_dir", lambda src, dest: 0)
    monkeypatch.setattr(paths, "belgeler_dir", lambda: env / "olmayan")
    monkeypatch.setattr(paths, "durum_dir", lambda: env / "olmayan-durum")
    assert backup_cli.main() == 0  # yok dizin -> atlanır, patlamaz
