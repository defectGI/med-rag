"""N-03 (kuru kosu) kilitleme testi: `classify_documents.main(dry_run=True)`
gercek kosuyla AYNI `scan_status` diff'ini (NEW/UNCHANGED/MODIFIED/MOVED/
DELETED) hesaplar ama `document_nodes.json`'a HICBIR SEY YAZMAZ.

Desen `test_move_delete_scenarios.py` ile ayni: gecici PRODUCT_INFO_DIR/
BELGELER_DIR/OUTPUT_PATH env degiskenleri, uretim verisine HIC dokunulmaz.

Calistirma: `pytest chatbot-corpus/document_info/tests/test_dry_run_scan.py`
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

_DOC_INFO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_DOC_INFO_DIR))


def _mini_product_nodes(code: str = "PN1008") -> list[dict]:
    return [{
        "node_id": "n1", "type": "product", "parent_id": None, "source_ref": "1",
        "family": None, "subfamily": None, "subfamily_2": None,
        "product": {"product_code": code, "variant_base": None, "display_name": "X"},
        "category": None,
    }]


def _write_product_nodes(product_info_dir: Path, code: str = "PN1008") -> None:
    product_info_dir.mkdir(parents=True, exist_ok=True)
    (product_info_dir / "product_nodes.json").write_text(
        json.dumps({"product_nodes": _mini_product_nodes(code)}), encoding="utf-8")


def _with_env(product_info_dir: Path, urunler_dir: Path, output_path: Path, fn):
    env_keys = ("PRODUCT_INFO_DIR", "BELGELER_DIR", "OUTPUT_PATH", "DOCUMENT_INFO_CONFIG")
    backup = {k: os.environ.get(k) for k in env_keys}
    os.environ["PRODUCT_INFO_DIR"] = str(product_info_dir)
    os.environ["BELGELER_DIR"] = str(urunler_dir)
    os.environ["OUTPUT_PATH"] = str(output_path)
    os.environ.pop("DOCUMENT_INFO_CONFIG", None)
    try:
        return fn()
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dry_run_writes_nothing_and_matches_real_run_diff(tmp_path):
    import classify_documents as cd

    product_info_dir = tmp_path / "product_info"
    urunler_dir = tmp_path / "BELGELER"
    output_path = tmp_path / "document_nodes.json"
    _write_product_nodes(product_info_dir)
    urunler_dir.mkdir(parents=True)

    rel = "klasor/PN1008 Datasheet.pdf"
    (urunler_dir / "klasor").mkdir()
    (urunler_dir / rel).write_bytes(b"icerik-v1")

    # 1) Ilk (gercek) tarama -- baseline dosyayi olusturur (NEW).
    baseline = _with_env(product_info_dir, urunler_dir, output_path, cd.main)
    assert baseline["summary"]["scan_status_counts"]["NEW"] == 1

    mtime_before = output_path.stat().st_mtime_ns
    sha_before = _sha256(output_path)

    # 2) Kuru kosu: ayni disk durumu -> UNCHANGED diff'i hesaplanmali, ama
    #    dosyaya HICBIR SEY yazilmamali.
    dry = _with_env(product_info_dir, urunler_dir, output_path,
                     lambda: cd.main(dry_run=True))

    assert output_path.stat().st_mtime_ns == mtime_before, \
        "kuru kosu dosyanin mtime'ini DEGISTIRMEMELI"
    assert _sha256(output_path) == sha_before, \
        "kuru kosu dosyanin icerigini DEGISTIRMEMELI"

    # 3) Kuru kosunun hesapladigi kume, gercek bir yeniden-taramanin
    #    uretecegi kumeyle AYNI olmali (kabul kriteri).
    real_rescan = _with_env(product_info_dir, urunler_dir, output_path, cd.main)
    assert dry["summary"]["scan_status_counts"] == real_rescan["summary"]["scan_status_counts"]
    assert dry["summary"]["scan_status_counts"]["UNCHANGED"] == 1
    dry_rel_paths = {d["location"]["rel_path"] for d in dry["documents"]}
    real_rel_paths = {d["location"]["rel_path"] for d in real_rescan["documents"]}
    assert dry_rel_paths == real_rel_paths == {rel}


def test_dry_run_reports_modified_without_writing(tmp_path):
    import classify_documents as cd

    product_info_dir = tmp_path / "product_info"
    urunler_dir = tmp_path / "BELGELER"
    output_path = tmp_path / "document_nodes.json"
    _write_product_nodes(product_info_dir)
    urunler_dir.mkdir(parents=True)

    rel = "klasor/PN1008 Datasheet.pdf"
    (urunler_dir / "klasor").mkdir()
    (urunler_dir / rel).write_bytes(b"icerik-v1")
    _with_env(product_info_dir, urunler_dir, output_path, cd.main)

    # Dosya degisir (MODIFIED bekleniyor).
    (urunler_dir / rel).write_bytes(b"icerik-v2-farkli")

    mtime_before = output_path.stat().st_mtime_ns
    sha_before = _sha256(output_path)

    dry = _with_env(product_info_dir, urunler_dir, output_path,
                     lambda: cd.main(dry_run=True))

    assert dry["summary"]["scan_status_counts"]["MODIFIED"] == 1
    assert output_path.stat().st_mtime_ns == mtime_before
    assert _sha256(output_path) == sha_before
