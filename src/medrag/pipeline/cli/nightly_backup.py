"""N-05 (I-40): `medrag-nightly`nin EN BASI -- hicbir yazma, bu modulun basarili
bir yedegi olmadan baslamaz.

K-63 baglantisi: gece kosusu elle duzeltilmis veriyi EZER (KARAR-KAYDI).
Bunun tek korumasi bu yedek -- yani `run_backup()` burada BASARISIZ olursa
`run_nightly.main()` HICBIR `stage_*`'i cagirmamali, gece *basarisiz*
raporlanmali (N-08/N-10'un IntegritySection'ina `backup_completed=False`
olarak dusuyor -- bkz. nightly_report.py'deki TODO notu).

Yedeklenen kalemler (task metninin listesi, I-42 ile ayni):
  1. `specs.db`      -- sqlite3'un CANLI baglanti uzerinden calisan
                        `.backup()` API'siyle (WAL modunda calisan bir DB'yi
                        duz `shutil.copy`/dosya kopyasiyla almak commit
                        edilmemis sayfalari (specs.db-wal) gormeyip BOZUK bir
                        .db uretebilir -- `sqlite3.Connection.backup()` canli
                        baglantidan sayfa sayfa okur, tutarli bir goruntu
                        garantiler).
  2. `document_nodes.json`
  3. `all_chunks.json`
  4. parse cikti dizini (PARSED_OUTPUT_DIR, tum agac -- `shutil.copytree`)
  5. Qdrant koleksiyonu -- snapshot API'siyle (K-63/I-42): BU MAKINEDE Qdrant
     CALISMIYOR (repo-genel kural, bkz. AGENTS.md/CLAUDE.md), bu yuzden bu
     adim GERCEK bir sunucuya bagli TEST EDILMEDI -- yalniz fonksiyon olarak
     yazildi (`backup_qdrant_snapshot`, istemci disaridan enjekte edilebilir,
     `qdrant_store.py`'nin ayni DI desenini aynalar). Gercek dogrulama
     Qdrant'in fiilen calistigi makinede yapilmali (bilinen sinir).
     `QDRANT_URL` tanimsizsa (ya da baglanti/snapshot cagrisi patlarsa) bu
     KALEM yedegi BASARISIZ SAYMAZ -- `skipped=True` ile not dusulur, cunku
     task metni acikca "varsa" diyor (Qdrant'in bu ortamda hic bulunmayan bir
     bagimlilik olabilecegini kabul ediyor); digerleri (1-4) ise ZORUNLU --
     biri bile eksik/basarisizsa `BackupError` firlatilir.

Hedef dizin (N-13/N-15 config/default.toml'a paralel yazdigi icin CAKISMA
riskini azaltmak amaciyla BILINCLI olarak TOML'a EKLENMEDI -- task metninin
acik tercihi): sabit `DEFAULT_BACKUP_ROOT`, `NIGHTLY_BACKUP_ROOT` env
degiskeniyle (testler ve dagitim ortami icin) override edilebilir -- ayni
desen `nightly_lock.py`nin `NIGHTLY_LOCK_PATH`i ile.

N-25: otomatik geri yukleme (restore). Onceki tasarim yalniz yedek ALIYORDU
-- bir asama yarida patlarsa ya da surec `kill -9`/cokme ile YOK OLURSA,
o ana kadar yazilmis (kismi/bozuk) veri diskte KALIYORDU, geri yukleme elle
yapiliyordu. Iki senaryo icin otomatik onarim eklendi:

  1. Tam zincir icinde bir asama YARIDA PATLARSA (surec HALA hayatta,
     exception yakalanabiliyor): `run_nightly._run_full_chain_with_report`
     `restore_backup()`u bu kosunun kendi yedegine (`backup_manifest.root`)
     cagirir, boylece o kosunun yazdigi HER SEY (basarili biten asamalar
     dahil) kosu ONCESI duruma doner -- "yarim ama kismen ilerlemis" bir
     durumda BIRAKMAMAK icin BUTUN kosu geri alinir, tek tek asama bazinda
     KISMI geri alma YAPILMAZ (basit ve ongorulebilir tutmak icin).
  2. Surec `kill -9`/guc kesintisi gibi bir sebeple YOK OLURSA (hicbir
     except/finally calismaz): `mark_in_progress()`/`clear_in_progress()`
     ile bir "devam ediyor" isaretcisi tutulur (bkz. asagi). Bir sonraki
     `medrag-nightly` cagrisi, YENI bir yedek almadan ONCE bu isaretciyi
     kontrol eder; isaretci hala varsa (onceki kosu TEMIZ cikmamis demektir)
     o kosunun yedegine geri doner, SONRA kendi isini yapmaya baslar.

K-63 ile CAKISMAMASI icin BILINCLI sinir: bu otonom geri alma YALNIZ
`medrag-nightly`nin KENDI yazdigi veriyi (bu kosunun/onceki yarim kosunun
yedegini) geri aliyor -- kullanicinin ELLE yaptigi bir duzeltmeyi asla
otomatik EZMEZ, cunku restore HER ZAMAN "bu kosu BASLAMADAN ONCEKI" bir
yedege doner (kosudan ONCE alinan yedek), kosu SONRASI/disi bir zamana
degil. `--stage`/`--from` (elle mudahale, K-60) yollarinda BASARISIZLIK
otomatik restore TETIKLEMEZ (kapsam kasitli dar, bkz. `run_nightly.py`daki
`_run_full_chain_with_report` docstring'i) -- yalniz `kill -9` kurtarmasi
(2. senaryo) bu yollari da kapsar, cunku o senaryo "surec zaten temiz
cikamadi" durumu, elle mudahalenin SONUCUNU degil KESINTIYE UGRAMASINI ele
alir.

Qdrant KAPSAM DISI: backup'taki gibi (server-side snapshot, bu makinede
Qdrant calismiyor) restore de qdrant kalemini ATLAR, sadece not duser --
gercek geri yukleme Qdrant'in calistigi makinede snapshot API'siyle elle
yapilmali (bilinen sinir, backup'taki AYNI kisitlama).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# src/medrag/pipeline/cli/nightly_backup.py -> repo koku (4 seviye yukari,
# run_nightly.py'nin _REPO_ROOT'uyla AYNI konvansiyon).
_REPO_ROOT = Path(__file__).resolve().parents[4]

NIGHTLY_BACKUP_ROOT_ENV = "NIGHTLY_BACKUP_ROOT"
DEFAULT_BACKUP_ROOT = _REPO_ROOT / "nightly_backups"

# N-25: "devam ediyor" isaretcisi -- ayni env-override deseni
# `nightly_lock.py`nin `NIGHTLY_LOCK_PATH`iyla (testler ve dagitim ortami
# icin), sabit varsayilan sistem gecici dizininde.
NIGHTLY_INPROGRESS_PATH_ENV = "NIGHTLY_INPROGRESS_PATH"
DEFAULT_INPROGRESS_PATH = Path(tempfile.gettempdir()) / "medrag-nightly-inprogress.json"


class BackupError(RuntimeError):
    """Yedeklemenin ZORUNLU kalemlerinden biri basarisiz oldu -- coagiran
    (`run_nightly.main`) bu istisnayi yakalayip HICBIR `stage_*`'i
    cagirmamali, gece *basarisiz* raporlanmali (N-05 kurali)."""


class RestoreError(RuntimeError):
    """Geri yuklemenin ZORUNLU kalemlerinden biri basarisiz oldu (N-25).
    Bu, backup HATASINDAN daha ciddi bir durum -- veri artik ne kosu-oncesi
    ne de kosu-sonrasi TUTARLI bir halde olabilir, coagiran bunu YUTMAMALI,
    yuksek sesle (traceback ile) durmali ki elle mudahale edilsin."""


@dataclass
class BackupEntry:
    name: str
    source: str
    destination: str
    size_bytes: int = 0
    skipped: bool = False
    note: str = ""


@dataclass
class BackupManifest:
    root: Path
    started_at: str
    finished_at: str
    duration_seconds: float
    entries: list[BackupEntry] = field(default_factory=list)

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self.entries if not e.skipped)

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "total_size_bytes": self.total_size_bytes,
            "entries": [
                {
                    "name": e.name, "source": e.source, "destination": e.destination,
                    "size_bytes": e.size_bytes, "skipped": e.skipped, "note": e.note,
                }
                for e in self.entries
            ],
        }


def _backup_root(root: Path | str | None) -> Path:
    if root is not None:
        return Path(root)
    raw = os.environ.get(NIGHTLY_BACKUP_ROOT_ENV)
    return Path(raw) if raw else DEFAULT_BACKUP_ROOT


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def _backup_sqlite(src: Path, dest: Path) -> int:
    """bkz. modul docstring'i -- WAL-guvenli SQLite yedegi."""
    if not src.is_file():
        raise BackupError(f"specs.db bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_con = sqlite3.connect(str(src))
    try:
        dest_con = sqlite3.connect(str(dest))
        try:
            src_con.backup(dest_con)
        finally:
            dest_con.close()
    finally:
        src_con.close()
    return dest.stat().st_size


def _backup_file(src: Path, dest: Path) -> int:
    if not src.is_file():
        raise BackupError(f"yedeklenecek dosya bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest.stat().st_size


def _backup_dir(src: Path, dest: Path) -> int:
    if not src.is_dir():
        raise BackupError(f"yedeklenecek dizin bulunamadi: {src}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def _restore_sqlite(backup_src: Path, live_dest: Path) -> None:
    """`_backup_sqlite`in TERSI. Yedek dosyasi zaten `.backup()` API'siyle
    alinmis TUTARLI/duz bir sqlite dosyasi (WAL yok) -- yine de canli hedefte
    bir onceki kosudan kalma `-wal`/`-shm` yan dosyalari olabilir, bunlar
    SILINIR (yoksa yeni yazilan ana dosyayla ESKI wal/shm UYUSMAZ, tutarsiz
    bir goruntu okunabilir); sonra AYNI `.backup()` deseni TERS yonde
    kullanilir (duz dosya kopyasi degil -- N-25'in de N-05'le AYNI WAL-guvenli
    garantiyi tasimasi icin)."""
    if not backup_src.is_file():
        raise RestoreError(f"yedekte specs.db bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("-wal", "-shm"):
        sidecar = live_dest.with_name(live_dest.name + suffix)
        sidecar.unlink(missing_ok=True)
    src_con = sqlite3.connect(str(backup_src))
    try:
        dest_con = sqlite3.connect(str(live_dest))
        try:
            src_con.backup(dest_con)
        finally:
            dest_con.close()
    finally:
        src_con.close()


def _restore_file(backup_src: Path, live_dest: Path) -> None:
    if not backup_src.is_file():
        raise RestoreError(f"yedekte dosya bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_src, live_dest)


def _restore_dir(backup_src: Path, live_dest: Path) -> None:
    if not backup_src.is_dir():
        raise RestoreError(f"yedekte dizin bulunamadi: {backup_src}")
    live_dest.parent.mkdir(parents=True, exist_ok=True)
    if live_dest.exists():
        shutil.rmtree(live_dest)
    shutil.copytree(backup_src, live_dest)


def restore_backup(root: Path | str) -> list[BackupEntry]:
    """N-25: `root` altindaki bir yedegi (manifest.json'daki her kalemin
    `destination`i) ilgili `source` (canli) konumuna GERI YAZAR -- `run_backup`
    ile TAM TERS yonde, ayni manifest.json'u okuyarak (kalemleri ELDE
    UYDURMAK yerine gercekten yedeklenmis olani temel alir).

    `qdrant` kalemi HER ZAMAN atlanir (bkz. modul docstring'i -- server-side
    snapshot, bu makineden geri yuklenemez). ZORUNLU kalemlerden (specs.db/
    document_nodes.json/all_chunks.json/parsed_output_dir) biri manifest'te
    `skipped=True` isaretliyse (backup sirasinda zaten basarisiz olmus
    olmali normalde -- ama boyle bir yedek BackupError firlatip diskte
    kalmamis olmaliydi) ya da geri yukleme sirasinda patlarsa, `RestoreError`
    firlatilir: coagiran bunu YUTMAMALI, veri artik BELIRSIZ bir halde
    olabilir."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RestoreError(f"manifest okunamadi ({manifest_path}): {exc}") from exc

    restored: list[BackupEntry] = []
    for item in raw.get("entries", []):
        name = item["name"]
        source = Path(item["source"])
        destination = Path(item["destination"])
        if name == "qdrant":
            restored.append(BackupEntry(name, str(source), str(destination), skipped=True,
                                         note="qdrant restore desteklenmiyor -- Qdrant'in "
                                              "calistigi makinede snapshot API'siyle elle "
                                              "geri yuklenmeli (bilinen sinir)"))
            continue
        if item.get("skipped"):
            raise RestoreError(f"yedekte '{name}' kalemi zaten basarisiz/eksik isaretli "
                                f"({item.get('note')}) -- geri yukleme guvenli degil")
        if name == "specs.db":
            _restore_sqlite(destination, source)
        elif name == "parsed_output_dir":
            _restore_dir(destination, source)
        else:
            _restore_file(destination, source)
        restored.append(BackupEntry(name, str(source), str(destination),
                                     size_bytes=item.get("size_bytes", 0)))
    return restored


def _inprogress_path(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    raw = os.environ.get(NIGHTLY_INPROGRESS_PATH_ENV)
    return Path(raw) if raw else DEFAULT_INPROGRESS_PATH


def mark_in_progress(backup_root: Path | str, *, path: Path | str | None = None) -> None:
    """N-25: bir kosu, kendi yedegini alip `stage_*`lari cagirmaya BASLAMADAN
    HEMEN ONCE bunu yazar. Surec temiz cikarsa (basarili YA DA yakalanmis bir
    hatayla, ikisinde de `_run_full_chain_with_report`/`main()` `finally`
    icinde `clear_in_progress()` cagirir) bu dosya SILINIR. `kill -9`/cokme
    gibi TEMIZ CIKMAYAN bir kosu bu dosyayi diskte BIRAKIR -- bir sonraki
    cagri bunu 'onceki kosu yarida kaldi, o yedege don' sinyali olarak okur."""
    p = _inprogress_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"backup_root": str(backup_root)}), encoding="utf-8")


def read_in_progress(*, path: Path | str | None = None) -> dict | None:
    p = _inprogress_path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def clear_in_progress(*, path: Path | str | None = None) -> None:
    p = _inprogress_path(path)
    p.unlink(missing_ok=True)


def backup_qdrant_snapshot(
    *, dest_dir: Path, collection: str, url: str | None, api_key: str | None,
    client_factory: Callable[[str, str | None], Any] | None = None,
) -> BackupEntry:
    """Qdrant koleksiyonunun snapshot API'siyle yedegi (I-42). `client_factory`
    testler icin enjekte edilebilir (`qdrant_store.py`nin `from_env`
    classmethod'uyla ayni DI deseni) -- verilmezse gercek `QdrantClient`
    kurulur. HICBIR durumda `BackupError` FIRLATMAZ: url tanimsiz ya da
    baglanti/snapshot cagrisi patlarsa `skipped=True` ile dondurur (task
    metni acikca 'varsa' diyor -- bu tek istisnai/opsiyonel kalem)."""
    if not url:
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            skipped=True, note="QDRANT_URL tanimsiz")
    try:
        if client_factory is not None:
            client = client_factory(url, api_key)
        else:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url, api_key=api_key or None)
        result = client.create_snapshot(collection_name=collection)
        snapshot_name = getattr(result, "name", None) or str(result)
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "snapshot_name.txt").write_text(snapshot_name, encoding="utf-8")
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            note=f"snapshot={snapshot_name} (server-side, sunucuda kaldi)")
    except Exception as exc:  # noqa: BLE001 -- best-effort, bkz. modul docstring'i
        return BackupEntry(name="qdrant", source=collection, destination=str(dest_dir),
                            skipped=True, note=f"snapshot alinamadi: {exc}")


def run_backup(
    *, backup_root: Path | str | None = None,
    qdrant_url: str | None = None, qdrant_api_key: str | None = None,
    qdrant_collection: str | None = None,
    qdrant_client_factory: Callable[[str, str | None], Any] | None = None,
) -> BackupManifest:
    """`run_nightly.main()`in ILK adimi (N-05). Herhangi bir ZORUNLU kalem
    (specs.db/document_nodes.json/all_chunks.json/parse cikti dizini)
    basarisiz olursa `BackupError` firlatir; yarim kalan hedef dizin
    TEMIZLENIR (basarisiz/eksik bir yedek gecerliymis gibi diskte kalmasin
    diye) -- coagiran bunu yakalayip HICBIR asamayi calistirmamali."""
    from medrag.pipeline.cli import run_parse_pipeline
    from medrag.pipeline.facts import discover

    started_monotonic = time.monotonic()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    root = _backup_root(backup_root) / _timestamp()
    entries: list[BackupEntry] = []

    try:
        run_parse_pipeline._bootstrap()

        dest = root / "specs.db"
        size = _backup_sqlite(discover.DB_PATH, dest)
        entries.append(BackupEntry("specs.db", str(discover.DB_PATH), str(dest), size))

        dest = root / "document_nodes.json"
        src = Path(run_parse_pipeline.DOCUMENT_NODES_PATH)
        size = _backup_file(src, dest)
        entries.append(BackupEntry("document_nodes.json", str(src), str(dest), size))

        dest = root / "all_chunks.json"
        size = _backup_file(discover.ALL_CHUNKS_PATH, dest)
        entries.append(BackupEntry("all_chunks.json", str(discover.ALL_CHUNKS_PATH),
                                    str(dest), size))

        dest = root / "parsed_output"
        src = Path(run_parse_pipeline.PARSED_OUTPUT_DIR)
        size = _backup_dir(src, dest)
        entries.append(BackupEntry("parsed_output_dir", str(src), str(dest), size))

        if qdrant_collection is None:
            from medrag.pipeline.vectorize.config import load_config
            qdrant_collection = load_config().qdrant.collection_name
        entries.append(backup_qdrant_snapshot(
            dest_dir=root / "qdrant", collection=qdrant_collection,
            url=qdrant_url if qdrant_url is not None else os.environ.get("QDRANT_URL"),
            api_key=qdrant_api_key if qdrant_api_key is not None
                    else os.environ.get("QDRANT_API_KEY"),
            client_factory=qdrant_client_factory))
    except BackupError:
        shutil.rmtree(root, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(root, ignore_errors=True)
        raise BackupError(f"yedekleme basarisiz: {exc}") from exc

    finished_at = datetime.now().astimezone().isoformat(timespec="seconds")
    duration = time.monotonic() - started_monotonic
    manifest = BackupManifest(root=root, started_at=started_at, finished_at=finished_at,
                               duration_seconds=duration, entries=entries)
    try:
        (root / "manifest.json").write_text(
            json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError:
        pass  # manifest yazimi basarisiz olsa bile yedegin KENDISI zaten tamam
    return manifest
