"""Tek giris noktasi: `medrag-nightly` (N-01).

`scan -> parse -> chunk -> ownership -> facts -> load -> vectorize` sirasini
calistirir (N-22: `ownership`/`load` ikisi de daha once zincire HIC BAGLI
DEGILDI -- `ownership` chunk-sahiplik tablosunu (`catalog_chunk_ownership/
build_all.py`) urettigi icin `facts`TEN ONCE gelir (facts'in cok-sahipli
fan-out daraltmasi bu tabloyu okur), `load` ise facts'in urettigi JSON'lari
`specs.db`ye YAZDIGI icin `facts`TEN SONRA gelir -- bu ikisi baglanmadan
gece kosusu specs.db'ye TEK SATIR YAZMIYORDU, bkz. asagidaki `stage_ownership`/
`stage_load` docstring'leri). Her
asama mevcut giris noktalarinin PROGRAMATIK fonksiyonunu cagirir --
subprocess YOK (bu dosyanin tek isi orkestrasyon, is mantigi zaten var
olan modullerde):

    scan       chatbot-corpus/document_info/classify_documents.py::main()
               (`medrag` paketinin DISINDA -- Grup D'nin tasima kapsamina
               henuz girmedi; kendi dizinindeki duz `config` modulunu ayni
               isimle import ettigi icin normal `import` yerine path-tabanli
               lazy-import + gecici sys.path eklemesi kullanilir, parser'in
               D-39 ONCESI kullandigi desenin ayni -- yalnizca stage_scan()
               gercekten cagrildiginda calisir, modul import zamaninda hicbir
               yan etki yok)
    parse      medrag.pipeline.cli.run_parse_pipeline.main(argv=[])
               (argv=[] ZORUNLU: bos verilmezse argparse orkestratorun kendi
               sys.argv'sini okur ve --stage/--from/--doc'u "bilinmeyen
               bayrak" diye reddeder)
    chunk      medrag.pipeline.cli.run_chunk_pipeline.main() (kendi ici zaten
               `python -m medrag.pipeline.chunker`i subprocess'liyor -- o
               chunker'in KENDI env-only sozlesmesi, bu N-01'in kapsami
               DEGIL; burada degisen sey yalniz ORKESTRATORUN bu scripti
               fonksiyon olarak cagirmasi)
    ownership  medrag.pipeline.facts.catalog_chunk_ownership.build_all.main(argv=[])
               (N-22: chunk'tan SONRA gelir cunku `all_chunks.json`a ihtiyaci
               var, facts'TEN ONCE gelir cunku facts'in cok-sahipli fan-out
               daraltmasi bu asamanin urettigi `chunk_ownership_all.json`i
               okur -- tablo yoksa facts cok-sahipli dokumanlari TAMAMEN
               dislar. `--doc` bu asamada NO-OP: build_all.py dokuman-bazli
               filtre desteklemiyor, HER KOSU tum korpusu (tek+cok-sahipli)
               tarar -- kendi checkpoint/devam mekanizmasi (`chunk_owner_
               progress.jsonl`) var, KARAR-012'nin ikinci surumune benzer.)
    facts      medrag.pipeline.facts.run_full.run_products(...) -- `main()`
               DEGIL: run_full.main() argv'siz `argparse.parse_args()`
               cagirir (orkestratorun kendi argv'sini okurdu). run_full.py'nin
               kendi docstring'i `run_products`u tam olarak bu amacla
               tanimliyor: "Grup N'in gecelik artimli kosusunun (bir doc_id
               degisince ondan etkilenen urunleri yeniden isleme) dogrudan
               cagirabilecegi programatik giris noktasi, CLI'a bagimli
               degil."
    load       medrag.pipeline.facts.load_to_db.main(argv=[...]) (N-22: facts'in
               urettigi `results/<CODE>.json`lari `specs.db`ye YAZAR --
               `run_products` specs.db'ye HICBIR SEY yazmadigi icin bu asama
               baglanmadan gece kosusu DB'de tek satir degistirmiyordu.
               `facts` asamasinin isledigi AYNI urun kumesini `_resolve_
               incremental_codes()` ile TEKRAR cozer (specs.db facts
               sirasinda hic degismedigi icin ayni kume) ve `--models <kume>
               --force` ile cagirir -- bu, `load_product_incremental`
               (O-12) ile TAM ESDEGERDIR (bkz. o fonksiyonun docstring'i).)
    vectorize  medrag.pipeline.vectorize.cli.calistir(...) -- `main()` DEGIL:
               main() yalnizca argv ayristirma + logging.basicConfig +
               .env yukleme sarmalayicisi, gercek is `calistir()`da.

Kural (I-26'nin korunan kabul cumlesi): `--stage`/`--from` ile baslatilan
zincirde plana DAHIL OLMAYAN asamalarin fonksiyonu HIC CAGRILMAZ. Bu modul
`stage_*` fonksiyonlarini `globals()` uzerinden, YALNIZ calisma zamaninda
plana giren adlar icin cozer -- plan disindaki `stage_*` adi bu surec
icinde bir kez bile referans edilmez (bkz. `tests/test_run_nightly.py`,
`--from chunk` ile `stage_scan`/`stage_parse`in hic cagrilmadigini bir
mock ile kanitlar; parse saatler surebilir, sessizce kosarsa fark
edilmez ama GPU'yu isgal eder).

Flags:
    --stage <stage>   run ONLY this stage (manual-intervention shortcut)
    --from <stage>    run this stage through the end of the chain
Environment variables (repo convention: settings in env, see CONFIG.md):
    FACTS_PROCESS_ALL=1    bypass the staleness gate, process ALL products
    FACTS_SKIP_EXISTING=1  manually to resume a half-finished run: skip
                           products whose `results/<CODE>.json` is COMPLETE
                           (DEFAULT OFF -- leaving it on forever would let
                           stale products go untainted)
    FACTS_PENDING_REWORK_PATH  path to the JSONL queue of doc_ids whose
                           facts/load step still has work pending
                           (default `facts/pending_rework.jsonl`)

    --doc <doc_id>    passed as `--doc-id` to the `facts` AND `load` stages
                      (reprocesses the products using that document -- the
                      other stages don't take a doc filter)
                      scan/parse/chunk/ownership/vectorize'in ALTINDAKI
                      mevcut giris noktalari dokuman-bazli filtre
                      DESTEKLEMIYOR (scan: KARAR-001 -- kismi tarama
                      gorulmeyen dosyalari yanlislikla DELETED'e dusurur;
                      parse: butun bekleyen korpusu isler; chunk/vectorize:
                      KARAR-012 ikinci surumu -- her kosu butun kapsami
                      yeniden isler; ownership: build_all.py'nin kendi
                      checkpoint/devam mekanizmasi var, dokuman-bazli filtre
                      yok) -- bu asamalarda `--doc` sessizce yoksayilmaz,
                      bir uyari basilir ve NO-OP olur (bilinen sinir, ileride
                      ayri bir gorev konusu).
    --dry-run         (N-03) hicbir veri yazmaz, ag cagrisi (Ollama/VLM/
                      Qdrant/embedding) yapmaz -- her asama icin GERCEK
                      `stage_*` yerine bir `_dry_run_*` raporlayicisi
                      cagrilir; bu, disk uzerindeki AYNI kaynaklari
                      (document_nodes.json, DOCUMENT_NODES_PATH, chunk
                      girdi klasoru, specs.db, chunk kapsamlari) okuyup
                      "bu asama gercekten kossaydi neyi islerdi" kumesini
                      basar ve durur.

Kullanim:
    python -m medrag.pipeline.cli.run_nightly                  # tam zincir
    python -m medrag.pipeline.cli.run_nightly --from chunk      # chunk->vectorize
    python -m medrag.pipeline.cli.run_nightly --stage facts --doc PN1057
    python -m medrag.pipeline.cli.run_nightly --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path

STAGES = ["scan", "parse", "chunk", "ownership", "facts", "load", "vectorize"]

# src/medrag/pipeline/cli/run_nightly.py -> repo koku (4 seviye yukari, diger
# pipeline scriptleriyle AYNI konvansiyon -- bkz. run_chunk_pipeline.py'nin
# CHUNKER_DIR yorumu).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCUMENT_INFO_DIR = _REPO_ROOT / "chatbot-corpus" / "document_info"
def _load_classify_documents():
    """`classify_documents.py`yi path'ten lazy-import eder. Modul kendi
    dizinindeki duz `config.py`yi `from config import load_config` ile
    import ediyor -- bu yuzden import sirasinda `document_info/` dizini
    gecici olarak `sys.path`e eklenir (yalnizca bu fonksiyon calisirken;
    finally'de geri cikarilir, kalici bir yan etki birakmaz)."""
    if not _DOCUMENT_INFO_DIR.is_dir():
        raise RuntimeError(
            f"chatbot-corpus/document_info bulunamadi: {_DOCUMENT_INFO_DIR}")
    ekliydi = str(_DOCUMENT_INFO_DIR) not in sys.path
    if ekliydi:
        sys.path.insert(0, str(_DOCUMENT_INFO_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "medrag_nightly_classify_documents",
            _DOCUMENT_INFO_DIR / "classify_documents.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if ekliydi:
            sys.path.remove(str(_DOCUMENT_INFO_DIR))


def _doc_noop(stage: str, reason: str, doc_id: str | None) -> None:
    if doc_id:
        print(f"{stage}: --doc no-op ({reason})")


def stage_scan(*, doc_id: str | None = None) -> dict:
    """N-21: `module.main()`in dondurdugu `output` dict (`summary`/
    `documents`) artik ATILMIYOR, cagirana geri veriliyor -- gecelik
    raporun Kaynak farki/Atlananlar bolumlerinin TEK gercek kaynagi bu
    (`_build_nightly_report` -- sayi UYDURULMAZ, scan zaten hesapliyor)."""
    _doc_noop("scan", "classify_documents.py dokuman-bazli filtre "
              "desteklemiyor, KARAR-001 -- tum BELGELER agaci taranir", doc_id)
    module = _load_classify_documents()
    return module.main()


def _read_document_nodes() -> dict | None:
    """K-99 (2026-08-26 fix, takip incelemesi maddesi 6 -- temizlik):
    `document_nodes.json`i okuyan HER yerin (`_forgettable_scan_records`/
    `_incremental_doc_ids`/`_parse_failed_docs`) TEK ortak noktasi -- ucu de
    kendi `_bootstrap()` + "dosya var mi" + `json.loads` KOPYASINI
    tasiyordu. Dosya YOKSA `None` doner (cagiran taraf bos liste/kume
    dondurur).

    BILINCLI OLARAK cache TUTMAZ -- HER CAGRIDA dosyayi YENIDEN okur: ucu de
    kosunun FARKLI zaman noktalarinda cagrilir (`_forgettable_scan_records`:
    parse ONCESI; `_parse_failed_docs`: parse SONRASI; `_incremental_doc_ids`:
    facts asamasinda, ayri bir `stage_*` cagrisinda) ve dosya bu noktalar
    ARASINDA GERCEKTEN degisir (`run_parse_pipeline.main()` her dokumandan
    sonra yazar) -- tek bir onbellek bayat/yanlis veri dondururdu. `run_parse_
    pipeline._bootstrap()`in coktan cozdugu `DOCUMENT_NODES_PATH`i kullanir --
    classify_documents.py'nin kendi `OUTPUT_PATH`i ile AYNI dosyayi gosterir
    (ikisi de scan'in yazdigi/parse'in okudugu tek kayit), ikinci bir
    yol-cozme mantigi ACILMAZ."""
    from medrag.pipeline.cli import run_parse_pipeline

    run_parse_pipeline._bootstrap()
    document_nodes_path = Path(run_parse_pipeline.DOCUMENT_NODES_PATH)
    if not document_nodes_path.is_file():
        return None
    return json.loads(document_nodes_path.read_text(encoding="utf-8"))


def _forgettable_scan_records() -> list[tuple[str, str | None, str, str | None]]:
    """N-19 (K-96 adim 2) + K-97 (2026-08-26 fix) + K-100 (2026-08-27 fix):
    `document_nodes.json`daki `scan_status IN ('DELETED', 'MODIFIED')`
    kayitlarin `(doc_id, doc_type, scan_status, content_hash)` listesi --
    `scan_status` UCUNCU alan olarak tasinir ki cagiran taraf
    (`_forget_deleted_sources`) pending-rework kuyruguna yazabilsin,
    `content_hash` DORDUNCU alan olarak tasinir ki cagiran taraf o hash'i
    `scan.forgotten_hash`e yazip AYNI kaydi bir sonraki gece TEKRAR
    unutmaya kalkismasin (asagidaki K-100 notu).
    `is_active=false` kayitlar BURADA YOK -- onlar "silinmis" degil
    "editoryal olarak devre disi" (farkli niyet, K-96), scan_status alani
    zaten onlari DELETED yapmaz.

    MODIFIED, DELETED KATILDI (eskiden yalniz DELETED isleniyordu): kullanici
    karari -- kaynak DEGISIMI koddaki modellemede "SIL + EKLE" olarak ele
    alinir, "yerinde guncelle" DEGIL. Bir dokuman MODIFIED oldugunda o
    dokumanin ESKI icerigine ait TUM turevler (parse ciktisi, chunk kaydi,
    Qdrant noktalari, spec kaniti) `forget_deleted_source` ile silinir; parse/
    chunk/facts zinciri o gece AYNI kosu icinde YENI icerikten yeniden uretir
    (dokuman "pending" listesinden hic dusmedi, yalniz silinmis DEGISTIRDI --
    bkz. `_forget_deleted_sources`in cagrildigi yer, `stage_parse`in EN BASI).
    MOVED BURAYA DAHIL DEGIL (N-07: bir dosyanin yeri degismesi silme+ekleme
    SAYILMAZ, turevleri korunur).

    K-100 (2026-08-27 fix): `scan_status=DELETED` STICKY'dir (klasor
    tarayicisi asla temizlemez, schema karari) -- bunun dogal sonucu: bu
    filtre HICBIR gate OLMADAN "su an DELETED yazan HER KAYIT" doner, yani
    GECMISTE zaten unutulmus (turevleri cazibasiyla silinmis) kayitlar da
    HER GECE tekrar tekrar secilirdi. Gercek dunyada bunun ikinci bir
    maliyeti var: bir kaynak SADECE GECICI olarak kayboldugunda (ag
    surucusu tasima/kopyalama sirasinda YARIM tarandiginda -- 2026-08-27
    olayi) forget_deleted_source GERCEKTEN calisir, sonra dosya GERI
    donunce (AYNI rel_path + AYNI content_hash) tarayici onu sessizce
    UNCHANGED'e cevirir -- ama `parse.parsed_from_hash` hala AYNI hash'i
    tasidigi icin (forget bu alana hic dokunmuyordu) parse/chunk/facts bir
    daha ASLA yeniden uretmezdi (bkz. `run_parse_pipeline.py`daki
    `parsed_from_hash != content_hash` kapisi). `_forget_deleted_sources`
    artik basariyla unuttugu her kayda `scan.forgotten_hash = content_hash`
    yazip `parse` blogunu PENDING'e resetliyor (bkz. o fonksiyonun notu) --
    bu filtre de `forgotten_hash == content_hash` olan (yani icerigi
    unutuldugundan beri hic degismemis) kayitlari ATLAR. Boylece: (1) ayni
    kayit sonsuza kadar tekrar islenmez (birikme biter), (2) icerik TEKRAR
    degisirse (forgotten_hash artik guncel content_hash'le uyusmaz) dogal
    olarak yeniden secilir, (3) `parse` sifirlandigi icin dosya geri
    donerse (UNCHANGED) bile hash-kapisi dogru tetiklenir (sessiz veri
    kaybi biter)."""
    data = _read_document_nodes()
    if data is None:
        return []
    result: list[tuple[str, str | None, str, str | None]] = []
    for rec in data.get("documents", []):
        scan = rec.get("scan", {})
        status = scan.get("scan_status")
        if status not in ("DELETED", "MODIFIED"):
            continue
        content_hash = scan.get("content_hash")
        forgotten_hash = scan.get("forgotten_hash")
        if forgotten_hash is not None and forgotten_hash == content_hash:
            continue  # K-100: bu icerikle daha once zaten unutuldu, atla
        result.append((rec["identity"]["doc_id"], rec.get("doc_type"), status, content_hash))
    return result


#: K-100 (2026-08-27 fix): `classify_documents.py::blank_pipeline_state()`in
#: `parse` alt-blogunun AYNI kopyasi -- o modul `medrag` paketinin DISINDA
#: (path-tabanli lazy-import gerektirir, bkz. modul dosyasinin EN USTUNDEKI
#: not), burasi icin sadece BU tek blogu tekrar tanimlamak, tum modulu
#: import etmekten daha ucuz VE katman kuralini bozmuyor. Bilerek SADECE
#: `parse` sifirlanir -- `chunk` KARAR-012 geregi zaten HER kosuda tum
#: korpusu yeniden uretir (kendi hash-kapisi yok, sifirlamaya gerek yok),
#: `facts` ise pending-rework kuyrugu uzerinden (asagida) kapsanir.
_BLANK_PARSE_STATE = {
    "parser": None, "parser_version": None, "status": "PENDING",
    "parsed_from_hash": None, "parsed_json_path": None,
    "last_parsed": None, "error": None, "stages": None,
}


def _forget_deleted_sources(*, dry_run: bool = False) -> list:
    """N-19 (K-96 adim 2): `forget_deleted_source`u (N-06, yazilmis ama
    hicbir yerden cagrilmiyordu) gece kosusuna baglar -- scan'DAN SONRA,
    parse'DAN ONCE (`stage_parse`in EN BASI) calisir ki parse gitmis/degismis
    bir dosyayi ESKI turevleriyle bosuna islemeye kalkmasin.

    `dry_run=True`: HICBIR SEY SILMEZ, yalniz hangi doc_id'lerin unutulacagini
    basar (digger `_dry_run_*` raporlayicilarinin AYNI deseni).

    K-100 (2026-08-27 fix, K-98'in yerini alir): gercek yolda (dry_run=False)
    HEM MODIFIED HEM DELETED doc_id'ler `_append_pending_rework_events` ile
    KALICI kuyruga 'pending' yazilir -- eskiden DELETED bilerek DISLANIYORDU
    ("kaynak sonsuza kadar gitti" varsayimiyla), ama 2026-08-27 olayi bu
    varsayimin YANLIS olabildigini gosterdi: bir kaynak GECICI olarak
    (tasima/kopyalama YARIDA kesilirken) DELETED gorunup gercekten
    unutulabilir, sonra AYNI icerikle GERI donebilir. Boyle bir doc_id
    pending-rework kuyrugunda oldugu icin `facts` onu scan_status'u
    UNCHANGED'e donmus olsa bile tekrar dener (bkz. `_incremental_doc_ids`
    ile birleseni, `_settle_pending_rework`). GERCEKTEN sonsuza kadar
    gitmis bir kaynak icin bu ZARARSIZ: `resolve_codes` bos doner,
    `_settle_pending_rework` onu 'orphaned' ile kuyruktan duser (bkz. o
    fonksiyonun docstring'i) -- kendiliginden temizlenir, sonsuza kadar
    yeniden denenmez.

    K-100 (2026-08-27 fix, devami): basariyla unutulan HER kayda
    `document_nodes.json`da `scan.forgotten_hash = content_hash` yazilir
    (bir sonraki gece `_forgettable_scan_records`in AYNI kaydi tekrar
    secmemesi icin, bkz. o fonksiyonun K-100 notu) VE `parse` blogu
    `_BLANK_PARSE_STATE`e resetlenir (dosya GERI donup UNCHANGED
    sayildiginda `parsed_from_hash != content_hash` kapisinin dogru
    tetiklenmesi icin -- forget_deleted_source'un KENDISI bu alanlara HIC
    dokunmuyordu, sessiz veri kaybinin asil kaynagi buydu). Log da ayni
    kosu icinde temizlenir: gercekten bir sey SILINEN kayitlar tek tek
    basilir, hicbir sey bulunamayan (zaten temiz) kayitlar tek bir ozet
    satirda toplanir -- eskiden HER kayit (cogunlugu no-op) ayri satir
    basiyordu, bu da her gece yuzlerce anlamsiz satirla logu bogyordu."""
    from medrag.pipeline.cli import run_parse_pipeline

    forgettable = _forgettable_scan_records()
    if not forgettable:
        print("forget_deleted_source: scan_status=DELETED/MODIFIED kayit yok "
              "(ya da hepsi zaten daha once unutulmustu)")
        return []

    if dry_run:
        doc_ids = [d for d, _, _, _ in forgettable]
        print(f"[dry-run] forget_deleted_source: {len(forgettable)} silinmis/degismis "
              f"kaynak unutulacakti -> {doc_ids[:5]}{', ...' if len(doc_ids) > 5 else ''} "
              "-> YAZILMAYACAK")
        return forgettable

    from medrag.pipeline.facts.discover import ALL_CHUNKS_PATH, DB_PATH
    from medrag.pipeline.forget_deleted_source import forget_deleted_source
    from medrag.pipeline.vectorize.config import load_config as _load_vectorize_config
    from medrag.pipeline.vectorize.store import QdrantVectorStore

    vec_cfg = _load_vectorize_config(os.environ.get("VECTORIZE_CONFIG") or None)
    qdrant_url = (os.environ.get("QDRANT_URL") or "").strip()
    if not qdrant_url:
        raise RuntimeError(
            "forget_deleted_source: QDRANT_URL tanimsiz -- silinen/degisen kaynaklarin "
            "Qdrant temizligi yapilamaz (bkz. .env.example)")
    store = QdrantVectorStore.from_env(
        collection_name=vec_cfg.qdrant.collection_name, distance=vec_cfg.qdrant.distance,
        on_dim_mismatch=vec_cfg.qdrant.on_dim_mismatch,
        upsert_batch_size=vec_cfg.qdrant.upsert_batch_size,
        url=qdrant_url, api_key=os.environ.get("QDRANT_API_KEY") or None)

    data = _read_document_nodes()
    records_by_id = {
        rec["identity"]["doc_id"]: rec for rec in (data.get("documents", []) if data else [])
    }

    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    try:
        results = []
        pending_events = []
        already_clean = 0
        for source_doc_id, doc_type, scan_status, content_hash in forgettable:
            result = forget_deleted_source(
                doc_id=source_doc_id, doc_type=doc_type,
                parsed_output_dir=Path(run_parse_pipeline.PARSED_OUTPUT_DIR),
                all_chunks_path=ALL_CHUNKS_PATH, vector_store=store, specs_con=con,
            )
            results.append(result)
            spec_rows_deleted = result.spec_evidence.get("rows_deleted", 0)
            if result.parse_dirs_removed or result.chunk_entry_removed or spec_rows_deleted:
                print(f"forget_deleted_source: {source_doc_id} -> "
                      f"parse_dirs_removed={len(result.parse_dirs_removed)} "
                      f"chunk_entry_removed={result.chunk_entry_removed} "
                      f"spec_rows_deleted={spec_rows_deleted}")
            else:
                already_clean += 1
            pending_events.append({"doc_id": source_doc_id, "status": "pending"})
            rec = records_by_id.get(source_doc_id)
            if rec is not None:
                rec.setdefault("scan", {})["forgotten_hash"] = content_hash
                rec["parse"] = dict(_BLANK_PARSE_STATE)
        if already_clean:
            print(f"forget_deleted_source: {already_clean} kayit zaten temizdi (no-op, "
                  "tek tek basilmadi)")
        _append_pending_rework_events(pending_events)
        if data is not None:
            run_parse_pipeline._atomic_write_json(run_parse_pipeline.DOCUMENT_NODES_PATH, data)
        return results
    finally:
        con.close()


def stage_parse(*, doc_id: str | None = None) -> dict:
    """N-21: `processed_count`/`duration_seconds` gercekten OLCULUR --
    `run_parse_pipeline.main()` artik islenen dokuman sayisini dondurur
    (once `None` donuyordu, sayi ATILIYORDU); sure bu fonksiyonun kendi
    `time.monotonic()` olcumu (forget_deleted_source adimi HARIC, yalniz
    gercek parse cagrisini kapsar). `model_call_count` BURADA YOK -- parser
    LLM/VLM cagri noktalarinin hicbirinde sayac yok, `nightly_report.py::
    ParseSection.model_call_count`in `None` ("olculmedi") varsayilanina
    dusuyor (N-21 bitti notu -- uydurmak yerine acikca eksik birak).

    N-10 fix (2026-08-26): `failed_docs` -- `run_parse_pipeline.main()`
    TEK TEK dokuman basarisizliklarini (bir sayfada VLM/OCR/LLM patlarsa
    `parse.status='FAILED'` olur) donus degerinde HIC TASIMIYORDU, yalniz
    TOPLAM islenen sayisini donuyordu (`counts` dict'i modul ICINDE
    hesaplanip yalniz PRINT ediliyordu). `_build_nightly_report` bu listeyi
    okuyup her satiri bir `FailureEntry`ye cevirir -- asama kendisi exception
    FIRLATMASA bile (parse `main()` tek tek dokuman hatalarini YUTAR, tum
    korpusu islemeye devam eder) tek bir FAILED dokuman `outcome()`i artik
    `full_success`ten dusurur.

    K-99 fix: `failed_docs` only carries FAILED records whose
    `parse.last_parsed` was UPDATED this run (`since=run_started_at`) --
    previously every FAILED record in `document_nodes.json` (including
    "tombstones" from previous nights -- e.g. a source file later DELETED
    but the record stuck at FAILED forever) entered the report every
    night, so even on a perfectly clean night `outcome()` could NEVER be
    `full_success` -- the INVERSE of what N-10 fixed (the report says
    "always bad", carries no information)."""
    _doc_noop("parse", "run_parse_pipeline.py does not support a doc-level filter "
              "-- the whole pending corpus is processed", doc_id)
    _forget_deleted_sources(dry_run=False)
    from datetime import datetime
    from time import monotonic

    from medrag.pipeline.cli import run_parse_pipeline
    run_started_at = datetime.now().astimezone()
    t0 = monotonic()
    processed_count = run_parse_pipeline.main(argv=[])
    duration_seconds = monotonic() - t0
    return {"processed_count": processed_count, "duration_seconds": duration_seconds,
            "failed_docs": _parse_failed_docs(since=run_started_at)}


def _parse_failed_docs(*, since) -> list[dict]:
    """N-10/K-99 fix (2026-08-26): `document_nodes.json`daki `parse.status ==
    'FAILED'` VE `parse.last_parsed >= since` olan kayitlarin `(doc_id,
    file_name, error)` listesi -- `stage_parse` kosusundan HEMEN SONRA
    cagirilir (`run_parse_pipeline.main()` her dokumandan sonra dosyayi
    diske yazar, bkz. o modulun `_run_phase`'i). `since` (kosu baslangici,
    `datetime.now().astimezone()` -- `run_parse_pipeline.py::_now()` ile
    AYNI tz-aware bicim) filtresi olmadan ONCEKI gecelerden kalma FAILED
    "tombstone" kayitlar (kaynak dosya silinmis, kayit sonsuza kadar FAILED
    kalan bir dokuman) HER GECE rapora girer ve `outcome()` asla `full_
    success` DIYEMEZDI -- bkz. `stage_parse`in K-99 notu. `last_parsed`i
    OLMAYAN (ya da ayristirilamayan) bir FAILED kayit -- gecerlilik suresi
    BILINMEDIGI icin -- GUVENLI TARAFTA kalinir, rapora GIRMEZ."""
    from datetime import datetime

    data = _read_document_nodes()
    if data is None:
        return []
    failed = []
    for rec in data.get("documents", []):
        parse = rec.get("parse", {})
        if parse.get("status") != "FAILED":
            continue
        last_parsed = parse.get("last_parsed")
        if not last_parsed:
            continue
        try:
            when = datetime.fromisoformat(last_parsed)
        except ValueError:
            continue
        if when < since:
            continue
        failed.append({
            "doc_id": rec["identity"]["doc_id"],
            "file_name": rec["identity"].get("file_name", "?"),
            "error": parse.get("error") or "parse.status=FAILED",
        })
    return failed


def _all_chunks_node_counts(path: Path) -> dict[str, int]:
    """`{doc_id: node_sayisi}` -- dosya yoksa/okunamazsa bos dict (KARAR-012:
    her kosu `all_chunks.json`i BASTAN yazar, bu yuzden "once/sonra" farki
    o dosyanin TAMAMININ diff'i anlamina gelir)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {doc_id: len(cs.get("nodes") or []) for doc_id, cs in (data.get("documents") or {}).items()}


def stage_chunk(*, doc_id: str | None = None) -> dict:
    """N-21: `produced_count`/`per_document_counts`/`removed_count` gercek
    OLCUM -- `run_chunk_pipeline.py` (chunker'i subprocess'liyor) bu
    sayilari hic DONDURMUYOR, bu yuzden `all_chunks.json`in kosu ONCESI/
    SONRASI durumu burada karsilastirilir (`resolve_all_chunks_path()` ile
    AYNI dosya, ikinci bir yol-cozme mantigi yok -- bkz. o fonksiyonun
    docstring'i). `produced_count` = kosu SONRASI toplam chunk (node)
    sayisi; `removed_count` = kosu ONCESINDE olup SONRASINDA hic gorunmeyen
    dokumanlarin (silinmis/artik chunk'lanamayan) chunk sayilarinin toplami."""
    _doc_noop("chunk", "run_chunk_pipeline.py her kosuda tum korpusu "
              "yeniden isler (KARAR-012 ikinci surumu)", doc_id)
    from medrag.pipeline.cli import run_chunk_pipeline

    all_chunks_path = run_chunk_pipeline.resolve_all_chunks_path()
    before = _all_chunks_node_counts(all_chunks_path)
    run_chunk_pipeline.main()
    after = _all_chunks_node_counts(all_chunks_path)

    removed_count = sum(count for doc_id_, count in before.items() if doc_id_ not in after)
    return {
        "produced_count": sum(after.values()),
        "per_document_counts": after,
        "removed_count": removed_count,
    }


def _env_flag(name: str) -> bool:
    """`1/true/yes/on` -> True (bosluk ve buyuk-kucuk harf onemsiz). Gece
    kosusunun env kapilarinin (`FACTS_PROCESS_ALL`, `FACTS_SKIP_EXISTING`)
    TEK ayristirma noktasi -- ikisi ayri ayri yazilirsa biri "on"u kabul
    ederken digeri etmeyebilir ve fark operatore SESSIZCE yansir."""
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _incremental_doc_ids() -> list[str]:
    """K-97 (2026-08-26 fix): bu geceki `facts` isinin BIRINCIL KAYNAGI --
    `document_nodes.json`daki `scan_status IN ('NEW', 'MODIFIED')` kayitlarin
    doc_id'leri. `_forgettable_scan_records` ile AYNI dosyayi okur (`_read_
    document_nodes()` uzerinden), farkli bir kume ister: DELETED burada YOK
    (kaynagi gitti, urun uretecek bir sey kalmadi -- `_forget_deleted_sources`
    zaten turevlerini sildi), MOVED burada YOK (N-07: yer degisikligi
    silme+ekleme SAYILMAZ, urun/facts etkilenmez), UNCHANGED burada YOK
    (hicbir sey degismedi -- bu fonksiyonun TUM amaci, degismeyen bir gecede
    facts'in SIFIR urun islemesini saglamak)."""
    data = _read_document_nodes()
    if data is None:
        return []
    return [
        rec["identity"]["doc_id"]
        for rec in data.get("documents", [])
        if rec.get("scan", {}).get("scan_status") in ("NEW", "MODIFIED")
    ]


def _pending_rework_doc_ids() -> list[str]:
    """K-98 (2026-08-26 fix, takip 1): pending-rework kuyrugundaki (append-
    only JSONL, `core/paths.py::resolve_pending_rework_path`) doc_id'lerden
    SON durumu 'pending' olanlar -- `chunk_owner_progress.jsonl` ile AYNI
    disiplin: dosya SATIR SATIR okunur, HER doc_id icin SON kayit kazanir
    (once 'pending' sonra 'done' gorulurse -- ya da tersi -- son yazilan
    gecerli)."""
    from medrag.core.paths import resolve_pending_rework_path

    path = resolve_pending_rework_path()
    if not path.is_file():
        return []
    latest: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[rec["doc_id"]] = rec["status"]
    return sorted(doc_id for doc_id, status in latest.items() if status == "pending")


def _append_pending_rework_events(events: list[dict]) -> None:
    """K-98: pending-rework kuyruguna append-only olay ekler -- dosya HICBIR
    ZAMAN UZERINE YAZILMAZ/KISALTILMAZ (`chunk_owner_progress.jsonl` ile AYNI
    disiplin), okuma tarafi (`_pending_rework_doc_ids`) SON kaydi esas alir."""
    if not events:
        return
    from medrag.core.paths import resolve_pending_rework_path

    path = resolve_pending_rework_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _settle_pending_rework(con, load_report: dict) -> None:
    """K-98 (2026-08-26 fix, takip 1): bu kosuda `load` BASARIYLA biten
    urunlerin ait oldugu pending doc_id'ler kuyruktan duser (append-only
    'done' olayi). Bir doc_id'nin `run_full.resolve_codes(con, doc_id=...)`
    ile cozulen urunlerinden HERHANGI BIRI bu kosuda denenmediyse ya da
    hata ile sonuclandiysa (`load_to_db.main()`in `reports[]`indeki `error`
    field) the doc_id STAYS on the queue -- partial success is NOT
    sufficient, the next night retries. An empty `load_report` (e.g.
    `stage_load` had an empty `codes` and so never called
    `load_to_db.main()`) drops NOTHING.

    Ownerless docs: a pending doc_id whose `resolve_codes` returns empty
    CANNOT be satisfied -- the queue exists to "reprocess this doc's
    products", and if there are none, there's nothing to do. Previously
    `continue`'d, the record stayed on the queue forever: re-tried every
    night with zero work, harmless but the queue grows unbounded. Now
    settled with an 'orphaned' event (NOT 'done' -- no products were
    processed); if the doc's content later changes, scan marks it MODIFIED
    and `_forget_deleted_sources` re-enqueues 'pending', so the protection
    is preserved. `stage_load` calls settle even in the empty-codes branch
    for this reason."""
    from medrag.pipeline.facts import run_full

    pending = _pending_rework_doc_ids()
    if not pending:
        return
    reports = load_report.get("reports") or []
    attempted = {r.get("model_code") for r in reports if r.get("model_code")}
    errored = {r.get("model_code") for r in reports if r.get("error")}

    events = []
    for pending_doc_id in pending:
        products = run_full.resolve_codes(con, doc_id=pending_doc_id)
        if not products:
            events.append({
                "doc_id": pending_doc_id, "status": "orphaned",
                "reason": "resolve_codes bos -- bu dokumana ait urun yok, islenecek sey yok",
            })
            continue
        if all(p in attempted and p not in errored for p in products):
            events.append({"doc_id": pending_doc_id, "status": "done"})
    _append_pending_rework_events(events)


def _resolve_incremental_codes(con, *, doc_id: str | None) -> tuple[list[str], bool]:
    """N-22 + K-97/K-98 (2026-08-26 fix): `--doc` + `FACTS_PROCESS_ALL` kacis
    kapisinin, VE varsayilan (kacis kapisi yok) yolda urun seciminin TEK
    cozumleme noktasi -- `stage_facts` VE `stage_load` (facts'in urettigi
    JSON'lari AYNI urun kumesi icin specs.db'ye yazar) bunu PAYLASIR.
    `run_products` specs.db'ye HICBIR SEY YAZMADIGI icin (bkz. run_full.py
    modul docstring'i) `facts` ile `load` arasinda DB durumu degismez -- iki
    cagri AYNI kumeyi vermek ZORUNDA, bu yuzden kod TEK yerde (iki ayri kopya
    sessizce ayrisirsa `load` facts'in isledigi urunlerden FARKLI bir kumeyi
    yazar).

    Varsayilan yolda (kacis kapisi yok) `codes` UC KAYNAGIN BIRLESIMI:
      1. **Birincil** -- NEW + MODIFIED dokumanlara sahip urunler
         (`_incremental_doc_ids()`, K-97). Buyuk cogunlugu bu karsilar.
      2. **K-98 (bu kosuda EKLENDI, takip 1)** -- pending-rework kuyrugundaki
         doc_id'lere sahip urunler (`_pending_rework_doc_ids()`). K-97'nin
         "bir sonraki gece ayni dokuman HALA MODIFIED gorunur, kendiliginden
         duzelir" varsayimi YANLIS cikti: `classify_documents.py::
         scan_status_for` MODIFIED/UNCHANGED kararini BIR ONCEKI TARAMANIN
         kaydettigi hash'e karsi verir ve tarama YENI hash'i HEMEN kaydeder
         -- yani `_forget_deleted_sources` (parse basi) ile bu fonksiyonun
         cagrildigi `facts`/`load` (zincirin sonlari) arasinda kosu PATLARSA,
         dokuman ERTESI GECE UNCHANGED gorunur ve BIR DAHA HIC SECILMEZ:
         sessiz, KALICI veri kaybi (2026-08-26 kosusu tam bu araliktaki
         `facts` asamasinda durduruldu -- varsayimsal degil). Kuyruk bu
         araligi KOSUDAN SAGKALAN bir yerde tutar (bkz. `_forget_deleted_
         sources`in K-98 notu, `_settle_pending_rework`).
      3. **K-96'nin denetim uclerinin ikincil AG olarak GERI EKLENMESI (bu
         kosuda, takip 2+3)** -- `staleness_audit.stale_product_codes`
         (`dangling_evidence`/`empty_evidence`/`stale_extractor_version`).
         Bu UC OLCUT birlikte "mutabakat agi"dir, BIRINCIL SECICI degildir:
         (a) `stale_extractor_version` olmadan `facts_version` (prompt/sema/
         esik) ilerledikten sonra HICBIR urun yeniden cikarilmiyordu --
         hicbir dokuman degismemis olsa da bu SECICI o kurulumda BOS
         DONMEMELI (regresyon, takip 2). (b) `dangling_evidence`/`empty_
         evidence` olmadan KAYNAK DEGISMEDEN chunker mantigi/surumu degisip
         chunk sha'lari degisirse (dokumanlar UNCHANGED KALIR) evidence artik
         var olmayan chunk'lara isaret eder ve HIC TAZELENMEZ -- bu "yeniden
         bolme kor noktasi" (takip 3). K-97'nin KALDIRDIGI dorduncu kural
         ("kaniti hic bulunmayan yeni chunk'a sahip urunler") BURAYA GERI
         EKLENMEDI -- o, fact URETMEYEN HER chunk'i sonsuza kadar "islenmemis"
         gorup HER GECE ~TUM korpusu (218 urun) isaretleyen, KOKTEN FARKLI ve
         yanlis bir olcuttu (K-97 kok nedeni); `stale_product_codes`in UC
         olcutu ONU icermez, dar ve sinirli kalir.

    Kabul: hicbir dokuman degismedigi, pending-rework kuyrugu bos oldugu VE
    `facts_version`/chunk kumesi degismedigi bir gecede bu fonksiyon BOS
    liste doner, facts SIFIR urun isler.

    Doner: `(codes, process_all)`."""
    from medrag.pipeline.facts import run_full

    process_all = _env_flag("FACTS_PROCESS_ALL")
    if doc_id or process_all:
        return run_full.resolve_codes(con, doc_id=doc_id), process_all

    from medrag.pipeline.facts.staleness_audit import stale_product_codes
    from medrag.pipeline.nightly_report import current_pipeline_versions

    codes: set[str] = set()
    for changed_doc_id in _incremental_doc_ids():
        codes.update(run_full.resolve_codes(con, doc_id=changed_doc_id))
    for pending_doc_id in _pending_rework_doc_ids():
        codes.update(run_full.resolve_codes(con, doc_id=pending_doc_id))
    facts_version = current_pipeline_versions()["facts"]
    codes.update(stale_product_codes(con, current_facts_version=facts_version))
    return sorted(codes), process_all


def stage_ownership(*, doc_id: str | None = None) -> dict | None:
    """N-22: `catalog_chunk_ownership/build_all.py`'nin belgeledigi
    programatik giris noktasi (`main(argv=[])`) uzerinden kosar -- bkz. modul
    docstring'i. Zincire baglanmadan `facts`in cok-sahipli fan-out daraltmasi
    (`chunk_ownership_index`) HICBIR ZAMAN dolmuyordu, katalog/brosur gibi
    dokumanlar TAMAMEN dislaniyordu (`docs_excluded_fanout`).

    `--doc` bu asamada NO-OP: build_all.py dokuman-bazli filtre
    desteklemiyor, HER KOSU tum korpusu (tek+cok-sahipli dokuman) tarar --
    kendi checkpoint/devam mekanizmasi var (Ctrl+C ile kesilirse bir SONRAKI
    kosu kaldigi yerden devam eder, `chunk_ownership_all.json` bozulmadan
    kalir)."""
    _doc_noop("ownership", "build_all.py dokuman-bazli filtre desteklemiyor, "
              "tum korpus (tek+cok-sahipli dokuman) taranir", doc_id)
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    return build_all.main(argv=[])


def stage_facts(*, doc_id: str | None = None) -> dict:
    """`run_full.py`'nin belgeledigi programatik giris noktasi
    (`run_products`) uzerinden kosar -- bkz. modul docstring'i.

    K-97/K-98 (2026-08-26 fix, eski N-20/K-96 adim 3'un YERINE): urun secimi
    `resolve_codes(con)` (specs.db'deki TUM urunler) DEGIL -- NEW+MODIFIED
    dokuman birlesimi (birincil) + pending-rework kuyrugu (K-98) + `staleness_
    audit.stale_product_codes`in UC olcutu (ikincil mutabakat agi) BIRLESIMI
    (bkz. `_resolve_incremental_codes`in docstring'i, TUM gerekce orada).
    **KRITIK KISIT**: bu SADECE hangi urunlerin islenecegini daraltir --
    secilen urun icin yazma yolu DEGISMEZ, `run_products` -> `load_product_
    incremental` yine `_reset_product` (force) ile TAM reset yapar (K-96,
    2026-08-06 olcumu: nokta atisi/kismi silme DENENMEMELI).

    Escape hatches (`--all` from K-96, also kept in K-97): `--doc`
    (manual intervention, always existed) AND `FACTS_PROCESS_ALL=1`
    (env var -- repo convention: settings in env, no new CLI flag) fully
    bypass the gate and revert to the old "all products" behavior.

    `FACTS_SKIP_EXISTING=1` (a SEPARATE gate, affects `run_products`'s
    inner loop, not product selection): products whose
    `results/<CODE>.json` is COMPLETE (not `partial`) are SKIPPED. MUST
    stay OFF by default -- if left on, a complete JSON from a previous
    night also counts as "already there", so a product whose chunk changed
    (looks stale) would never be refreshed (silent staleness, opposite of
    what this gate is for). The only legitimate use is manual resume: if
    the nightly run was killed mid-`facts`, the next run sees the same set
    as stale and would redo all the LLM work.
    yerden surdurmek.

    N-22: urun secimi (`codes`) artik `_resolve_incremental_codes()`
    uzerinden -- `stage_load` (bu asamanin urettigi JSON'lari specs.db'ye
    yazan asama) AYNI fonksiyonu cagirir, iki asama arasinda urun kumesi
    ASLA sessizce ayrismaz."""
    from dotenv import load_dotenv

    from medrag.pipeline.facts import run_full

    load_dotenv(run_full.ENV_PATH)
    provider, model, base_url, api_key = run_full._resolve_llm_env(os.environ)
    system = run_full.build_system_message()
    con = run_full._connect_ro()
    (all_chunks_index, atoms_bridge_index, exclude_doc_types,
     chunk_ownership_index, chunk_anchor_index) = run_full.load_indexes(
        run_full.DEFAULT_ATOMS_BRIDGE)
    owner_counts = run_full.owner_count_by_doc(con)

    codes, process_all = _resolve_incremental_codes(con, doc_id=doc_id)
    if doc_id:
        print(f"facts: --doc-id={doc_id} -> {len(codes)} urun etkileniyor")
    elif process_all:
        print(f"facts: FACTS_PROCESS_ALL=1 -- bayatlik kapisi atlaniyor, "
              f"{len(codes)} urun (TUMU) isleniyor")
    else:
        print(f"facts: bayatlik kapisi -> {len(codes)} urun etkileniyor")

    skip_existing = _env_flag("FACTS_SKIP_EXISTING")
    if skip_existing:
        print("facts: FACTS_SKIP_EXISTING=1 -- TAM (`partial` OLMAYAN) "
              "`results/<CODE>.json` dosyasi olan urunler ATLANIYOR")

    return run_full.run_products(
        codes, con=con, provider=provider, model=model, base_url=base_url,
        api_key=api_key, all_chunks_index=all_chunks_index,
        atoms_bridge_index=atoms_bridge_index,
        exclude_doc_types=exclude_doc_types, owner_counts=owner_counts,
        chunk_ownership_index=chunk_ownership_index,
        chunk_anchor_index=chunk_anchor_index, system=system,
        skip_existing=skip_existing)


def stage_load(*, doc_id: str | None = None) -> dict:
    """N-22: `load_to_db.py`'nin belgeledigi programatik giris noktasi
    (`main(argv=[...])`) uzerinden kosar. Bu asama baglanmadan `stage_facts`
    yalniz `results/<CODE>.json` URETIYORDU -- `run_products` specs.db'ye
    HICBIR SEY YAZMAZ (bkz. o fonksiyonun docstring'i), yani gece kosusu
    DB'de tek satir bile degistirmiyordu.

    `--doc <id>` verilmisse `load_to_db.py --doc-id <id>` ile cagrilir --
    `load_product_incremental` (O-12) yolu, doc_id'yi KULLANAN urunleri
    REPLACE semantigiyle isler (bkz. `load_to_db.py::load_product_
    incremental` docstring'i).

    Verilmemisse: `stage_facts`in ISLEDIGI AYNI urun kumesi `_resolve_
    incremental_codes()` ile YENIDEN cozulur (facts specs.db'ye yazmadigi
    icin DB durumu degismedi, ayni girdi ayni kumeyi verir) ve `--models
    <kume> --force` ile cagirilir. **KRITIK**: `--force`SUZ `load_to_db.py`
    zaten-yuklenmis urunleri "already loaded" diye ATLAR -- facts'in
    bayatlik kapisinden GECEN urunler her gece ZATEN yuklenmis olur, `--force`
    OLMADAN bu asama HICBIR SEY yazmazdi (K-96'nin "TAM reset" ilkesiyle
    AYNI gerekce, bkz. `stage_facts` docstring'i). `--force` + `--models`,
    `load_product_incremental`in doc_id'siz dalinin (`load_product(force=
    True) is EXACTLY EQUIVALENT to `load_product_incremental` (O-12) --
    see that function's docstring. There is NO way to call
    `load_product_incremental` without doc_id.

    If no products are due (the staleness gate selected none),
    `load_to_db.main()` is NEVER called -- empty `--models ""` would error;
    instead `n_products=0` is returned.

    After `load_to_db.main()` returns successfully, `_settle_pending_rework`
    is called -- the pending-rework doc_ids whose products were loaded
    successfully this run drop from the queue (see that function's
    docstring). Both branches (manual `--doc` AND automatic `codes`) do
    this -- both answer the same "did the product load OK?" question.
    The empty-`codes` branch also calls settle
    cagrilir (2026-08-26 gece kosusu incelemesi): sahipsiz pending
    kayitlar ancak boyle duser, `load_to_db.main()`siz bir gecede bile."""
    from medrag.pipeline.facts import load_to_db, run_full

    if doc_id:
        print(f"load: --doc-id={doc_id} -> load_to_db.py'ye devrediliyor")
        result = load_to_db.main(argv=["--doc-id", doc_id])
        _settle_pending_rework(run_full._connect_ro(), result)
        return result

    con = run_full._connect_ro()
    codes, process_all = _resolve_incremental_codes(con, doc_id=None)

    if not codes:
        print("load: islenecek urun yok (facts asamasi hicbir urunu bayat bulmadi)")
        # Sahipsiz pending kayitlari (resolve_codes BOS donen doc'ler) hicbir
        # kosunun `codes` kumesine GIREMEZ -- settle edilmedikleri muddetce
        # kuyrukta sonsuza kadar kalirlar (bkz. `_settle_pending_rework`un
        # sahipsiz-doc notu). Bos `reports` verilir: URUNLU pending doc'ler
        # denenmedikleri icin dogal olarak kuyrukta KALIR, yalniz sahipsizler
        # 'orphaned' ile duser.
        _settle_pending_rework(con, {"reports": []})
        return {"n_products": 0}

    print(f"load: {'FACTS_PROCESS_ALL=1 -- ' if process_all else ''}"
          f"{len(codes)} urun specs.db'ye yazilacak (facts ile AYNI kume)")
    result = load_to_db.main(argv=["--models", ",".join(codes), "--force"])
    _settle_pending_rework(con, result)
    return result


def stage_vectorize(*, doc_id: str | None = None):
    """N-21: donus degeri artik duz `int` DEGIL -- `calistir()` (bkz. o
    fonksiyonun `VectorizeRunStats` dataclass'i) `points_written`/
    `embedding_model`i de tasir, gecelik rapor bunlari GERCEK sayi olarak
    kullanir. `points_deleted`/`total_points_after` o dataclass'ta YOK
    (bilinen sinir, bkz. dataclass docstring'i) -- rapor bu ikisini
    `None` ("olculmedi") yazar.

    2026-08-27 fix: `vectorize_cli.calistir()` kendi `logging.basicConfig`ini
    KURMUYOR (o yalniz `main()`de var, bkz. cli.py) -- `run_nightly.py` bu
    fonksiyonu DOGRUDAN cagirdigi icin (main() degil) INFO/ERROR loglari
    HICBIR YERE yazilmiyordu (2026-08-26 kosusunda `=== vectorize ===`
    basligindan sonra TEK satir bile yoktu -- QDRANT_URL eksik olsa,
    embedder kurulamasa, hic kapsam bulunamasa BILE sessiz kalirdi).
    AYRICA eskiden `VectorizeRunStats.exit_code` HIC KONTROL EDILMIYORDU --
    `exit_code=2` (yapilandirma/baglanti hatasi) ya da `exit_code=1` (bir
    kapsam basarisiz) durumunda bile asama basariyla `return` ediyordu,
    `run()`un `except Exception` yakalamasi (bkz. o fonksiyonun N-21 notu)
    hic tetiklenmiyordu, gece raporu bunu `full_success`un bir PARCASI
    sayiyordu. Artik loglama kurulur VE `exit_code != 0` acikca bir
    istisnaya cevrilir -- boylece asama BASARISIZLIGI raporda `failures[]`
    olarak GORUNUR, sessizce yutulmaz."""
    _doc_noop("vectorize", "kapsam-bazli calisir, dokuman-bazli filtre yok",
              doc_id)
    from medrag.pipeline.vectorize import cli as vectorize_cli
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = vectorize_cli.calistir(env=os.environ)
    if stats.exit_code != 0:
        raise RuntimeError(f"vectorize basarisiz (exit_code={stats.exit_code}) -- "
                           "yukaridaki loglarda ilk ERROR/exception satirina bakin")
    return stats


# --- N-03: kuru kosu raporlayicilari -----------------------------------
#
# Her biri HICBIR DOSYAYA YAZMAZ, AG CAGRISI (Ollama/VLM/Qdrant/embedding)
# YAPMAZ -- yalniz mevcut disk durumunu okuyup "bu asama calissaydi neyi
# islerdi" kumesini hesaplar ve basar. `run()` dry_run=True oldugunda
# `stage_*` yerine bunlari cagirir (bkz. asagidaki `_DRY_RUN_FUNC_NAMES`).


def _dry_run_scan(*, doc_id: str | None = None) -> None:
    _doc_noop("scan", "classify_documents.py dokuman-bazli filtre "
              "desteklemiyor, KARAR-001 -- tum BELGELER agaci taranir", doc_id)
    module = _load_classify_documents()
    # classify_documents.main(dry_run=True): AYNI tarama/eslesme/scan_status
    # diff'i hesaplanir (NEW/UNCHANGED/MODIFIED/MOVED/DELETED), output_path'e
    # HICBIR SEY YAZILMAZ -- N-03'un "gercek kosuyla ayni kumeleri listeler"
    # kabul kriteri budur.
    module.main(dry_run=True)


def _dry_run_parse(*, doc_id: str | None = None) -> None:
    _doc_noop("parse", "run_parse_pipeline.py dokuman-bazli filtre "
              "desteklemiyor, tum bekleyen korpus islenir", doc_id)
    from medrag.pipeline.cli import run_parse_pipeline
    run_parse_pipeline._bootstrap()  # yalniz .env'den yol/DOC_TYPES cozer, ag cagrisi yok
    _forget_deleted_sources(dry_run=True)
    if run_parse_pipeline.DOC_TYPES:
        print(f"parse: doc_type filter: {', '.join(sorted(run_parse_pipeline.DOC_TYPES))}")
    todo, info = run_parse_pipeline.compute_pending()
    print(f"[dry-run] parse: {len(todo)} / {info['total']} dokuman islenecekti "
          f"({', '.join(r['identity']['file_name'] for r in todo[:5])}"
          f"{', ...' if len(todo) > 5 else ''}) -> YAZILMAYACAK")


def _dry_run_chunk(*, doc_id: str | None = None) -> None:
    """KARAR-012 geregi chunk her kosuda TUM korpusu yeniden isler -- "islenecek
    kume" ile "girdi klasorundeki tum dokumanlar" ayni sey. Girdi klasoru
    var olan `run_chunk_pipeline.main()`in kullandigi AYNI iki env
    degiskeninden (`CHUNKER_INPUT_DIR`/`PARSED_OUTPUT_DIR`) cozulur, ama
    subprocess (`python -m medrag.pipeline.chunker`) HIC baslatilmaz."""
    _doc_noop("chunk", "run_chunk_pipeline.py her kosuda tum korpusu "
              "yeniden isler (KARAR-012 ikinci surumu)", doc_id)
    from medrag.pipeline.cli import run_chunk_pipeline
    input_raw = os.environ.get("CHUNKER_INPUT_DIR") or os.environ.get("PARSED_OUTPUT_DIR")
    if not input_raw:
        print("[dry-run] chunk: CHUNKER_INPUT_DIR de PARSED_OUTPUT_DIR de "
              "tanimsiz, hesaplanamadi")
        return
    input_dir = run_chunk_pipeline._resolve(input_raw)
    if not input_dir.is_dir():
        print(f"[dry-run] chunk: girdi klasoru yok: {input_dir}")
        return
    doc_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    print(f"[dry-run] chunk: KARAR-012 geregi butun korpus yeniden islenir "
          f"-> {len(doc_dirs)} dokuman klasoru ({input_dir}) -> YAZILMAYACAK")


def _dry_run_ownership(*, doc_id: str | None = None) -> None:
    """`build_all.split_ownership_work` disinda HICBIR SEY cagirmaz -- o
    fonksiyonun KENDI docstring'i (O-06 kabul kriteri) `_chat_ollama`/
    `_tag_chunk_llm`i HIC IMPORT ETMEDIGINI/CAGIRMADIGINI garanti eder, bu
    yuzden burada LLM/Ollama istegi YAPISAL olarak imkansiz -- ayrica bir
    mock/monkeypatch disiplinine ihtiyac yok."""
    _doc_noop("ownership", "build_all.py dokuman-bazli filtre desteklemiyor, "
              "tum korpus (tek+cok-sahipli dokuman) taranir", doc_id)
    from medrag.pipeline.facts.catalog_chunk_ownership import build_all

    con = build_all._connect_ro()
    try:
        config = build_all.load_config()
        exclude_doc_types = frozenset(config.scope.exclude_doc_types)
        docs = build_all._load_documents(con, exclude_doc_types)
        primary = build_all._load_chunks_from_all_chunks_json()
        fallback = build_all._load_chunk_bridge()

        def _chunks_for(did: str):
            return primary.get(did) or fallback.get(did)

        all_codes: set[str] = set()
        for d in docs:
            all_codes.update(d["product_codes"])
        product_info = build_all._load_product_info(con, all_codes)
        _rows, _missing, llm_jobs, single_docs, multi_docs = build_all.split_ownership_work(
            docs, _chunks_for, product_info, verbose=False)
    finally:
        con.close()

    print(f"[dry-run] ownership: {len(single_docs)} tek-sahipli dokuman (LLM'siz, "
          f"deterministik), {len(multi_docs)} cok-sahipli dokuman -> {len(llm_jobs)} "
          "chunk LLM'e gonderilecekti -> YAZILMAYACAK")


def _dry_run_facts(*, doc_id: str | None = None) -> None:
    """`resolve_codes` disinda HICBIR SEY cagirmaz -- `run_products` (ve
    dolayisiyla LLM/Ollama istegi) hic tetiklenmez."""
    from dotenv import load_dotenv

    from medrag.pipeline.facts import run_full

    load_dotenv(run_full.ENV_PATH)
    con = run_full._connect_ro()
    codes = run_full.resolve_codes(con, doc_id=doc_id)
    print(f"[dry-run] facts: {len(codes)} urun islenecekti"
          + (f" (--doc-id={doc_id})" if doc_id else "")
          + f" ({', '.join(codes[:5])}{', ...' if len(codes) > 5 else ''})"
          + " -> YAZILMAYACAK")


def _dry_run_load(*, doc_id: str | None = None) -> None:
    """`_resolve_incremental_codes` disinda HICBIR SEY cagirmaz -- `load_to_db.
    main` (ve dolayisiyla specs.db'ye GERCEK yazma) hic tetiklenmez. `facts`in
    dry-run raporundan (basitce `resolve_codes`) FARKLI olarak burada K-96
    bayatlik kapisinin GERCEK sonucu kullanilir -- `stage_load`in gercek
    kosuda isleyecegi kumeyi DAHA DOGRU yansitir."""
    from medrag.pipeline.facts import run_full

    if doc_id:
        print(f"[dry-run] load: --doc-id={doc_id} -> load_to_db.py --doc-id "
              "yoluna gidecekti -> YAZILMAYACAK")
        return

    con = run_full._connect_ro()
    codes, process_all = _resolve_incremental_codes(con, doc_id=None)
    print(f"[dry-run] load: {'FACTS_PROCESS_ALL=1 -- ' if process_all else ''}"
          f"{len(codes)} urun specs.db'ye yazilacakti "
          f"({', '.join(codes[:5])}{', ...' if len(codes) > 5 else ''}) -> YAZILMAYACAK")


def _dry_run_vectorize(*, doc_id: str | None = None) -> None:
    """Embedder/Qdrant istemcisi HIC KURULMAZ (ag cagrisi riski) -- yalniz
    diskteki chunk kapsamlari sayilir, hangilerinin bayat oldugu (staleness
    state karsilastirmasi embedder'a bagli oldugu icin) bu raporun kapsami
    DEGIL (bilinen sinir)."""
    _doc_noop("vectorize", "kapsam-bazli calisir, dokuman-bazli filtre yok",
              doc_id)
    from medrag.pipeline.vectorize.discover import discover_chunk_scopes

    input_dir_ham = os.environ.get("VECTORIZE_INPUT_DIR")
    if not input_dir_ham:
        print("[dry-run] vectorize: VECTORIZE_INPUT_DIR tanimsiz, hesaplanamadi")
        return
    burasi = Path(__file__).resolve().parent.parent
    input_dir = (burasi / input_dir_ham).resolve() if not Path(input_dir_ham).is_absolute() \
        else Path(input_dir_ham)
    kapsamlar = discover_chunk_scopes(input_dir)
    print(f"[dry-run] vectorize: {len(kapsamlar)} kapsam bulundu ({input_dir}); "
          "hangisinin bayat oldugu (gercekten islenecek alt kume) embedder "
          "kurulumu gerektirir, bu rapor kapsami DEGIL -> YAZILMAYACAK")


# Ad -> modul-seviyesi fonksiyon adi (fonksiyon NESNESI degil): `run()` bunu
# her cagrida `globals()` uzerinden cozer, boylece testler `run_nightly.
# stage_parse` gibi bir adi monkeypatch/Mock ile degistirdiginde `run()`
# gercekten o degistirilmis surumu gorur (import-zamaninda donmus bir dict
# olsaydi patch gorunmezdi).
_STAGE_FUNC_NAMES = {
    "scan": "stage_scan",
    "parse": "stage_parse",
    "chunk": "stage_chunk",
    "ownership": "stage_ownership",
    "facts": "stage_facts",
    "load": "stage_load",
    "vectorize": "stage_vectorize",
}

# N-03: dry_run=True'da her `stage_*` yerine bunun karsiligi cagrilir --
# ayni `globals()` cozumleme deseni, ayni sebep (testler tekil fonksiyonlari
# monkeypatch edebilsin).
_DRY_RUN_FUNC_NAMES = {
    "scan": "_dry_run_scan",
    "parse": "_dry_run_parse",
    "chunk": "_dry_run_chunk",
    "ownership": "_dry_run_ownership",
    "facts": "_dry_run_facts",
    "load": "_dry_run_load",
    "vectorize": "_dry_run_vectorize",
}


def _stage_index(name: str) -> int:
    try:
        return STAGES.index(name)
    except ValueError:
        raise SystemExit(
            f"bilinmeyen asama: {name!r} (gecerli: {', '.join(STAGES)})"
        ) from None


def plan(*, stage: str | None = None, from_stage: str | None = None) -> list[str]:
    """Calistirilacak asama listesi. `--stage`/`--from` birlikte verilemez
    (biri "yalniz bu", digeri "buradan sona kadar" -- ayni anda ikisi
    anlamsiz)."""
    if stage and from_stage:
        raise SystemExit("--stage ve --from birlikte verilemez")
    if stage:
        _stage_index(stage)
        return [stage]
    if from_stage:
        i = _stage_index(from_stage)
        return STAGES[i:]
    return list(STAGES)


def run(*, stage: str | None = None, from_stage: str | None = None,
        doc_id: str | None = None, dry_run: bool = False,
        collect: dict | None = None) -> list[str]:
    """Orkestratorun govdesi. Planlanan asamalarin listesini dondurur
    (testlerin dogrulamasi icin de kullanisli). Plan disindaki `stage_*`
    fonksiyonlarina bu fonksiyon icinde HICBIR referans yoktur -- yalnizca
    `asamalar` listesindeki adlar `globals()`tan cozulup cagrilir.

    N-21: `collect` (varsayilan `None`) verilirse her tamamlanan asamanin
    donus degeri `collect["stages"][ad]`e yazilir -- gecelik raporun TEK
    gercek veri kaynagi budur (bkz. `_build_nightly_report`). Bir asama
    PATLARSA `collect["failure"]`e (stage/reason/error_text) yazilir ve
    istisna YINE FIRLATILIR (yutulmaz) -- cagiran taraf (`main()`) onu
    yakalayip KISMI raporu yine de yazar, K-62'nin "tavana carpilirsa
    kontrollu durur, kismi raporlanir" niyetini karsilar. `collect=None`
    (varsayilan, mevcut testlerin TAMAMININ kullandigi yol) davranisi HIC
    DEGISTIRMEZ -- istisna dogrudan cagirana firlar, once oldugu gibi."""
    asamalar = plan(stage=stage, from_stage=from_stage)
    print(f"medrag-nightly: plan = {' -> '.join(asamalar)}"
          + (f" (doc={doc_id})" if doc_id else ""))
    if dry_run:
        print("--dry-run: hicbir asama gercekten calistirilmiyor, yalniz "
              "her asamanin isleyecegi kume hesaplanip raporlanacak (N-03)")
        for name in asamalar:
            fn = globals()[_DRY_RUN_FUNC_NAMES[name]]
            print(f"\n=== {name} (dry-run) ===")
            fn(doc_id=doc_id)
        return asamalar
    for name in asamalar:
        fn = globals()[_STAGE_FUNC_NAMES[name]]
        print(f"\n=== {name} ===")
        try:
            result = fn(doc_id=doc_id)
        except Exception as exc:
            if collect is not None:
                collect["failure"] = {
                    "stage": name, "reason": type(exc).__name__, "error_text": str(exc),
                }
            raise
        if collect is not None:
            collect.setdefault("stages", {})[name] = result
    return asamalar


def _scan_paths_by_status(scan_output: dict) -> dict[str, list[str]]:
    """`stage_scan`in dondurdugu `output["documents"]`i `scan_status`a gore
    grupla -- `location.rel_path` (yoksa `identity.file_name`) YOL etiketi
    olarak kullanilir. `scan_output` bos/None ise (scan bu kosuda hic
    calismadi) tum gruplar bos -- cagiran taraf bunu, eslik eden bir
    `FailureEntry` ile birlikte "olculmedi" olarak okur."""
    out: dict[str, list[str]] = {}
    for rec in (scan_output or {}).get("documents", []):
        status = rec.get("scan", {}).get("scan_status")
        etiket = rec.get("location", {}).get("rel_path") or rec.get("identity", {}).get("file_name", "?")
        out.setdefault(status, []).append(etiket)
    return out


def _build_nightly_report(
    *, collect: dict, started_at, finished_at, backup_manifest, restore_note: str = "",
):
    """N-21: `collect` (bkz. `run()`) icindeki HAM asama sonuclarini TEK
    `NightlyReport` nesnesine cevirir. SAYI UYDURMAZ -- bir asama hic
    calismadiysa (collect'te yok) o bolumun sayisal alanlari ya bos/0
    varsayilanina duser (liste/sozluk alanlar -- YALNIZ o asama basarisiz
    olmussa, ki bu durumda asagida MUTLAKA esdeger bir `FailureEntry`
    esclik eder, "0 olcum" ile "hic calismadi" KARISMAZ) ya da modelin
    zaten Optional (`None` = olculmedi) alanlarina duser (bkz.
    `nightly_report.py`deki section docstring'leri -- model_call_count,
    points_deleted/total_points_after HALA HICBIR ZAMAN hesaplanmiyor.
    N-22: `ownership`in ucu VE `product_features.features_added` artik
    `stage_ownership`/`stage_load` zincire baglandiginda GERCEKTEN
    hesaplaniyor (asagidaki ilgili blok) -- `evidence_added`/
    `evidence_removed`/`features_removed_for_no_evidence` ise HALA
    olculmuyor, `forget_source` bu akista kullanilmiyor)."""
    from medrag.pipeline.nightly_report import (
        ChunkSection,
        FailureEntry,
        IntegritySection,
        NightlyReport,
        OwnershipSection,
        ParseSection,
        ProductFeaturesSection,
        SkippedSection,
        SourceDiffSection,
        VectorizeSection,
    )

    stages = collect.get("stages", {})
    failure = collect.get("failure")

    by_status = _scan_paths_by_status(stages.get("scan"))
    source_diff = SourceDiffSection(
        changed=by_status.get("MODIFIED", []), added=by_status.get("NEW", []),
        removed=by_status.get("DELETED", []), moved=by_status.get("MOVED", []),
    )
    skipped = SkippedSection(
        unchanged_count=len(by_status.get("UNCHANGED", [])),
        unchanged_paths=by_status.get("UNCHANGED", []),
    )

    parse_out = stages.get("parse") or {}
    parse = ParseSection(
        processed_count=parse_out.get("processed_count") or 0,
        duration_seconds=parse_out.get("duration_seconds") or 0.0,
        model_call_count=None,
    )

    chunk_out = stages.get("chunk") or {}
    chunk = ChunkSection(
        produced_count=chunk_out.get("produced_count") or 0,
        per_document_counts=chunk_out.get("per_document_counts") or {},
        removed_count=chunk_out.get("removed_count") or 0,
    )

    vec_stats = stages.get("vectorize")
    vectorize = VectorizeSection(
        points_written=getattr(vec_stats, "points_written", 0) or 0,
        points_deleted=None,
        total_points_after=None,
        embedding_model=getattr(vec_stats, "embedding_model", None) or "olculmedi",
    )

    # N-22: `ownership` zincire baglandi -- `stage_ownership`in dondurdugu
    # `build_all.main()` ciktisindan (meta + rows) GERCEK chunk sayilari
    # cikarilir. `stages.get("ownership")` `None`/bos ise (asama hic
    # calismadi, Ctrl+C ile kesildi, ya da `--stage`/`--from` gibi bu asamayi
    # PLANA dahil etmeyen bir yol izlendi) SAYI UYDURULMAZ, ucu de None
    # kalir -- "olculmedi" ile "hic belirsiz chunk yok" KARISTIRILMAZ.
    ownership_out = stages.get("ownership")
    ownership = OwnershipSection(deterministic_count=None, model_routed_count=None,
                                 unresolved_count=None)
    if isinstance(ownership_out, dict) and not ownership_out.get("interrupted") \
            and isinstance(ownership_out.get("rows"), list):
        rows = ownership_out["rows"]
        deterministic_count = sum(1 for r in rows if r.get("source") == "deterministic")
        llm_chunk_ids: set[tuple] = set()
        accepted_chunk_ids: set[tuple] = set()
        for r in rows:
            if r.get("source") != "llm":
                continue
            key = (r.get("doc_id"), r.get("chunk_id"))
            llm_chunk_ids.add(key)
            if r.get("accepted"):
                accepted_chunk_ids.add(key)
        ownership = OwnershipSection(
            deterministic_count=deterministic_count,
            model_routed_count=ownership_out.get("meta", {}).get("n_llm_chunks", len(llm_chunk_ids)),
            unresolved_count=len(llm_chunk_ids - accepted_chunk_ids),
        )

    # N-22: `load` zincire baglandi -- `stage_load`in dondurdugu `load_to_db.
    # main()` ozetindeki `n_present` (bu kosuda `present` durumuyla YAZILAN
    # spec_value satiri sayisi) "eklenen ozellik" (`features_added`) ile
    # DOGRUDAN eslesir (`load_product_incremental` her urunu ONCE
    # not_specified'e DONDURUP SONRA yeniden yazdigi icin bu kosunun
    # `n_present`i o kosuda present OLAN ozellik sayisidir). DIGER UC alan
    # (evidence_added/evidence_removed/features_removed_for_no_evidence)
    # `load_to_db.py`nin raporunda YOK -- `forget_source` (bu akista
    # KULLANILMAYAN, bkz. `load_product_incremental` docstring'i) gerektirir,
    # SAYI UYDURULMAZ, ucu de None kalir.
    load_out = stages.get("load")
    product_features = ProductFeaturesSection(
        features_added=load_out.get("n_present") if isinstance(load_out, dict) else None,
        evidence_added=None, evidence_removed=None,
        features_removed_for_no_evidence=None,
    )

    # N-10 fix (2026-08-26): eskiden `failures` YALNIZ bir asama exception
    # FIRLATIRSA doluyordu -- asama ICINDEKI tek tek basarisizliklar (bir
    # urunun `load_to_db` yazimi patladi, bir dokuman parse FAILED oldu)
    # asama kendisi BASARIYLA DONDUGU icin (exception yok, sadece raporunda
    # hata satiri var) rapora hic ULASMIYORDU -- 2026-08-26 kosusunda 34
    # urun yazilamadi + 1 dokuman parse FAILED oldugu halde rapor `full_
    # success` diyordu. Asama-ICI basarisizliklar burada, top-level exception
    # kontrolunden ONCE, ayri satirlar olarak eklenir.
    failures: list[FailureEntry] = []

    for failed_doc in parse_out.get("failed_docs") or []:
        failures.append(FailureEntry(
            file=failed_doc.get("file_name") or failed_doc.get("doc_id", "-"),
            stage="parse", reason="document_parse_failed",
            error_text=f"{failed_doc.get('doc_id', '?')}: "
                       f"{failed_doc.get('error') or 'parse.status=FAILED'}",
        ))

    for product_report in (stages.get("load") or {}).get("reports") or []:
        if product_report.get("error"):
            failures.append(FailureEntry(
                file=product_report.get("model_code", "-"), stage="load",
                reason="product_write_failed", error_text=product_report["error"],
            ))

    if failure is not None:
        failures.append(FailureEntry(
            file="-", stage=failure["stage"], reason=failure["reason"],
            error_text=failure["error_text"],
        ))
        # Basarisiz asamadan SONRAKI planlanan asamalar hic calismadi --
        # ayri bir kayit, "0 olcum" ile "hic calismadi"yi KARISTIRMASIN.
        idx = STAGES.index(failure["stage"])
        for name in STAGES[idx + 1:]:
            failures.append(FailureEntry(
                file="-", stage=name, reason="skipped_upstream_failure",
                error_text=f"{failure['stage']} basarisiz oldugu icin bu asamaya hic gelinmedi",
            ))

    integrity = IntegritySection(
        # O-08 butunluk kapisi (verify_db_integrity) run_nightly.py'ye HENUZ
        # baglanmadi -- N-05'in ayni durumdaki backup_completed alani icin
        # kullandigi AYNI konvansiyon: False = "henuz entegre edilmedi",
        # "kontrol yapildi ve BASARISIZ oldu" DEGIL (bkz. IntegritySection
        # docstring'i).
        pre_write_check_passed=False, post_write_check_passed=False,
        backup_completed=True, backup_location=str(backup_manifest.root),
        notes=("O-08 butunluk kapisi henuz run_nightly'e baglanmadi (ayri gorev, "
               "N-21 kapsami disi)" + (f" | {restore_note}" if restore_note else "")),
    )

    return NightlyReport(
        run_id=started_at.strftime("%Y-%m-%dT%H-%M-%SZ"),
        started_at=started_at, finished_at=finished_at,
        source_diff=source_diff, skipped=skipped, parse=parse, chunk=chunk,
        vectorize=vectorize, ownership=ownership, product_features=product_features,
        failures=failures, integrity=integrity,
    )


def _nightly_email_subject(report) -> str:
    return f"[medrag-nightly] {report.run_id} -- {report.outcome()}"


def build_nightly_email_summary(report) -> str:
    """Body text -- a few lines of numeric summary, NOT the report (per
    operator's preference: short numeric summary, not a wall of prose).
    The full report is attached as `.md` (see `_send_nightly_email`).

    K-101: `sd.removed` (`scan_status=DELETED`) is STICKY -- it means
    "every record currently DELETED", not "deleted tonight" (may have
    accumulated over previous nights). The label says so explicitly so a
    reader doesn't misread "638 removed" as "638 files deleted tonight" --
    the exact confusion that caused the K-101 incident."""
    sd, sk, ch, ve = report.source_diff, report.skipped, report.chunk, report.vectorize
    pf = report.product_features
    return (
        f"Outcome: {report.outcome()}\n"
        f"Kaynak: +{len(sd.added)} yeni, {len(sd.changed)} degisen, "
        f"{len(sd.removed)} DELETED (sistemde toplam, sticky -- bu gece degil), "
        f"{len(sd.moved)} tasinan, {sk.unchanged_count} degismedi\n"
        f"Chunk: {ch.produced_count} produced, {ch.removed_count} removed\n"
        f"Vector: {ve.points_written} points written\n"
        f"Product features: {pf.features_added if pf.features_added is not None else 'not measured'} added\n"
        f"Failures: {len(report.failures)}\n"
    )


def build_nightly_email(*, report, markdown_text: str, md_path: Path,
                        from_addr: str, to_addrs: list[str]):
    """Builds the `EmailMessage` to send (no network call, pure -- kept
    separate from `_send_nightly_email` so tests can verify without SMTP).

    Body is ALWAYS a short numeric summary (`build_nightly_email_summary`) --
    the full report is NEVER embedded in the body, it's always a `.md`
    attachment."""
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = _nightly_email_subject(report)
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.set_content(build_nightly_email_summary(report))
    msg.add_attachment(markdown_text.encode("utf-8"), maintype="text",
                      subtype="plain", filename=md_path.name)
    return msg


def _send_nightly_email(*, report, markdown_text: str, md_path: Path) -> None:
    """Nightly report email (repo convention: settings in env):

        NIGHTLY_REPORT_EMAIL_TO     comma-separated recipient list. EMPTY/
                                    UNSET -- email is NEVER sent (default off,
                                    no surprise mail).
        NIGHTLY_REPORT_EMAIL_FROM   sender address. Default: nightly@<hostname>.
        SMTP_HOST / SMTP_PORT       default 127.0.0.1 / 25 (local postfix relay).

    Mail gonderimi BASARISIZ olsa da gece kosusunu DUSURMEZ -- rapor zaten
    diske yazildi (JSON+MD), mail sadece bir bildirim kanali; SMTP hatasi
    burada YUTULUR, sadece uyari basilir."""
    to_addrs_raw = (os.environ.get("NIGHTLY_REPORT_EMAIL_TO") or "").strip()
    if not to_addrs_raw:
        print("nightly-email: NIGHTLY_REPORT_EMAIL_TO tanimsiz, mail atlaniyor")
        return
    to_addrs = [a.strip() for a in to_addrs_raw.split(",") if a.strip()]

    import socket

    from_addr = os.environ.get("NIGHTLY_REPORT_EMAIL_FROM") or f"nightly@{socket.gethostname()}"
    smtp_host = os.environ.get("SMTP_HOST") or "127.0.0.1"
    smtp_port = int(os.environ.get("SMTP_PORT") or "25")

    msg = build_nightly_email(
        report=report, markdown_text=markdown_text, md_path=md_path,
        from_addr=from_addr, to_addrs=to_addrs,
    )

    import smtplib

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as smtp:
            smtp.send_message(msg)
        print(f"nightly-email: gonderildi -> {', '.join(to_addrs)} "
              f"(ozet govdede, tam rapor ek olarak -- {md_path.name})")
    except OSError as exc:
        print(f"nightly-email: gonderilemedi ({exc}) -- rapor diskte kaldi ({md_path})")


def _run_full_chain_with_report(*, doc_id: str | None, backup_manifest) -> None:
    """N-21: tam zincir (`--stage`/`--from` YOK) icin `run()`u sarmalar --
    kosu basariyla BITSIN ya da bir asama YARIDA PATLASIN, HER IKI DURUMDA
    da bir rapor (JSON + markdown) yazilir (K-62 niyeti: "tavana carpilirsa
    kontrollu durur, kismi raporlanir" -- bugune kadar bir asama patlayinca
    traceback'le cikiliyordu, HICBIR IZ kalmiyordu).

    Kapsam BILINCLI dar tutuldu: yalniz TAM zincir (`--stage`/`--from` YOK)
    rapor uretir -- K-60'in "elle mudahale" araclarindan (`--stage facts`
    gibi) sonra rapor beklemek bu gorevin (N-21) kapsami DISINDA, bkz. bitti
    notu. `--dry-run` zaten HIC bu fonksiyona GELMEZ (N-03: hicbir veri
    yazmaz, `main()` dry-run'i erken DONER).

    N-25: bir asama YARIDA PATLARSA, rapor yazilmadan ONCE bu kosunun kendi
    yedegine (`backup_manifest.root`) `restore_backup()` ile GERI DONULUR --
    o kosunun o ana kadar yazdigi HER SEY (basariyla biten asamalar dahil)
    geri alinir, boylece diskte "yarim ama ilerlemis" bozuk bir durum
    KALMAZ (bkz. `nightly_backup.py`nin N-25 notu -- tek koruma budur, K-63
    ile cakismaz cunku donulen yedek HER ZAMAN bu kosu BASLAMADAN ONCEKI
    durumdur). Restore'un kendisi patlarsa (`RestoreError`) YUTULMAZ --
    veri artik belirsiz bir halde olabilir, bu YUKSEK SESLE durmali."""
    from datetime import UTC, datetime

    from medrag.core.paths import resolve_reports_dir
    from medrag.pipeline.cli.nightly_backup import restore_backup
    from medrag.pipeline.nightly_report import render_markdown, save_nightly_report

    started_at = datetime.now(UTC)
    collect: dict = {}
    stage_exc: Exception | None = None
    restore_note = ""
    try:
        run(doc_id=doc_id, dry_run=False, collect=collect)
    except Exception as exc:  # noqa: BLE001 -- kismi bile olsa RAPOR yazilsin, sonra yeniden firlatilir
        stage_exc = exc
        print(f"medrag-nightly: asama patladi ({type(exc).__name__}: {exc}) -- "
              f"kosu-oncesi yedege ({backup_manifest.root}) geri donuluyor")
        restored = restore_backup(backup_manifest.root)
        restore_note = (f"asama basarisiz oldugu icin bu kosunun yazdigi veri "
                         f"{backup_manifest.root} yedegine GERI ALINDI "
                         f"({len(restored)} kalem)")
        print(f"medrag-nightly: geri yukleme tamam -- {restore_note}")
    finished_at = datetime.now(UTC)

    report = _build_nightly_report(collect=collect, started_at=started_at,
                                    finished_at=finished_at, backup_manifest=backup_manifest,
                                    restore_note=restore_note)
    reports_dir = resolve_reports_dir()
    json_path = save_nightly_report(report, reports_dir=reports_dir)
    md_path = json_path.with_suffix(".md")
    markdown_text = render_markdown(report)
    md_path.write_text(markdown_text, encoding="utf-8")
    print(f"medrag-nightly: rapor yazildi ({json_path}, {md_path}) -- "
          f"outcome={report.outcome()}")

    # K-101: `build_nightly_email_summary` used to only go to the email body.
    # When the run was triggered via cron's MAILTO (which posts the process's
    # raw stdout without SMTP), the short summary was never visible, and the
    # recipient had to read hundreds of lines of raw log (including the
    # forget_deleted_source dump). The summary is now ALSO printed to the
    # console INDEPENDENTLY of the SMTP path -- whatever channel posts the
    # log, a searchable summary block sits at the very end.
    print("\n=== SUMMARY ===")
    print(build_nightly_email_summary(report).rstrip())
    print("=== END SUMMARY ===\n")

    # Whether the run finishes cleanly OR a stage CRASHES mid-run -- in
    # BOTH cases the email is tried (report is already on disk, before
    # `stage_exc` is re-raised below so failed nights still notify).
    _send_nightly_email(report=report, markdown_text=markdown_text, md_path=md_path)

    if stage_exc is not None:
        # Traceback printed EXPLICITLY: `SystemExit` does NOT print a
        # traceback (Python default), and the `from stage_exc` chain is
        # also invisible. The report's `reason`/`error_text` carries the
        # exception TEXT but not the traceback -- two separate incidents
        # (VLM canary,
        # facts'in mkdir'i) teshisi dogrudan log'daki traceback'in son
        # last line in the log. Losing the traceback means gaining the
        # report but losing diagnosis.
        import traceback
        traceback.print_exception(type(stage_exc), stage_exc, stage_exc.__traceback__)
        raise SystemExit(1) from stage_exc


def _parse_argv(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="medrag-nightly", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=STAGES, default=None,
                    help="run only this stage (manual intervention)")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default=None,
                    help="run this stage through the end of the chain")
    ap.add_argument("--doc", dest="doc_id", default=None,
                    help="only this doc_id (currently only effective in the "
                         "facts stage, passed as --doc-id)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, run no stage")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_argv(argv)
    if args.dry_run:
        # N-03: kuru kosu hicbir veri yazmaz -- kilide (N-02) gerek yok,
        # gercek bir kosuyla AYNI ANDA calismasi zararsizdir.
        run(stage=args.stage, from_stage=args.from_stage, doc_id=args.doc_id,
            dry_run=True)
        return

    from medrag.pipeline.cli.nightly_backup import (
        BackupError,
        RestoreError,
        clear_in_progress,
        mark_in_progress,
        read_in_progress,
        restore_backup,
        run_backup,
    )
    from medrag.pipeline.cli.nightly_lock import NightlyLock, NightlyLockHeld

    # N-21: rapor UZERETICI yalniz TAM zincir (--stage/--from YOK) icin
    # calisir -- bkz. `_run_full_chain_with_report` docstring'i (kapsam
    # kasitli dar, K-60'in elle mudahale araclari rapor beklemez).
    is_full_chain = args.stage is None and args.from_stage is None

    try:
        with NightlyLock() as lock:
            # N-25: isaretci dosyasi kilit dosyasinin YANINA konur (ayni
            # `NIGHTLY_LOCK_PATH` cozumlemesini paylasir) -- ayri bir env
            # degiskeni yonetmeye gerek kalmaz, testlerin zaten set ettigi
            # `NIGHTLY_LOCK_PATH` otomatik olarak isaretciyi de izole eder.
            inprogress_path = lock.path.with_name(lock.path.stem + ".inprogress.json")

            # Onceki kosu TEMIZ CIKMAMISSA (kill -9/cokme -- hicbir
            # except/finally calisamadan surec yok oldu) bir "devam ediyor"
            # isaretcisi diskte kalir (bkz. nightly_backup.py'nin N-25 notu).
            # YENI bir yedek almadan ONCE bunu kontrol et: varsa, o kosunun
            # kendi yedegine geri don -- yoksa bu kosu, bir onceki kosunun
            # YARIM/BOZUK verisinin USTUNE binerdi.
            pending = read_in_progress(path=inprogress_path)
            if pending is not None:
                prev_root = pending.get("backup_root")
                print(f"medrag-nightly: onceki kosu TEMIZ CIKMAMIS (kill/cokme "
                      f"supheli) -- yarim kalan veri, o kosunun yedegine "
                      f"({prev_root}) geri donuluyor")
                restore_backup(prev_root)
                clear_in_progress(path=inprogress_path)
                print("medrag-nightly: onceki yarim kosunun verisi geri yuklendi")

            # N-05 (I-40): hicbir yazma, basarili bir yedek alinmadan
            # BASLAMAZ -- kilit alindiktan HEMEN sonra, ilk `stage_*`den
            # ONCE. Yedek basarisiz olursa (BackupError) hicbir `stage_*`
            # CAGRILMAZ, gece *basarisiz* raporlanmali (K-63'un tek
            # koruması budur -- elle duzeltilmis veriyi ezen gece kosusuna
            # karsi tek guvenlik agi).
            manifest = run_backup()
            print(f"medrag-nightly: yedek tamam ({manifest.root}, "
                  f"{manifest.total_size_bytes} byte, "
                  f"{manifest.duration_seconds:.1f}s)")
            # N-25: `stage_*` cagirmaya BASLAMADAN hemen once isaretcimizi
            # yaz -- surec buradan sonra kill/cokme ile giderse bir sonraki
            # cagri yukarida bu isaretciyi gorup geri yukleme yapar.
            mark_in_progress(manifest.root, path=inprogress_path)
            try:
                if is_full_chain:
                    _run_full_chain_with_report(doc_id=args.doc_id, backup_manifest=manifest)
                else:
                    run(stage=args.stage, from_stage=args.from_stage,
                        doc_id=args.doc_id, dry_run=False)
            except RestoreError:
                # Geri yukleme kendisi PATLADI -- veri belirsiz durumda,
                # isaretci BILEREK diskte birakilir ki bir sonraki cagri
                # kurtarmayi TEKRAR denesin; burada yutmak/temizlemek bu
                # korumayi tek kullanimlik yapardi.
                print("medrag-nightly: GERI YUKLEME BASARISIZ -- veri belirsiz "
                      "durumda, isaretci elle mudahale icin diskte birakildi")
                raise
            except BaseException:
                # Surec HALA hayatta, kontrollu cikiyor (basarili bicimde
                # geri alinmis bir asama hatasi -- SystemExit(1) -- ya da
                # `--stage`/`--from` elle mudahale yolunun kendi hatasi,
                # K-60 kapsaminda otomatik restore YOK ama bu bir cokme
                # DEGIL): isaretciyi temizle, sonra HANGI istisna ise onu
                # ayni sekilde yeniden firlat.
                clear_in_progress(path=inprogress_path)
                raise
            else:
                clear_in_progress(path=inprogress_path)
    except NightlyLockHeld as exc:
        print(f"medrag-nightly: {exc}")
        raise SystemExit(1) from exc
    except BackupError as exc:
        print(f"medrag-nightly: BASARISIZ -- yedek alinamadi, hicbir asama "
              f"calistirilmadi: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
