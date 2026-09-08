"""Uçtan uca entrypoint: chunk JSON klasörü → Qdrant koleksiyonu.

Kullanım: ``python -m vectorize`` — yapılandırma env değişkenleriyledir
(chunker/pipeline ile aynı konvansiyon: koşu `.env`/ortamdan yapılandırılır).
İKİ istisna komut satırı bayrağı (chunker'ın `--limit` kararıyla aynı
gerekçe: "ne kadarını koşayım" kalıcı bir ayar değil):

    --limit N   en fazla N kapsam işlenir (keşif sırasına göre ilk N;
                deneme/örnekleme için).
    --force     bayatlık kapısını (yerel state.json) yok sayar, TÜM
                kapsamları yeniden embed eder (tek seferlik bypass).

    VECTORIZE_INPUT_DIR   zorunlu — chunk JSON'larının kökü (rekürsif taranır;
                          `discover.py`, hem eski per-doküman hem yeni
                          all_chunks/all_raptor/all_combined yerleşimini tanır)
    VECTORIZE_OUTPUT_DIR  default ./storage — yerel state.json + run_log.json
                          (gerçek çıktı Qdrant koleksiyonudur, bkz.
                          storage/README.md)
    VECTORIZE_CONFIG      opsiyonel — default.toml üstüne kısmi override TOML
    VECTORIZE_PROGRESS    on | off (default on) — tqdm ilerleme çubuğu
    EMBEDDING_*           sağlayıcı bağlantısı (bkz. .env.example)
    QDRANT_URL/QDRANT_API_KEY  bağlantı (bkz. .env.example)

Bayatlık kapısı: her kapsamın imzası (`core.ChunkSet.signature`) yerel
state.json'a yazılır; bir sonraki koşuda aynı imzalı kapsam atlanır
(config/default.toml `[staleness] skip_unchanged`, varsayılan açık —
embedding gerçek bir uzak LLM/GPU çağrısıdır, chunker'ın "her koşu tüm
korpusu yeniden üret" kararının (KARAR-012) aksine burada maliyet önemli).

Kapsam güncelleme kontratı: bir kapsam (yeniden) embed edildiğinde önce
Qdrant'taki o kapsama ait TÜM eski noktalar silinir, sonra yeni set baştan
yazılır (`store.replace_scope` — gerekçe: chunk node_id'leri set-kapsamlıdır,
KARAR-004; "yalnız değişeni upsert et" yanlış eşleşme üretir).

Hata modeli: kapsam başına hata loglanır, koşu diğer kapsamlarla sürer.
Çıkış kodu 0 = hepsi tamam, 1 = en az bir kapsam başarısız, 2 = yapılandırma
hatası (eksik env, hiç kapsam bulunamaması dahil — boş girdi klasörü
neredeyse her zaman yanlış yol demektir).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from medrag.pipeline.vectorize.config import VectorizeConfig, load_config
from medrag.pipeline.vectorize.core import ChunkNode, ChunkSet
from medrag.pipeline.vectorize.discover import ChunkScope, discover_chunk_scopes
from medrag.pipeline.vectorize.embedder import ProviderError, embedder_from_env
from medrag.pipeline.vectorize.layout import OutputLayout
from medrag.pipeline.vectorize.state import is_unchanged, load_state, save_state
from medrag.pipeline.vectorize.store import QdrantVectorStore, VectorPoint

log = logging.getLogger("vectorize.cli")

try:
    from tqdm import tqdm as _tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm as _redirect_tqdm
except ImportError:
    _tqdm = None
    _redirect_tqdm = None


def _progress_enabled(env: Mapping[str, str]) -> bool:
    acik = (env.get("VECTORIZE_PROGRESS") or "on").strip().lower() \
        not in ("off", "0", "false")
    return acik and _tqdm is not None and sys.stderr.isatty()


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def _select_nodes(chunk_set: ChunkSet, *, include_summary_nodes: bool) -> list[ChunkNode]:
    secilen = []
    for node in chunk_set.nodes:
        if not node.text.strip():
            continue
        if node.tree_level > 0 and not include_summary_nodes:
            continue
        secilen.append(node)
    return secilen


def _node_payload(node: ChunkNode, chunk_set: ChunkSet) -> dict:
    # `chunk_schema_version`/`chunk_generated_at` (PROTOCOL KARAR-067):
    # evidence chunk'ın HANGİ chunk üretiminden geldiği yanıt zincirinin
    # (retrieval metadata -> chatbot conversation log) sonuna kadar izlenebilir
    # olmalı -- ikisi de zaten chunker'ın ChunkSet'inde var, burada sadece
    # payload'a AKTARILIYOR (yeni bir alan icat edilmiyor).
    prov = chunk_set.provenance
    return {
        "node_id": node.node_id,
        "doc_id": chunk_set.doc_id,
        "scope": chunk_set.scope,
        "tree_level": node.tree_level,
        "text": node.text,
        "keywords": node.keywords,
        "heading_path": node.heading_path,
        "page_start": node.page_start,
        "page_end": node.page_end,
        "source_path": node.source_path,
        "fmt": node.fmt,
        "images": [img.model_dump() for img in node.images],
        "chunk_schema_version": chunk_set.schema_version,
        "chunker_version": prov.chunker_version if prov else None,
        "chunk_generated_at": prov.generated_at if prov else None,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m vectorize")
    parser.add_argument("--limit", type=int, default=None,
                        help="en fazla N kapsam işle (keşif sırasına göre ilk N)")
    parser.add_argument("--force", action="store_true",
                        help="bayatlık kapısını yok say, tüm kapsamları yeniden embed et")
    return parser.parse_args(argv)


def _embed_scope(kapsam: ChunkScope, *, cfg: VectorizeConfig, embedder,
                 store: QdrantVectorStore) -> int:
    """Bir kapsamı embed edip Qdrant'a yazar, o kapsamdaki nokta sayısını
    döndürür (0 dahil — metinsiz/boş kapsam yalnız eski noktaları siler)."""
    nodes = _select_nodes(kapsam.chunk_set,
                          include_summary_nodes=cfg.embedding.include_summary_nodes)
    if not nodes:
        store.replace_scope(kapsam.scope_id, [])
        return 0

    vektorler: list[list[float]] = []
    for i in range(0, len(nodes), cfg.embedding.batch_size):
        parca = nodes[i:i + cfg.embedding.batch_size]
        metinler = [cfg.embed_text(heading_path=n.heading_path, text=n.text)
                    for n in parca]
        vektorler.extend(embedder.embed(metinler))

    store.ensure_collection(vector_size=len(vektorler[0]))
    points = [VectorPoint(node_id=n.node_id, vector=v,
                          payload=_node_payload(n, kapsam.chunk_set))
              for n, v in zip(nodes, vektorler)]
    store.replace_scope(kapsam.scope_id, points)
    return len(points)


@dataclass
class VectorizeRunStats:
    """N-21: `calistir()`in ESKİ tek `int` (çıkış kodu) dönüşü, gecelik
    raporun `VectorizeSection`ına yazılabilecek gerçek sayıları TAŞIMIYORDU
    -- `basarili`/`nokta_sayisi` yalnız `log.info` ile YAZDIRILIYOR, hiçbir
    yere DÖNDÜRÜLMÜYORDU. `main()` hâlâ düz `int` döner (CLI çıkış kodu
    sözleşmesi DEĞİŞMEDİ, `main().exit_code` DEĞİL) -- yalnız `calistir()`i
    doğrudan çağıran taraf (`run_nightly.py::stage_vectorize`) bu zengin
    nesneyi görür.

    `points_deleted`/`total_points_after` burada YOK -- ikisi de bu
    fonksiyonun yerel sayaçlarından gelmiyor: `store.replace_scope` silinen
    nokta sayısını hiç döndürmüyor, "son toplam" ise gerçek bir Qdrant
    `count()` ağ çağrısı ister (bu makinenin sahte `_SahteStore`'ları bunu
    desteklemiyor). `nightly_report.py::VectorizeSection` bu ikisini
    `None` ("ölçülmedi") kabul eder -- bkz. o modülün docstring'i."""

    exit_code: int
    points_written: int = 0
    embedding_model: str | None = None


def calistir(env: Mapping[str, str] = os.environ, *, limit: int | None = None,
             force: bool = False) -> VectorizeRunStats:
    """Koşunun tamamı; `env` injectable (testler sözlük verir). `limit`/`force`
    `main`'in argv ayrıştırmasından geçirilir. `VectorizeRunStats` döner
    (bkz. o dataclass'ın docstring'i) -- `main()` yalnız `.exit_code`ini
    kullanır, CLI çıkış kodu sözleşmesi DEĞİŞMEDİ."""
    burasi = Path(__file__).resolve().parent.parent

    input_dir_ham = env.get("VECTORIZE_INPUT_DIR")
    if not input_dir_ham:
        log.error("VECTORIZE_INPUT_DIR tanımsız (bkz. .env.example)")
        return VectorizeRunStats(exit_code=2)

    try:
        cfg = load_config(env.get("VECTORIZE_CONFIG") or None)
    except Exception:
        log.exception("yapılandırma yüklenemedi")
        return VectorizeRunStats(exit_code=2)

    try:
        embedder = embedder_from_env(timeout=cfg.embedding.timeout_seconds)
    except ProviderError as exc:
        log.error("embedding sağlayıcısı kurulamadı: %s", exc)
        return VectorizeRunStats(exit_code=2)

    qdrant_url = (env.get("QDRANT_URL") or "").strip()
    if not qdrant_url:
        log.error("QDRANT_URL tanımsız (bkz. .env.example)")
        return VectorizeRunStats(exit_code=2)

    try:
        store = QdrantVectorStore.from_env(
            collection_name=cfg.qdrant.collection_name, distance=cfg.qdrant.distance,
            on_dim_mismatch=cfg.qdrant.on_dim_mismatch,
            upsert_batch_size=cfg.qdrant.upsert_batch_size,
            url=qdrant_url, api_key=env.get("QDRANT_API_KEY") or None)
    except Exception:
        log.exception("qdrant istemcisi kurulamadı")
        return VectorizeRunStats(exit_code=2)

    input_dir = _resolve(input_dir_ham, burasi)
    try:
        kapsamlar = discover_chunk_scopes(input_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return VectorizeRunStats(exit_code=2)

    if not kapsamlar:
        log.error("%s altında hiç chunk kapsamı bulunamadı — girdi yolu "
                  "yanlış olabilir (bkz. VECTORIZE_INPUT_DIR)", input_dir)
        return VectorizeRunStats(exit_code=2)

    if limit is not None:
        kapsamlar = kapsamlar[:limit]

    out = OutputLayout(_resolve(env.get("VECTORIZE_OUTPUT_DIR") or "./storage", burasi))
    out.ensure()
    state = load_state(out.state_file)

    progress = _progress_enabled(env)
    dongu = (_tqdm(kapsamlar, desc="vectorize", unit="scope", dynamic_ncols=True)
             if progress else kapsamlar)
    ctx = _redirect_tqdm() if progress else nullcontext()

    basarili = hatali = atlandi = 0
    toplam_nokta_yazildi = 0
    with ctx:
        for kapsam in dongu:
            imza = kapsam.chunk_set.signature(embedding_model=embedder.name)
            if not force and cfg.staleness.skip_unchanged and \
                    is_unchanged(state, kapsam.scope_id, imza):
                log.info("%s değişmemiş, atlanıyor (--force ile bypass edilir)",
                         kapsam.scope_id)
                atlandi += 1
                continue
            try:
                nokta_sayisi = _embed_scope(kapsam, cfg=cfg, embedder=embedder, store=store)
                state[kapsam.scope_id] = imza
                basarili += 1
                toplam_nokta_yazildi += nokta_sayisi
                log.info("%s → %d nokta", kapsam.scope_id, nokta_sayisi)
                # N-04: her kapsamdan SONRA anında kaydedilir (aşama-sonu
                # commit noktası) -- eskiden `save_state` yalnız döngü
                # BİTİNCE bir kez çağrılıyordu, yani koşu 50 kapsamın
                # 49'unu Qdrant'a yazıp tam da 50.de kesilirse state.json
                # hâlâ BOŞTU: bir sonraki koşu o 49 kapsamı da (gereksiz
                # ama YANLIŞ değil) yeniden embed ediyordu -- "kaldığı
                # yerden devam" kabul kriterini karşılamıyordu. `save_state`
                # zaten write-then-replace (atomic), bu yüzden döngü
                # içinde sık çağırmak güvenli.
                save_state(state, out.state_file)
            except Exception:
                log.exception("kapsam işlenemedi: %s (%s)", kapsam.scope_id,
                              kapsam.source_file)
                hatali += 1

    save_state(state, out.state_file)
    log.info("bitti: %d başarılı, %d atlandı, %d hatalı", basarili, atlandi, hatali)
    return VectorizeRunStats(exit_code=1 if hatali else 0,
                             points_written=toplam_nokta_yazildi,
                             embedding_model=embedder.name)


def main(argv: list[str] | None = None) -> int:
    """``python -m vectorize`` girişi: log + argv + .env, sonra `calistir`."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")  # .env yoksa no-op; tanımlı gerçek env değişkenini ezmez

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return calistir(env=os.environ, limit=args.limit, force=args.force).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
