"""End-to-end entrypoint: chunk JSON folder → Qdrant collection.

Usage: ``python -m vectorize`` -- configuration is via env vars
(same convention as chunker/pipeline: the run is configured from .env/environment).
TWO exceptions are command-line flags (same reasoning as chunker's `--limit`
decision: "how much to run" is not a persistent setting):

    --limit N   at most N scopes are processed (the first N in discovery order;
                for trial/sampling).
    --force     ignores the staleness gate (local state.json), re-embeds ALL
                scopes (one-off bypass).

    VECTORIZE_INPUT_DIR   required -- root of the chunk JSONs (scanned recursively;
                          `discover.py`, recognizes both the old per-document and the
                          new all_chunks/all_raptor/all_combined layouts)
    VECTORIZE_OUTPUT_DIR  default ./storage -- local state.json + run_log.json
                          (the real output is the Qdrant collection, see
                          storage/README.md)
    VECTORIZE_CONFIG      optional -- partial override TOML on top of default.toml
    VECTORIZE_PROGRESS    on | off (default on) -- tqdm progress bar
    EMBEDDING_*           provider connection (see .env.example)
    QDRANT_URL/QDRANT_API_KEY  connection (see .env.example)

Staleness gate: each scope's signature (`core.ChunkSet.signature`) is written
to the local state.json; on the next run a scope with the same signature is
skipped (config/default.toml `[staleness] skip_unchanged`, default on -- embedding
is a real remote LLM/GPU call, so unlike chunker's "re-produce the whole corpus
each run" decision (KARAR-012) here cost matters).

Scope update contract: when a scope is (re-)embedded, first ALL old points of
that scope in Qdrant are deleted, then the new set is written from scratch
(`store.replace_scope` -- rationale: chunk node_ids are set-scoped, KARAR-004;
"upsert only the changed one" produces wrong matches).

Error model: per-scope errors are logged, the run continues with the other scopes.
Exit code 0 = all OK, 1 = at least one scope failed, 2 = configuration error
(including missing env, no scopes found -- an empty input folder almost always
means the wrong path).
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
    # which chunk production an evidence chunk came from must be traceable to
    # the end of the response chain (retrieval metadata -> chatbot conversation
    # log) -- both already exist in chunker's ChunkSet, here they are just
    # COPIED into the payload (no new field is invented).
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
    """Embeds a scope and writes it to Qdrant, returns the number of points in
    that scope (including 0 -- a textless/empty scope only deletes old points)."""
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
    """N-21: `calistir()`'s OLD single `int` (exit code) return did NOT carry the
    real numbers that the nightly report's `VectorizeSection` can write --
    `basarili`/`nokta_sayisi` were only PRINTED via `log.info`, never RETURNED
    anywhere. `main()` still returns a plain `int` (the CLI exit-code contract
    is UNCHANGED, NOT `main().exit_code`) -- only the side that directly calls
    `calistir()` (`run_nightly.py::stage_vectorize`) sees this rich object.

    `points_deleted`/`total_points_after` are NOT here -- neither comes from
    this function's local counters: `store.replace_scope` never returns the
    deleted point count, and "the final total" would require a real Qdrant
    `count()` network call (this machine's fake `_SahteStore`s do not support
    it). `nightly_report.py::VectorizeSection` accepts both as `None`
    ("not measured") -- see that module's docstring."""

    exit_code: int
    points_written: int = 0
    embedding_model: str | None = None


def calistir(env: Mapping[str, str] = os.environ, *, limit: int | None = None,
             force: bool = False) -> VectorizeRunStats:
    """The whole run; `env` is injectable (tests pass a dict). `limit`/`force`
    are passed through from `main`'s argv parsing. Returns `VectorizeRunStats`
    (see that dataclass's docstring) -- `main()` only uses its `.exit_code`,
    the CLI exit-code contract is UNCHANGED."""
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
                # N-04: saved immediately after EACH scope (stage-end commit
                # point) -- previously `save_state` was called only once after
                # the loop FINISHED, so if a run wrote 49 of 50 scopes and was
                # cut off exactly at the 50th, state.json was STILL EMPTY: the
                # next run would re-embed those 49 scopes too (unnecessary but
                # not WRONG) -- it failed the "resume where it left off"
                # acceptance criterion. `save_state` is already
                # write-then-replace (atomic), so calling it frequently inside
                # the loop is safe.
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
    """Entry point for ``python -m vectorize``: log + argv + .env, then `calistir`."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")  # no-op if .env missing; does not override a defined real env var

    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return calistir(env=os.environ, limit=args.limit, force=args.force).exit_code


if __name__ == "__main__":
    raise SystemExit(main())
