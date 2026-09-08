"""Yeni kosunun COZEMEDIGI ama ONCEKI kosunun cozdugu chunk'larin atamasini
ONCEKI tablodan TASIR -- yalniz KARMA OLMAYAN chunk'lar icin (2026-08-06).

NEDEN: 2026-08-06 ownership kosusunda (PROTOCOL KARAR-064) `PN5085_CATALOGUE`
`::c17` chunk'i once timeout, sonra retry'da yarim JSON dondurdu ve YENI
tabloda 45 -> **0** urune dustu. Bu bir REGRESYON: PN1260'in
`storage_temperature -55..125` degeri tam da o chunk'tan geliyor (ayni gun
ROADMAP 20a ile kurtarilan deger). Bu tabloyla extraction kosulursa PN1253
ailesi o degeri kaybederdi.

NEDEN MESRU: tasima YALNIZ `heading_path`i DOLU (yani ortak atasi olan, TEK
bolumlu) chunk'lar icin yapilir. Boyle bir chunk'in span'a IHTIYACI YOKTUR --
KARAR-064'un getirdigi tek yenilik karma chunk'i bolumlere ayirmakti; tek
bolumlu bir chunk'ta eski ve yeni promptun URETMESI GEREKEN cevap aynidir.
Karma bir chunk'in atamasi ASLA tasinmaz (orada eski cevap zaten kusurlu
olabilir -- duzeltmeye calistigimiz sey o).

Tasinan satirlar `source="carried_over"` ile ISARETLENIR: sessiz sapma yok,
tabloya bakan biri hangi satirin bu kosudan, hangisinin oncekinden geldigini
gorur. (KARAR-062'nin `manual_family_wide` isaretiyle ayni ilke.)

Kullanim:
  cd facts/experiments/catalog_chunk_ownership
  python carry_over_unresolved.py --old <onceki_tablo.json>            # DRY RUN
  python carry_over_unresolved.py --old <onceki_tablo.json> --apply

O-07 (2026-08-20): `--apply` ile birlikte `skipped` listesi (halen
cozulememis chunk'lar) ayrica `issues` tablosuna (`issues_bridge.py`,
`UNRESOLVED_CHUNK` kodu) yazilir -- sessiz dusme yok, panelde (M-09)
tek tek gorulebilir hale gelir (I-14).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from medrag.core.db.issues import connect_issues
from medrag.core.paths import resolve_issues_db_path
from medrag.pipeline.facts.issues_bridge import record_unresolved_chunks

HERE = Path(__file__).resolve().parent  # src/medrag/pipeline/facts/catalog_chunk_ownership/ (kod, O-03)
REPO_ROOT = HERE.parents[4]  # medrag/ (catalog_chunk_ownership/facts/pipeline/urun/src/<repo>)
RESULTS_PATH = HERE / "results" / "chunk_ownership_all.json"
ALL_CHUNKS = REPO_ROOT / "chunker" / "storage" / "all_chunks.json"
HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)


def load_nodes() -> dict[str, dict]:
    data = json.loads(ALL_CHUNKS.read_text(encoding="utf-8"))["documents"]
    return {n["node_id"]: n
            for cs in data.values()
            for n in cs.get("nodes", []) if n.get("tree_level", 0) == 0}


def plan(new: dict, old: dict, nodes: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """`(tasinacak satirlar, tasinmayacak hatalar)`."""
    old_by_chunk: dict[str, list[dict]] = {}
    for r in old.get("rows", []):
        if r.get("accepted") and r.get("product_code"):
            old_by_chunk.setdefault(r["chunk_id"], []).append(r)

    carried, skipped = [], []
    for err in [r for r in new["rows"] if r.get("error")]:
        cid = err["chunk_id"]
        node = nodes.get(cid)
        prior = old_by_chunk.get(cid, [])
        if node is None:
            skipped.append({**err, "_why": "chunk metni bulunamadi"})
            continue
        if not node.get("heading_path"):
            # Ortak atasi YOK -> karma olma ihtimali yuksek -> eski cevap
            # zaten kusurlu olabilir, TASIMA.
            skipped.append({**err, "_why": "KARMA chunk (heading_path bos) -- tasinmaz"})
            continue
        if not prior:
            skipped.append({**err, "_why": "onceki tabloda da cozulmemis"})
            continue
        for r in prior:
            carried.append({**r, "source": "carried_over"})
    return carried, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--apply", action="store_true", help="Gercekten yaz (varsayilan: dry run).")
    ap.add_argument("--issues-db", default=str(resolve_issues_db_path()),
                    help="Cozulememis chunk'larin yazilacagi issues.db yolu (O-07, "
                         "varsayilan: ISSUES_DB_PATH ortam degiskeni, 2026-08-26 fix).")
    args = ap.parse_args()

    new = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    old = json.loads(Path(args.old).read_text(encoding="utf-8"))
    carried, skipped = plan(new, old, load_nodes())

    by_chunk: dict[str, int] = {}
    for r in carried:
        by_chunk[r["chunk_id"]] = by_chunk.get(r["chunk_id"], 0) + 1
    for cid, n in by_chunk.items():
        print(f"  TASINDI  {cid[-8:]}: {n} atama (onceki kosudan)")
    for s in skipped:
        print(f"  BIRAKILDI {s['chunk_id'][-8:]}: {s['_why']}")

    if args.apply and carried:
        # Tasinan chunk'larin `error` satirlarini DUSUR (artik cozuldu).
        resolved = set(by_chunk)
        new["rows"] = [r for r in new["rows"] if not (r.get("error") and r["chunk_id"] in resolved)]
        new["rows"].extend(carried)
        new["meta"]["n_rows"] = len(new["rows"])
        new["meta"]["n_llm_errors"] = sum(1 for r in new["rows"] if r.get("error"))
        new["meta"]["n_carried_over"] = len(carried)
        RESULTS_PATH.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    n_issues = 0
    if args.apply and skipped:
        # O-07: hala cozulememis chunk'lar sessizce dusmez -- issues.db'ye
        # UNRESOLVED_CHUNK olarak yazilir (panelde tek tek gorulebilir, M-09).
        con = connect_issues(args.issues_db)
        try:
            n_issues = record_unresolved_chunks(con, skipped)
        finally:
            con.close()

    mode = "UYGULANDI" if args.apply else "DRY RUN"
    print(f"[{mode}] {len(carried)} satir / {len(by_chunk)} chunk tasindi, "
          f"{len(skipped)} chunk cozulmemis kaldi"
          + (f", {n_issues} issue yazildi -> {args.issues_db}" if args.apply else ""))


if __name__ == "__main__":
    main()
