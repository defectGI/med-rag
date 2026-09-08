"""Drop chunks whose body consists ONLY of an image description.

CONTEXT: the strip script leaves nodes whose body becomes empty after
description removal ALONE (same principle as `strip_injected_ocr.py`:
"OCR was the only content source — emptying it deletes real information").
For image descriptions that principle does NOT hold: the description is a
VLM interpretation of a photo, not "real information" — so having it as
sole content makes the chunk LESS valuable, not more.

Measured: 31 nodes, all `tree_level=0`, none a summary child; all 31/31
become COMPLETELY empty when the description is stripped. Typical body:
"The image shows a promotional flyer for the PN1183 LXI Switching and DAQ
Mainframe...".

CHAIN REPAIR (the real work): the deleted node's `prev_node_id`/
`next_node_id` neighbors are connected to each other — otherwise the
`prev/next` chain is broken and anything that walks neighbors (RAPTOR
sweep, context expansion) would look at a dead id. `child_ids`/summary
nodes are not affected (measured), but the script STILL halts if any node
being deleted is a summary's child.

WHAT IT DOES NOT DELETE: any node whose body retains text outside the
description. The criterion is mechanical — empty after
`strip_image_descriptions` + `strip()`.

Usage:
  cd chunker
  python scripts/drop_description_only_chunks.py storage/all_chunks.json [--apply]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _strip_mod():
    spec = importlib.util.spec_from_file_location(
        "_strip", HERE / "strip_injected_image_description.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--apply", action="store_true", help="Actually write (default: dry run).")
    a = ap.parse_args()

    m = _strip_mod()
    data = json.loads(a.path.read_text(encoding="utf-8"))
    docs = data["documents"]
    doc_iter = list(docs.values()) if isinstance(docs, dict) else list(docs)

    silinecek: list[str] = []
    for doc in doc_iter:
        for n in doc["nodes"]:
            t = n.get("text") or ""
            if not m.OPENERS.search(t):
                continue
            yeni, _c, _a = m.strip_image_descriptions(t, has_image_ref=bool(n.get("images")))
            if not yeni.strip():
                silinecek.append(n["node_id"])
    hedef = set(silinecek)

    # SAFETY: none of them may be a summary node's child (measured: 0).
    cocuk_olan = [
        n["node_id"] for doc in doc_iter for n in doc["nodes"]
        for cid in (n.get("child_ids") or []) if cid in hedef
    ]
    if cocuk_olan:
        raise SystemExit(f"STOPPED: {len(cocuk_olan)} nodes are children of a "
                         f"summary node — deleting them would break the tree; "
                         f"inspect manually.")

    # Chain repair: deleted nodes' neighbors are connected. For consecutive
    # deletions, walk the chain to find the right surviving neighbor.
    by_id = {n["node_id"]: n for doc in doc_iter for n in doc["nodes"]}

    def _canli(nid, alan):
        while nid in hedef:
            nid = by_id[nid].get(alan)
        return nid

    onarilan = 0
    for doc in doc_iter:
        for n in doc["nodes"]:
            if n["node_id"] in hedef:
                continue
            for alan in ("prev_node_id", "next_node_id"):
                v = n.get(alan)
                if v in hedef:
                    n[alan] = _canli(v, alan)
                    onarilan += 1

    for doc in doc_iter:
        doc["nodes"] = [n for n in doc["nodes"] if n["node_id"] not in hedef]

    print(f"chunks to drop       : {len(hedef)}")
    print(f"prev/next links repaired: {onarilan}")
    for nid in silinecek[:5]:
        print(f"  {nid}")
    if len(silinecek) > 5:
        print(f"  ... +{len(silinecek)-5}")

    if not a.apply:
        print("\n--dry-run: file was not written")
        return 0
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    yedek = a.path.with_name(f"{a.path.name}.pre_drop_{stamp}.json")
    shutil.copy2(a.path, yedek)
    a.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten -> {a.path}\nbackup   -> {yedek}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())