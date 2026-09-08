"""Clears the shared VLM visual cache BY HAND — after a quality improvement.

Why it exists: full-corpus runs write IR/visual/chunk output to their own
`runs/<ts>/` folder, BUT the VLM classify + describe cache (`STORAGE_LABELS_DIR`,
default `parser/storage/labels`) is INTENTIONALLY SHARED: keyed by crop-sha
+ prompt version, it survives IR/parser version bumps so each full run
doesn't pay the expensive vision calls again.

The cost: if you make a quality improvement WITHOUT bumping the prompt
version (better crop logic, better model, threshold tweak, ...), the
cache keeps serving the old results -- the change is silently invisible.
At that point DELETING THIS CACHE ON PURPOSE is the developer's job;
this script is exactly for that. If you bumped the prompt version, you
DON'T NEED THIS -- the key already changed.

What it deletes:
  - STORAGE_LABELS_DIR contents   -> both visual_classify (`<sha>.json`)
                                     and describe (`desc-<sha>.json`) caches
  - with --images, also STORAGE_IMAGES_DIR (crop/blob store) contents.
    Note: full-corpus now writes visuals to runs/<ts>/images (per-run,
    always fresh); this flag is only for setups that still keep
    STORAGE_IMAGES_DIR in a shared location (pre-`runs/` era).

Safety: DEFAULT dry-run -- only shows what would be deleted (file count +
size), does NOT TOUCH. To actually delete: `--yes`. The parent directory
and its README.md are preserved; only cache files/sub-directories are
deleted.

Usage:
    python reset_visual_cache.py              # dry-run: show what would be deleted
    python reset_visual_cache.py --yes        # actually delete the labels/ cache
    python reset_visual_cache.py --images      # + images blob store (dry-run)
    python reset_visual_cache.py --images --yes
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PARSER_DIR = (REPO_ROOT / "src" / "urun" / "pipeline" / "parser").resolve()
PARSER_ENV = PARSER_DIR / ".env"


def _cozumle(anahtar: str, default_alt: str) -> Path:
    """Resolves the parser storage path with the SAME precedence as
    storage_paths.py: env var > parser/.env > default (relative to parser/).
    This script doesn't import parser (relative defaults would depend on
    cwd); instead it reads parser/.env and resolves relative to the parser
    directory to find the actual location."""
    import os
    val = os.environ.get(anahtar)
    if not val and PARSER_ENV.is_file():
        val = dotenv_values(PARSER_ENV).get(anahtar)
    if not val:
        val = default_alt
    p = Path(val)
    return p if p.is_absolute() else (PARSER_DIR / p).resolve()


def _ozet(dizin: Path) -> tuple[int, int]:
    """(file count, total bytes) -- excluding README.md. Returns (0, 0) if the directory does not exist."""
    if not dizin.is_dir():
        return 0, 0
    adet = boyut = 0
    for yol in dizin.rglob("*"):
        if yol.is_file() and yol.name != "README.md":
            adet += 1
            boyut += yol.stat().st_size
    return adet, boyut


def _mb(byte: int) -> str:
    return f"{byte / (1024 * 1024):.1f} MB"


def _temizle(dizin: Path) -> int:
    """Deletes the directory's contents (README.md and the parent itself are
    preserved). Returns: number of top-level entries deleted."""
    if not dizin.is_dir():
        return 0
    silinen = 0
    for yol in dizin.iterdir():
        if yol.name == "README.md":
            continue
        if yol.is_dir():
            shutil.rmtree(yol)
        else:
            yol.unlink()
        silinen += 1
    return silinen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python reset_visual_cache.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", action="store_true",
                    help="STORAGE_IMAGES_DIR (crop/blob store) icerigini de sil "
                         "-- yalnizca images'i paylasilan tutan (runs/ oncesi) "
                         "kurulumlar icin gerekli")
    ap.add_argument("--yes", action="store_true",
                    help="GERCEKTEN sil (varsayilan: dry-run, yalniz goster)")
    args = ap.parse_args(argv)

    hedefler: list[tuple[str, Path]] = [
        ("labels (VLM classify + describe cache)",
         _cozumle("STORAGE_LABELS_DIR", "storage/labels")),
    ]
    if args.images:
        hedefler.append(
            ("images (crop/blob store)",
             _cozumle("STORAGE_IMAGES_DIR", "storage/images")))

    print("=" * 72)
    print("paylasilan VLM gorsel cache temizligi")
    print("=" * 72)
    toplam_adet = 0
    for etiket, dizin in hedefler:
        adet, boyut = _ozet(dizin)
        toplam_adet += adet
        durum = "" if dizin.is_dir() else "  (dizin yok)"
        print(f"  {etiket}")
        print(f"    {dizin}{durum}")
        print(f"    {adet} dosya, {_mb(boyut)}")
    print("-" * 72)

    if toplam_adet == 0:
        print("silinecek bir sey yok -- cache zaten bos.")
        return 0

    if not args.yes:
        print(f"dry-run: {toplam_adet} dosya silinecekti. Gercekten silmek "
              f"icin --yes ekle.")
        return 0

    print("siliniyor...")
    for etiket, dizin in hedefler:
        n = _temizle(dizin)
        print(f"  {etiket}: {n} girdi silindi")
    print("tamam -- bir sonraki kosu bu gorselleri sifirdan uretir.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
