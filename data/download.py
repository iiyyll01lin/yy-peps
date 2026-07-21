"""Dataset fetcher — Kodak, AmbientCG PBR sets, Stanford meshes.

繁體中文:資料下載腳本。抓取 Kodak 無損影像、AmbientCG PBR 材質、Stanford 網格。
所有資料皆為公開/CC0。用法:
    python data/download.py --kodak
    python data/download.py --ambientcg MetalPlates013 Metal032
    python data/download.py --all
下載內容放在 data/raw/(已被 .gitignore 忽略)。
"""

from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile

RAW = os.path.join(os.path.dirname(__file__), "raw")

# r0k.us (the canonical Kodak host) blocks direct PNG hotlinks, so we pull the
# 24 images from a GitHub mirror. Files are named 01.png..24.png there; we save
# them locally as kodim01.png..kodim24.png (the name the rest of the repo uses).
KODAK_MIRROR = (
    "https://raw.githubusercontent.com/MohamedBakrAli/"
    "Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992"
)
KODAK_IMAGES = [(f"{i:02d}.png", f"kodim{i:02d}.png") for i in range(1, 25)]

# AmbientCG download pattern (CC0). 2K PNG bundles.
AMBIENTCG_TMPL = "https://ambientcg.com/get?file={name}_2K-PNG.zip"

DEFAULT_TEXTURES = ["MetalPlates013", "Metal032", "PlanksDiffuse", "Rock023"]


def _get(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        print(f"  [skip] {dst}")
        return
    print(f"  [get ] {url}")
    urllib.request.urlretrieve(url, dst)


def fetch_kodak() -> None:
    print("Kodak Lossless True Color Suite (24 images)")
    d = os.path.join(RAW, "kodak")
    for remote_name, local_name in KODAK_IMAGES:
        _get(f"{KODAK_MIRROR}/{remote_name}", os.path.join(d, local_name))


def fetch_ambientcg(names) -> None:
    print(f"AmbientCG PBR sets: {names}")
    for name in names:
        dst_zip = os.path.join(RAW, "ambientcg", f"{name}.zip")
        _get(AMBIENTCG_TMPL.format(name=name), dst_zip)
        out_dir = os.path.join(RAW, "ambientcg", name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            try:
                with zipfile.ZipFile(dst_zip) as z:
                    z.extractall(out_dir)
                print(f"  [unzip] {out_dir}")
            except zipfile.BadZipFile:
                print(f"  [warn] not a zip (maybe HTML error page): {dst_zip}")


def fetch_stanford() -> None:
    print("Stanford meshes — download manually from:")
    print("  http://graphics.stanford.edu/data/3Dscanrep/  (Armadillo, Thai Statue)")
    print("Place .ply files under data/raw/stanford/ then run App3 preprocessing.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kodak", action="store_true")
    ap.add_argument("--ambientcg", nargs="*", metavar="NAME")
    ap.add_argument("--stanford", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all or args.kodak:
        fetch_kodak()
    if args.all or args.ambientcg is not None:
        names = args.ambientcg if args.ambientcg else DEFAULT_TEXTURES
        fetch_ambientcg(names)
    if args.all or args.stanford:
        fetch_stanford()
    if not any([args.kodak, args.ambientcg is not None, args.stanford, args.all]):
        ap.print_help()


if __name__ == "__main__":
    main()
