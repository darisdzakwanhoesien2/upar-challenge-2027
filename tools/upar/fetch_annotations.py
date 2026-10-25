"""Download only the UPAR Track 1/2 annotation CSVs (about 40 MB) from the organizers' public starter kit.

    python tools/upar/fetch_annotations.py --out data/annotations

The images are not included; get them with `python download_datasets.py` from a clone of
https://github.com/speckean/UPAR-Challenge-2027 (Market-1501, PA-100K and PETA, several GB from Google Drive and Dropbox).
"""
from __future__ import annotations

import argparse
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/speckean/UPAR-Challenge-2027/HEAD/data/annotations/"
FILES = ["task1/train/gt.csv", "task1/val/gt.csv", "task2/train/gt.csv", "task2/train/ids.csv", "task2/train/queries.csv",
         "task2/val/gt.csv", "task2/val/ids.csv", "task2/val/queries.csv"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/annotations")
    a = ap.parse_args()
    for f in FILES:
        dst = os.path.join(a.out, f)
        if os.path.isfile(dst):
            print("have", dst)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print("get ", BASE + f)
        urllib.request.urlretrieve(BASE + f, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
