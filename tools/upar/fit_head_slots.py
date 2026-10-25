#!/usr/bin/env python3
"""Fit the linear maps that give the PoseTrack head_bottom and head_top slots from a COCO model's nose and shoulders.

    python tools/upar/fit_head_slots.py --ann ~/upar_data/pose/train_upar_all.json

For every labelled person (at least 8 labelled keypoints, and nose, head_bottom, head_top and both shoulders labelled), with
neck = midpoint of the shoulders and w = shoulder width (at least 3 px), the inputs are (nose - neck) / w and a constant 1, and the
targets are (head_bottom - neck) / w and (head_top - neck) / w. Least squares on a random 80% of persons (seed 0); the held-out 20%
gives the error in shoulder widths. Prints the two 3 x 2 matrices in the form pasted into
shared_task_upar_pose_2026/submission/run.py (B_HEAD_BOTTOM, B_HEAD_TOP).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ann", required=True, help="train_upar_all.json from the Track 3 starter kit")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = json.load(open(a.ann))
    X, Yb, Yt = [], [], []
    for p in d["annotations"]:
        k = np.array(p["keypoints"], dtype=float).reshape(17, 3)
        if p.get("num_keypoints", 0) < 8 or (k[[0, 1, 2, 5, 6], 2] == 0).any():
            continue
        neck = (k[5, :2] + k[6, :2]) / 2
        w = np.linalg.norm(k[5, :2] - k[6, :2])
        if w < 3:
            continue
        X.append(np.concatenate([(k[0, :2] - neck) / w, [1.0]]))
        Yb.append((k[1, :2] - neck) / w)
        Yt.append((k[2, :2] - neck) / w)
    X, Yb, Yt = map(np.array, (X, Yb, Yt))
    n = len(X)
    idx = np.random.default_rng(a.seed).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    print(f"{n} usable persons, {len(tr)} train / {len(te)} held out")
    for name, Y in (("B_HEAD_BOTTOM", Yb), ("B_HEAD_TOP", Yt)):
        B, *_ = np.linalg.lstsq(X[tr], Y[tr], rcond=None)
        err = float(np.mean(np.linalg.norm(X[te] @ B - Y[te], axis=1)))
        print(f"{name} = [" + ", ".join("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in B) + f"]   held-out error {err:.3f} shoulder widths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
