"""Tiny synthetic stand-in for the UPAR data, so the scripts can be exercised without the real images.

Each image is a 'person': a coloured upper-body block over a coloured lower-body block on a grey background. The upper
and lower colour attributes follow the pixels; the other attributes are random. This tests code paths and that a head
can learn something. It says nothing about real performance.
"""
from __future__ import annotations

import os

import numpy as np

from upar_attr import ATTRS

COLORS = {"Black": (20, 20, 20), "Blue": (30, 60, 200), "Red": (200, 30, 30), "White": (240, 240, 240),
          "Green": (30, 160, 60)}


def make_dataset(root: str, n: int, split: str, seed: int = 0, size=(96, 48)):
    """Write images under root/synth/<split>/ and return (relative image names, (n, 40) float32 labels)."""
    from PIL import Image

    rng = np.random.default_rng(seed + (0 if split == "train" else 1000))
    d = os.path.join(root, "synth", split)
    os.makedirs(d, exist_ok=True)
    names, rows = [], []
    cn = list(COLORS)
    for i in range(n):
        up, lo = cn[rng.integers(len(cn))], cn[rng.integers(len(cn))]
        h, w = size
        a = np.full((h, w, 3), 128, dtype=np.uint8)
        a[h // 5: h // 2, w // 6: 5 * w // 6] = COLORS[up]
        a[h // 2: 9 * h // 10, w // 4: 3 * w // 4] = COLORS[lo]
        a = np.clip(a.astype(int) + rng.integers(-12, 13, a.shape), 0, 255).astype(np.uint8)
        y = (rng.random(len(ATTRS)) < 0.15).astype(np.float32)
        for k, name in enumerate(ATTRS):
            if name.startswith("UpperBody-Color-"):
                y[k] = float(name.split("-")[-1] == up)
            elif name.startswith("LowerBody-Color-"):
                y[k] = float(name.split("-")[-1] == lo)
        rel = f"synth/{split}/{i:05d}.png"
        Image.fromarray(a).save(os.path.join(root, rel))
        names.append(rel)
        rows.append(y)
    return names, np.stack(rows)


def write_gt(path: str, names, y) -> None:
    import pandas as pd

    df = pd.DataFrame(y.astype(int), columns=ATTRS)
    df.insert(0, "# image", names)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
