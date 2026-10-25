"""Mock of the Codabench ingestion program: imports a submission's run.py and calls it the way the platform does,
validating the output against the rules on the Submission pages. It cannot reproduce the real scorer or hidden data.

    python tools/upar/mock_ingest.py --track 1 --submission DIR --images DIR_OR_GT [--gt gt.csv] [--limit N]
    python tools/upar/mock_ingest.py --track 2 --submission DIR --gt val_gt.csv --data-dir data [--limit N]
    python tools/upar/mock_ingest.py --track 3 --submission DIR --images DIR_OF_IMAGES [--limit N]

DIR is a submission folder (or an unzipped archive) with run.py at its root.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upar_attr import ATTRS  # noqa: E402
import metrics  # noqa: E402


def load_run(sub: str):
    if not os.path.isfile(os.path.join(sub, "run.py")):
        raise SystemExit(f"{sub}/run.py not found: run.py must be at the archive root")
    sys.path.insert(0, sub)
    os.chdir(sub)  # the platform's working directory is the submission folder
    spec = importlib.util.spec_from_file_location("run", os.path.join(sub, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_gt(gt: str, data_dir: str):
    import pandas as pd

    df = pd.read_csv(gt)
    df = df.rename(columns={df.columns[0]: "image"})
    return [os.path.join(data_dir, n) for n in df["image"]], df[ATTRS].values.astype(np.float32)


def check_probs(rows) -> np.ndarray:
    a = np.asarray(rows, dtype=np.float64)
    assert a.ndim == 2 and a.shape[1] == 40, f"each image needs 40 probabilities, got shape {a.shape}"
    assert np.isfinite(a).all(), "NaN/Inf in predictions"
    assert a.min() >= 0 and a.max() <= 1, f"values outside [0, 1]: {a.min()}..{a.max()}"
    return a


def run_track1(mod, a) -> None:
    paths, y = read_gt(a.gt, a.data_dir)
    if a.limit:
        paths, y = paths[:a.limit], y[:a.limit]
    samples = [{"index": i, "image": p, "image_path": p, "attribute_names": ATTRS} for i, p in enumerate(paths)]
    t = time.time()
    if hasattr(mod, "load_model"):
        mod.load_model()
    if hasattr(mod, "predict_batch"):
        bs = getattr(mod, "BATCH_SIZE", 32)
        out = [r for i in range(0, len(samples), bs) for r in mod.predict_batch(samples[i:i + bs])]
    else:
        out = [mod.predict_image(s) for s in samples]
    p = check_probs(out)
    assert len(p) == len(samples), "one row per image"
    dt = time.time() - t
    print(f"[mock_ingest] track 1 OK: {len(p)} images, {dt:.1f}s ({dt / max(len(p), 1) * 1000:.0f} ms/image)")
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.par_metrics(p, y).items()))


def run_track2(mod, a) -> None:
    paths, y = read_gt(a.gt, a.data_dir)
    if a.limit:
        paths, y = paths[:a.limit], y[:a.limit]
    queries = np.unique(y, axis=0)
    sample = {"attribute_names": ATTRS, "queries": queries.astype(int).tolist(),
              "gallery": [{"index": i, "image": p, "image_path": p} for i, p in enumerate(paths)]}
    t = time.time()
    if hasattr(mod, "load_model"):
        mod.load_model()
    out = mod.rank_gallery(sample)
    assert isinstance(out, dict) and len(out) == 1 and next(iter(out)) in ("distances", "similarities", "ranking"), \
        'return a dict with exactly one of "distances", "similarities", "ranking"'
    key, m = next(iter(out.items()))
    m = np.asarray(m)
    Q, G = len(queries), len(paths)
    assert m.shape == (Q, G), f"expected {(Q, G)}, got {m.shape}"
    if key == "ranking":
        assert all(sorted(r) == list(range(G)) for r in m), "each ranking row must be a permutation of 0..G-1"
        dist = np.argsort(np.argsort(m, axis=1), axis=1)
    else:
        assert np.isfinite(m.astype(float)).all(), "NaN/Inf"
        dist = m if key == "distances" else -m
    dt = time.time() - t
    print(f"[mock_ingest] track 2 OK: {Q} queries x {G} gallery images, key={key}, {dt:.1f}s")
    print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in
                           metrics.retrieval_metrics(dist, queries, y).items()))


def run_track3(mod, a) -> None:
    from PIL import Image

    files = sorted(f for f in os.listdir(a.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        files = files[:a.limit]
    t = time.time()
    if hasattr(mod, "load_model"):
        mod.load_model()
    n_boxes = 0
    for i, f in enumerate(files):
        path = os.path.abspath(os.path.join(a.images, f))
        with Image.open(path) as im:
            w, h = im.size
        boxes = [[w * 0.2, h * 0.1, w * 0.4, h * 0.8], [w * 0.6, h * 0.3, w * 0.3, h * 0.6]]
        sample = {"image_id": i, "file_name": f, "image_path": path, "width": w, "height": h, "boxes": boxes,
                  "box_ids": [10 * i, 10 * i + 1]}
        out = mod.predict_image(sample)
        kps = out["keypoints"] if isinstance(out, dict) else out
        scores = out.get("scores") if isinstance(out, dict) else None
        assert len(kps) == len(boxes), "one prediction per box, in order"
        for k in kps:
            arr = np.asarray(k, dtype=np.float64).reshape(-1)
            assert arr.size == 51 and np.isfinite(arr).all(), "17 keypoints x (x, y, v), all finite"
        if scores is not None:
            assert len(scores) == len(boxes) and all(np.isfinite(scores)), "one finite score per box"
        n_boxes += len(boxes)
    dt = time.time() - t
    print(f"[mock_ingest] track 3 OK: {len(files)} images, {n_boxes} boxes, {dt:.1f}s ({dt / max(len(files), 1):.2f} s/image)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--gt", help="gt.csv (tracks 1 and 2)")
    ap.add_argument("--data-dir", default="data", help="images in gt.csv are relative to this folder")
    ap.add_argument("--images", help="folder of images (track 3)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    mod = load_run(os.path.abspath(a.submission))
    {1: run_track1, 2: run_track2, 3: run_track3}[a.track](mod, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
