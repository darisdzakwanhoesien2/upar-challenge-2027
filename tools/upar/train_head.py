"""Train the linear attribute head (Tracks 1 and 2) on top of a frozen backbone, on CPU or GPU.

    python tools/upar/fetch_annotations.py                      # annotations only (small), into data/annotations
    python download_datasets.py                                  # organizers' script, in a clone of their starter kit: images
    python tools/upar/train_head.py --data-dir data --backbone-weights imagenet --out dist/assets

--backbone-weights imagenet downloads torchvision's ImageNet resnet18 (about 45 MB) on first use; the rules allow
ImageNet pre-training and forbid external data for fine-tuning, so only the organizers' images are used here.
Features are cached under --cache-dir so re-training the head takes seconds.
The test images are never touched: tune on the validation split only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from upar_attr import ATTRS, AttrModel, build_backbone, input_hw  # noqa: E402
from metrics import par_metrics  # noqa: E402


def read_gt(path: str):
    import pandas as pd

    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "image"})
    assert list(df.columns[1:]) == ATTRS, "gt.csv columns differ from the 40 UPAR attributes"
    return df["image"].tolist(), df[ATTRS].values.astype(np.float32)


def cached_features(model: AttrModel, paths: list[str], cache: str, tag: str, batch: int) -> np.ndarray:
    key = hashlib.md5((tag + "\n" + "\n".join(paths)).encode()).hexdigest()[:12]
    f = os.path.join(cache, f"feat_{key}.npy")
    if os.path.isfile(f):
        return np.load(f)
    x = model.features(paths, batch, progress=True)
    os.makedirs(cache, exist_ok=True)
    np.save(f, x)
    return x


def fit_head(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, wd: float, seed: int, pos_power: float = 0.5):
    import torch

    torch.manual_seed(seed)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xt = torch.from_numpy((x - mu) / sd)
    yt = torch.from_numpy(y)
    pos = yt.mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = ((1 - pos) / pos).pow(pos_power).clamp(1.0, 20.0)  # up-weight rare attributes: mA weighs them equally
    lin = torch.nn.Linear(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    n = len(xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            opt.zero_grad()
            loss_fn(lin(xt[idx]), yt[idx]).backward()
            opt.step()
    W = lin.weight.detach().numpy().astype(np.float32)
    b = lin.bias.detach().numpy().astype(np.float32)
    return W, b, mu.astype(np.float32), sd.astype(np.float32)


def calibrate(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-attribute logit shift maximising balanced accuracy on held-out data (the metric thresholds at 0.5)."""
    grid = np.linspace(-3, 3, 25)
    off = np.zeros(z.shape[1], dtype=np.float32)
    for j in range(z.shape[1]):
        pos, neg = y[:, j] > 0.5, y[:, j] <= 0.5
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        best = max(grid, key=lambda o: 0.5 * ((z[pos, j] + o > 0).mean() + (z[neg, j] + o <= 0).mean()))
        off[j] = best
    return off


def calibrate_avg(z: np.ndarray, y: np.ndarray, rounds: int = 2) -> np.ndarray:
    """Coordinate ascent on per-attribute logit offsets to maximise Challenge_Avg itself (couples mA and Instance_F1)."""
    grid = np.linspace(-3, 3, 25)
    off = np.zeros(z.shape[1], dtype=np.float32)
    best = par_metrics(1 / (1 + np.exp(-z)), y)["Challenge_Avg"]
    for _ in range(rounds):
        for j in range(z.shape[1]):
            keep = off[j]
            for o in grid:
                off[j] = o
                v = par_metrics(1 / (1 + np.exp(-(z + off))), y)["Challenge_Avg"]
                if v > best + 1e-6:
                    best, keep = v, o
            off[j] = keep
    return off


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder that contains Market1501/, PA100k/, PETA/ and annotations/")
    ap.add_argument("--train-gt", default=None)
    ap.add_argument("--val-gt", default=None)
    ap.add_argument("--out", required=True, help="assets directory to write")
    ap.add_argument("--backbone-weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--max-train", type=int, default=0, help="random subset of the training rows (0 = all); use on CPU")
    ap.add_argument("--max-val", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--calibrate", nargs="?", const="ba", choices=["ba", "avg"], default=None,
                    help="fit per-attribute logit offsets on the validation split: ba = balanced accuracy, avg = Challenge_Avg")
    ap.add_argument("--pos-power", type=float, default=0.5, help="positive-class weight = (neg/pos)**power (0 = unweighted)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    train_gt = a.train_gt or os.path.join(a.data_dir, "annotations/task1/train/gt.csv")
    val_gt = a.val_gt or os.path.join(a.data_dir, "annotations/task1/val/gt.csv")
    cache = a.cache_dir or os.path.join(a.data_dir, "features")
    rng = np.random.default_rng(a.seed)

    def load(gt, cap):
        names, y = read_gt(gt)
        keep = np.sort(rng.choice(len(names), cap, replace=False)) if 0 < cap < len(names) else np.arange(len(names))
        return [os.path.join(a.data_dir, names[i]) for i in keep], y[keep]

    tr_paths, ytr = load(train_gt, a.max_train)
    va_paths, yva = load(val_gt, a.max_val)
    missing = [p for p in tr_paths[:50] + va_paths[:50] if not os.path.isfile(p)]
    if missing:
        print(f"[train_head] image not found, e.g. {missing[0]}\n  run the organizers' download_datasets.py first "
              f"and check --data-dir ({a.data_dir})", file=sys.stderr)
        return 2

    os.makedirs(a.out, exist_ok=True)
    backbone, dim = build_backbone("resnet18", a.backbone_weights)
    torch.save(backbone.state_dict(), os.path.join(a.out, "backbone.pth"))
    meta = {"backbone": "resnet18", "feat_dim": dim, "input_hw": list(input_hw())}
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    model = AttrModel(a.out, device="cpu" if not torch.cuda.is_available() else None)
    model.backbone = backbone.to(model.device).eval()
    model.hw = tuple(meta["input_hw"])
    tag = f"resnet18-{a.backbone_weights}-{model.hw}"

    xtr = cached_features(model, tr_paths, cache, tag, a.batch)
    xva = cached_features(model, va_paths, cache, tag, a.batch)
    W, b, mu, sd = fit_head(xtr, ytr, a.epochs, a.lr, a.wd, a.seed, a.pos_power)
    off = np.zeros(len(ATTRS), dtype=np.float32)
    model.head = {"W": W, "b": b, "mu": mu, "sd": sd, "offset": off}
    zva = model.logits_from_features(xva)
    base = par_metrics(1 / (1 + np.exp(-zva)), yva)
    honest = None
    if a.calibrate:
        fit = calibrate if a.calibrate == "ba" else calibrate_avg
        half = len(yva) // 2  # honest estimate: fit the offsets on one half of val, score on the other
        o1 = fit(zva[:half], yva[:half])
        honest = par_metrics(1 / (1 + np.exp(-(zva[half:] + o1))), yva[half:])
        uncal2 = par_metrics(1 / (1 + np.exp(-zva[half:])), yva[half:])
        off = fit(zva, yva)
        model.head["offset"] = off
    cal = par_metrics(1 / (1 + np.exp(-model.logits_from_features(xva))), yva)
    prior = ytr.mean(0)
    pri = par_metrics(np.tile(prior, (len(yva), 1)), yva)
    # The challenge scores every dataset in the test set separately and averages them, so report that too.
    ds = np.array([os.path.relpath(p, a.data_dir).split(os.sep)[0] for p in va_paths])
    pva = 1 / (1 + np.exp(-model.logits_from_features(xva)))
    per = {d: (par_metrics(pva[ds == d], yva[ds == d]), par_metrics(np.tile(prior, ((ds == d).sum(), 1)), yva[ds == d]))
           for d in sorted(set(ds))}

    np.savez(os.path.join(a.out, "head.npz"), W=W, b=b, mu=mu, sd=sd, offset=off)
    with open(os.path.join(a.out, "prior.json"), "w", encoding="utf-8") as fh:
        json.dump({"attribute_names": ATTRS, "positive_rate": [float(v) for v in prior]}, fh)
    fmt = lambda m: "  ".join(f"{k}={m[k]:.4f}" for k in ("Challenge_Avg", "mA", "Instance_F1", "Label_F1"))
    print(f"[train_head] {len(ytr)} train / {len(yva)} val rows")
    print(f"  prior only      {fmt(pri)}")
    print(f"  head            {fmt(base)}")
    if a.calibrate:
        print(f"  head+calibrated {fmt(cal)}  (offsets fitted on all val rows, so optimistic)")
        print(f"  held-out half   uncalibrated {fmt(uncal2)}")
        print(f"  held-out half   calibrated   {fmt(honest)}  <- offsets fitted on the other half: the honest number")
    print("  per dataset (head vs prior, Challenge_Avg):  " + "  ".join(f"{d} {h['Challenge_Avg']:.4f} vs {q['Challenge_Avg']:.4f}" for d, (h, q) in per.items()))
    print(f"  mean over datasets (organizer-style): head {np.mean([h['Challenge_Avg'] for h, _ in per.values()]):.4f}  "
          f"prior {np.mean([q['Challenge_Avg'] for _, q in per.values()]):.4f}")
    print(f"[train_head] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
