"""Shared attribute model for UPAR Tracks 1 and 2: a frozen CNN backbone and a linear multi-label head.

Used three ways: by tools/upar/train_head.py (feature extraction + training), by the submissions' run.py (inference
inside the evaluation container, no network), and by tools/upar/selftest.py. With no trained head in `assets/` the model
falls back to the attribute prior (the organizers' own baseline), so a zip is always valid.

Assets directory layout:
    meta.json     {"backbone": "resnet18", "feat_dim": 512, "input_hw": [256, 128]}
    backbone.pth  torch state_dict of the backbone (shipped in the zip; nothing is downloaded at run time)
    head.npz      W (40 x feat_dim), b (40), mu (feat_dim), sd (feat_dim), offset (40, logit shift per attribute)
    prior.json    {"attribute_names": [...], "positive_rate": [...]}
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ATTRS = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female", "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown",
    "UpperBody-Color-Green", "UpperBody-Color-Grey", "UpperBody-Color-Orange", "UpperBody-Color-Pink",
    "UpperBody-Color-Purple", "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black", "LowerBody-Color-Blue",
    "LowerBody-Color-Brown", "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange",
    "LowerBody-Color-Pink", "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat",
]
assert len(ATTRS) == 40

DEFAULT_HW = (256, 128)  # (height, width): pedestrian crops are tall
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def input_hw() -> tuple[int, int]:
    """(H, W). Override with UPAR_INPUT_HW=128x64 (used by the selftest to stay fast)."""
    v = os.environ.get("UPAR_INPUT_HW")
    if v:
        h, w = v.lower().split("x")
        return int(h), int(w)
    return DEFAULT_HW


def build_backbone(name: str = "resnet18", weights: str = "none"):
    """Return (module mapping (N,3,H,W) to (N,D) features, D). weights: 'imagenet' downloads via torchvision."""
    import torch
    import torchvision

    if name != "resnet18":
        raise ValueError("only resnet18 is wired up; add others here")
    w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if weights == "imagenet" else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = torch.nn.Identity()
    return m, 512


def load_image(path: str, hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    h, w = hw
    with Image.open(path) as im:
        im = im.convert("RGB").resize((w, h), Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    return ((a - MEAN) / STD).transpose(2, 0, 1)


def read_prior(assets: str) -> np.ndarray:
    with open(os.path.join(assets, "prior.json"), "r", encoding="utf-8") as fh:
        p = json.load(fh)
    rate = dict(zip(p["attribute_names"], p["positive_rate"]))
    return np.array([rate.get(a, 0.0) for a in ATTRS], dtype=np.float32)


class AttrModel:
    def __init__(self, assets: str, device: str | None = None, threads: int = 4):
        self.assets = assets
        self.device = device
        self.threads = threads
        self.backbone = None
        self.head = None
        self.prior = None
        self.hw = input_hw()

    def load(self) -> "AttrModel":
        import torch

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        prior_path = os.path.join(self.assets, "prior.json")
        if os.path.isfile(prior_path):
            self.prior = read_prior(self.assets)
        meta_path = os.path.join(self.assets, "meta.json")
        head_path = os.path.join(self.assets, "head.npz")
        bb_path = os.path.join(self.assets, "backbone.pth")
        if os.path.isfile(meta_path) and os.path.isfile(head_path) and os.path.isfile(bb_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.hw = tuple(meta.get("input_hw", self.hw))
            self.backbone, _ = build_backbone(meta["backbone"], "none")
            self.backbone.load_state_dict(torch.load(bb_path, map_location="cpu"))
            self.backbone.to(self.device).eval()
            z = np.load(head_path)
            self.head = {k: z[k].astype(np.float32) for k in ("W", "b", "mu", "sd", "offset")}
        elif self.prior is None:
            raise FileNotFoundError(f"{self.assets} has neither a trained head nor prior.json")
        return self

    @property
    def has_head(self) -> bool:
        return self.head is not None

    def features(self, paths: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        import torch

        out = []
        with ThreadPoolExecutor(self.threads) as ex, torch.no_grad():
            for i in range(0, len(paths), batch_size):
                chunk = paths[i:i + batch_size]
                x = np.stack(list(ex.map(lambda p: load_image(p, self.hw), chunk)))
                out.append(self.backbone(torch.from_numpy(x).to(self.device)).cpu().numpy())
                if progress and (i // batch_size) % 20 == 0:
                    print(f"[upar_attr] {i + len(chunk)}/{len(paths)} images", flush=True)
        return np.concatenate(out) if out else np.zeros((0, 512), dtype=np.float32)

    def logits_from_features(self, f: np.ndarray) -> np.ndarray:
        h = self.head
        return ((f - h["mu"]) / h["sd"]) @ h["W"].T + h["b"] + h["offset"]

    def probs(self, paths: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        """(N, 40) probabilities in the order of ATTRS."""
        if not self.has_head:
            return np.tile(self.prior[None, :], (len(paths), 1)).astype(np.float32)
        z = self.logits_from_features(self.features(paths, batch_size, progress))
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
