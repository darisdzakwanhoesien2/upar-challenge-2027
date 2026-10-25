"""UPAR Track 1 entry: pedestrian attribute recognition (40 binary attributes, one probability each).

The platform imports this file, calls load_model() once, then predict_batch(samples). With a trained head in assets/
the model looks at the pixels (frozen ResNet-18 features + linear head); with none it returns the attribute prior, which
is the organizers' own example baseline. Nothing is downloaded at run time.
"""
from __future__ import annotations

import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "..", "tools", "upar")):  # zip root first, repo tools/ when run from the tree
    if os.path.isfile(os.path.join(_p, "upar_attr.py")):
        sys.path.insert(0, _p)
        break
from upar_attr import ATTRS, AttrModel  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
BATCH_SIZE = 64
_MODEL: AttrModel | None = None


def load_model() -> None:
    global _MODEL
    if _MODEL is None:
        _MODEL = AttrModel(ASSETS).load()
        print(f"[submission] attribute model ready (head={'yes' if _MODEL.has_head else 'no, prior only'}, "
              f"device={_MODEL.device})", flush=True)


def predict_batch(samples: list[dict[str, Any]]) -> list[list[float]]:
    load_model()
    names = samples[0]["attribute_names"] if samples else ATTRS
    p = _MODEL.probs([s["image_path"] for s in samples], batch_size=len(samples) or 1)
    order = [ATTRS.index(n) for n in names]  # the platform's order is the contract; ours is the same unless it changes
    return p[:, order].tolist()
