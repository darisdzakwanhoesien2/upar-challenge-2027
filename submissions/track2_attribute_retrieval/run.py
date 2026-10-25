"""UPAR Track 2 entry: attribute-based person retrieval.

rank_gallery() predicts the 40 attribute probabilities of every gallery image once (same model as Track 1), then turns
them into a Q x G distance matrix. DISTANCE selects the rule: "l1" is the organizers' example recipe; "nll" is the
negative log-likelihood of the query. Pick one by measuring mADM on the validation queries (tools/upar/mock_ingest.py --track 2 prints it when given a labelled gt.csv);
the default stays on the organizers' recipe until that is measured on real data.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..", "..", "tools", "upar")):
    if os.path.isfile(os.path.join(_p, "upar_attr.py")):
        sys.path.insert(0, _p)
        break
from upar_attr import ATTRS, AttrModel  # noqa: E402
from metrics import distances_l1, distances_nll  # noqa: E402

ASSETS = os.path.join(HERE, "assets")
DISTANCE = os.environ.get("UPAR_DISTANCE", "nll")
_MODEL: AttrModel | None = None


def load_model() -> None:
    global _MODEL
    if _MODEL is None:
        _MODEL = AttrModel(ASSETS).load()
        print(f"[submission] attribute model ready (head={'yes' if _MODEL.has_head else 'no, prior only'}, "
              f"device={_MODEL.device}, distance={DISTANCE})", flush=True)


def rank_gallery(sample: dict[str, Any]) -> dict[str, Any]:
    load_model()
    paths = [g["image_path"] for g in sample["gallery"]]
    probs = _MODEL.probs(paths, batch_size=64, progress=True)
    order = [ATTRS.index(n) for n in sample["attribute_names"]]
    probs = probs[:, order]
    queries = np.asarray(sample["queries"], dtype=np.float32)
    fn = distances_nll if DISTANCE == "nll" else distances_l1
    return {"distances": fn(probs, queries)}  # (Q x G), smaller = better; ties keep gallery order
