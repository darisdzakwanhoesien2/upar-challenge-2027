"""UPAR Track 3 entry: top-down 2D pose, 17 PoseTrack18 slots, per provided box.

Backend: torchvision Keypoint R-CNN with its COCO weights (pre-training on COCO is allowed; see documentation.md for the
open question about the organizers' ViTPose+ example). For each given box the image is cropped with padding, the model
runs on the crop, the detection that best overlaps the box is kept and its COCO keypoints are mapped to the PoseTrack
slots. Where the model finds nobody, a fixed standing-pose template inside the box is returned with a low score so the
output is always valid. The head_bottom and head_top slots are NOT predicted by COCO models: they come from a linear map
of the predicted nose and shoulders fitted on the organizers' training annotations (see coco_to_posetrack).

Weights: assets/kprcnn_coco.pth, written by tools/upar/prepare_pose_weights.py. Nothing is downloaded at run time.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "assets", "kprcnn_coco.pth")
PAD = 0.15
FALLBACK_SCORE = 0.05

# standing pose inside a box, as (x, y) fractions of the box (person facing the camera); hand-set, unvalidated
TEMPLATE = np.array([[.50, .08], [.50, .14], [.50, .02], [.58, .07], [.42, .07], [.62, .20], [.38, .20], [.68, .35],
                     [.32, .35], [.70, .50], [.30, .50], [.57, .52], [.43, .52], [.58, .73], [.42, .73], [.58, .94],
                     [.42, .94]], dtype=np.float32)

_MODEL = None
_DEVICE = None


def load_model() -> None:
    global _MODEL, _DEVICE
    if _MODEL is not None:
        return
    import torch
    import torchvision

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if os.path.isfile(WEIGHTS):
        m = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                                                   box_score_thresh=0.05)
        m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        _MODEL = m.to(_DEVICE).eval()
        print(f"[submission] Keypoint R-CNN (COCO) ready on {_DEVICE}", flush=True)
    else:
        _MODEL = False  # template-only mode: valid output, no model
        print("[submission] no weights in assets/: returning the template pose for every box", flush=True)


# head_bottom / head_top are not COCO keypoints. Each is predicted as neck + w * ([dx, dy, 1] @ B), where neck is the
# shoulder midpoint, w the shoulder width and (dx, dy) = (nose - neck) / w. B (3 x 2) was fitted by least squares on the
# organizers' labelled training annotations (train_upar_all.json, 33,624 persons, 80/20 split). Held-out error with
# ground-truth inputs, in shoulder widths: head_bottom 0.098 (old heuristic 0.136), head_top 0.231 (old 0.533).
# Assumes an upright person; the old geometric heuristic remains the fallback for degenerate shoulders.
B_HEAD_BOTTOM = np.array([[0.2244, 0.0041], [-0.0233, 0.4297], [-0.0045, -0.0007]], dtype=np.float32)
B_HEAD_TOP = np.array([[0.5580, 0.0071], [-0.0201, 1.6302], [-0.0101, -0.1355]], dtype=np.float32)


def coco_to_posetrack(k: np.ndarray) -> np.ndarray:
    """(17, 2) COCO keypoints -> (17, 2) PoseTrack18 slots (nose, head_bottom, head_top, ears, then limbs)."""
    nose, lsh, rsh = k[0], k[5], k[6]
    neck = (lsh + rsh) / 2.0
    w = float(np.linalg.norm(lsh - rsh))
    if w > 2.0:
        f = np.array([*((nose - neck) / w), 1.0], dtype=np.float32)
        head_bottom = neck + w * (f @ B_HEAD_BOTTOM)
        head_top = neck + w * (f @ B_HEAD_TOP)
    else:
        head_bottom = neck + 0.25 * (nose - neck)
        head_top = nose + 0.8 * (nose - head_bottom)
    out = np.zeros((17, 2), dtype=np.float32)
    out[0], out[1], out[2], out[3], out[4] = nose, head_bottom, head_top, k[3], k[4]
    out[5:] = k[5:]
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _one_box(img, box) -> tuple[list[list[float]], float]:
    import torch

    x, y, w, h = [float(v) for v in box[:4]]
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x - PAD * w)), max(0, int(y - PAD * h))
    x1, y1 = min(W, int(x + w + PAD * w)), min(H, int(y + h + PAD * h))
    if _MODEL and x1 - x0 > 4 and y1 - y0 > 4:
        crop = torch.from_numpy(img[y0:y1, x0:x1].astype(np.float32) / 255.0).permute(2, 0, 1).to(_DEVICE)
        res = _MODEL([crop])[0]
        inner = np.array([x - x0, y - y0, x + w - x0, y + h - y0], dtype=np.float32)
        best, best_iou = None, 0.0
        for i in range(len(res["scores"])):
            iou = _iou(res["boxes"][i].cpu().numpy(), inner)
            if iou > best_iou:
                best, best_iou = i, iou
        if best is not None and best_iou > 0.3:
            kp = res["keypoints"][best].cpu().numpy()[:, :2] + np.array([x0, y0], dtype=np.float32)
            kp = coco_to_posetrack(kp)
            return [[float(a), float(b), 1.0] for a, b in kp], float(res["scores"][best]) * best_iou
    pts = TEMPLATE * np.array([w, h], dtype=np.float32) + np.array([x, y], dtype=np.float32)
    return [[float(a), float(b), 1.0] for a, b in pts], FALLBACK_SCORE


def predict_image(sample: dict[str, Any]) -> dict:
    from PIL import Image

    load_model()
    with Image.open(sample["image_path"]) as im:
        img = np.asarray(im.convert("RGB"))
    kps, scores = [], []
    for box in sample["boxes"]:
        k, s = _one_box(img, box)
        kps.append(k)  # per person: 17 x [x, y, v], the same shape as the organizers' example
        scores.append(s)
    return {"keypoints": kps, "scores": scores}
