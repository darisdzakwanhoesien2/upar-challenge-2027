"""Write assets/kprcnn_coco.pth for Track 3: torchvision's Keypoint R-CNN (ResNet-50 FPN) with its COCO weights.

    python tools/upar/prepare_pose_weights.py --out dist/pose_assets            # downloads about 160 MB
    python tools/upar/prepare_pose_weights.py --out dist/pose_assets --random   # untrained stub, for plumbing tests only

The COCO weights are trained on COCO only, which the Track 3 rules allow as pre-training. Check with the organizers before
swapping in any other checkpoint (the rules forbid other external pre-training data and foundation models).
"""
from __future__ import annotations

import argparse
import os

import torch
import torchvision


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    w = None if a.random else torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    m = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=w, weights_backbone=None)
    path = os.path.join(a.out, "kprcnn_coco.pth")
    torch.save(m.state_dict(), path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB){' [RANDOM STUB]' if a.random else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
