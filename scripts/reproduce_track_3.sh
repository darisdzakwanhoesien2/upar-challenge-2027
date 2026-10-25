#!/usr/bin/env bash
# Track 3 has no training on UPAR images. It uses torchvision's Keypoint R-CNN with its public COCO weights, plus a six-number linear
# map per head slot fitted on the organizers' training annotations (train_upar_all.json from the Track 3 starter kit).
set -euo pipefail
python tools/upar/prepare_pose_weights.py --out weights            # downloads about 230 MB of COCO weights (writes weights/kprcnn_coco.pth)
python tools/upar/fit_head_slots.py --ann "${1:-$HOME/upar_data/pose/train_upar_all.json}"   # prints the constants in submissions/track3_pose_estimation/run.py
