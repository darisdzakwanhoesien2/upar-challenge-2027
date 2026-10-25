#!/usr/bin/env bash
# Retrain the Track 1 / Track 2 attribute model from scratch. About 35 minutes of feature extraction on an Apple-silicon CPU
# plus seconds for the head; a GPU makes the extraction faster. Needs about 2 GB of disk for the images.
set -euo pipefail
DATA="${1:-$HOME/upar_data}"
pip install -r requirements.txt
python tools/upar/fetch_annotations.py --out "$DATA/annotations"
# Images: clone https://github.com/speckean/UPAR-Challenge-2027 and run its download_datasets.py with --data-dir "$DATA"
# (Market-1501, PA-100K and PETA, from Google Drive and Dropbox). Skipping this step makes train_head.py stop with a clear message.
python tools/upar/train_head.py --data-dir "$DATA" --out weights/track12 \
  --backbone-weights imagenet --max-val 10000 --pos-power 0.5 --seed 0
scripts/install_weights.sh weights/track12
