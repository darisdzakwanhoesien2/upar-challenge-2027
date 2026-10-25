#!/usr/bin/env bash
# Put the trained weights where the submissions look for them (submissions/<track>/assets/).
# Usage: scripts/install_weights.sh weights/track12 [weights/kprcnn_coco.pth]
set -euo pipefail
T12="${1:?directory with backbone.pth head.npz meta.json prior.json}"
POSE="${2:-}"
for t in track1_attribute_recognition track2_attribute_retrieval; do
  mkdir -p "submissions/$t/assets"
  cp "$T12"/backbone.pth "$T12"/head.npz "$T12"/meta.json "$T12"/prior.json "submissions/$t/assets/"
done
if [ -n "$POSE" ] && [ -f "$POSE" ]; then
  mkdir -p submissions/track3_pose_estimation/assets
  cp "$POSE" submissions/track3_pose_estimation/assets/kprcnn_coco.pth
else
  echo "no pose weights given: Track 3 will return the template pose only"
fi
echo "weights installed"
