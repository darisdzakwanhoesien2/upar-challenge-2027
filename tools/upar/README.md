# UPAR scripts (RWS 2027 @ WACV 2027, Codabench 18180 / 18196 / 18273)

Scripts to build, check and package code submissions for the three UPAR tracks. Rules, metrics and deadlines are in
`shared_task_upar_{pedestrian,retrieval,pose}_2026/documentation.md`. Tracks 1 and 2 share one attribute model.

**Status (2026-10-07):** all three entries are built as zips, trained or assembled on the real data, and checked through the
mock ingestion; local validation numbers are in each track's `documentation.md`. Nothing has been uploaded. The local
metrics are re-implementations of the formulas on the Codabench pages, not the organizers' scorers, and the validation sets
are in-domain, so expect lower scores on the hidden test. `selftest.py` still tests the plumbing on synthetic data.

| File | Purpose |
|---|---|
| `upar_attr.py` | Frozen ResNet-18 + linear head for the 40 attributes; falls back to the attribute prior when no head is present |
| `train_head.py` | Extract and cache features, train the head, optional per-attribute calibration, write the assets folder |
| `fetch_annotations.py` | Download only the annotation CSVs (about 40 MB) from the organizers' public starter kit |
| `metrics.py` | Track 1 (Challenge_Avg, mA, F1) and Track 2 (mADM, mAP, Rank-k, mINP) metrics; L1 and NLL distances |
| `mock_ingest.py` | Import a submission's `run.py` the way the platform does and validate its output (and score it if labelled) |
| `package_submission.py` | Build the zip (run.py at the root, metadata.yaml unchanged), unpack it and import-check it |
| `eval_pose.py` | Track 3: COCO keypoint AP with pycocotools on the organizers' labelled training annotations |
| `prepare_pose_weights.py` | Track 3: save torchvision's COCO Keypoint R-CNN weights into the assets folder |
| `synth.py`, `selftest.py` | Synthetic data and an end-to-end test of all of the above (about a minute on CPU) |

Submissions live next to the docs: `shared_task_upar_pedestrian_2026/submission/run.py` (Track 1),
`shared_task_upar_retrieval_2026/submission/run.py` (Track 2), `shared_task_upar_pose_2026/submission/run.py` (Track 3).

## Check the scripts (no downloads)

```bash
python3 tools/upar/selftest.py
```

## Tracks 1 and 2 on the real data

Needs the images (Market-1501, PA-100K, PETA: several GB from Google Drive and Dropbox) and ideally a GPU. Keep the data
outside the repository, for example `~/upar_data`.

```bash
python3 tools/upar/fetch_annotations.py --out ~/upar_data/annotations
# images: clone https://github.com/speckean/UPAR-Challenge-2027 and run its download_datasets.py into ~/upar_data
python3 tools/upar/train_head.py --data-dir ~/upar_data --out dist/assets --backbone-weights imagenet --calibrate
# on a CPU add --max-train 20000 --max-val 5000 first, to see a number in minutes
python3 tools/upar/package_submission.py --track 1 --assets dist/assets --out dist/upar_track1.zip
python3 tools/upar/package_submission.py --track 2 --assets dist/assets --out dist/upar_track2.zip
python3 tools/upar/mock_ingest.py --track 1 --submission <unzipped dir> --gt ~/upar_data/annotations/task1/val/gt.csv --data-dir ~/upar_data --limit 2000
```

`--backbone-weights imagenet` downloads torchvision's ImageNet ResNet-18 (about 45 MB). The rules allow ImageNet
pre-training and forbid external data for fine-tuning and any use of test images. Tune only on the validation split. The
image path in `gt.csv` is relative to the data folder (for example `Market1501/bounding_box_train/....jpg`).

Track 2 uses the organizers' L1 distance by default. Compare `UPAR_DISTANCE=nll` on the validation queries with
`mock_ingest.py --track 2` before choosing.

A prior-only zip (the organizers' own baseline) needs only `prior.json` in the assets folder.

## Track 3

```bash
python3 tools/upar/prepare_pose_weights.py --out dist/pose_assets        # about 160 MB of COCO weights
python3 tools/upar/package_submission.py --track 3 --assets dist/pose_assets --out dist/upar_track3.zip
python3 tools/upar/mock_ingest.py --track 3 --submission <unzipped dir> --images <folder of images>
```

The head_bottom and head_top slots come from a linear map of COCO's nose and shoulders fitted on the organizers' training
annotations. Check the pipeline with `tools/upar/eval_pose.py --submission <unzipped dir> --ann <train_*.json> --data-dir <data> --n 200`
(needs `pip install pycocotools`; the annotation files are in the Track 3 starter kit). First settle with the organizers
whether the ViTPose+ example checkpoint is allowed (see the Track 3 documentation).
