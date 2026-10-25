# UPAR Challenge 2027 (RWS @ WACV 2027): entries for Tracks 1, 2 and 3

Code, weights and instructions for three entries to the UPAR Challenge at the Real-World Surveillance workshop, WACV 2027
(Codabench competitions 18180, 18196 and 18273). The Track 1 and 2 entries share one deliberately simple model; the Track 3 entry is
zero-shot. This is a baseline-class submission, not a state-of-the-art system.

| Track | Task | Method | Submission code |
|---|---|---|---|
| 1 | Pedestrian attribute recognition (40 attributes) | frozen ImageNet ResNet-18 + class-weighted linear head | [`submissions/track1_attribute_recognition`](submissions/track1_attribute_recognition) |
| 2 | Attribute-based person retrieval | the same model; gallery ranked by negative log-likelihood of the query | [`submissions/track2_attribute_retrieval`](submissions/track2_attribute_retrieval) |
| 3 | 2D human pose estimation (PoseTrack18 slots) | torchvision Keypoint R-CNN, COCO weights, per given box; head slots from a fitted linear map | [`submissions/track3_pose_estimation`](submissions/track3_pose_estimation) |

The fact sheets (one per track, in the organizers' template) are in [`fact_sheets/`](fact_sheets). Leaderboard results are in them.

## Rules followed

- Tracks 1 and 2: trained only on the organizers' training images (Market-1501, PA-100K, PETA). The only other data is ImageNet pre-training of the backbone, which the rules allow.
  The test images were never touched; the validation split was used only to choose one hyper-parameter and the retrieval distance.
- Track 3: only COCO pre-training (torchvision's published Keypoint R-CNN weights). The organizers' training annotations were used only to fit twelve numbers
  (the head-slot map, `tools/upar/fit_head_slots.py`). No pose model was trained or fine-tuned.
- No proprietary models, no vision-language models.

## Weights

Attached to the GitHub release of this repository (nothing is stored in git):

| File | Contents | SHA-256 |
|---|---|---|
| `upar_weights_track12.zip` | `backbone.pth` (ImageNet ResNet-18 state dict), `head.npz`, `meta.json`, `prior.json` for Tracks 1 and 2 | see [`weights/SHA256SUMS.txt`](weights/SHA256SUMS.txt) |
| `kprcnn_coco.pth` | torchvision `keypointrcnn_resnet50_fpn` COCO weights for Track 3 (identical to what `tools/upar/prepare_pose_weights.py` downloads) | same file |

```bash
gh release download v1.0 --repo <owner>/upar-challenge-2027 --dir weights     # or download the two files from the release page
cd weights && unzip -o upar_weights_track12.zip && cd ..
shasum -a 256 -c weights/SHA256SUMS.txt                                        # optional check
scripts/install_weights.sh weights/track12 weights/kprcnn_coco.pth
```

## Run the entries (inference)

Each submission folder has a `run.py` with the interface of the organizers' evaluation program (`predict_batch` for Track 1, `rank_gallery` for Track 2,
`predict_image` for Track 3) and the unchanged `metadata.yaml`.

```bash
python tools/upar/package_submission.py --track 1 --assets submissions/track1_attribute_recognition/assets --out dist/upar_track1.zip   # the Codabench zip
python tools/upar/mock_ingest.py --track 1 --submission submissions/track1_attribute_recognition \
    --gt DATA/annotations/task1/val/gt.csv --data-dir DATA --limit 500                                                               # runs it like the platform
```

`mock_ingest.py --track 2` and `--track 3` work the same way. Local metrics in `tools/upar/metrics.py` are my own implementation of the formulas on the Codabench
Evaluation pages; the organizers' scorers are not public, so they may differ in edge cases.

## Reproduce training

- Tracks 1 and 2: `scripts/reproduce_tracks_1_2.sh DATA_DIR`. Downloads the annotations (small), expects the images from the organizers' `download_datasets.py`
  (about 1.8 GB), extracts ResNet-18 features for all 97,669 training images (about 35 minutes on an Apple-silicon CPU) and fits the head.
  The final command is
  `python tools/upar/train_head.py --data-dir DATA --out weights/track12 --backbone-weights imagenet --max-val 10000 --pos-power 0.5 --seed 0`.
  Retraining the head from the cached features reproduces the submitted `head.npz` and `backbone.pth` byte for byte on the machine used.
- Track 3: `scripts/reproduce_track_3.sh [train_upar_all.json]` downloads the COCO weights and prints the two head-slot matrices, which match the constants in `run.py`.

## Local results (not the leaderboard)

Own metric code, 10,000 random validation images from the same three datasets as training (an in-domain estimate; the hidden test is a new domain, so expect lower scores).

| Track | This entry | Reference |
|---|---|---|
| 1, challenge average | 0.703 (mA 0.686, instance F1 0.720); 0.690 averaged over the three datasets | attribute prior 0.548 (0.546) |
| 2, mADM | 0.331 (negative log-likelihood) | 0.292 (organizers' L1), about 0.125 prior only |
| 3, AP | 0.661 on 200 labelled PETA crops, assumed OKS sigmas | 0.625 first head heuristic, 0.739 with ground-truth head slots |

## What was and was not tested

- Tested: the offline self-test (`python tools/upar/selftest.py`, synthetic data); the three zips unpacked and run through `mock_ingest.py`; Track 1 on real validation images; the
  Track 3 AP check above; byte-identical retraining of the head from cached features. Environment used: Python 3.10, torch 2.12, torchvision 0.27, macOS (Apple silicon), CPU.
- **Not tested:** the `Dockerfile` (no Docker daemon was available to build it), the evaluation image `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (Python 3.11, torch 2.4.1),
  CUDA, and a full feature extraction from scratch on another machine (the head retraining used cached features).
- The leaderboard scores of the submitted zips are in the fact sheets; they were not computed by this repository.

## Licence and data

Code: MIT ([`LICENSE`](LICENSE)). The UPAR annotations are CC BY-NC-SA 3.0; Market-1501, PA-100K and PETA keep their own licences. torchvision weights are used as published.
Please cite the UPAR papers (Specker et al. WACV 2023, Cormier et al. WACVW 2023 and 2024).
