"""Local COCO keypoint AP for the Track 3 entry, on the organizers' own labelled training annotations.

    python tools/upar/eval_pose.py --submission <unzipped dir> --ann ~/upar_data/pose/train_peta.json --data-dir ~/upar_data --n 150

Needs pycocotools. The images in these annotations are small pedestrian crops (PETA, Market-1501, PA-100K), not the
full-scene PoseTrack-style images of the hidden test, so the AP is only a rough check of the pipeline and of the
head_bottom / head_top geometry. Ear slots are not annotated and are ignored (as in the challenge). OKS sigmas are the
COCO ones mapped to the PoseTrack slot order, with 0.035 for the two head slots: an assumption, not the organizers' values.
Also prints an upper bound where the two head slots are replaced by ground truth, to show how much they cost.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys

import numpy as np

# slot order: nose, head_bottom, head_top, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr, l_hip, r_hip, l_kn, r_kn, l_an, r_an
SIGMAS = np.array([.026, .035, .035, .035, .035, .079, .079, .072, .072, .062, .062, .107, .107, .087, .087, .089, .089])


def find_image(data_dir: str, rel: str) -> str | None:
    parts = rel.split("/")
    cur = data_dir
    for p in parts:
        hit = next((e for e in os.listdir(cur) if e.lower() == p.lower()), None) if os.path.isdir(cur) else None
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur


def main() -> int:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--n", type=int, default=150, help="number of images (first n with a findable file)")
    a = ap.parse_args()

    sub = os.path.abspath(a.submission)
    sys.path.insert(0, sub)
    os.chdir(sub)
    spec = importlib.util.spec_from_file_location("run", os.path.join(sub, "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    run.load_model()

    d = json.load(open(a.ann))
    cat = d["categories"][0]
    anns = {}
    for x in d["annotations"]:
        if x.get("num_keypoints", 0) > 0 and not x.get("iscrowd", 0):
            anns.setdefault(x["image_id"], []).append(x)
    imgs = []
    for im in d["images"]:
        if im["id"] in anns:
            p = find_image(a.data_dir, im["file_name"])
            if p:
                imgs.append((im, p))
        if len(imgs) >= a.n:
            break
    gt = {"images": [im for im, _ in imgs], "categories": [cat], "annotations": [x for im, _ in imgs for x in anns[im["id"]]]}
    dts, dts_oracle = [], []
    for i, (im, p) in enumerate(imgs):
        boxes = [x["bbox"] for x in anns[im["id"]]]
        out = run.predict_image({"image_id": im["id"], "file_name": im["file_name"], "image_path": p, "width": im["width"],
                                 "height": im["height"], "boxes": boxes, "box_ids": [x["id"] for x in anns[im["id"]]]})
        for x, k, s in zip(anns[im["id"]], out["keypoints"], out["scores"]):
            flat = [float(v) for pt in k for v in pt]
            dts.append({"image_id": im["id"], "category_id": 1, "keypoints": flat, "score": float(s)})
            o = list(flat)
            for slot in (1, 2):  # oracle: ground-truth head_bottom / head_top where labelled
                if x["keypoints"][3 * slot + 2] > 0:
                    o[3 * slot], o[3 * slot + 1] = x["keypoints"][3 * slot], x["keypoints"][3 * slot + 1]
            dts_oracle.append({**dts[-1], "keypoints": o})
        if (i + 1) % 25 == 0:
            print(f"[eval_pose] {i + 1}/{len(imgs)} images", flush=True)

    coco = COCO()
    coco.dataset = copy.deepcopy(gt)
    coco.createIndex()

    def ap_of(dt):
        e = COCOeval(coco, coco.loadRes(dt), "keypoints")
        e.params.kpt_oks_sigmas = SIGMAS
        e.params.maxDets = [20]
        e.evaluate(); e.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            e.summarize()
        return e.stats[0], e.stats[1]

    ap_, ap50 = ap_of(dts)
    ap_o, ap50_o = ap_of(dts_oracle)
    fallbacks = sum(1 for x in dts if abs(x["score"] - run.FALLBACK_SCORE) < 1e-9)
    print(f"[eval_pose] {len(imgs)} images, {len(dts)} persons, template fallback on {fallbacks}")
    print(f"  AP {ap_:.4f}   AP50 {ap50:.4f}")
    print(f"  with ground-truth head slots (upper bound for the head geometry): AP {ap_o:.4f}   AP50 {ap50_o:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
