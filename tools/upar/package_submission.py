"""Build the zip to upload on Codabench: run.py at the archive root, metadata.yaml unchanged, assets, helper modules.

    python tools/upar/package_submission.py --track 1 --assets dist/assets --out dist/upar_track1.zip
    python tools/upar/package_submission.py --track 2 --assets dist/assets --out dist/upar_track2.zip
    python tools/upar/package_submission.py --track 3 --assets dist/pose_assets --out dist/upar_track3.zip

Tracks 1 and 2 share one assets folder from train_head.py. Pass --assets for a prior-only zip made from just prior.json.
The zip is unpacked into a temp folder and run.py is imported there as a smoke test before the size is reported.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
# Two layouts: the research-collaboration repo keeps each submission next to its project docs; the standalone challenge
# repository (exported for the organizers) keeps them under submissions/.
SUB_OLD = {1: "shared_task_upar_pedestrian_2026/submission", 2: "shared_task_upar_retrieval_2026/submission", 3: "shared_task_upar_pose_2026/submission"}
SUB_NEW = {1: "submissions/track1_attribute_recognition", 2: "submissions/track2_attribute_retrieval", 3: "submissions/track3_pose_estimation"}
SUB = {t: (SUB_NEW[t] if os.path.isdir(os.path.join(ROOT, SUB_NEW[t])) else SUB_OLD[t]) for t in SUB_OLD}
HELPERS = {1: ["upar_attr.py"], 2: ["upar_attr.py", "metrics.py"], 3: []}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = os.path.join(ROOT, SUB[a.track])
    stage = tempfile.mkdtemp(prefix="upar_pkg_")
    for f in ("run.py", "metadata.yaml"):
        shutil.copy(os.path.join(src, f), os.path.join(stage, f))
    for f in HELPERS[a.track]:
        shutil.copy(os.path.join(HERE, f), os.path.join(stage, f))
    shutil.copytree(a.assets, os.path.join(stage, "assets"))
    need = {1: ["prior.json"], 2: ["prior.json"], 3: []}[a.track]
    for n in need:
        if not os.path.isfile(os.path.join(stage, "assets", n)):
            raise SystemExit(f"{a.assets}/{n} missing")
    if a.track == 3 and not os.path.isfile(os.path.join(stage, "assets", "kprcnn_coco.pth")):
        print("WARNING: no kprcnn_coco.pth in assets: this zip returns the template pose only (valid, but not a model)")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        os.remove(a.out)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(stage):
            for f in sorted(files):
                if f.endswith((".pyc", ".DS_Store")) or "__pycache__" in d:
                    continue
                full = os.path.join(d, f)
                z.write(full, os.path.relpath(full, stage))
    with zipfile.ZipFile(a.out) as z:
        names = z.namelist()
    assert "run.py" in names and "metadata.yaml" in names, "run.py and metadata.yaml must be at the archive root"

    check = tempfile.mkdtemp(prefix="upar_unzip_")
    with zipfile.ZipFile(a.out) as z:
        z.extractall(check)
    sys.path.insert(0, check)
    spec = importlib.util.spec_from_file_location("run_check", os.path.join(check, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"[package] {a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB, {len(names)} files, run.py imports from the archive")
    for n in sorted(names):
        print("   ", n)
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(check, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
