"""End-to-end check of the UPAR scripts on synthetic data (no downloads, no GPU, about a minute on a laptop CPU).

    python tools/upar/selftest.py

Trains a head on synthetic images, packages the Track 1, 2 and 3 zips, unpacks each and runs it through the mock
ingestion program. Passing means the plumbing works and a head learns colour attributes better than the prior. It says
nothing about real UPAR performance.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from synth import make_dataset, write_gt  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, UPAR_INPUT_HW="64x32")


def run(*args) -> str:
    r = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, env=ENV)
    if r.returncode != 0:
        print(r.stdout, r.stderr, sep="\n")
        raise SystemExit(f"FAILED: {' '.join(map(str, args))}")
    return r.stdout


def unzip(z: str, d: str) -> str:
    shutil.unpack_archive(z, d)
    return d


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="upar_selftest_")
    try:
        data = os.path.join(tmp, "data")
        for split, n in (("train", 500), ("val", 200)):
            names, y = make_dataset(data, n, split)
            write_gt(os.path.join(data, f"annotations/task1/{split}/gt.csv"), names, y)
        print("[selftest] synthetic data written")

        out = run(os.path.join(HERE, "train_head.py"), "--data-dir", data, "--out", os.path.join(tmp, "assets"),
                  "--backbone-weights", "none", "--epochs", 60, "--calibrate")
        print(out.strip())
        prior = float(re.search(r"prior only\s+Challenge_Avg=([0-9.]+)", out).group(1))
        head = float(re.search(r"\n  head\s+Challenge_Avg=([0-9.]+)", out).group(1))
        assert head > prior + 0.05, f"head ({head}) should beat the prior ({prior}) on colour attributes"
        print(f"[selftest] head {head:.3f} beats prior {prior:.3f}")

        gt = os.path.join(data, "annotations/task1/val/gt.csv")
        for track, name in ((1, "track1"), (2, "track2")):
            z = os.path.join(tmp, f"{name}.zip")
            run(os.path.join(HERE, "package_submission.py"), "--track", track, "--assets", os.path.join(tmp, "assets"), "--out", z)
            d = unzip(z, os.path.join(tmp, name))
            print(run(os.path.join(HERE, "mock_ingest.py"), "--track", track, "--submission", d, "--gt", gt, "--data-dir", data).strip())

        # prior-only zips (no head, no backbone): the always-valid fallback
        prior_assets = os.path.join(tmp, "prior_assets")
        os.makedirs(prior_assets)
        shutil.copy(os.path.join(tmp, "assets", "prior.json"), prior_assets)
        for track in (1, 2):
            z = os.path.join(tmp, f"prior{track}.zip")
            run(os.path.join(HERE, "package_submission.py"), "--track", track, "--assets", prior_assets, "--out", z)
            d = unzip(z, os.path.join(tmp, f"prior{track}"))
            print(run(os.path.join(HERE, "mock_ingest.py"), "--track", track, "--submission", d, "--gt", gt, "--data-dir", data, "--limit", 60).strip())

        # Track 3: untrained stub weights (plumbing only) and template-only
        imgs = os.path.join(data, "synth", "val")
        pose_assets = os.path.join(tmp, "pose_assets")
        run(os.path.join(HERE, "prepare_pose_weights.py"), "--out", pose_assets, "--random")
        for label, assets in (("stub weights", pose_assets), ("template only", os.path.join(tmp, "pose_empty"))):
            os.makedirs(assets, exist_ok=True)
            z = os.path.join(tmp, "track3.zip")
            run(os.path.join(HERE, "package_submission.py"), "--track", 3, "--assets", assets, "--out", z)
            d = unzip(z, os.path.join(tmp, "track3_" + label.split()[0]))
            print(f"[selftest] track 3, {label}: " + run(os.path.join(HERE, "mock_ingest.py"), "--track", 3, "--submission", d, "--images", imgs, "--limit", 3).strip())

        # COCO -> PoseTrack mapping is the only hand-written geometry: check shape and that ears/limbs pass through
        _root = os.path.join(HERE, "..", "..")
        _new = os.path.join(_root, "submissions", "track3_pose_estimation")
        sys.path.insert(0, _new if os.path.isdir(_new) else os.path.join(_root, "shared_task_upar_pose_2026", "submission"))
        import run as pose_run  # noqa: E402
        k = np.arange(34, dtype=np.float32).reshape(17, 2)
        pt = pose_run.coco_to_posetrack(k)
        assert pt.shape == (17, 2) and np.allclose(pt[5:], k[5:]) and np.allclose(pt[3:5], k[3:5]) and np.allclose(pt[0], k[0])
        print("[selftest] ALL PASSED")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
