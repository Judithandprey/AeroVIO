"""Merge the original motion cache with pose-only no-hover flights.

The original cache (prepare_motion_data.py on data/hyperdrone-motion-v2) keeps
its images, depth and visibility, so those pairs still get the warp loss. The
no-hover flights come from `viobench simulate --hover 0 --no-video` and have no
depth: their images, depth and visibility are zero, so photometric_loss skips
them and they only contribute the pose loss. The network is IMU-only, so images
never enter the predictor either way.

Optionally writes a fixed sampling plan (--balanced-plan) that draws a fixed
fraction of every epoch from the original pairs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "workspace/solution")]
from preintegrate import make_state, exp_so3, log_so3
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices, invert

# Scenes used for evaluation or the original training/validation split.
FORBIDDEN_SCENES = {1, 2, 3, 4, 5, 6, 40, 60, 80, 100, 120, 140, 50, 90, 130, 170}
FORBIDDEN_SEEDS = {2000, 3000}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nohover_arrays(paths, width=64, height=48):
    state, target, intrinsics, episode_id, evidence = [], [], [], [], []
    for number, path in enumerate(paths):
        episode = read_episode(path)
        scene = int(re.search(r"ProcTHOR-Train-(\d+)", episode.meta["scene"]).group(1))
        if scene in FORBIDDEN_SCENES or int(episode.meta["seed"]) in FORBIDDEN_SEEDS:
            raise ValueError(f"{path}: scene/seed overlaps an evaluation or original split")
        if episode.config["episode"]["hover_seconds"] != 0:
            raise ValueError(f"{path}: expected a no-hover flight")
        config, _ = episode.adapt((320, 240))
        poses = poses_to_matrices(episode.ground_truth_relative())
        k = np.array([config["fx"] * width / config["width"], config["fy"] * height / config["height"],
                      (config["cx"] + .5) * width / config["width"] - .5,
                      (config["cy"] + .5) * height / config["height"] - .5], dtype=np.float32)
        for index in range(1, episode.num_frames):
            t0, t1 = episode.frame_times[index - 1:index + 1]
            s = make_state(episode.imu_window(index), config["T_cam_imu"], config, t0, t1)
            relative = invert(poses[index - 1]) @ poses[index]
            residual = log_so3(exp_so3(s[:3].astype(np.float64)).T @ relative[:3, :3])
            state.append(s)
            target.append(np.concatenate((relative[:3, 3], residual)).astype(np.float32))
            intrinsics.append(k)
            episode_id.append(number)
        evidence.append({"path": str(Path(path).resolve()), "scene": episode.meta["scene"],
                         "seed": int(episode.meta["seed"]), "frames": episode.num_frames,
                         "archive_sha256": digest(path)})
        print(f"prepared {Path(path).name}: {episode.num_frames - 1} pairs", flush=True)
    count = len(target)
    arrays = {"state": np.asarray(state, np.float32), "target": np.asarray(target, np.float32),
              "intrinsics": np.asarray(intrinsics, np.float32), "episode_id": np.asarray(episode_id, np.int32),
              "images": np.zeros((count, 2, height, width), np.float32),
              "depth": np.zeros((count, 1, height, width), np.float32)}
    return arrays, evidence


def speed_summary(target, dt=.05):
    speed = np.linalg.norm(target[:, :3], axis=1) / dt
    return {"mean_speed_mps": float(speed.mean()), "p50_speed_mps": float(np.median(speed)),
            "p95_speed_mps": float(np.percentile(speed, 95)), "max_speed_mps": float(speed.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", type=Path, required=True)
    parser.add_argument("--nohover", type=Path, required=True, help="directory with train/ and validation/ zips")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--balanced-plan", type=Path)
    parser.add_argument("--old-fraction", type=float, default=.25)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve earlier caches")
    args.out.mkdir(parents=True)
    record = {"kind": "original warp cache + pose-only no-hover flights",
              "old_cache": str(args.old_cache.resolve()),
              "old_cache_record": json.loads((args.old_cache / "cache.json").read_text()),
              "nohover_dir": str(args.nohover.resolve()), "splits": {}}
    sizes = {}
    for split in ["train", "validation"]:
        with np.load(args.old_cache / f"{split}.npz", allow_pickle=False) as archive:
            old = {key: archive[key] for key in archive.files}
        paths = sorted((args.nohover / split).glob("*.zip"))
        if not paths:
            raise ValueError(f"no no-hover episodes in {args.nohover / split}")
        new, evidence = nohover_arrays(paths)
        new["visibility"] = np.zeros((len(new["target"]),) + old["visibility"].shape[1:], old["visibility"].dtype)
        new["episode_id"] += int(old["episode_id"].max()) + 1
        if set(new) != set(old):
            raise ValueError(f"key mismatch: {sorted(set(new) ^ set(old))}")
        merged = {key: np.concatenate((old[key], new[key].astype(old[key].dtype))) for key in old}
        if not all(np.isfinite(value).all() for value in merged.values() if value.dtype.kind == "f"):
            raise ValueError("non-finite data")
        output = args.out / f"{split}.npz"
        np.savez(output, **merged)
        sizes[split] = (len(old["target"]), len(new["target"]))
        record["splits"][split] = {"old_pairs": len(old["target"]), "nohover_pairs": len(new["target"]),
                                   "pairs": len(merged["target"]), "nohover_episodes": evidence,
                                   "old_motion": speed_summary(old["target"]),
                                   "nohover_motion": speed_summary(new["target"]),
                                   "state_std": merged["state"].std(0).tolist(),
                                   "sha256": digest(output)}
        print(split, json.dumps({k: v for k, v in record["splits"][split].items()
                                 if k not in ("nohover_episodes", "state_std")}), flush=True)
    if args.balanced_plan is not None:
        old_count, new_count = sizes["train"]
        total = old_count + new_count
        rng = np.random.default_rng(args.seed)
        draws_old = int(round(args.old_fraction * total))
        plan = np.empty((args.epochs, total), np.int64)
        for epoch in range(args.epochs):
            row = np.concatenate((rng.integers(0, old_count, draws_old),
                                  old_count + rng.integers(0, new_count, total - draws_old)))
            plan[epoch] = rng.permutation(row)
        np.savez(args.balanced_plan, order=plan)
        record["balanced_plan"] = {"path": str(args.balanced_plan.resolve()), "old_fraction": args.old_fraction,
                                   "seed": args.seed, "sha256": digest(args.balanced_plan)}
    (args.out / "cache.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
