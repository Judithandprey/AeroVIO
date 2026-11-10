"""Convert separately generated episodes into deterministic frame-pair training arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "workspace/solution")]
from preintegrate import make_state, prepare_frame, exp_so3, log_so3
from training.motion_model import backward_warp
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices, invert


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare(entries, width=64, height=48):
    arrays = {key: [] for key in ["images", "state", "target", "depth", "source_depth", "intrinsics", "episode_id"]}
    evidence = []
    for episode_id, entry in enumerate(entries):
        path = resolve(entry["path"])
        episode = read_episode(path)
        config, frames = episode.adapt((320, 240))
        gray = np.stack([prepare_frame(frame, (width, height)) for frame in frames])
        poses = poses_to_matrices(episode.ground_truth_relative())
        depth_path = resolve(entry["depth_path"])
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != (episode.num_frames, height, width):
            raise ValueError(f"depth/image alignment mismatch: {depth.shape}, episode {episode.num_frames}")
        depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        intrinsics = np.array([config["fx"] * width / config["width"],
                               config["fy"] * height / config["height"],
                               (config["cx"] + .5) * width / config["width"] - .5,
                               (config["cy"] + .5) * height / config["height"] - .5], dtype=np.float32)
        for index in range(1, episode.num_frames):
            t0, t1 = episode.frame_times[index - 1:index + 1]
            state = make_state(episode.imu_window(index), config["T_cam_imu"], config, t0, t1)
            relative = invert(poses[index - 1]) @ poses[index]
            residual = log_so3(exp_so3(state[:3].astype(np.float64)).T @ relative[:3, :3])
            target = np.concatenate((relative[:3, 3], residual)).astype(np.float32)
            arrays["images"].append(np.stack((gray[index - 1], gray[index])))
            arrays["state"].append(state)
            arrays["target"].append(target)
            arrays["depth"].append(depth[index:index + 1])
            arrays["source_depth"].append(depth[index - 1:index])
            arrays["intrinsics"].append(intrinsics)
            arrays["episode_id"].append(episode_id)
        evidence.append({"path": str(path), "scene": entry["scene"], "frames": episode.num_frames,
                         "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                         "episode_json_sha256": hashlib.sha256((path / "episode.json").read_bytes()).hexdigest()})
        print(f"prepared {entry['scene']}: {episode.num_frames - 1} pairs", flush=True)
    arrays = {key: np.asarray(value, dtype=np.int32 if key == "episode_id" else np.float32)
              for key, value in arrays.items()}
    masks = []
    with torch.no_grad():
        for start in range(0, len(arrays["target"]), 128):
            batch = {key: torch.from_numpy(value[start:start + 128]) for key, value in arrays.items()}
            _, in_bounds, projected_z, grid = backward_warp(batch["images"][:, :1], batch["depth"],
                batch["intrinsics"], batch["target"], batch["state"][:, :3])
            old_depth = F.grid_sample(batch["source_depth"], grid, align_corners=False)
            valid = in_bounds & (batch["depth"] > 0) & (old_depth > 0)
            # Fixed true-geometry visibility: exclude disocclusions and depth boundaries.
            valid &= (old_depth - projected_z).abs() < torch.maximum(old_depth * .03, old_depth.new_tensor(.04))
            masks.append(valid.numpy())
    arrays["visibility"] = np.concatenate(masks)
    arrays.pop("source_depth")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("non-finite training data")
    return arrays, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("status") != "complete" or not manifest.get("train") or not manifest.get("validation"):
        raise ValueError("use a completed dataset with nonempty train and validation splits")
    train_scenes = {entry["scene"] for entry in manifest["train"]}
    validation_scenes = {entry["scene"] for entry in manifest["validation"]}
    if train_scenes & validation_scenes or "ProcTHOR-Train-1" in train_scenes | validation_scenes:
        raise ValueError("train, validation, and public comparison scenes must be disjoint")
    if args.out.exists():
        raise ValueError("cache output already exists; preserve earlier evidence")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    record = {"source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "source_manifest": str(args.manifest.resolve()), "input_width": 64, "input_height": 48,
              "guest_resolution": [320, 240], "splits": {}}
    for split in ["train", "validation"]:
        arrays, evidence = prepare(manifest[split])
        output = args.out / f"{split}.npz"
        np.savez_compressed(output, **arrays)
        record["splits"][split] = {"pairs": len(arrays["target"]), "episodes": evidence,
                                  "visible_pixel_fraction": float(arrays["visibility"].mean()),
                                  "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (args.out / "cache.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
