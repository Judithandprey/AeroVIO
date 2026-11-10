"""Prepare reproducible, scene-disjoint pure-rotation auxiliary examples."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.rotation_augmentation import make_rotation_pair, sample_rotation, exp_so3
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def prepare_split(entries, seed, crop, max_angle):
    rng = np.random.default_rng(seed)
    rows, episodes = [], []
    rejected = 0
    max_rotation_error = 0.
    for episode_id, entry in enumerate(entries):
        path = resolve(entry["path"])
        ep = read_episode(path)
        ep.check_consistency()
        if ep.config["camera"].get("distortion"):
            raise ValueError("rotation reprojection requires a calibrated undistorted pinhole image")
        config, frames = ep.adapt((320, 240))
        world_camera = poses_to_matrices(ep.camera_poses_world())
        imu_config = ep.config["imu"]
        for index, frame in enumerate(frames):
            axis_id = len(rows) % 4
            for attempt in range(100):
                vector = sample_rotation(rng, axis_id, max_angle)
                try:
                    row = make_rotation_pair(frame, config, world_camera[index, :3, :3], imu_config,
                                             vector, rng, crop=crop, duration=.05)
                    break
                except ValueError as error:
                    if "outside source support" not in str(error):
                        raise
                    rejected += 1
            else:
                raise RuntimeError("100 rejected virtual rotations; inspect camera crop support")
            angle = np.linalg.norm(vector)
            signed_angle = angle * (1. if vector[np.argmax(np.abs(vector))] >= 0 else -1.)
            axis_vector = vector / signed_angle if signed_angle != 0 else np.array([1., 0., 0.])
            row.update({"episode_id": np.int32(episode_id), "frame_index": np.int32(index),
                        "axis_id": np.int32(axis_id), "rotation_vector_gt": vector.astype(np.float32),
                        "angle_degrees": np.float32(angle * 180 / np.pi),
                        "signed_angle_degrees": np.float32(signed_angle * 180 / np.pi),
                        "axis_vector": axis_vector.astype(np.float32)})
            reconstructed = exp_so3(row["state"][:3]) @ exp_so3(row["rotation_residual"])
            max_rotation_error = max(max_rotation_error, float(np.linalg.norm(reconstructed - row["rotation_gt"])))
            rows.append(row)
        episodes.append({"episode_id": episode_id, "path": str(path), "scene": entry["scene"],
                         "frames": ep.num_frames, "episode_json_sha256": digest(path / "episode.json"),
                         "source_archive_sha256": entry["archive_sha256"],
                         "imu_priors": imu_config})
        print(json.dumps({"scene": entry["scene"], "source": ep.name, "pairs": len(rows), "rejected": rejected}), flush=True)
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("nonfinite augmentation data")
    assert max_rotation_error < 1e-6
    audit = {"all_source_rays_inside_margin_pixels": 1.,
             "minimum_source_clearance_pixels": float(arrays["source_clearance_pixels"].min()),
             "maximum_rotation_reconstruction_frobenius_error": max_rotation_error,
             "visible_fraction": float(arrays["visibility"].mean()),
             "axis_counts": np.bincount(arrays["axis_id"], minlength=4).tolist(),
             "angle_degrees_quantiles": np.percentile(arrays["angle_degrees"], [0, 25, 50, 75, 100]).tolist(),
             "imu_residual_radians_std": arrays["rotation_residual"].std(0).tolist(),
             "rejected_attempts": rejected}
    return arrays, episodes, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve prior evidence and choose a new directory")
    manifest = json.loads(args.manifest.read_text())
    real_record = json.loads((args.real_cache / "cache.json").read_text())
    if manifest.get("status") != "complete" or real_record["source_manifest_sha256"] != digest(args.manifest):
        raise ValueError("use the same completed manifest as the existing real-motion cache")
    scene_sets = {s: {entry["scene"] for entry in manifest[s]} for s in ("train", "validation")}
    if not all(scene_sets.values()) or scene_sets["train"] & scene_sets["validation"] or "ProcTHOR-Train-1" in set.union(*scene_sets.values()):
        raise ValueError("augmentation train/validation/public scenes must be disjoint")
    cache_hashes = {s: digest(args.real_cache / f"{s}.npz") for s in scene_sets}
    if any(cache_hashes[s] != real_record["splits"][s]["sha256"] for s in scene_sets):
        raise ValueError("real-motion cache changed since its provenance record")
    for split in scene_sets:
        expected = [(str(resolve(e["path"])), e["scene"]) for e in manifest[split]]
        actual = [(e["path"], e["scene"]) for e in real_record["splits"][split]["episodes"]]
        if expected != actual:
            raise ValueError("real-motion cache source episodes differ from the manifest")
    args.out.mkdir(parents=True)
    record = {"schema_version": 1, "kind": "pure_rotation_augmentation", "status": "generating",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_manifest": {"path": str(args.manifest.resolve()), "sha256": digest(args.manifest)},
              "source_real_cache": {"path": str(args.real_cache.resolve()), "cache_json_sha256": digest(args.real_cache / "cache.json"),
                                    "train_sha256": cache_hashes["train"], "validation_sha256": cache_hashes["validation"]},
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                [Path(__file__), ROOT / "training/rotation_augmentation.py", ROOT / "workspace/solution/preintegrate.py"]},
              "seeds": {"train": 314159, "validation": 314160},
              "config": {"crop": .85, "max_angle_degrees": 3., "duration_s": .05,
                         "axis_classes": ["camera_x_pitch", "camera_y_yaw", "camera_z_roll", "mixed"],
                         "sampling": "balanced axis classes; uniform magnitude 0 to 3 deg; independent random sign",
                         "signed_angle_convention": "axis_vector has nonnegative dominant component; rotation_vector = axis_vector * signed_angle_radians",
                         "preprocessing": "upstream RGB640-to320 area resize, symmetric SO3 reprojection at320, shared grayscale64x48 area resize",
                         "imu": "constant camera-centre angular velocity, actual extrinsic/gravity/centripetal acceleration; independent configured bias/noise/randomwalk",
                         "supervision": "rotation only; no synthetic translation target or depth label"},
              "splits": {}, "geometry_audit": {},
              "limitations": ["One-frame textures model static geometry under pure camera-centre rotation; no translation/disocclusion generation.",
                              "Fixed virtual crop changes calibration; network state contains the updated intrinsic values.",
                              "Synthetic bias/noise follows stored sensor priors, not a measured new sensor sequence.",
                              "Auxiliary rotation supervision must be mixed with real-motion translation/rotation losses."]}
    record_path = args.out / "cache.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    started = time.perf_counter()
    for split in ("train", "validation"):
        arrays, episodes, audit = prepare_split(manifest[split], record["seeds"][split], .85, 3.)
        path = args.out / f"{split}.npz"
        np.savez_compressed(path, **arrays)
        record["splits"][split] = {"pairs": len(arrays["state"]), "sha256": digest(path),
                                   "scenes": sorted(scene_sets[split]), "episodes": episodes}
        record["geometry_audit"][split] = audit
        record_path.write_text(json.dumps(record, indent=2) + "\n")
    record.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_s": time.perf_counter() - started})
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"path": str(record_path), "status": record["status"], "elapsed_s": record["elapsed_s"],
                      "audit": record["geometry_audit"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
