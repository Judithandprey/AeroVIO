"""Post-fit diagnostic: independent projection and train/validation matching metrics."""
from pathlib import Path
import sys
import json
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from train_correspondence import construct, load, load_labels, matching_metrics, digest, write_json


def main():
    out = ROOT / "experiments/correspondence-v6/matching-audit.json"
    if out.exists(): raise ValueError("preserve audit")
    torch.set_num_threads(4); cv2.setNumThreads(1)
    paths = {"v5-temporal": (ROOT / "models/correlation-temporal-v5", "temporal"),
             "v5-single-frame": (ROOT / "models/correlation-single-frame-v5-complete", "single-frame"),
             "continuation": (ROOT / "models/correspondence-continuation-v6", "temporal"),
             "supervised": (ROOT / "models/correspondence-supervised-v6", "temporal")}
    record = {"source_sha256": digest(Path(__file__)), "scope": "Post-fit diagnosis only; no training or selection changes",
              "projection": {}, "matching": {}}
    for split in ("train", "validation"):
        labels = load_labels(split)
        rng = np.random.default_rng(161803)
        with np.load(ROOT / f"data/motion-cache-v2/{split}.npz") as raw:
            indices = rng.choice(len(raw["images"]), 128, replace=False)
            maximum = 0.; count = 0
            for index in indices:
                fx, fy, cx, cy = raw["intrinsics"][index].astype(np.float64)
                camera = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
                rotation = cv2.Rodrigues(raw["state"][index, :3].astype(np.float64))[0] @ cv2.Rodrigues(raw["target"][index, 3:].astype(np.float64))[0]
                ys, xs = np.nonzero(labels["valid"][index].numpy())
                px, py = 4*xs, 4*ys
                depth = raw["depth"][index, 0, py, px].astype(np.float64)
                points = np.stack(((px-cx)*depth/fx, (py-cy)*depth/fy, depth), axis=1)
                projected, _ = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], raw["target"][index, :3].astype(np.float64), camera, None)
                expected = projected.reshape(-1, 2) - np.stack((px, py), 1)
                observed = labels["displacement"][index].numpy()[:, ys, xs].T*4
                maximum = max(maximum, float(np.abs(observed-expected).max())); count += len(xs)
            if maximum > 5e-4: raise ValueError("independent projection mismatch")
            record["projection"][split] = {"sampled_pairs": 128, "valid_points_checked": count,
                "max_abs_difference_image_pixels": maximum, "tolerance_pixels": 5e-4, "seed": 161803}
        record["matching"][split] = {}
        for name, (directory, variant) in paths.items():
            net, meta = construct(directory, variant)
            record["matching"][split][name] = matching_metrics(net, load(split, meta), labels)
    write_json(out, record)
    print(json.dumps({split: {name: r['all']['mean_episode_epe_pixels'] for name, r in rows.items()}
                      for split, rows in record['matching'].items()}, indent=2))


if __name__ == "__main__": main()
