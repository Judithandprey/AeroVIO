"""Per-set summary of viobench evaluate reports, with a tilt/yaw split of the rotation error.

For each report directory prints the episode means of: unaligned ATE, Sim(3) ATE,
Sim(3) scale, rotation RMSE, tilt RMSE, yaw RMSE, and the max fuel per frame.
Tilt is the angle between the true and the estimated up direction; yaw is the
twist of the world-frame error rotation about the true up direction.
Episodes are found by archive SHA-1 in --data directories and the conta cache.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices


def index_archives(directories):
    found = {}
    for directory in directories:
        for path in Path(directory).expanduser().iterdir():
            if path.suffix == ".zip":
                found[hashlib.sha1(path.read_bytes()).hexdigest()] = path
            elif len(path.name) == 40 and path.is_file():
                found[path.name] = path
    return found


def tilt_yaw(episode_path, estimate_path):
    episode = read_episode(episode_path, frames="none")
    gt = poses_to_matrices(episode.ground_truth_relative())[:, :3, :3]
    rows = np.loadtxt(estimate_path, comments="#")
    est = poses_to_matrices(rows[:, 1:8])[:, :3, :3]
    if len(est) != len(gt):
        raise ValueError(f"{estimate_path}: {len(est)} poses vs {len(gt)} ground truth")
    T_wc0 = poses_to_matrices(episode.camera_poses_world()[:1])[0]
    up = T_wc0[:3, :3].T @ np.array([0., 0., 1.])  # world up in camera-0 axes
    up_true = np.einsum("nji,j->ni", gt, up)
    up_est = np.einsum("nji,j->ni", est, up)
    tilt = np.arccos(np.clip(np.sum(up_true * up_est, axis=1), -1, 1))
    error = est @ np.swapaxes(gt, 1, 2)  # world-frame error, R_est R_gt^T
    # Twist angle about `up`: quaternion of the error, projected on the axis.
    w = np.sqrt(np.clip(1 + np.trace(error, axis1=1, axis2=2), 0, None)) / 2
    v = np.stack([error[:, 2, 1] - error[:, 1, 2], error[:, 0, 2] - error[:, 2, 0],
                  error[:, 1, 0] - error[:, 0, 1]], axis=1) / (4 * np.maximum(w, 1e-12)[:, None])
    # v = sin(theta/2) * axis (valid below 180 degrees)
    yaw = 2 * np.arctan2(np.abs(v @ up), w)
    rms = lambda x: float(np.degrees(np.sqrt(np.mean(x ** 2))))
    return rms(tilt), rms(yaw)


def summarize(report_dir, archives):
    report = json.loads((Path(report_dir) / "report.json").read_text())
    rows = []
    for episode in report["episodes"]:
        m = episode["metrics"]
        tilt = yaw = float("nan")
        estimate = Path(report_dir) / "logs" / episode["hash"] / "estimate.tum"
        if episode["hash"] in archives and estimate.exists():
            tilt, yaw = tilt_yaw(archives[episode["hash"]], estimate)
        rows.append({"hash": episode["hash"], "completed": episode["completed"], "ate": m["ate_rmse"],
                     "sim3_ate": m["sim3_ate_rmse"], "scale": m["sim3_scale"],
                     "rot": float(np.degrees(m["rotation_rmse"])), "tilt": tilt, "yaw": yaw,
                     "fuel": episode["fuel_per_frame_max"]})
    keys = ["ate", "sim3_ate", "scale", "rot", "tilt", "yaw"]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    mean["fuel"] = max(row["fuel"] for row in rows)
    mean["completed"] = f"{sum(row['completed'] for row in rows)}/{len(rows)}"
    return mean, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--data", nargs="+", default=["data/val-nohover", "data/val-agile", "~/.cache/conta"])
    parser.add_argument("--json", type=Path, help="write per-episode rows here")
    args = parser.parse_args()
    archives = index_archives(args.data)
    out = {}
    print("| Report | Done | ATE (m) | Sim(3) ATE (m) | Sim(3) scale | Rot (°) | Tilt (°) | Yaw (°) | Max fuel |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for report in args.reports:
        mean, rows = summarize(report, archives)
        out[str(report)] = {"mean": mean, "episodes": rows}
        print(f"| {report} | {mean['completed']} | {mean['ate']:.3f} | {mean['sim3_ate']:.3f} | {mean['scale']:.2f} | "
              f"{mean['rot']:.1f} | {mean['tilt']:.1f} | {mean['yaw']:.1f} | {mean['fuel']:,} |")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
