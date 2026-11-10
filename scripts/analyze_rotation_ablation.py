"""Post-hoc SO(3) diagnostics from completed WASM runs; never estimator input.

Use --run NAME DIR repeatedly, --rotation-on NAME --rotation-off NAME,
and --out experiments/motion-ablation-v2/rotation-diagnostics.
Only metadata/IMU/ground truth are read from the cached episodes; no RGB is decoded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from viobench.tools.geometry import poses_to_matrices, rotation_vectors

ROOT = Path(__file__).resolve().parents[1]
DEG = 180.0 / np.pi


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def increments(rotations):
    return np.swapaxes(rotations[:-1], 1, 2) @ rotations[1:]


def rotate_vectors(rotations, vectors):
    return np.einsum("nij,nj->ni", rotations, vectors)


def vector_stats(vectors, dt):
    """Signed components share fixed first-camera axes unless explicitly labelled.

    Coherence is the norm of the vector sum divided by summed vector magnitudes.
    The sum describes first-order consistency; it is not exact SO(3) integration.
    """
    vectors = np.asarray(vectors)
    dt = np.asarray(dt)
    rate = vectors / dt[:, None]
    total = np.linalg.norm(vectors, axis=1).sum()
    return {
        "mean_vector_deg_per_frame": (vectors.mean(axis=0) * DEG).tolist(),
        "duration_weighted_mean_rate_deg_s": (vectors.sum(axis=0) / dt.sum() * DEG).tolist(),
        "rms_rate_deg_s": float(np.sqrt(np.mean(np.sum(rate * rate, axis=1))) * DEG),
        "rms_deg_per_frame": float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))) * DEG),
        "first_order_sum_deg": (vectors.sum(axis=0) * DEG).tolist(),
        "directional_coherence": float(np.linalg.norm(vectors.sum(axis=0)) / total) if total else 0.0,
    }


def rotation_diagnostics(times, truth, estimate, bucket_seconds=10.0):
    times = np.asarray(times, float)
    truth, estimate = np.asarray(truth), np.asarray(estimate)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("need increasing timestamps and at least two frames")
    if truth.shape != estimate.shape or truth.shape != (len(times), 3, 3):
        raise ValueError("rotation arrays must have matching N x 3 x 3 shape")
    if not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("rotations must be finite")
    dt = np.diff(times)
    delta_gt, delta_est = increments(truth), increments(estimate)
    # Right increment error is expressed at the interval's current camera.
    local = rotation_vectors(np.swapaxes(delta_gt, 1, 2) @ delta_est)
    fixed = rotate_vectors(truth[1:], local)
    # Left absolute error is already in the fixed first-camera coordinate system.
    absolute = rotation_vectors(estimate @ np.swapaxes(truth, 1, 2))
    angles = np.linalg.norm(absolute, axis=1) * DEG
    elapsed = times - times[0]
    buckets = []
    for begin in np.arange(0, elapsed[-1] + 1e-9, bucket_seconds):
        selected = (elapsed >= begin) & (elapsed < begin + bucket_seconds)
        if not selected.any():
            continue
        intervals = selected[1:]
        buckets.append({
            "begin_s": float(begin), "end_s": float(begin + bucket_seconds),
            "frames": int(selected.sum()),
            "absolute_rmse_deg": float(np.sqrt(np.mean(angles[selected] ** 2))),
            "absolute_last_deg": float(angles[selected][-1]),
            "absolute_last_vector_first_camera_deg": (absolute[selected][-1] * DEG).tolist(),
            "increment_error_first_camera": vector_stats(fixed[intervals], dt[intervals]) if intervals.any() else None,
        })
    return {
        "frames": len(times), "duration_s": float(elapsed[-1]),
        "absolute_rmse_deg": float(np.sqrt(np.mean(angles ** 2))),
        "absolute_final_deg": float(angles[-1]), "absolute_max_deg": float(angles.max()),
        "absolute_final_vector_first_camera_deg": (absolute[-1] * DEG).tolist(),
        "increment_error_first_camera": vector_stats(fixed, dt),
        "increment_error_camera_channels": vector_stats(local, dt),
        "time_buckets": buckets,
    }


def paired_diagnostics(times, truth, on, off):
    """Recover the deployed residual from two poses, not a host NN replay.

    Input transforms map camera k into camera 0. With matching image/model inputs,
    dR_off is the deployed gyro-only increment and dR_on = dR_off Exp(residual).
    residual and ideal correction are in current-camera channels; transporting
    each with GT R[k] gives a shared fixed frame for signed averages.
    """
    gt, ron, roff = truth[:, :3, :3], on[:, :3, :3], off[:, :3, :3]
    dg, dn, do = increments(gt), increments(ron), increments(roff)
    residual = rotation_vectors(np.swapaxes(do, 1, 2) @ dn)
    required = rotation_vectors(np.swapaxes(do, 1, 2) @ dg)
    fixed_residual = rotate_vectors(gt[1:], residual)
    fixed_required = rotate_vectors(gt[1:], required)
    dt = np.diff(times)
    # The same translational head is composed through different global rotations.
    local_on = rotate_vectors(np.swapaxes(ron[:-1], 1, 2), np.diff(on[:, :3, 3], axis=0))
    local_off = rotate_vectors(np.swapaxes(roff[:-1], 1, 2), np.diff(off[:, :3, 3], axis=0))
    translation_difference = np.linalg.norm(local_on - local_off, axis=1)
    denom = np.linalg.norm(fixed_residual) * np.linalg.norm(fixed_required)
    return {
        "recovered_residual_first_camera": vector_stats(fixed_residual, dt),
        "recovered_residual_camera_channels": vector_stats(residual, dt),
        "required_correction_first_camera": vector_stats(fixed_required, dt),
        "required_correction_camera_channels": vector_stats(required, dt),
        "residual_required_uncentered_cosine": float(np.sum(fixed_residual * fixed_required) / denom) if denom else None,
        "max_local_translation_difference_m": float(translation_difference.max()),
        "rms_local_translation_difference_m": float(np.sqrt(np.mean(translation_difference ** 2))),
        "final_world_position_difference_m": float(np.linalg.norm(on[-1, :3, 3] - off[-1, :3, 3])),
    }


def load_run(name, directory, expected):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    reports = list(directory.glob("report-*/report.json")) + list(directory.glob("report.json"))
    if len(reports) != 1:
        raise ValueError(f"expected one report in {directory}")
    report_path = reports[0]
    report = json.loads(report_path.read_text())
    if not manifest.get("finished_utc") or manifest.get("exit_code") != 0:
        raise ValueError(f"run is not successfully finished: {name}")
    if sha256(report_path) != manifest["report_sha256"]:
        raise ValueError(f"report hash differs: {name}")
    if [e["hash"] for e in report["episodes"]] != expected or [e["hash"] for e in manifest["episodes"]] != expected:
        raise ValueError("requires the complete pinned public selection in order")
    if report["summary"]["completed"] != len(expected) or any(
            not e["completed"] or e["failure"] or e["metrics"]["filled_frames"] for e in report["episodes"]):
        raise ValueError(f"requires complete trajectories without filled frames: {name}")
    return {"name": name, "directory": str(directory), "manifest": manifest,
            "report": report, "report_path": report_path,
            "manifest_sha256": sha256(manifest_path)}


def validate_pair(on, off):
    a, b = on["manifest"], off["manifest"]
    for key in ["upstream_commit", "runtime_sha256", "selection_sha256", "limits_sha256", "host", "python", "packages"]:
        if a[key] != b[key]:
            raise ValueError(f"rotation pair differs in {key}")
    files = lambda m: {k: v for k, v in m["bundle_files_sha256"].items() if k != "meta.json"}
    if files(a) != files(b):
        raise ValueError("rotation pair must have identical embedded code and ONNX")
    ma, mb = [json.loads(json.dumps(m["embedded_model_metadata"])) for m in (a, b)]
    if not ma.pop("rotation_residual_enabled", True) or mb.pop("rotation_residual_enabled", True):
        raise ValueError("expected enabled residual in on and disabled residual in off")
    if ma != mb or ma.get("map", {}).get("enabled"):
        raise ValueError("pair metadata must differ only in rotation_residual_enabled, with map off")
    if ma.get("input_mode", "image_imu") != "image_imu":
        raise ValueError("this causal pair must retain the image and IMU inputs")


def load_trajectory(run, episode):
    path = run["report_path"].parent / "logs" / episode.hash / "estimate.tum"
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape != (episode.num_frames, 8) or not np.isfinite(rows).all():
        raise ValueError(f"incomplete/nonfinite raw trajectory: {path}")
    if not np.allclose(rows[:, 0], episode.frame_times, atol=1e-7, rtol=0):
        raise ValueError(f"trajectory timestamps differ: {path}")
    norms = np.linalg.norm(rows[:, 4:8], axis=1)
    if not np.allclose(norms, 1, atol=1e-6, rtol=0):
        raise ValueError(f"invalid saved quaternion: {path}")
    return poses_to_matrices(rows[:, 1:8]), {"path": str(path), "sha256": sha256(path)}


def markdown(result):
    lines = ["# Offline rotation ablation diagnostics", "",
             "Computed from saved WASM output poses and reference camera poses, after evaluation. Ground truth never enters the estimator.", "",
             "| Variant | Median rotation RMSE (deg) | Median final error (deg) | Mean step-error rate RMS (deg/s) |",
             "| --- | ---: | ---: | ---: |"]
    for name, row in result["runs"].items():
        s = row["aggregate"]
        lines.append(f"| {name} | {s['median_absolute_rmse_deg']:.5f} | {s['median_absolute_final_deg']:.5f} | {s['mean_increment_error_rate_rms_deg_s']:.5f} |")
    lines += ["", "## Accumulation over time", "",
              "Median across episodes of orientation RMSE within each elapsed-time bucket. These are measured absolute errors, not sums of rotation vectors.", "",
              "| Seconds | " + " | ".join(result["runs"]) + " |",
              "| --- | " + " | ".join("---:" for _ in result["runs"]) + " |"]
    first = next(iter(result["runs"].values()))["episodes"][0]
    for bucket in first["time_buckets"]:
        values = []
        for run in result["runs"].values():
            errors = [b["absolute_rmse_deg"] for e in run["episodes"] for b in e["time_buckets"]
                      if b["begin_s"] == bucket["begin_s"]]
            values.append(f"{statistics.median(errors):.4f}")
        lines.append(f"| {bucket['begin_s']:.0f}–{bucket['end_s']:.0f} | " + " | ".join(values) + " |")
    lines += ["", "## Controlled residual toggle", "", result["interpretation"][0], "",
              "| Episode | On RMSE (deg) | Off RMSE (deg) | On final (deg) | Off final (deg) | Residual mean camera channels (deg/s) | Required mean camera channels (deg/s) |",
              "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for episode in result["pair"]["episodes"]:
        digest = episode["hash"]
        a = next(e for e in result["runs"][result['rotation_on']]["episodes"] if e['hash'] == digest)
        b = next(e for e in result["runs"][result['rotation_off']]["episodes"] if e['hash'] == digest)
        learned = episode['recovered_residual_camera_channels']['duration_weighted_mean_rate_deg_s']
        required = episode['required_correction_camera_channels']['duration_weighted_mean_rate_deg_s']
        fmt = lambda values: ', '.join(f'{x:+.4f}' for x in values)
        lines.append(f"| {digest[:8]} | {a['absolute_rmse_deg']:.5f} | {b['absolute_rmse_deg']:.5f} | {a['absolute_final_deg']:.5f} | {b['absolute_final_deg']:.5f} | {fmt(learned)} | {fmt(required)} |")
    lines += ["", "## Interpretation boundaries", ""]
    lines += ["- " + text for text in result["interpretation"][1:]]
    lines += ["", "The JSON contains fixed-first-camera signed vectors, camera-channel vectors, 10-second buckets, per-file hashes and complete per-episode statistics."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", required=True, metavar=("NAME", "DIRECTORY"))
    parser.add_argument("--rotation-on", required=True)
    parser.add_argument("--rotation-off", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.out.with_suffix(suffix) for suffix in [".json", ".md"]]
    if any(p.exists() for p in outputs):
        parser.error("diagnostic output already exists; choose a new prefix")
    if len({name for name, _ in args.run}) != len(args.run):
        parser.error("run names must be unique")
    expected = [e["hash"] for e in json.loads((ROOT / "configs/public-episodes.json").read_text())]
    runs = {name: load_run(name, directory, expected) for name, directory in args.run}
    validate_pair(runs[args.rotation_on], runs[args.rotation_off])
    result = {
        "schema_version": 1, "analysis_source_sha256": sha256(__file__),
        "rotation_on": args.rotation_on, "rotation_off": args.rotation_off,
        "pose_serialization": "Actual runner estimate.tum files, nine significant digits; recomputation tolerance is 0.00002 degrees.",
        "coordinates": {
            "relative_error": "Log(dR_GT.T @ dR_est), transported by R_GT[k] to the first-camera axes before signed fixed-frame statistics.",
            "absolute_error": "Log(R_est[k] @ R_GT[k].T), in first-camera axes.",
            "residual": "Log(dR_off.T @ dR_on); required correction is Log(dR_off.T @ dR_GT). Camera-channel vectors rotate with camera; fixed-frame copies use R_GT[k].",
        },
        "runs": {}, "pair": {"metadata_only_rotation_toggle_verified": True, "episodes": []},
    }
    for name, run in runs.items():
        result["runs"][name] = {"directory": run["directory"], "manifest_sha256": run["manifest_sha256"],
            "report_sha256": sha256(run["report_path"]), "wasm_sha256": run["manifest"]["wasm_sha256"], "episodes": []}
    from viobench.tools.datasets.sources import episodes
    for episode in episodes(["conta:" + digest for digest in expected], frames="none"):
        truth = poses_to_matrices(episode.ground_truth_relative())
        saved = {}
        gt_hash = hashlib.sha256(episode.ground_truth.astype('<f8').tobytes() + episode.T_body_camera.astype('<f8').tobytes() + episode.frame_times.astype('<f8').tobytes()).hexdigest()
        for name, run in runs.items():
            saved[name], provenance = load_trajectory(run, episode)
            row = rotation_diagnostics(episode.frame_times, truth[:, :3, :3], saved[name][:, :3, :3])
            reported = next(e for e in run["report"]["episodes"] if e["hash"] == episode.hash)
            if abs(row["absolute_rmse_deg"] - reported["metrics"]["rotation_rmse"] * DEG) > 2e-5:
                raise ValueError(f"saved trajectory does not reproduce reported orientation metric: {name} {episode.hash}")
            row.update(hash=episode.hash, trajectory=provenance, reference_arrays_sha256=gt_hash)
            result["runs"][name]["episodes"].append(row)
        paired = paired_diagnostics(episode.frame_times, truth, saved[args.rotation_on], saved[args.rotation_off])
        if paired["max_local_translation_difference_m"] > 2e-6:
            raise ValueError("rotation toggle changed local translation predictions beyond text-rounding tolerance")
        paired["hash"] = episode.hash
        result["pair"]["episodes"].append(paired)
    for run in result["runs"].values():
        rows = run["episodes"]
        run["aggregate"] = {
            "median_absolute_rmse_deg": statistics.median(e["absolute_rmse_deg"] for e in rows),
            "median_absolute_final_deg": statistics.median(e["absolute_final_deg"] for e in rows),
            "mean_increment_error_rate_rms_deg_s": statistics.mean(e["increment_error_first_camera"]["rms_rate_deg_s"] for e in rows),
        }
    on, off = result["runs"][args.rotation_on], result["runs"][args.rotation_off]
    wins = sum(b["absolute_rmse_deg"] < a["absolute_rmse_deg"] for a, b in zip(on["episodes"], off["episodes"]))
    on_wins = sum(a["absolute_rmse_deg"] < b["absolute_rmse_deg"] for a, b in zip(on["episodes"], off["episodes"]))
    on_rmse = on["aggregate"]["median_absolute_rmse_deg"]
    off_rmse = off["aggregate"]["median_absolute_rmse_deg"]
    cosines = [e["residual_required_uncentered_cosine"] for e in result["pair"]["episodes"]]
    valid_cosines = [c for c in cosines if c is not None]
    agreement = (f"The uncentered cosine between recovered residual and reference-required correction ranges from {min(valid_cosines):.4f} to {max(valid_cosines):.4f}. "
                 if valid_cosines else "The residual/correction cosine is undefined because at least one sequence has zero norm. ")
    result["pair"]["rotation_off_rmse_wins"] = wins
    result["pair"]["rotation_on_rmse_wins"] = on_wins
    result["pair"]["median_rmse_change_when_residual_enabled_percent"] = 100 * (on_rmse / off_rmse - 1) if off_rmse else None
    result["interpretation"] = [
        f"Keeping the learned rotation residual gives median orientation RMSE {on_rmse:.5f} degrees versus {off_rmse:.5f} with it disabled. Residual ON wins {on_wins}/{len(expected)} episodes; OFF wins {wins}/{len(expected)}. The bundle/model/source/metadata checks isolate this deployment toggle, and paired camera-local translations match within serialization tolerance. These results must determine whether the residual helps; a larger error than a different baseline does not establish that the residual caused it.",
        agreement + "This measures per-frame directional agreement, not unbiasedness or successful cancellation of all drift components.",
        "The learned residual and the correction required by the reference are recovered from actual relative rotations. Sustained signed errors and their time evolution can diagnose a drift mechanism, but do not prove which training-distribution or label mechanism caused it.",
        "The pinned KLT baseline estimates gyro bias during an initial 0.5-second calibration when its rest test passes, and initializes its pose reference there. The learned runtime does neither. These are additional differences that may contribute to the remaining baseline gap; no matched calibration toggle was performed, so causation is unproven.",
        "Signed averages in fixed axes use the first camera separately for each episode. Camera-channel averages describe sensor-axis bias; they are not fixed-world cumulative rotations. No cross-episode vector is labelled as a universal world bias.",
        "Vector sums/coherence are first-order diagnostics, not exact composition on SO(3). Absolute pose errors and the controlled toggle establish the measured accumulated effect.",
        "World-position differences are expected when identical camera-local translations are composed through different rotations. This diagnostic does not attribute all trajectory error to rotation.",
        "All ten public episodes are post-hoc diagnostics of this fixed model, not new validation data for selecting correction parameters. No bias subtraction or estimator change is fitted here.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    outputs[1].write_text(markdown(result))
    print(json.dumps({"outputs": [str(p) for p in outputs], "rotation_off_rmse_wins": wins,
                      "aggregates": {name: row["aggregate"] for name, row in result["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
