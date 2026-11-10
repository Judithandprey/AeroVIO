"""Post-hoc IMU-only replay and observable-prefix checks for completed v4 runs.

Run only after all public timing work is complete. This does not read RGB or
use ground truth, run a model, tune parameters, or change any estimator output.
Host helper replay is distinct from observing guest internal calibration state.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace/solution"))
from gyro_bias import InitialGyroBias
from preintegrate import IMU_TIMESTAMP_TOLERANCE_SECONDS
from analyze_rotation_ablation import load_run
from viobench.tools.geometry import poses_to_matrices, invert

HORIZON_SECONDS = 1.1
ROTATION_FROBENIUS_TOLERANCE = 2e-6  # saved TUM quaternion/text serialization
CELLS = ("current", "gyro-only", "calibrated-residual", "calibrated-gyro")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    array = np.ascontiguousarray(value, dtype=np.float64)
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def validate_manifests(runs, protocol):
    reference = runs["current"]["manifest"]
    expected = {item["name"]: item for item in protocol["variants"]}
    base_meta = None
    for name, run in runs.items():
        manifest = run["manifest"]
        for key in ("upstream_commit", "runtime_sha256", "selection_sha256", "limits_sha256"):
            if manifest[key] != reference[key]:
                raise ValueError("factorial run mismatch: " + key)
        files = {k: v for k, v in manifest["bundle_files_sha256"].items() if k != "meta.json"}
        reference_files = {k: v for k, v in reference["bundle_files_sha256"].items() if k != "meta.json"}
        if files != reference_files or files["net.onnx"] != protocol["onnx_sha256"]:
            raise ValueError("factorial payloads/model are not identical")
        for source in ("gyro_bias.py", "preintegrate.py", "solution.py", "motion_inference.py"):
            if files[source] != sha(ROOT / "workspace/solution" / source):
                raise ValueError("host replay source differs from actual WASM: " + source)
        meta = json.loads(json.dumps(manifest["embedded_model_metadata"]))
        bias = meta.pop("gyro_bias", {"enabled": False})
        residual = meta.pop("rotation_residual_enabled", True)
        settings = expected[name]
        if bias != {"enabled": settings["gyro_bias_enabled"]} or residual is not settings["rotation_residual_enabled"]:
            raise ValueError("actual metadata does not match factorial cell: " + name)
        if meta.get("map", {}).get("enabled", False) or meta.get("input_mode", "image_imu") != "image_imu":
            raise ValueError("audit requires map-off image+IMU models")
        if base_meta is None:
            base_meta = meta
        elif meta != base_meta:
            raise ValueError("factorial metadata differs beyond the two switches")


def replay_startup(episode):
    helper = InitialGyroBias()
    trace, windows = [], []
    horizon = float(episode.frame_times[0]) + HORIZON_SECONDS
    for index, t in enumerate(episode.frame_times):
        if t > horizon + IMU_TIMESTAMP_TOLERANCE_SECONDS:
            break
        window = episode.imu_window(index).copy()
        windows.append(window)
        helper.observe_after_output(float(t), window)
        diagnostics = helper.diagnostics()
        trace.append({"frame_index": index, "time_s": float(t), "diagnostics": diagnostics})
        if helper.anchor_t is not None:
            horizon = helper.anchor_t + HORIZON_SECONDS
    if helper.status not in ("accepted", "rejected"):
        raise ValueError("startup replay did not finish within fixed horizon")
    fixed = next(item for item in trace if item["diagnostics"]["status"] in ("accepted", "rejected"))
    if any(item["diagnostics"] != fixed["diagnostics"] for item in trace[fixed["frame_index"]:]):
        raise ValueError("replayed state changed after its frozen decision")
    return {"diagnostics": helper.diagnostics(), "finalized_frame_index": fixed["frame_index"],
            "replayed_frames": len(trace), "first_frame_time_s": float(episode.frame_times[0]),
            "last_replayed_frame_time_s": trace[-1]["time_s"],
            "imu_rows_replayed": sum(len(window) for window in windows),
            "imu_windows_sha256": hashlib.sha256("".join(array_sha(window) for window in windows).encode()).hexdigest()}, windows


def saved_prefix(run, episode, frames):
    directory = run["report_path"].parent / "logs" / episode.hash
    pose_path, twist_path = directory / "estimate.tum", directory / "twist.txt"
    pose = np.loadtxt(pose_path, comments="#", ndmin=2, max_rows=frames)
    twist = np.loadtxt(twist_path, comments="#", ndmin=2, max_rows=frames)
    if pose.shape != (frames, 8) or twist.shape != (frames, 7):
        raise ValueError("unexpected saved prefix dimensions")
    if not np.isfinite(pose).all() or not np.isfinite(twist).all():
        raise ValueError("nonfinite actual saved output")
    times = episode.frame_times[:frames]
    if not np.allclose(pose[:, 0], times, atol=1e-7, rtol=0) or not np.allclose(twist[:, 0], times, atol=1e-7, rtol=0):
        raise ValueError("saved prefix timestamps differ from pinned episode")
    if not np.allclose(np.linalg.norm(pose[:, 4:8], axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("invalid saved quaternion")
    return {"pose": pose, "twist": twist, "matrices": poses_to_matrices(pose[:, 1:8]),
            "provenance": {"estimate_path": str(pose_path), "estimate_sha256": sha(pose_path),
                           "twist_path": str(twist_path), "twist_sha256": sha(twist_path)}}


def independent_rotation(window, start_t, end_t, bias, T_cam_imu):
    """Independent explicit sample product; does not call runtime integration."""
    rotation = np.eye(3)
    previous = float(start_t)
    for row in np.asarray(window, dtype=np.float64):
        timestamp = min(float(row[0]), float(end_t))
        dt = timestamp - previous
        if dt <= 0:
            continue
        rotation = rotation @ cv2.Rodrigues((row[4:7] - bias) * dt)[0]
        previous = timestamp
        if timestamp >= end_t:
            break
    camera = np.asarray(T_cam_imu, dtype=np.float64).reshape(4, 4)[:3, :3]
    return camera @ rotation @ camera.T


def audit_episode(episode, runs):
    replay, windows = replay_startup(episode)
    accepted = replay["diagnostics"]["status"] == "accepted"
    activation = replay["finalized_frame_index"] + 1 if accepted else None
    prefix_count = activation if accepted else replay["replayed_frames"]
    read_count = prefix_count + 1 if accepted else prefix_count
    if read_count > len(windows):
        raise ValueError("first permitted correction falls outside audited IMU horizon")
    saved = {name: saved_prefix(run, episode, read_count) for name, run in runs.items()}
    checks = {}
    for calibrated, raw in [("calibrated-residual", "current"), ("calibrated-gyro", "gyro-only")]:
        pose_equal = np.array_equal(saved[calibrated]["pose"][:prefix_count], saved[raw]["pose"][:prefix_count])
        twist_equal = np.array_equal(saved[calibrated]["twist"][:prefix_count], saved[raw]["twist"][:prefix_count])
        if not (pose_equal and twist_equal):
            raise ValueError("actual WASM changed its pre-activation output prefix")
        row = {"uncorrected_prefix_frames": prefix_count, "pose_prefix_exact": True, "twist_prefix_exact": True}
        if not accepted:
            row["full_pose_file_identical_after_rejection"] = saved[calibrated]["provenance"]["estimate_sha256"] == saved[raw]["provenance"]["estimate_sha256"]
            row["full_twist_file_identical_after_rejection"] = saved[calibrated]["provenance"]["twist_sha256"] == saved[raw]["provenance"]["twist_sha256"]
            if not all(value for key, value in row.items() if key.startswith("full_")):
                raise ValueError("rejected calibration changed actual WASM output files")
        checks[calibrated] = row
    correction = None
    if accepted:
        index = activation
        start_t, end_t = [float(t) for t in episode.frame_times[index - 1:index + 1]]
        T_cam_imu = invert(episode.T_body_camera)  # calibration metadata only; no frame adapter
        independent = independent_rotation(windows[index], start_t, end_t,
                                            np.asarray(replay["diagnostics"]["bias_imu"]), T_cam_imu)
        increments = {name: record["matrices"][index - 1, :3, :3].T @ record["matrices"][index, :3, :3]
                      for name, record in saved.items()}
        independent_raw = independent_rotation(windows[index], start_t, end_t, np.zeros(3), T_cam_imu)
        raw_error = float(np.linalg.norm(independent_raw - increments["gyro-only"], ord="fro"))
        if raw_error > ROTATION_FROBENIUS_TOLERANCE:
            raise ValueError("raw gyro check does not match the saved input interval")
        residual = increments["gyro-only"].T @ increments["current"]
        expected = {"calibrated-gyro": independent, "calibrated-residual": independent @ residual}
        correction = {"frame_index": index, "interval_start_s": start_t, "interval_end_s": end_t,
                      "imu_window_sha256": array_sha(windows[index]),
                      "independent_raw_gyro_frobenius_error": raw_error,
                      "recovered_learned_residual_rotation": residual.tolist(),
                      "independent_debiased_imu_rotation": independent.tolist(), "cells": {}}
        for name, raw in [("calibrated-gyro", "gyro-only"), ("calibrated-residual", "current")]:
            error = float(np.linalg.norm(increments[name] - expected[name], ord="fro"))
            unchanged_error = float(np.linalg.norm(increments[name] - increments[raw], ord="fro"))
            if error > ROTATION_FROBENIUS_TOLERANCE:
                raise ValueError("actual corrected rotation does not match independent expectation: " + name)
            correction["cells"][name] = {"expected_rotation_frobenius_error": error,
                "uncalibrated_alternative_frobenius_error": unchanged_error,
                "correction_resolved_above_saved_text_tolerance": unchanged_error > ROTATION_FROBENIUS_TOLERANCE}
    return {"episode": episode.hash, "replay": replay, "actual_saved_prefix_checks": checks,
            "first_post_window_rotation_check": correction,
            "saved_outputs": {name: record["provenance"] for name, record in saved.items()}}


def markdown(result):
    lines = ["# Public calibration replay and observable output audit", "", result["evidence_scope"], "",
             "The replay reads no RGB and uses no ground-truth fields. The upstream frames='none' storage reader loads metadata/IMU/reference arrays, but this script consults only timestamps, the initial IMU windows, and camera/IMU extrinsics.", "",
             "| Episode | Host replay decision | Samples | Finalized frame | First permitted corrected frame | Largest corrected-rotation check error (Frobenius) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in result["episodes"]:
        replay = row["replay"]
        correction = row["first_post_window_rotation_check"]
        error = max(v["expected_rotation_frobenius_error"] for v in correction["cells"].values()) if correction else None
        error_text = f"{error:.3g}" if error is not None else "n/a (rejected)"
        frame_text = str(correction["frame_index"]) if correction else "none"
        lines.append(f"| {row['episode'][:8]} | {replay['diagnostics']['status']} | {replay['diagnostics']['samples']} | {replay['finalized_frame_index']} | {frame_text} | {error_text} |")
    lines += ["", "Frame indices are zero-based. Both calibrated variants exactly reproduce their corresponding uncalibrated saved pose/twist prefixes through the inferred decision frame. For a rejection, the full saved output-file hashes must also match.", "",
              "The first later rotation is checked against an independent ordered product of gyro-minus-bias exponentials and camera-frame conjugation. The unchanged learned residual is recovered from the actual current/gyro-only saved relative rotations. Comparisons use a fixed 2e-6 Frobenius tolerance for serialized quaternion precision; the JSON also states whether the difference from no correction is large enough to resolve.", "",
              "This is a post-hoc behavior and provenance check. It does not directly observe a WASM internal bias variable, prove the rest assumption, establish bias accuracy, or select/tune a model.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/gyro-bias-v4/public-calibration-audit")
    args = parser.parse_args()
    if args.out.with_suffix(".json").exists() or args.out.with_suffix(".md").exists():
        parser.error("preserve earlier audit; choose a new output prefix")
    protocol_path = ROOT / "configs/gyro-bias-v4.json"
    protocol = json.loads(protocol_path.read_text())
    selection_path = ROOT / "configs/public-episodes.json"
    expected = [entry["hash"] for entry in json.loads(selection_path.read_text())]
    runs = {name: load_run(name, ROOT / ("results/gyro-v4-" + name), expected) for name in CELLS}
    validate_manifests(runs, protocol)
    revision = subprocess.check_output(["git", "-C", str(ROOT / "external/viobench"), "rev-parse", "HEAD"], text=True).strip()
    if revision != json.loads((ROOT / "upstream.lock.json").read_text())["commit"] or revision != runs["current"]["manifest"]["upstream_commit"]:
        raise ValueError("upstream reader revision differs from pinned runs")
    import viobench
    if not Path(viobench.__file__).resolve().is_relative_to((ROOT / "external/viobench").resolve()):
        raise ValueError("reader must come from the pinned checkout")
    from viobench.tools.datasets.sources import episodes
    result = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
              "evidence_scope": "Calibration state is inferred by host replay of the exact deployed helper. Actual WASM internal diagnostics were not logged. Saved output prefixes and an independent first-correction calculation verify observable behavior separately.",
              "public_gt_used": False, "rgb_decoded": False, "model_inference_run": False,
              "replay_horizon_seconds": HORIZON_SECONDS,
              "rotation_frobenius_tolerance": ROTATION_FROBENIUS_TOLERANCE,
              "protocol_sha256": sha(protocol_path), "selection_sha256": sha(selection_path),
              "upstream_commit": revision, "script_sha256": sha(Path(__file__)),
              "helper_sha256": sha(ROOT / "workspace/solution/gyro_bias.py"),
              "runs": {name: {"manifest_sha256": run["manifest_sha256"], "report_sha256": sha(run["report_path"]),
                               "wasm_sha256": run["manifest"]["wasm_sha256"]} for name, run in runs.items()},
              "episodes": []}
    for episode in episodes(["conta:" + value for value in expected], frames="none"):
        result["episodes"].append(audit_episode(episode, runs))
    result["replay_decision_counts"] = {status: sum(row["replay"]["diagnostics"]["status"] == status for row in result["episodes"])
                                         for status in ("accepted", "rejected")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    args.out.with_suffix(".md").write_text(markdown(result))
    print(json.dumps({"status": result["status"], "episodes": len(result["episodes"]),
                      "replay_decision_counts": result["replay_decision_counts"]}, indent=2))


if __name__ == "__main__":
    main()
