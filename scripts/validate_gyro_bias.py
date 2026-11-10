"""Freeze v4 selection from actual host Solution.step on simulated validation.

Runs every frozen factorial cell twice using the four original validation
flights, never public episodes. Ground truth is used only after each episode's
estimator rollouts. Host validation is not a WASM timing or fuel measurement.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def source_hashes():
    sources = [Path(__file__), *sorted((ROOT / "workspace/solution").glob("*.py"))]
    if not (ROOT / "workspace/solution/gyro_bias.py").exists():
        raise ValueError("gyro calibration helper is not ready")
    return {str(path.relative_to(ROOT)): digest(path) for path in sources}


def summarize(episodes):
    mapping = {
        "mean_episode_rotation_rmse_rad": "rotation_rmse",
        "mean_episode_unaligned_position_rmse_m": "ate_rmse",
        "mean_episode_sim3_ate_m": "sim3_ate_rmse",
        "mean_episode_velocity_rmse_m_s": "velocity_rmse",
        "mean_episode_angular_velocity_rmse_rad_s": "angular_velocity_rmse",
        "mean_episode_rotation_rpe_rad": "rpe_rotation_rmse",
    }
    result = {key: statistics.mean(e["metrics"][metric] for e in episodes) for key, metric in mapping.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("nonfinite aggregate validation metric")
    result["mean_episode_rotation_rmse_deg"] = float(np.degrees(result["mean_episode_rotation_rmse_rad"]))
    result["episodes"] = len(episodes)
    result["frames"] = sum(e["metrics"]["frames"] for e in episodes)
    return result


def select_variant(variants, protocol):
    """Apply the frozen engineering thresholds; no fitted acceptance rule."""
    validation = protocol["validation"]
    rules = validation["eligibility"]
    tolerance = rules["numeric_absolute_tolerance"]
    reference_name = validation["reference"]
    reference = variants[reference_name]
    base = reference["summary"]
    checks = {}
    eligible = []
    order = [v["name"] for v in protocol["variants"]]
    for rank, name in enumerate(order):
        row = variants[name]
        summary = row["summary"]
        if [e["episode_id"] for e in row["episodes"]] != [e["episode_id"] for e in reference["episodes"]]:
            raise ValueError("candidate/reference episode order differs")
        wins = sum(e["metrics"]["rotation_rmse"] < b["metrics"]["rotation_rmse"] - tolerance
                   for e, b in zip(row["episodes"], reference["episodes"]))
        passed = {
            "rotation_material_improvement": summary["mean_episode_rotation_rmse_rad"] <= rules["mean_episode_rotation_rmse_ratio_max"] * base["mean_episode_rotation_rmse_rad"] + tolerance,
            "aligned_ate_nonregression": summary["mean_episode_sim3_ate_m"] <= rules["mean_episode_sim3_ate_ratio_max"] * base["mean_episode_sim3_ate_m"] + tolerance,
            "unaligned_position_nonregression": summary["mean_episode_unaligned_position_rmse_m"] <= rules["mean_episode_unaligned_position_rmse_ratio_max"] * base["mean_episode_unaligned_position_rmse_m"] + tolerance,
            "rotation_win_count": wins >= rules["minimum_rotation_wins"],
        }
        accepted = name != reference_name and all(passed.values())
        checks[name] = {"summary": summary, "rotation_wins": wins, "conditions": passed,
                        "eligible": accepted, "is_reference": name == reference_name}
        if accepted:
            eligible.append((summary["mean_episode_sim3_ate_m"], summary["mean_episode_rotation_rmse_rad"], rank, name))
    chosen = min(eligible)[-1] if eligible else reference_name
    return {"selected_variant": chosen, "candidate_eligibility": checks,
            "decision": "accepted_candidate" if eligible else "retain_current",
            "selection_rule": validation["selection"],
            "win_definition": "episode orientation RMSE lower than current by more than numeric_absolute_tolerance"}


class Bundle:
    def __init__(self, directory, meta):
        self.directory, self.meta = Path(directory), meta

    def read_text(self, name):
        return json.dumps(self.meta) if name == "meta.json" else (self.directory / name).read_text()

    def read_bytes(self, name):
        return (self.directory / name).read_bytes()


class RecordingPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.meta, self.size = predictor.meta, predictor.size
        self.states, self.predictions, self.pair_hashes = [], [], []

    def predict(self, pair, state):
        self.states.append(np.asarray(state).copy())
        self.pair_hashes.append(array_digest(pair))
        prediction = self.predictor.predict(pair, state)
        self.predictions.append(np.asarray(prediction).copy())
        return prediction


def rollout(config, frames, times, imu_windows, model_path, metadata):
    from motion_inference import MotionPredictor
    from solution import Solution
    from viobench.tools.geometry import poses_to_matrices
    predictor = RecordingPredictor(MotionPredictor(Bundle(model_path, metadata)))
    solution = Solution(copy.deepcopy(config), predictor=predictor)
    outputs, diagnostics = [], []
    for t, frame, imu in zip(times, frames, imu_windows):
        value = np.asarray(solution.step(float(t), frame, imu), dtype=np.float64)
        if value.shape != (13,) or not np.isfinite(value).all():
            raise ValueError("incomplete or nonfinite actual Solution.step output")
        outputs.append(value)
        diagnostics.append(json.loads(json.dumps(solution.gyro_bias_diagnostics, allow_nan=False)))
    outputs = np.asarray(outputs)
    if len(predictor.predictions) != len(times) - 1:
        raise ValueError("predictor must be called once for every noninitial frame")
    expected_initial = np.array([0., 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0])
    if not np.array_equal(outputs[0], expected_initial):
        raise ValueError("first-camera identity was changed")
    poses = poses_to_matrices(outputs[:, :7])
    local = np.einsum("nji,nj->ni", poses[:-1, :3, :3], np.diff(poses[:, :3, 3], axis=0))
    return {"outputs": outputs, "states": np.asarray(predictor.states),
            "predictions": np.asarray(predictor.predictions), "image_pair_sha256": np.asarray(predictor.pair_hashes),
            "local_translation": local, "diagnostics": diagnostics}


def invariant_checks(reference, candidate, repeat=False):
    exact_keys = ["states", "predictions", "image_pair_sha256"]
    if repeat:
        exact_keys += ["outputs", "local_translation"]
    for key in exact_keys:
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError("factorial/repeat invariant failed: " + key)
    discrepancy = float(np.max(np.abs(reference["local_translation"] - candidate["local_translation"])))
    prediction_discrepancy = float(np.max(np.abs(candidate["local_translation"] - candidate["predictions"][:, :3])))
    if max(discrepancy, prediction_discrepancy) > 1e-9:
        raise ValueError("camera-local translation differs from unchanged network prediction")
    if repeat and reference["diagnostics"] != candidate["diagnostics"]:
        raise ValueError("repeat calibration states differ")
    return {"raw_states_identical": True, "raw_predictions_identical": True, "image_pairs_identical": True,
            "max_local_translation_difference_m": discrepancy,
            "max_local_translation_vs_prediction_difference_m": prediction_discrepancy,
            "outputs_identical": True if repeat else None,
            "calibration_trace_identical": True if repeat else None}


def calibration_summary(trace, times, protocol):
    final = trace[-1]
    changes = []
    previous = None
    for index, item in enumerate(trace):
        state = (item["status"], item.get("reason"))
        if state != previous:
            changes.append({"frame_index": index, "time_s": float(times[index]), "status": state[0], "reason": state[1]})
            previous = state
    if final["enabled"]:
        if final["status"] not in ("accepted", "rejected"):
            raise ValueError("calibration did not reach a frozen decision")
        if final.get("anchor_t") is not None and final.get("deadline_t") is not None:
            if abs(final["deadline_t"] - final["anchor_t"] - protocol["calibration"]["duration_seconds"]) > 1e-9:
                raise ValueError("helper duration differs from frozen protocol")
        fixed = next(i for i, item in enumerate(trace) if item["status"] in ("accepted", "rejected"))
        if any(item != trace[fixed] for item in trace[fixed:]):
            raise ValueError("calibration state changed after final decision")
    return {"final": final, "transitions": changes}


def verify_causal_prefix(calibrated, uncalibrated, times):
    final = calibrated["diagnostics"][-1]
    if final["status"] == "accepted":
        prefix = times <= final["activation_after_t"] + 1e-6
    else:
        prefix = np.ones(len(times), bool)
    if not np.array_equal(calibrated["outputs"][prefix], uncalibrated["outputs"][prefix]):
        raise ValueError("calibration changed a past/deadline-frame output or a rejected rollout")
    return {"uncorrected_prefix_frames": int(prefix.sum()), "prefix_outputs_identical": True,
            "first_permitted_corrected_frame": int(prefix.sum()) if prefix.sum() < len(times) else None}


def markdown(result, selection):
    lines = ["# Gyro-bias factorial validation", "",
             "Original-flight validation uses four simulated HyperDrone flights with translation and rotation; it is not physical robot/sensor data. Every cell runs actual host Solution.step twice. This is not a WASM speed or fuel measurement.", "",
             "| Variant | Mean episode rotation RMSE (deg) | Mean Sim(3) ATE (m) | Mean unaligned position RMSE (m) | Rotation wins | Eligible change |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, row in result["variants"].items():
        summary, choice = row["summary"], selection["candidate_eligibility"][name]
        lines.append(f"| {name} | {summary['mean_episode_rotation_rmse_deg']:.6f} | {summary['mean_episode_sim3_ate_m']:.6f} | {summary['mean_episode_unaligned_position_rmse_m']:.6f} | {choice['rotation_wins']}/4 | {choice['eligible']} |")
    lines += ["", f"Frozen selected variant: **{selection['selected_variant']}** ({selection['decision']}).", "",
              "Acceptance requires at least 20% lower mean orientation RMSE, no increase in aligned ATE or unaligned position RMSE beyond the fixed 1e-6 tolerance, and orientation improvement in at least three flights. Eligible candidates are ordered by aligned ATE, then orientation error, then protocol order. The current model remains the fallback.", "",
              "## Per-episode metrics", "",
              "| Episode | Variant | Rotation RMSE (deg) | Sim(3) ATE (m) | Unaligned RMSE (m) | Calibration |",
              "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, row in result["variants"].items():
        for episode in row["episodes"]:
            metrics = episode["metrics"]
            lines.append(f"| {episode['episode_id']} | {name} | {np.degrees(metrics['rotation_rmse']):.6f} | {metrics['sim3_ate_rmse']:.6f} | {metrics['ate_rmse']:.6f} | {episode['calibration']['final']['status']} |")
    lines += ["", "## Verification and interpretation", "",
              "All four cells use the same ONNX, raw paired images, raw IMU-derived predictor state, and network outputs. Camera-local translation agrees to numerical tolerance; world positions can differ through rotation composition. Ground truth is read for scoring after the rollouts and never enters Solution.step.", "",
              "All four complete rollouts repeat exactly in outputs, predictions and calibration states. Calibrated cells preserve every output through the activation frame; rejected calibration preserves the entire corresponding uncalibrated trajectory. The current cell also matches the historical original-flight orientation/position audit within the frozen numerical tolerance.", "",
              "Low-variation/prior checks are not proof of rest. Initial calibration assumes the documented one-second hover; small true motion can contaminate the estimate. The factorial effects and their interaction are diagnostic for this fixed estimator, not proof of a sensor-bias mechanism or robustness to other startups.", "",
              "Selection is frozen before any public comparison. Repetition establishes deterministic reproduction, not statistical significance. The JSON stores complete per-episode metrics, bias decisions, invariants, and hashes; raw full trajectories/traces are retained under the ignored results directory."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/gyro-bias-v4.json")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "results/gyro-bias-v4-validation")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments/gyro-bias-v4")
    args = parser.parse_args()
    args.protocol, args.raw_dir, args.out_dir = map(Path.resolve, (args.protocol, args.raw_dir, args.out_dir))
    if args.raw_dir.exists() or any((args.out_dir / name).exists() for name in ["validation.json", "validation.md", "selection.json"]):
        parser.error("validation evidence exists; use new output directories")
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = digest(args.protocol)
    sources_before = source_hashes()
    model_path = resolve(protocol["model"])
    if digest(model_path / "net.onnx") != protocol["onnx_sha256"]:
        raise ValueError("model differs from frozen ONNX")
    original_meta = json.loads((model_path / "meta.json").read_text())
    if original_meta["model_sha256"] != protocol["onnx_sha256"]:
        raise ValueError("model metadata hash differs")
    manifest_path = resolve(protocol["validation"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["validation"]
    if manifest["status"] != "complete" or len(entries) != 4 or {e["scene"] for e in entries} != {"ProcTHOR-Train-5", "ProcTHOR-Train-6"}:
        raise ValueError("requires all four original validation flights, excluding public scene")
    cells = protocol["variants"]
    if len(cells) != 4 or {(v['gyro_bias_enabled'], v['rotation_residual_enabled']) for v in cells} != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError("protocol must define the complete 2x2 factorial")
    upstream = ROOT / "external/viobench"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != json.loads((ROOT / "upstream.lock.json").read_text())["commit"]:
        raise ValueError("upstream revision differs from pin")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("upstream checkout is not clean")
    sys.path.insert(0, str(ROOT / "workspace/solution"))
    import viobench
    if not Path(viobench.__file__).resolve().is_relative_to(upstream.resolve()):
        raise ValueError("wrong imported viobench checkout")
    from viobench.tools.datasets.storage import read_episode
    from viobench.tools.runner.metrics import evaluate_trajectory
    cv2.setNumThreads(1)
    with np.load(ROOT / "data/motion-cache-v2/validation.npz", allow_pickle=False) as cache:
        old_states, old_ids, old_images = cache["state"], cache["episode_id"], cache["images"]
    args.raw_dir.mkdir(parents=True)
    result = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Original simulated validation flights; actual host Solution.step, no public evaluation or estimator GT input",
              "command": [sys.executable, *sys.argv], "protocol_sha256": protocol_hash,
              "model_sha256": protocol["onnx_sha256"], "model_metadata_sha256": digest(model_path / "meta.json"),
              "validation_manifest_sha256": digest(manifest_path), "historical_cache_sha256": digest(ROOT / "data/motion-cache-v2/validation.npz"),
              "source_sha256": sources_before, "upstream_commit": revision,
              "upstream_metric_source_sha256": digest(upstream / "viobench/tools/runner/metrics.py"),
              "packages": {"numpy": np.__version__, "opencv": cv2.__version__},
              "raw_directory": str(args.raw_dir), "variants": {v["name"]: {"settings": v, "episodes": []} for v in cells},
              "repeat_checks": [], "causal_prefix_checks": [], "input_provenance": []}
    for episode_id, entry in enumerate(entries):
        path = resolve(entry["path"])
        episode = read_episode(path, frames="lazy")
        config, adapted = episode.adapt((320, 240))
        frames = np.stack([frame for frame in adapted])
        frames.setflags(write=False)
        windows = [episode.imu_window(i).copy() for i in range(episode.num_frames)]
        for window in windows:
            window.setflags(write=False)
        times = episode.frame_times.copy()
        if len(times) != entry["frames"] or len(times) != 200:
            raise ValueError("validation episode is incomplete")
        input_hash = hashlib.sha256()
        for t, frame, window in zip(times, frames, windows):
            for value in [np.asarray([t]), frame, window]:
                input_hash.update(array_digest(value).encode())
        result["input_provenance"].append({"episode_id": episode_id, "scene": entry["scene"], "path": str(path),
            "seed": entry["seed"], "episode_json_sha256": digest(path / "episode.json"),
            "actual_adapted_rgb_times_imu_stream_sha256": input_hash.hexdigest(), "frames": len(times)})
        runs = [{}, {}]
        episode_dir = args.raw_dir / f"episode-{episode_id}"
        episode_dir.mkdir()
        for repeat in range(2):
            for variant in cells:
                name = variant["name"]
                meta = {**copy.deepcopy(original_meta), "gyro_bias": {"enabled": variant["gyro_bias_enabled"]},
                        "rotation_residual_enabled": variant["rotation_residual_enabled"], "map": {"enabled": False}}
                record = rollout(config, frames, times, windows, model_path, meta)
                runs[repeat][name] = record
                raw_path = episode_dir / f"repeat-{repeat}-{name}.npz"
                np.savez_compressed(raw_path, times=times, **{key: value for key, value in record.items() if key != "diagnostics"})
                diag_path = raw_path.with_suffix(".calibration.json")
                write_json(diag_path, record["diagnostics"])
                record["raw_provenance"] = {"path": str(raw_path), "sha256": digest(raw_path),
                    "calibration_path": str(diag_path), "calibration_sha256": digest(diag_path),
                    "outputs_sha256": array_digest(record["outputs"]), "states_sha256": array_digest(record["states"]),
                    "predictions_sha256": array_digest(record["predictions"]), "image_pair_hashes_sha256": array_digest(record["image_pair_sha256"])}
            current = runs[repeat][protocol["validation"]["reference"]]
            indices = old_ids == episode_id
            if not np.array_equal(current["states"], old_states[indices]) or not np.array_equal(current["image_pair_sha256"], np.asarray([array_digest(pair) for pair in old_images[indices]])):
                raise ValueError("actual raw predictor inputs differ from historical original-flight cache")
            for variant in cells:
                name = variant["name"]
                record = runs[repeat][name]
                record["invariants"] = invariant_checks(current, record)
                if variant["gyro_bias_enabled"]:
                    counterpart = next(v["name"] for v in cells if not v["gyro_bias_enabled"] and v["rotation_residual_enabled"] == variant["rotation_residual_enabled"])
                    result["causal_prefix_checks"].append({"episode_id": episode_id, "repeat": repeat, "variant": name,
                        **verify_causal_prefix(record, runs[repeat][counterpart], times)})
        # Reference poses and twist are first requested after all estimator runs.
        truth = episode.ground_truth_state()
        reference_hash = array_digest(truth)
        for variant in cells:
            name = variant["name"]
            first, second = runs[0][name], runs[1][name]
            repeat_check = invariant_checks(first, second, repeat=True)
            result["repeat_checks"].append({"episode_id": episode_id, "variant": name, **repeat_check})
            metrics = evaluate_trajectory(truth, first["outputs"], times).as_dict()
            if metrics["filled_frames"] or metrics["frames"] != len(times):
                raise ValueError("incomplete output would enter scoring")
            result["variants"][name]["episodes"].append({"episode_id": episode_id, "scene": entry["scene"],
                "metrics": metrics, "reference_state_sha256": reference_hash,
                "calibration": calibration_summary(first["diagnostics"], times, protocol),
                "invariants": first["invariants"], "repeats": [first["raw_provenance"], second["raw_provenance"]]})
        print(f"validated episode {episode_id}: all four complete cells, repeated exactly", flush=True)
    for variant in result["variants"].values():
        variant["summary"] = summarize(variant["episodes"])
    historical_path = ROOT / "experiments/rotation-focus-v3/rollout-audit.json"
    historical = json.loads(historical_path.read_text())["baseline_validation"]
    current = result["variants"][protocol["validation"]["reference"]]["summary"]
    tolerance = protocol["validation"]["eligibility"]["numeric_absolute_tolerance"]
    deltas = {"rotation_rmse_rad": abs(current["mean_episode_rotation_rmse_rad"] - historical["rotation_rmse_mean_rad"]),
              "unaligned_position_rmse_m": abs(current["mean_episode_unaligned_position_rmse_m"] - historical["trajectory_position_rmse_mean_m"])}
    if any(value > tolerance for value in deltas.values()):
        raise ValueError("current validation does not match historical original-flight audit: " + str(deltas))
    result["historical_control_check"] = {"passed": True, "audit_sha256": digest(historical_path),
        "absolute_differences": deltas, "tolerance": tolerance, "raw_inputs_match_original_cache": True}
    if source_hashes() != sources_before or digest(args.protocol) != protocol_hash:
        raise ValueError("source or frozen protocol changed during validation")
    result.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                  validation_complete=True, repeat_verified=True)
    selection = {**select_variant(result["variants"], protocol), "validation_complete": True, "repeat_verified": True,
                 "protocol_sha256": protocol_hash, "source_sha256": sources_before,
                 "model_sha256": protocol["onnx_sha256"], "validation_manifest_sha256": digest(manifest_path),
                 "frozen_utc": datetime.now(timezone.utc).isoformat(), "public_data_used": False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "validation.json", result)
    selection["validation_sha256"] = digest(args.out_dir / "validation.json")
    write_json(args.out_dir / "selection.json", selection)
    (args.out_dir / "validation.md").write_text(markdown(result, selection))
    print(json.dumps({"selected_variant": selection["selected_variant"], "decision": selection["decision"],
                      "summary": {name: row["summary"] for name, row in result["variants"].items()},
                      "output_directory": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
