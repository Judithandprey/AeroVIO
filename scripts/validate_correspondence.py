"""Actual runtime validation and frozen v6 acceptance before any public scoring."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "workspace/solution")]
from train_correspondence import PROTOCOL, source_hashes
from train_feature_correlation import digest, write_json, eligible
from validate_gyro_bias import Bundle, array_digest
from motion_inference import MotionPredictor
from solution import Solution
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.metrics import evaluate_trajectory


def main():
    cv2.setNumThreads(1)
    out = ROOT / "experiments/correspondence-v6"; raw = ROOT / "results/correspondence-v6-validation"
    if raw.exists() or (out / "selection.json").exists(): raise ValueError("preserve earlier evidence")
    out.mkdir(parents=True, exist_ok=True); raw.mkdir()
    protocol = json.loads(PROTOCOL.read_text())
    paths = {"continuation": ROOT / "models/correspondence-continuation-v6", "supervised": ROOT / "models/correspondence-supervised-v6",
        "v5-temporal": ROOT / "models/correlation-temporal-v5", "v5-single-frame": ROOT / "models/correlation-single-frame-v5-complete",
        "calibrated-gyro": ROOT / "experiments/learned-motion-v1/warp"}
    runs = {name: json.loads((paths[name] / "run.json").read_text()) for name in ("continuation", "supervised")}
    for run in runs.values():
        if run["status"] != "complete" or run["protocol_sha256"] != digest(PROTOCOL) or run["source_sha256"] != source_hashes():
            raise ValueError("fit incomplete or source/protocol changed")
    if runs["continuation"]["initial_weights_sha256"] != runs["supervised"]["initial_weights_sha256"]:
        raise ValueError("initial weights differ")
    if [x["order_sha256"] for x in runs["continuation"]["history"]] != [x["order_sha256"] for x in runs["supervised"]["history"]]:
        raise ValueError("minibatches differ")
    metadata = {name: json.loads((path / "meta.json").read_text()) for name, path in paths.items()}
    metadata["calibrated-gyro"].update(gyro_bias={"enabled": True}, rotation_residual_enabled=False, map={"enabled": False})
    with np.load(ROOT / "data/motion-cache-v2/validation.npz") as data: states, ids = data["state"], data["episode_id"]
    with np.load(ROOT / "data/correlation-cache-v5/validation.npz") as data: priors = data["prior"]
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    record = {"protocol_sha256": digest(PROTOCOL), "matching_initialization_and_minibatches": True,
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "workspace/solution").glob("*.py"))},
        "variants": {name: {"episodes": []} for name in paths}, "checks": []}
    for episode_id, entry in enumerate(manifest["validation"]):
        episode = read_episode(Path(entry["path"]), frames="lazy")
        config, adapted = episode.adapt((320, 240)); frames = list(adapted)
        times = episode.frame_times; windows = [episode.imu_window(i) for i in range(len(times))]
        indices = np.flatnonzero(ids == episode_id); all_predictions = {}
        for name, path in paths.items():
            predictions, outputs = [], []
            predictor = MotionPredictor(Bundle(path, metadata[name]))
            class Recorder:
                meta, size = predictor.meta, predictor.size
                requires_rotation_prior = predictor.requires_rotation_prior
                def predict(self, pair, state, rotation_prior=None):
                    index = indices[len(predictions)]
                    if not np.array_equal(state, states[index]): raise ValueError("raw inputs differ")
                    if self.requires_rotation_prior and not np.allclose(rotation_prior, priors[index], atol=1e-10, rtol=0):
                        raise ValueError("causal prior differs")
                    value = predictor.predict(pair, state, rotation_prior); predictions.append(value)
                    return value
            solution = Solution(copy.deepcopy(config), Recorder())
            for t, frame, imu in zip(times, frames, windows): outputs.append(solution.step(t, frame, imu))
            predictions, outputs = np.asarray(predictions), np.asarray(outputs); all_predictions[name] = predictions
            np.savez_compressed(raw / f"{name}-{episode_id}.npz", predictions=predictions, outputs=outputs)
            metrics = evaluate_trajectory(episode.ground_truth_state(), outputs, times).as_dict()
            if metrics["filled_frames"] or metrics["frames"] != 200: raise ValueError("incomplete validation")
            record["variants"][name]["episodes"].append({"episode_id": episode_id, "metrics": metrics,
                "outputs_sha256": array_digest(outputs), "calibration": solution.gyro_bias_diagnostics})
            if name in ("v5-temporal", "v5-single-frame", "calibrated-gyro"):
                historical = ROOT / "results/correlation-v5-validation" / f"{name.removeprefix('v5-')}-{episode_id}.npz"
                with np.load(historical) as data:
                    if not np.array_equal(outputs, data["outputs"]): raise ValueError("preserved validation reference changed")
        for name in ("supervised", "continuation"):
            maximum = float(np.max(np.abs(all_predictions[name][:, :3] - all_predictions["v5-temporal"][:, :3])))
            if maximum > 1e-7: raise ValueError("translation changed")
            record["checks"].append({"variant": name, "episode_id": episode_id, "translation_max_abs_difference_m": maximum,
                "raw_state_exact": True, "causal_prior_matches": True})
    for name, row in record["variants"].items():
        row["summary"] = {"rotation_deg": float(np.degrees(np.mean([e["metrics"]["rotation_rmse"] for e in row["episodes"]]))),
            "sim3_ate_m": float(np.mean([e["metrics"]["sim3_ate_rmse"] for e in row["episodes"]])),
            "unaligned_ate_m": float(np.mean([e["metrics"]["ate_rmse"] for e in row["episodes"]])), "episodes": row["episodes"]}
        if name in runs:
            delta = {k: abs(row["summary"][k]-runs[name]["best_validation"][k]) for k in ("rotation_deg", "sim3_ate_m", "unaligned_ate_m")}
            if max(delta.values()) > 1e-4: raise ValueError("actual and training rollout metrics differ")
            row["training_rollout_absolute_differences"] = delta
    reference = record["variants"]["calibrated-gyro"]["summary"]
    decisions = {name: eligible(record["variants"][name]["summary"], reference, protocol["acceptance"]) for name in runs}
    match = runs["supervised"]["best_matching"]
    matching_checks = {name+"_"+subset: bool(match[subset]["mean_episode_epe_pixels"] <= .95*other[subset]["mean_episode_epe_pixels"])
        for name, other in (("v5", runs["supervised"]["initial_matching"]), ("continuation", runs["continuation"]["best_matching"]))
        for subset in ("all", "moving")}
    selected = decisions["supervised"]["eligible"] and all(matching_checks.values())
    record.update(status="complete", frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "validation.json", record)
    write_json(out / "selection.json", {"selected_v6_variant": "supervised" if selected else "retain_previous",
        "pose_decisions": decisions, "matching_checks": matching_checks, "public_data_used": False,
        "protocol_sha256": digest(PROTOCOL), "validation_sha256": digest(out / "validation.json"),
        "source_sha256": record["source_sha256"], "model_sha256": {name: metadata[name]["model_sha256"] for name in paths},
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "previous_selection_unchanged": True})
    print(json.dumps({"selected": selected, "pose_decisions": decisions, "matching_checks": matching_checks,
        "metrics": {name: {k: v for k, v in row["summary"].items() if k != "episodes"} for name, row in record["variants"].items()}}, indent=2))


if __name__ == "__main__": main()
