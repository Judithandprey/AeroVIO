"""Verify v5 exported candidates with actual causal Solution.step, then freeze decision."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "workspace/solution"), str(ROOT / "scripts")]
from train_feature_correlation import PROTOCOL, CACHE, digest, write_json, eligible
from validate_gyro_bias import Bundle, array_digest
from motion_inference import MotionPredictor
from solution import Solution
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.metrics import evaluate_trajectory


def main():
    cv2.setNumThreads(1)
    out = ROOT / "experiments/feature-correlation-v5"
    raw = ROOT / "results/correlation-v5-validation"
    if raw.exists() or (out / "selection.json").exists():
        raise ValueError("preserve existing evidence")
    out.mkdir(parents=True, exist_ok=True); raw.mkdir()
    protocol = json.loads(PROTOCOL.read_text())
    paths = {"single-frame": ROOT / "models/correlation-single-frame-v5-complete",
             "temporal": ROOT / "models/correlation-temporal-v5"}
    runs = {name: json.loads((path / "run.json").read_text()) for name, path in paths.items()}
    if any(run["status"] != "complete" or run["protocol_sha256"] != digest(PROTOCOL) for run in runs.values()):
        raise ValueError("incomplete fit or changed protocol")
    if runs["single-frame"]["initial_weights_sha256"] != runs["temporal"]["initial_weights_sha256"]:
        raise ValueError("initialization differs")
    if [e["order_sha256"] for e in runs["single-frame"]["history"]] != [e["order_sha256"] for e in runs["temporal"]["history"]]:
        raise ValueError("training minibatches differ")
    for run in runs.values():
        for path, expected in run["source_sha256"].items():
            if digest(ROOT / path) != expected: raise ValueError("training source changed")
    reference_path = ROOT / "experiments/learned-motion-v1/warp"
    reference_meta = json.loads((reference_path / "meta.json").read_text())
    reference_meta.update(gyro_bias={"enabled": True}, rotation_residual_enabled=False, map={"enabled": False})
    metadata = {name: json.loads((path / "meta.json").read_text()) for name, path in paths.items()}
    paths["calibrated-gyro"] = reference_path
    metadata["calibrated-gyro"] = reference_meta
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    with np.load(ROOT / protocol["cache"] / "validation.npz") as cache:
        cached_states, ids = cache["state"], cache["episode_id"]
    with np.load(CACHE / "validation.npz") as cache: priors = cache["prior"]
    record = {"protocol_sha256": digest(PROTOCOL), "matching_initialization_and_orders": True,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "workspace/solution").glob("*.py"))},
              "variants": {name: {"episodes": []} for name in paths}, "checks": []}
    for episode_id, entry in enumerate(manifest["validation"]):
        episode = read_episode(Path(entry["path"]), frames="lazy")
        config, adapted = episode.adapt((320, 240))
        frames = list(adapted); times = episode.frame_times
        windows = [episode.imu_window(i) for i in range(len(times))]
        indices = np.flatnonzero(ids == episode_id)
        all_predictions = {}
        for name, path in paths.items():
            outputs, predictions = [], []
            predictor = MotionPredictor(Bundle(path, metadata[name]))
            class Recorder:
                meta, size = predictor.meta, predictor.size
                requires_rotation_prior = predictor.requires_rotation_prior
                def predict(self, pair, state, rotation_prior=None):
                    index = indices[len(predictions)]
                    if not np.array_equal(state, cached_states[index]): raise ValueError("raw state mismatch")
                    if self.requires_rotation_prior and not np.allclose(rotation_prior, priors[index], atol=1e-10, rtol=0):
                        raise ValueError("causal prior mismatch")
                    value = predictor.predict(pair, state, rotation_prior)
                    predictions.append(value)
                    return value
            solution = Solution(copy.deepcopy(config), Recorder())
            for t, frame, imu in zip(times, frames, windows): outputs.append(solution.step(t, frame, imu))
            outputs, predictions = np.asarray(outputs), np.asarray(predictions)
            all_predictions[name] = predictions
            if len(outputs) != 200 or len(predictions) != 199: raise ValueError("incomplete flight")
            np.savez_compressed(raw / f"{name}-{episode_id}.npz", outputs=outputs, predictions=predictions)
            metrics = evaluate_trajectory(episode.ground_truth_state(), outputs, times).as_dict()
            record["variants"][name]["episodes"].append({"episode_id": episode_id, "metrics": metrics,
                "outputs_sha256": array_digest(outputs), "calibration": solution.gyro_bias_diagnostics})
        for name in ("single-frame", "temporal"):
            maximum = float(np.max(np.abs(all_predictions[name][:, :3] - all_predictions["calibrated-gyro"][:, :3])))
            if maximum > 1e-7: raise ValueError("frozen translation differs")
            record["checks"].append({"episode_id": episode_id, "variant": name, "max_translation_difference_m": maximum,
                                    "raw_state_exact": True, "causal_rotation_prior_matches": True})
    for name, row in record["variants"].items():
        row["summary"] = {"rotation_deg": float(np.degrees(np.mean([e["metrics"]["rotation_rmse"] for e in row["episodes"]]))),
            "sim3_ate_m": float(np.mean([e["metrics"]["sim3_ate_rmse"] for e in row["episodes"]])),
            "unaligned_ate_m": float(np.mean([e["metrics"]["ate_rmse"] for e in row["episodes"]])), "episodes": row["episodes"]}
        expected = runs["temporal"]["reference"] if name == "calibrated-gyro" else runs[name]["best_validation"]
        deltas = {key: abs(row["summary"][key]-expected[key]) for key in ("rotation_deg", "sim3_ate_m", "unaligned_ate_m")}
        if max(deltas.values()) > 1e-4: raise ValueError("training/actual rollout parity failed: " + str(deltas))
        row["training_rollout_absolute_differences"] = deltas
    reference = record["variants"]["calibrated-gyro"]["summary"]
    decisions = {name: eligible(record["variants"][name]["summary"], reference, protocol["acceptance"])
                 for name in ("single-frame", "temporal")}
    # Only the requested temporal branch can replace this round's reference.
    selected = "temporal" if decisions["temporal"]["eligible"] else "calibrated-gyro"
    record.update(status="complete", frozen_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "validation.json", record)
    write_json(out / "selection.json", {"selected_v5_variant": selected, "decisions": decisions,
        "v4_selected_variant_unchanged": "current", "public_data_used": False,
        "protocol_sha256": digest(PROTOCOL), "validation_sha256": digest(out / "validation.json"),
        "source_sha256": record["source_sha256"], "model_sha256": {name: metadata[name]["model_sha256"] for name in paths},
        "frozen_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"selected": selected, "decisions": decisions,
        "metrics": {name: {k: v for k, v in r["summary"].items() if k != "episodes"} for name, r in record["variants"].items()}}, indent=2))


if __name__ == "__main__": main()
