"""Fixed, post-training image-pair shuffle sensitivity probe and coverage audit.

Shuffle whole pairs with seed 314162; keep every state and label in place.
This is an out-of-distribution sensitivity test, never a selection criterion or
standalone evidence that images improve deployed odometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SHUFFLE_SEED = 314162
ANGLE_EDGES_DEG = [0., .1, .5, 1., 2., 3., 5., float("inf")]
DEG = 180.0 / np.pi


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shuffle_indices(count, seed=SHUFFLE_SEED):
    if count < 2:
        raise ValueError("a shuffle probe needs at least two pairs")
    return np.random.default_rng(seed).permutation(count)


def rotation_vectors(rotations):
    rotations = np.asarray(rotations, dtype=np.float64)
    skew = np.stack((rotations[..., 2, 1] - rotations[..., 1, 2],
                     rotations[..., 0, 2] - rotations[..., 2, 0],
                     rotations[..., 1, 0] - rotations[..., 0, 1]), -1) * .5
    sine = np.linalg.norm(skew, axis=-1)
    cosine = (np.trace(rotations, axis1=-2, axis2=-1) - 1) * .5
    angle = np.arctan2(sine, cosine)
    scale = np.divide(angle, sine, out=1 + angle * angle / 6, where=sine > 1e-8)
    result = skew * scale[..., None]
    # Small-angle probes should not reach pi, but preserve correct diagnostics.
    for index in np.flatnonzero((sine.reshape(-1) < 1e-6) & (cosine.reshape(-1) < 0)):
        result.reshape(-1, 3)[index] = cv2.Rodrigues(rotations.reshape(-1, 3, 3)[index])[0].ravel()
    return result


def composed_rotations(state, motion, residual_enabled=True):
    state, motion = np.asarray(state), np.asarray(motion)
    if state.shape != (len(motion), 14) or motion.shape != (len(motion), 6):
        raise ValueError("expected matching Nx14 state and Nx6 motion")
    if not np.isfinite(state).all() or not np.isfinite(motion).all():
        raise ValueError("nonfinite motion/state")
    result = []
    for gyro, residual in zip(state[:, :3], motion[:, 3:]):
        rotation = cv2.Rodrigues(gyro.astype(np.float64))[0]
        if residual_enabled:
            rotation = rotation @ cv2.Rodrigues(residual.astype(np.float64))[0]
        result.append(rotation)
    return np.asarray(result)


def true_rotations(data):
    if "rotation_gt" in data:
        truth = data["rotation_gt"].astype(np.float64)
        if "rotation_residual" in data:
            motion = np.column_stack((np.zeros((len(truth), 3)), data["rotation_residual"]))
            reconstructed = composed_rotations(data["state"], motion)
            discrepancy = rotation_vectors(np.swapaxes(truth, 1, 2) @ reconstructed)
            if np.max(np.linalg.norm(discrepancy, axis=1)) > 4e-6:
                raise ValueError("auxiliary direct rotation and residual label disagree")
    else:
        truth = composed_rotations(data["state"], data["target"])
    if truth.shape != (len(data["state"]), 3, 3) or not np.isfinite(truth).all():
        raise ValueError("invalid rotation ground truth")
    if np.max(np.abs(np.swapaxes(truth, 1, 2) @ truth - np.eye(3))) > 1e-5:
        raise ValueError("reference rotations are not orthonormal")
    return truth


def predict_motion(predictor, images, states, order=None):
    order = np.arange(len(states)) if order is None else np.asarray(order)
    if sorted(order.tolist()) != list(range(len(states))):
        raise ValueError("order must be a permutation of image-pair indices")
    # The same state k is passed in both conditions; neither label nor order is
    # an estimator input. Channel 0 and channel 1 always come from the same pair.
    return np.stack([predictor.predict(images[j], states[k]) for k, j in enumerate(order)])


def quantiles(values):
    labels = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
    return dict(zip(labels, np.percentile(values, [0, 10, 25, 50, 75, 90, 95, 99, 100]).tolist()))


def coverage(data):
    truth = true_rotations(data)
    vector = rotation_vectors(truth)
    angles = np.linalg.norm(vector, axis=1) * DEG
    residual = data["rotation_residual"] if "rotation_residual" in data else data["target"][:, 3:]
    row = {
        "pairs": len(angles), "episodes": len(np.unique(data["episode_id"])),
        "rotation_norm_deg": quantiles(angles),
        "absolute_rotation_components_deg": {str(q): np.percentile(np.abs(vector) * DEG, q, axis=0).tolist() for q in [50, 95, 100]},
        "gyro_residual_norm_deg": quantiles(np.linalg.norm(residual, axis=1) * DEG),
        "mean_residual_camera_channels_deg": (residual.astype(float).mean(axis=0) * DEG).tolist(),
        "below_angle_deg_counts": {str(a): int((angles < a).sum()) for a in [.01, .05, .1, .25, .5, 1, 2, 3]},
        "frame_dt_range_s": [float(data["state"][:, 13].min()), float(data["state"][:, 13].max())],
    }
    if "target" in data:
        translation = np.linalg.norm(data["target"][:, :3], axis=1)
        row["translation_norm_m"] = quantiles(translation)
        row["below_translation_m_counts"] = {str(t): int((translation < t).sum()) for t in [1e-5, 1e-4, 1e-3]}
    if "axis_id" in data:
        row["aux_axis_counts"] = {str(int(a)): int((data["axis_id"] == a).sum()) for a in np.unique(data["axis_id"])}
    return row


def error_summary(truth, paired, shuffled, mask=None):
    mask = np.ones(len(truth), bool) if mask is None else mask
    if not mask.any():
        return {"pairs": 0}
    row = {"pairs": int(mask.sum())}
    for name, estimate in [("paired", paired), ("shuffled", shuffled)]:
        error = rotation_vectors(np.swapaxes(truth[mask], 1, 2) @ estimate[mask]) * DEG
        angles = np.linalg.norm(error, axis=1)
        row[name] = {
            "rotation_rmse_deg": float(np.sqrt(np.mean(angles * angles))),
            "rotation_median_deg": float(np.median(angles)),
            "rotation_p95_deg": float(np.percentile(angles, 95)),
            "error_component_rmse_deg": np.sqrt(np.mean(error * error, axis=0)).tolist(),
            "mean_signed_error_camera_channels_deg": error.mean(axis=0).tolist(),
        }
    row["shuffled_minus_paired_rmse_deg"] = row["shuffled"]["rotation_rmse_deg"] - row["paired"]["rotation_rmse_deg"]
    changed = rotation_vectors(np.swapaxes(paired[mask], 1, 2) @ shuffled[mask]) * DEG
    changed_angles = np.linalg.norm(changed, axis=1)
    row["paired_shuffled_prediction_difference"] = {
        "rotation_rms_deg": float(np.sqrt(np.mean(changed_angles ** 2))),
        "rotation_median_deg": float(np.median(changed_angles)),
        "rotation_max_deg": float(changed_angles.max()),
    }
    return row


def grouped_errors(data, paired, shuffled):
    truth = true_rotations(data)
    vector = rotation_vectors(truth)
    angles = np.linalg.norm(vector, axis=1) * DEG
    axis = data.get("axis_id", np.argmax(np.abs(vector), axis=1))
    axes = {0: "camera_x", 1: "camera_y", 2: "camera_z", 3: "mixed"}
    return {
        "overall": error_summary(truth, paired, shuffled),
        "axis_grouping": "synthetic labelled axis" if "axis_id" in data else "dominant ground-truth camera-axis component; real motion may be mixed",
        "by_axis": {axes[int(a)]: error_summary(truth, paired, shuffled, axis == a) for a in np.unique(axis)},
        "by_angle_deg": {f"[{lo:g},{hi:g})": error_summary(truth, paired, shuffled, (angles >= lo) & (angles < hi))
                         for lo, hi in zip(ANGLE_EDGES_DEG[:-1], ANGLE_EDGES_DEG[1:])},
    }


def load_cache(path, include_images):
    with np.load(path, allow_pickle=False) as archive:
        wanted = {"state", "target", "episode_id", "rotation_gt", "rotation_residual", "axis_id", "angle_degrees", "rotation_vector_gt"}
        if include_images:
            wanted.add("images")
        return {key: archive[key] for key in archive.files if key in wanted}


def coverage_markdown(result):
    lines = ["# Cached rotation coverage", "", "Computed from cached ground-truth rotations, not from public evaluation results.", "",
             "| Split | Pairs | Rotation median / p95 (deg) | Axis-wise p95 absolute rotation (deg) | Residual median / p95 (deg) |",
             "| --- | ---: | --- | --- | --- |"]
    for name, item in result["datasets"].items():
        row = item["coverage"]
        angle, residual = row["rotation_norm_deg"], row["gyro_residual_norm_deg"]
        axes = ", ".join(f"{v:.4f}" for v in row["absolute_rotation_components_deg"]["95"])
        lines.append(f"| {name} | {row['pairs']} | {angle['p50']:.4f} / {angle['p95']:.4f} | {axes} | {residual['p50']:.4f} / {residual['p95']:.4f} |")
    lines += ["", "Axis order is camera x, y, z. Real rotations are reconstructed as Exp(IMU rotation) Exp(target residual). Synthetic rotations use their explicit labels and are checked against that residual convention.",
              "Coverage is a dataset description; it does not establish that a particular data bias caused deployed drift."]
    return "\n".join(lines) + "\n"


def probe_markdown(result):
    lines = ["# Rotation image-pair shuffle probe", "", result["scope"], "",
             "Whole image pairs are independently permuted with fixed seed 314162 while states and labels stay fixed. The two channels remain together.", "",
             "| Dataset | Model | Paired rotation RMSE (deg) | Shuffled rotation RMSE (deg) | Shuffled minus paired (deg) | Prediction change RMS (deg) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for dataset, item in result["datasets"].items():
        for name, row in item["models"].items():
            overall = row["errors"]["overall"]
            lines.append(f"| {dataset} | {name} | {overall['paired']['rotation_rmse_deg']:.6f} | {overall['shuffled']['rotation_rmse_deg']:.6f} | {overall['shuffled_minus_paired_rmse_deg']:+.6f} | {overall['paired_shuffled_prediction_difference']['rotation_rms_deg']:.6f} |")
    for dataset, item in result["datasets"].items():
        for name, row in item["models"].items():
            lines += ["", f"## {dataset}: {name}", "", row["errors"]["axis_grouping"], "",
                      "| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for group in ["by_axis", "by_angle_deg"]:
                for label, values in row["errors"][group].items():
                    if values["pairs"]:
                        lines.append(f"| {label} | {values['pairs']} | {values['paired']['rotation_rmse_deg']:.6f} | {values['shuffled']['rotation_rmse_deg']:.6f} | {values['shuffled_minus_paired_rmse_deg']:+.6f} |")
    lines += ["", "Positive task-error changes indicate an advantage for correct image/state correspondence under this artificial intervention. Prediction change measures output sensitivity separately: unchanged task RMSE does not imply unchanged predictions. Shuffling can also mismatch camera intrinsics across episodes. Negative or small changes do not establish that vision is useless. This probe does not measure real-world generalization, trajectory drift, or image usefulness under a matched retrained control. No checkpoint or hyperparameter is selected here.",
              "Error components in JSON are Log(R_gt.T R_pred) in current-camera channels. Source/cache/model/permutation hashes are retained there."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-cache", type=Path, required=True)
    parser.add_argument("--aux-cache", type=Path)
    parser.add_argument("--model", nargs=2, action="append", metavar=("NAME", "DIRECTORY"))
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.out.with_suffix(ext) for ext in [".json", ".md"]]
    if any(p.exists() for p in outputs):
        parser.error("output exists; preserve previous evidence with a new prefix")
    if not args.coverage_only and (not args.model or not args.aux_cache):
        parser.error("probe requires --aux-cache and one or more --model")
    if args.model and len({name for name, _ in args.model}) != len(args.model):
        parser.error("model names must be unique")
    cv2.setNumThreads(1)
    paths = {"real_train": args.real_cache / "train.npz", "real_validation": args.real_cache / "validation.npz"} if args.coverage_only else {"real_validation": args.real_cache / "validation.npz"}
    if args.aux_cache:
        paths["aux_validation"] = args.aux_cache / "validation.npz"
        if args.coverage_only:
            paths["aux_train"] = args.aux_cache / "train.npz"
    result = {"schema_version": 1, "analysis_source_sha256": sha256(__file__),
              "scope": "Post-training, out-of-distribution image correspondence sensitivity; not standalone proof of visual usefulness and never checkpoint selection.",
              "shuffle_seed": SHUFFLE_SEED, "angle_edges_deg": [None if np.isinf(x) else x for x in ANGLE_EDGES_DEG],
              "packages": {"numpy": np.__version__, "opencv": cv2.__version__},
              "datasets": {}}
    models = {}
    if not args.coverage_only:
        sys.path.insert(0, str(ROOT / "workspace/solution"))
        from motion_inference import MotionPredictor
        result["inference_source_sha256"] = sha256(ROOT / "workspace/solution/motion_inference.py")
        class Bundle:
            def __init__(self, path): self.path = Path(path)
            def read_text(self, name): return (self.path / name).read_text()
            def read_bytes(self, name): return (self.path / name).read_bytes()
        for name, directory in args.model:
            path = Path(directory).resolve()
            predictor = MotionPredictor(Bundle(path))
            if sha256(path / "net.onnx") != predictor.meta["model_sha256"]:
                raise ValueError(f"model metadata hash mismatch: {name}")
            if predictor.meta.get("map", {}).get("enabled"):
                raise ValueError("probe requires map-disabled network metadata")
            models[name] = (predictor, {"directory": str(path), "model_sha256": sha256(path / "net.onnx"), "metadata_sha256": sha256(path / "meta.json"), "input_mode": predictor.meta.get("input_mode", "image_imu")})
    for name, path in paths.items():
        data = load_cache(path, include_images=not args.coverage_only)
        item = {"path": str(path.resolve()), "sha256": sha256(path), "coverage": coverage(data)}
        result["datasets"][name] = item
        if args.coverage_only:
            continue
        order = shuffle_indices(len(data["state"]))
        item["shuffle"] = {"seed": SHUFFLE_SEED, "permutation_sha256": hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                           "unchanged_pair_indices": int(np.sum(order == np.arange(len(order))))}
        item["models"] = {}
        for model_name, (predictor, provenance) in models.items():
            if data["images"].shape[1:] != (2, predictor.size[1], predictor.size[0]):
                raise ValueError("cache image shape differs from model input")
            paired = predict_motion(predictor, data["images"], data["state"])
            shuffled = predict_motion(predictor, data["images"], data["state"], order)
            maximum_change = float(np.max(np.abs(paired - shuffled)))
            if provenance["input_mode"] == "imu_only" and maximum_change > 1e-7:
                raise ValueError("IMU-only sanity model changed after shuffling images")
            enabled = predictor.meta.get("rotation_residual_enabled", True)
            item["models"][model_name] = {**provenance, "max_motion_component_change": maximum_change,
                "imu_only_invariance_passed": maximum_change <= 1e-7 if provenance["input_mode"] == "imu_only" else None,
                "errors": grouped_errors(data, composed_rotations(data["state"], paired, enabled), composed_rotations(data["state"], shuffled, enabled))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    outputs[1].write_text(coverage_markdown(result) if args.coverage_only else probe_markdown(result))
    print(json.dumps({"outputs": [str(p) for p in outputs], "datasets": {name: row["coverage"]["pairs"] for name, row in result["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
