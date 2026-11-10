"""Compare complete public runs only when runtime, episodes and limits match."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def normalized_meta(meta):
    value = json.loads(json.dumps(meta))
    value.setdefault("input_mode", "image_imu")
    value.setdefault("network_inputs", "images_state")
    value.setdefault("rotation_residual_enabled", True)
    calibration = dict(value.get("gyro_bias", {}))
    calibration.setdefault("enabled", False)
    value["gyro_bias"] = calibration
    return value


def files_except(manifest, *names):
    return {k: v for k, v in manifest["bundle_files_sha256"].items() if k not in names}


def load(name, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = list(directory.glob("report-*/report.json")) + list(directory.glob("report.json"))
    if len(reports) != 1:
        raise ValueError(f"expected one report in {directory}")
    report = json.loads(reports[0].read_text())
    if not manifest.get("finished_utc") or manifest.get("exit_code") not in (0, 1):
        raise ValueError(f"evaluation is not finished in {directory}")
    if hashlib.sha256(reports[0].read_bytes()).hexdigest() != manifest["report_sha256"]:
        raise ValueError(f"report hash mismatch in {directory}")
    return {"name": name, "directory": str(directory), "manifest": manifest, "report": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "DIRECTORY"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    runs = [load(name, directory) for name, directory in args.run]
    base = runs[0]
    expected = json.loads((ROOT / "configs/public-episodes.json").read_text())
    expected_hashes = [entry["hash"] for entry in expected]
    for run in runs:
        manifest, report = run["manifest"], run["report"]
        for key in ["upstream_commit", "limits_sha256", "selection_sha256", "runtime_sha256", "bundle_files_sha256"]:
            if not manifest.get(key):
                raise ValueError(f"missing required {key} for {run['name']}")
        if len(expected_hashes) != 10 or [e["hash"] for e in report["episodes"]] != expected_hashes:
            raise ValueError("comparison requires all ten public episodes in pinned order")
        if [e["hash"] for e in manifest["episodes"]] != expected_hashes:
            raise ValueError("manifest selection differs from pinned public selection")
        if report["summary"]["episodes"] != 10 or report["limits"] != base["report"]["limits"]:
            raise ValueError("report episode count or actual limits differ")
    for run in runs[1:]:
        for key in ["upstream_commit", "limits_sha256", "selection_sha256", "runtime_sha256"]:
            if run["manifest"][key] != base["manifest"][key]:
                raise ValueError(f"incomparable {key} for {run['name']}")
        if [e["hash"] for e in run["report"]["episodes"]] != [e["hash"] for e in base["report"]["episodes"]]:
            raise ValueError("episode selection/order differs")
        if run["report"]["wasmtime"] != base["report"]["wasmtime"]:
            raise ValueError("Wasmtime version differs")
        for key in ["host", "python", "packages"]:
            if run["manifest"].get(key) != base["manifest"].get(key):
                raise ValueError(f"incomparable environment {key} for {run['name']}")
    learned = [run for run in runs if run["manifest"].get("embedded_model_metadata")]
    if learned:
        reference_sources = {k: v for k, v in learned[0]["manifest"]["bundle_files_sha256"].items() if k.endswith(".py")}
        for run in learned:
            sources = {k: v for k, v in run["manifest"]["bundle_files_sha256"].items() if k.endswith(".py")}
            if sources != reference_sources:
                raise ValueError("learned variants must use identical inference Python sources")
            meta = run["manifest"]["embedded_model_metadata"]
            calibration = normalized_meta(meta)["gyro_bias"]
            if set(calibration) != {"enabled"} or type(calibration["enabled"]) is not bool:
                raise ValueError("gyro calibration metadata must contain only a Boolean enabled flag")
            if calibration["enabled"] and meta.get("map", {}).get("enabled", False):
                raise ValueError("gyro calibration requires map correction disabled")
            if meta["model_sha256"] != run["manifest"]["bundle_files_sha256"].get("net.onnx"):
                raise ValueError("metadata model hash differs from the actual embedded ONNX")
            if not meta.get("rotation_residual_enabled", True):
                candidates = [other for other in learned
                              if other["manifest"]["embedded_model_metadata"].get("rotation_residual_enabled", True)
                              and not other["manifest"]["embedded_model_metadata"].get("map", {}).get("enabled")
                              and normalized_meta(other["manifest"]["embedded_model_metadata"])["gyro_bias"] == calibration
                              and other["manifest"]["bundle_files_sha256"].get("net.onnx") == meta["model_sha256"]]
                if len(candidates) != 1:
                    raise ValueError("rotation ablation needs exactly one rotation-on counterpart with identical ONNX weights")
                counterpart = candidates[0]["manifest"]
                a, b = normalized_meta(meta), normalized_meta(counterpart["embedded_model_metadata"])
                a["rotation_residual_enabled"] = True
                if files_except(run["manifest"], "meta.json") != files_except(counterpart, "meta.json") or a != b:
                    raise ValueError("rotation on/off bundles must differ only in rotation_residual_enabled")
            if calibration["enabled"]:
                candidates = [other for other in learned
                              if not normalized_meta(other["manifest"]["embedded_model_metadata"])["gyro_bias"]["enabled"]
                              and other["manifest"]["embedded_model_metadata"].get("rotation_residual_enabled", True) == meta.get("rotation_residual_enabled", True)
                              and not other["manifest"]["embedded_model_metadata"].get("map", {}).get("enabled")
                              and other["manifest"]["bundle_files_sha256"].get("net.onnx") == meta["model_sha256"]]
                if len(candidates) != 1:
                    raise ValueError("gyro calibration needs exactly one calibration-off counterpart with identical ONNX and residual setting")
                counterpart = candidates[0]["manifest"]
                a, b = normalized_meta(meta), normalized_meta(counterpart["embedded_model_metadata"])
                a["gyro_bias"] = {"enabled": False}
                if files_except(run["manifest"], "meta.json") != files_except(counterpart, "meta.json") or a != b:
                    raise ValueError("calibration on/off bundles must differ only in gyro_bias.enabled")
            if meta.get("network_inputs", "images_state") == "state":
                candidates = [other for other in learned
                              if other["manifest"]["bundle_files_sha256"].get("net.onnx") == meta.get("source_model_sha256")
                              and other["manifest"]["embedded_model_metadata"].get("network_inputs", "images_state") == "images_state"]
                if len(candidates) != 1 or meta.get("input_mode") != "imu_only":
                    raise ValueError("compact model needs exactly one full-graph IMU-only parent")
                counterpart = candidates[0]["manifest"]
                a, b = normalized_meta(meta), normalized_meta(counterpart["embedded_model_metadata"])
                a["network_inputs"] = "images_state"
                a["model_sha256"] = a.pop("source_model_sha256")
                if files_except(run["manifest"], "meta.json", "net.onnx") != files_except(counterpart, "meta.json", "net.onnx") or a != b:
                    raise ValueError("compact metadata must preserve its full-graph IMU-only training identity")
            if meta.get("map", {}).get("enabled"):
                candidates = [other for other in learned
                              if not other["manifest"]["embedded_model_metadata"].get("map", {}).get("enabled")
                              and normalized_meta(other["manifest"]["embedded_model_metadata"])["gyro_bias"] == calibration
                              and other["manifest"]["embedded_model_metadata"].get("rotation_residual_enabled", True) == meta.get("rotation_residual_enabled", True)
                              and other["manifest"]["bundle_files_sha256"].get("net.onnx") == run["manifest"]["bundle_files_sha256"].get("net.onnx")]
                if len(candidates) != 1:
                    raise ValueError("map ablation needs exactly one map-off run with identical ONNX weights")
                counterpart = candidates[0]["manifest"]
                files = {k: v for k, v in run["manifest"]["bundle_files_sha256"].items() if k != "meta.json"}
                other_files = {k: v for k, v in counterpart["bundle_files_sha256"].items() if k != "meta.json"}
                a, b = json.loads(json.dumps(meta)), json.loads(json.dumps(counterpart["embedded_model_metadata"]))
                a["map"]["enabled"] = False
                if files != other_files or a != b:
                    raise ValueError("map on/off bundles must differ only in map.enabled metadata")
    result = []
    baseline = base["report"]["summary"]
    for run in runs:
        report, manifest = run["report"], run["manifest"]
        s = report["summary"]
        result.append({"name": run["name"], "episodes": s["episodes"], "completed": s["completed"],
            "filled_frames": sum(e["metrics"]["filled_frames"] for e in report["episodes"]),
            "sim3_ate_mean_m": s["sim3_ate_rmse_mean"], "fuel_per_frame_max": s["fuel_per_frame_max"],
            "fuel_per_frame_mean": s["fuel_per_frame_mean"], "wall_per_frame_median_ms": s["wall_per_frame_median_ms"],
            "unaligned_ate_median_m": s["ate_rmse_median"], "velocity_rmse_median_m_s": s["velocity_rmse_median"],
            "rotation_rmse_median_rad": s["rotation_rmse_median"],
            "rotation_rmse_median_deg": math.degrees(s["rotation_rmse_median"]),
            "rpe_translation_rmse_median_m": s["rpe_translation_rmse_median"],
            "rpe_rotation_rmse_median_deg": math.degrees(s["rpe_rotation_rmse_median"]),
            "angular_velocity_rmse_median_rad_s": s["angular_velocity_rmse_median"],
            "sim3_scale_median": statistics.median(e["metrics"]["sim3_scale"] for e in report["episodes"]),
            "sim3_scale_range": [min(e["metrics"]["sim3_scale"] for e in report["episodes"]),
                                 max(e["metrics"]["sim3_scale"] for e in report["episodes"])],
            "ate_change_percent": 100 * (s["sim3_ate_rmse_mean"] / baseline["sim3_ate_rmse_mean"] - 1),
            "max_fuel_change_percent": 100 * (s["fuel_per_frame_max"] / baseline["fuel_per_frame_max"] - 1),
            "wasm_sha256": manifest["wasm_sha256"], "model_metadata": manifest.get("embedded_model_metadata"),
            "per_episode": [{"hash": e["hash"], "sim3_ate_m": e["metrics"]["sim3_ate_rmse"],
                              "max_fuel": e["fuel_per_frame_max"], "completed": e["completed"]} for e in report["episodes"]]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Local public-set comparison", "", "All runs use the same ten public episodes, pinned runner/runtime and saved limits.",
             "Wall time is local step execution time, not end-to-end latency or the grading score.", "",
             "| Variant | Completed | Mean Sim(3) ATE (m) | Max fuel/frame | Median step (ms) | Filled frames |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result:
        lines.append(f"| {row['name']} | {row['completed']}/{row['episodes']} | {row['sim3_ate_mean_m']:.6f} | {row['fuel_per_frame_max']:,} | {row['wall_per_frame_median_ms']:.3f} | {row['filled_frames']} |")
    lines += ["", "## Additional accuracy metrics", "",
              "Sim(3) alignment fits scale, rotation and translation. It does not measure metric-scale accuracy.",
              "Values below are medians across episodes; rotation and unaligned ATE expose tradeoffs hidden by alignment.", "",
              "| Variant | Unaligned ATE (m) | Rotation RMSE (deg) | 1 s translation RPE (m) | Velocity RMSE (m/s) | Alignment scale |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in result:
        lines.append(f"| {row['name']} | {row['unaligned_ate_median_m']:.4f} | {row['rotation_rmse_median_deg']:.3f} | {row['rpe_translation_rmse_median_m']:.4f} | {row['velocity_rmse_median_m_s']:.4f} | {row['sim3_scale_median']:.3f} |")
    lines += ["", "| Variant | Rotation RMSE (deg) | 1 s rotation RPE (deg) | Angular velocity RMSE (rad/s) |",
              "| --- | ---: | ---: | ---: |"]
    for row in result:
        lines.append(f"| {row['name']} | {row['rotation_rmse_median_deg']:.3f} | {row['rpe_rotation_rmse_median_deg']:.3f} | {row['angular_velocity_rmse_median_rad_s']:.5f} |")
    lines += ["", "## Per-episode Sim(3) ATE", "", "| Episode | " + " | ".join(r["name"] for r in result) + " |",
              "| --- | " + " | ".join("---:" for _ in result) + " |"]
    for index, episode in enumerate(result[0]["per_episode"]):
        lines.append("| `" + episode["hash"][:8] + "` | " + " | ".join(f"{r['per_episode'][index]['sim3_ate_m']:.6f}" for r in result) + " |")
    lines += ["", "These are pilot results from one training seed and a small independent synthetic training set.",
              "They do not establish statistical significance, unseen-scene generalization, or course-server performance."]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
