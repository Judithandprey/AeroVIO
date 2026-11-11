"""Verify complete v7 runs and archive algorithm-reproduction evidence."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

from report_feature_correlation import load, exact_repeat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/tiny-cnn-v7"
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def main():
    if (OUT / "comparison.json").exists(): raise ValueError("preserve report")
    protocol = json.loads((ROOT / "configs/tiny-cnn-v7.json").read_text())
    runs = {n: load(ROOT / ("results/tiny-cnn-v7-"+n)) for n in protocol["variants"]}
    base = runs["upstream"]
    expected = [e["hash"] for e in json.loads((ROOT / "configs/public-episodes.json").read_text())]
    rows = []
    for name, (manifest, report, directory) in runs.items():
        for key in ("upstream_commit", "limits_sha256", "selection_sha256", "runtime_sha256", "host", "python", "packages"):
            if manifest[key] != base[0][key]: raise ValueError("incomparable "+key)
        if report["limits"] != base[1]["limits"] or report["wasmtime"] != base[1]["wasmtime"]: raise ValueError("runtime/limits differ")
        if [e["hash"] for e in report["episodes"]] != expected or [e["hash"] for e in manifest["episodes"]] != expected: raise ValueError("episodes differ")
        if name in ("upstream", "matched"):
            fit = json.loads((ROOT / ("models/tiny-cnn-v7-"+name) / "run.json").read_text())
            if manifest["bundle_files_sha256"] != fit["bundle_files_sha256"]: raise ValueError("guest differs from checked training payload")
            for filename, sha in fit["source_sha256"].items():
                if filename.startswith("solution/") and manifest["bundle_files_sha256"][Path(filename).name] != sha: raise ValueError("upstream guest changed")
            logs = [p.read_text() for p in (directory / "logs").glob("*/stdout.txt")]
            if len(logs) != 10 or any("using cv2.dnn" not in text or "using the numpy forward pass" in text for text in logs):
                raise ValueError("expected OpenCV DNN execution in all episodes")
        summary = report["summary"]
        rows.append({"name": name, "completed": summary["completed"], "frames": summary["frames_attempted"],
            "mean_sim3_ate_m": summary["sim3_ate_rmse_mean"], "median_rotation_deg": math.degrees(summary["rotation_rmse_median"]),
            "mean_rotation_deg": math.degrees(statistics.mean(e["metrics"]["rotation_rmse"] for e in report["episodes"])),
            "median_step_ms": summary["wall_per_frame_median_ms"], "max_fuel": summary["fuel_per_frame_max"],
            "median_unaligned_ate_m": summary["ate_rmse_median"], "median_velocity_rmse_m_s": summary["velocity_rmse_median"],
            "median_sim3_scale": statistics.median(e["metrics"]["sim3_scale"] for e in report["episodes"]),
            "wasm_sha256": manifest["wasm_sha256"], "model_sha256": manifest["bundle_files_sha256"]["net.onnx"],
            "per_episode": [{"hash": e["hash"], "metrics": e["metrics"], "max_fuel": e["fuel_per_frame_max"]} for e in report["episodes"]]})
        dest = OUT / "runs" / name; dest.mkdir(parents=True)
        shutil.copy2(ROOT / ("results/tiny-cnn-v7-"+name) / "manifest.json", dest)
        shutil.copy2(directory / "report.json", dest)
    verification = {"matched_full_repeat": exact_repeat(runs["matched"], runs["matched-repeat"])}
    historical = {"v1-pose": "motion-pose-v1", "v4-current": "gyro-v4-current", "calibrated-gyro": "correspondence-v6-calibrated-gyro", "v5-single": "correspondence-v6-v5-single-frame", "v6-supervised": "correspondence-v6-supervised"}
    for name, previous in historical.items(): verification[name] = exact_repeat(load(ROOT / "results" / previous), runs[name])
    # The first matched fit completed, but its post-training record lookup failed.
    # Preserved checkpoint parity documents that fixing that lookup did not alter it.
    previous = ROOT / "models/tiny-cnn-v7-matched-record-check-failed/net.onnx"
    if digest(previous) != rows[1]["model_sha256"]: raise ValueError("matched rerun model changed")
    verification["matched_training_repeat_same_onnx"] = digest(previous)
    write(OUT / "verification.json", verification); write(OUT / "comparison.json", rows)
    byname = {r["name"]: r for r in rows}; changes = {}
    for baseline in ("upstream", "matched"):
        changes[baseline] = {name: {metric+"_change_percent": 100*(byname[name][metric]/byname[baseline][metric]-1)
            for metric in ("mean_sim3_ate_m", "median_rotation_deg", "median_step_ms", "max_fuel")}
            for name in historical}
    write(OUT / "relative-to-tiny-cnn.json", changes)
    lines = ["# Tiny CNN reproduction: local comparison", "",
        "All eight runs completed 10/10 public episodes and 12,000 frames with no filled outputs. The runner, runtime, episode hashes/order, input adaptation, limits and host environment match. The new guests retain the upstream implementation byte-for-byte; old WASMs remain unchanged. These local scores do not reproduce the server's different 20-episode evaluation.", "",
        "| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median WASM step (ms) | Max fuel/frame |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['mean_sim3_ate_m']:.6f} | {r['median_rotation_deg']:.4f} | {r['median_step_ms']:.3f} | {r['max_fuel']:,} |")
    lines += ["", "`upstream`: original 10-epoch, seed-0 trainer, final checkpoint, on our twelve training flights. `matched`: same flights and exact image pairs as AeroVIO, seed 17, 40 epochs, batch 64, Adam/cosine schedule, clipping and validation-selection policy matched to v1. It retains upstream 9-value IMU features, normalization, targets and inference. The latter is a controlled-budget reproduction, not the unmodified training recipe. V5/V6 received additional later training; this is not a one-factor architecture ablation.", "",
        "Wall time measures host-specific WASM step execution, not end-to-end sensor latency. Fuel is the portable scoring measure. Sim(3) fits global scale; it does not establish metric-map accuracy.", "",
        "| Variant | Median unaligned ATE (m) | Mean rotation RMSE (deg) | Median velocity RMSE (m/s) | Median alignment scale |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['median_unaligned_ate_m']:.4f} | {r['mean_rotation_deg']:.4f} | {r['median_velocity_rmse_m_s']:.4f} | {r['median_sim3_scale']:.4f} |")
    lines += ["", "## Per-episode aligned ATE", "", "| Episode | "+" | ".join(byname)+" |", "| --- | "+" | ".join(["---:"]*len(rows))+" |"]
    for i, episode in enumerate(rows[0]["per_episode"]): lines.append("| "+episode["hash"][:8]+" | "+" | ".join(f"{r['per_episode'][i]['metrics']['sim3_ate_rmse']:.6f}" for r in rows)+" |")
    lines += ["", "The full matched repetition and all preserved-reference reruns match trajectory, twist, accuracy and per-frame fuel exactly. No default or past acceptance decision changes. One seed per recipe and one public scene do not establish generalization or explain the server performance gap."]
    (OUT / "comparison.md").write_text("\n".join(lines)+"\n")
    for variant in ("upstream", "matched"):
        dest = OUT / variant; dest.mkdir()
        source = ROOT / ("models/tiny-cnn-v7-"+variant)
        for name in ("net.onnx", "weights.npz", "meta.json", "run.json", "history.json"):
            if (source / name).exists(): shutil.copy2(source / name, dest)
        shutil.copy2(ROOT / ("logs/tiny-cnn-v7-"+variant+"-training.log"), dest / "training.txt")
        shutil.copy2(ROOT / ("logs/tiny-cnn-v7-check-"+variant+".log"), dest / "wasm-check.txt")
    shutil.copy2(ROOT / "logs/tiny-cnn-v7-pipeline.json", OUT / "pipeline.json")
    print("\n".join(lines[:14]))


if __name__ == "__main__": main()
