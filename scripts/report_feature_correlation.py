"""Verify and archive the fixed v5 public comparison without changing selection."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/feature-correlation-v5"
NAMES = ("baseline", "current", "calibrated-gyro", "single-frame", "temporal", "temporal-repeat")


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    reports = list(directory.glob("report-*/report.json"))
    if len(reports) != 1: raise ValueError("expected one report")
    report = json.loads(reports[0].read_text())
    if manifest.get("exit_code") != 0 or manifest["report_sha256"] != digest(reports[0]): raise ValueError("failed or changed report")
    if report["summary"]["completed"] != 10 or report["summary"]["frames_attempted"] != 12000:
        raise ValueError("incomplete run")
    if any(e["metrics"]["filled_frames"] or not e["completed"] or e["failure"] for e in report["episodes"]):
        raise ValueError("failed/filled episode")
    return manifest, report, reports[0].parent


def exact_repeat(before, after):
    ma, ra, pa = before; mb, rb, pb = after
    if ma["wasm_sha256"] != mb["wasm_sha256"]: raise ValueError("repeat artifact differs")
    rows = []
    for a, b in zip(ra["episodes"], rb["episodes"]):
        if a["hash"] != b["hash"] or a["metrics"] != b["metrics"]: raise ValueError("repeat accuracy differs")
        hashes = {}
        for name in ("estimate.tum", "twist.txt", "fuel.npy"):
            left, right = [p / "logs" / a["hash"] / name for p in (pa, pb)]
            if digest(left) != digest(right): raise ValueError("repeat outputs/fuel differ: " + name)
            hashes[name] = digest(right)
        rows.append({"hash": a["hash"], "accuracy_exact": True, "output_sha256": hashes})
    return rows


def main():
    if (OUT / "comparison.json").exists(): raise ValueError("preserve previous report")
    loaded = {name: load(ROOT / "results" / ("correlation-v5-" + name)) for name in NAMES}
    base = loaded["baseline"]
    expected = [e["hash"] for e in json.loads((ROOT / "configs/public-episodes.json").read_text())]
    for name, (manifest, report, _) in loaded.items():
        for key in ("upstream_commit", "limits_sha256", "selection_sha256", "runtime_sha256", "host", "python", "packages"):
            if manifest[key] != base[0][key]: raise ValueError("incomparable " + key)
        if report["limits"] != base[1]["limits"] or report["wasmtime"] != base[1]["wasmtime"]:
            raise ValueError("actual limits/runtime differ")
        if [e["hash"] for e in report["episodes"]] != expected or [e["hash"] for e in manifest["episodes"]] != expected:
            raise ValueError("episode order differs")
        meta = manifest.get("embedded_model_metadata")
        if meta and meta["model_sha256"] != manifest["bundle_files_sha256"]["net.onnx"]:
            raise ValueError("embedded model differs")
    sources = lambda m: {k: v for k, v in m["bundle_files_sha256"].items() if k.endswith(".py")}
    if sources(loaded["single-frame"][0]) != sources(loaded["temporal"][0]): raise ValueError("new variants have different guest code")
    selection = json.loads((OUT / "selection.json").read_text())
    for name in ("single-frame", "temporal"):
        manifest = loaded[name][0]
        if manifest["bundle_files_sha256"]["net.onnx"] != selection["model_sha256"][name]: raise ValueError("model changed after validation")
        for path, expected_hash in selection["source_sha256"].items():
            if manifest["bundle_files_sha256"][Path(path).name] != expected_hash: raise ValueError("guest changed after validation")
        meta = manifest["embedded_model_metadata"]
        if not meta["rotation_prior_input"] or not meta["gyro_bias"]["enabled"] or not meta["rotation_residual_enabled"] or meta["map"]["enabled"]:
            raise ValueError("v5 inference switches differ")
    repeats = {"temporal_repeat": exact_repeat(loaded["temporal"], loaded["temporal-repeat"])}
    for name in ("baseline", "current", "calibrated-gyro"):
        repeats[name + "_historical_repeat"] = exact_repeat(load(ROOT / "results" / ("gyro-v4-" + name)), loaded[name])
    # Old artifacts deliberately retain old source. Prove the default-disabled
    # extension also preserves reference host trajectories on the four flights.
    compatibility = []
    for episode_id in range(4):
        a = ROOT / f"results/gyro-bias-v4-validation/episode-{episode_id}/repeat-0-calibrated-gyro.npz"
        b = ROOT / f"results/correlation-v5-validation/calibrated-gyro-{episode_id}.npz"
        with np.load(a) as x, np.load(b) as y:
            if not np.array_equal(x["outputs"], y["outputs"]): raise ValueError("old reference changed with new guest")
        compatibility.append({"episode_id": episode_id, "outputs_exact": True, "old_raw_sha256": digest(a), "new_raw_sha256": digest(b)})
    write_json(OUT / "output-verification.json", {"repeats": repeats, "default_compatibility": compatibility,
        "scope": "Exact deterministic repetition is not statistical/generalization evidence; old reference WASM payloads are preserved"})
    rows = []
    for name, (manifest, report, _) in loaded.items():
        s = report["summary"]
        rows.append({"name": name, "completed": s["completed"], "frames": s["frames_attempted"],
                     "mean_sim3_ate_m": s["sim3_ate_rmse_mean"], "median_rotation_deg": math.degrees(s["rotation_rmse_median"]),
                     "mean_rotation_deg": math.degrees(statistics.mean(e["metrics"]["rotation_rmse"] for e in report["episodes"])),
                     "median_step_ms": s["wall_per_frame_median_ms"], "max_fuel": s["fuel_per_frame_max"],
                     "median_unaligned_ate_m": s["ate_rmse_median"],
                     "median_angular_velocity_rmse_rad_s": s["angular_velocity_rmse_median"],
                     "median_rotation_rpe_deg": math.degrees(s["rpe_rotation_rmse_median"]),
                     "median_sim3_scale": statistics.median(e["metrics"]["sim3_scale"] for e in report["episodes"]),
                     "wasm_sha256": manifest["wasm_sha256"], "model_sha256": manifest["bundle_files_sha256"].get("net.onnx"),
                     "per_episode": [{"hash": e["hash"], "metrics": e["metrics"]} for e in report["episodes"]]})
        destination = OUT / "runs" / name
        destination.mkdir(parents=True)
        shutil.copy2(ROOT / "results" / ("correlation-v5-" + name) / "manifest.json", destination)
        shutil.copy2(loaded[name][2] / "report.json", destination)
        shutil.copy2(ROOT / "results" / ("correlation-v5-" + name) / "evaluate.log", destination / "evaluate.txt")
    write_json(OUT / "comparison.json", rows)
    byname = {r["name"]: r for r in rows}
    changes = {}
    for ref in ("baseline", "current", "calibrated-gyro", "single-frame"):
        changes[ref] = {key + "_change_percent": 100*(byname["temporal"][key]/byname[ref][key]-1)
                       for key in ("mean_sim3_ate_m", "median_rotation_deg", "median_step_ms", "max_fuel", "median_unaligned_ate_m")}
        changes[ref]["rotation_wins"] = sum(a["metrics"]["rotation_rmse"] < b["metrics"]["rotation_rmse"]
             for a, b in zip(byname["temporal"]["per_episode"], byname[ref]["per_episode"]))
        changes[ref]["aligned_ate_wins"] = sum(a["metrics"]["sim3_ate_rmse"] < b["metrics"]["sim3_ate_rmse"]
             for a, b in zip(byname["temporal"]["per_episode"], byname[ref]["per_episode"]))
    write_json(OUT / "interpretation-metrics.json", changes)
    lines = ["# Feature-correlation v5: completed public comparison", "",
        "All six serial runs completed all ten public episodes (12,000 frames each), without failures or filled outputs. The preserved baseline/current/calibrated-gyro WASM files were rerun unchanged. Both new branches use the same new guest source and pinned runtime/limits. Reference payloads retain their original source; the extension's default compatibility was verified separately.", "",
        "| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Max fuel/frame |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['mean_sim3_ate_m']:.6f} | {r['median_rotation_deg']:.4f} | {r['median_step_ms']:.3f} | {r['max_fuel']:,} |")
    lines += ["", "Step time is host-specific WASM execution time, not complete sensor-to-output latency. Fuel is reported separately. Sim(3) alignment removes scale and does not establish metric-map accuracy.", "",
        "| Variant | Mean rotation RMSE (deg) | Median unaligned ATE (m) | Median 1 s rotation RPE (deg) | Median angular-velocity RMSE (rad/s) | Median fitted scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['mean_rotation_deg']:.4f} | {r['median_unaligned_ate_m']:.4f} | {r['median_rotation_rpe_deg']:.4f} | {r['median_angular_velocity_rmse_rad_s']:.6f} | {r['median_sim3_scale']:.3f} |")
    lines += ["", "## Temporal candidate versus references", "",
              "| Reference | Aligned ATE change | Median rotation change | Step-time change | Max-fuel change | Rotation wins |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, c in changes.items(): lines.append(f"| {name} | {c['mean_sim3_ate_m_change_percent']:+.2f}% | {c['median_rotation_deg_change_percent']:+.2f}% | {c['median_step_ms_change_percent']:+.2f}% | {c['max_fuel_change_percent']:+.2f}% | {c['rotation_wins']}/10 |")
    lines += ["", "## Per-episode rotation and aligned position", "",
              "| Episode | " + " | ".join(NAMES[:-1]) + " |", "| --- | " + " | ".join(["---:"]*5) + " |"]
    for i, episode in enumerate(rows[0]["per_episode"]):
        lines.append("| " + episode["hash"][:8] + " | " + " | ".join(f"{math.degrees(r['per_episode'][i]['metrics']['rotation_rmse']):.3f} deg / {r['per_episode'][i]['metrics']['sim3_ate_rmse']:.3f} m" for r in rows[:-1]) + " |")
    lines += ["", "The temporal repeat reproduces all trajectories, twists, accuracy and per-frame fuel exactly. The three preserved references also reproduce their v4 outputs/fuel exactly. Timing is measured separately and need not match.", "",
              "Validation had already rejected both trained branches; public diagnostics do not override that decision. The v5 reference remains calibrated-gyro, itself an unselected v4 diagnostic; v4's selected current model and workspace pose-only default are unchanged. One training seed, four short validation flights and one public scene do not establish generalization or robot performance."]
    (OUT / "comparison.md").write_text("\n".join(lines) + "\n")
    for variant, path in (("single-frame", ROOT / "models/correlation-single-frame-v5-complete"), ("temporal", ROOT / "models/correlation-temporal-v5")):
        dest = OUT / variant; dest.mkdir()
        for name in ("net.onnx", "meta.json", "run.json", "visual-probe.json"): shutil.copy2(path / name, dest)
    for name in ("tests", "validation", "prepare", "single-frame-complete", "temporal", "check-single-frame", "check-temporal"):
        filename = "correlation-v5-" + name + (".txt" if name == "tests" else ".log")
        shutil.copy2(ROOT / "logs" / filename, OUT / (name + ".txt"))
    shutil.copy2(ROOT / "logs/correlation-v5-pipeline.json", OUT / "pipeline-status.json")
    shutil.copy2(ROOT / "data/correlation-cache-v5/cache.json", OUT / "cache-provenance.json")
    print("\n".join(lines[:12]))


if __name__ == "__main__": main()
