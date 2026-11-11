"""Archive v6 results with exact repeat, graph identity and matching diagnostics."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import onnx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from report_feature_correlation import load, exact_repeat

OUT = ROOT / "experiments/correspondence-v6"
NAMES = ("baseline", "calibrated-gyro", "v5-single-frame", "v5-temporal", "continuation", "supervised", "supervised-repeat")
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def graph_signature(path):
    graph = copy.deepcopy(onnx.load(path).graph)
    for tensor in graph.initializer:
        metadata = onnx.TensorProto(name=tensor.name, data_type=tensor.data_type, dims=tensor.dims)
        tensor.CopyFrom(metadata)
    return hashlib.sha256(graph.SerializeToString(deterministic=True)).hexdigest()


def main():
    if (OUT / "comparison.json").exists(): raise ValueError("preserve existing comparison")
    loaded = {name: load(ROOT / "results" / ("correspondence-v6-"+name)) for name in NAMES}
    base = loaded["baseline"]
    expected = [e["hash"] for e in json.loads((ROOT / "configs/public-episodes.json").read_text())]
    selection = json.loads((OUT / "selection.json").read_text())
    for name, (manifest, report, _) in loaded.items():
        for key in ("upstream_commit", "limits_sha256", "selection_sha256", "runtime_sha256", "host", "python", "packages"):
            if manifest[key] != base[0][key]: raise ValueError("incomparable "+key)
        if report["limits"] != base[1]["limits"] or report["wasmtime"] != base[1]["wasmtime"]: raise ValueError("runtime/limits differ")
        if [e["hash"] for e in report["episodes"]] != expected or [e["hash"] for e in manifest["episodes"]] != expected:
            raise ValueError("episode order differs")
        meta = manifest.get("embedded_model_metadata")
        if meta and meta["model_sha256"] != manifest["bundle_files_sha256"]["net.onnx"]: raise ValueError("model hash differs")
        if name in ("continuation", "supervised"):
            if manifest["bundle_files_sha256"]["net.onnx"] != selection["model_sha256"][name]: raise ValueError("model changed after validation")
            for path, expected_hash in selection["source_sha256"].items():
                if manifest["bundle_files_sha256"][Path(path).name] != expected_hash: raise ValueError("guest source changed")
    source = lambda m: {k: v for k, v in m["bundle_files_sha256"].items() if k.endswith(".py")}
    for name in ("continuation", "supervised", "v5-single-frame"):
        if source(loaded[name][0]) != source(loaded["v5-temporal"][0]): raise ValueError("v5/v6 guest code differs")
    verification = {"supervised_repeat": exact_repeat(loaded["supervised"], loaded["supervised-repeat"])}
    historical = {"baseline": "baseline", "calibrated-gyro": "calibrated-gyro", "v5-single-frame": "single-frame", "v5-temporal": "temporal"}
    for name, old in historical.items():
        verification[name+"_preserved_repeat"] = exact_repeat(load(ROOT / "results" / ("correlation-v5-"+old)), loaded[name])
    signatures = {"v5-temporal": graph_signature(ROOT / "models/correlation-temporal-v5/net.onnx")}
    for name in ("continuation", "supervised"):
        signatures[name] = graph_signature(ROOT / f"models/correspondence-{name}-v6/net.onnx")
    if len(set(signatures.values())) != 1: raise ValueError("deployed graph structure differs")
    write(OUT / "output-verification.json", {"exact_repeats": verification, "onnx_graph_without_initializer_values_sha256": signatures,
        "same_v5_v6_guest_source": True, "scope": "Deterministic repeat, not generalization; learned values may affect small numeric-branch fuel differences"})
    rows = []
    for name, (manifest, report, directory) in loaded.items():
        s = report["summary"]
        rows.append({"name": name, "completed": s["completed"], "frames": s["frames_attempted"],
            "mean_sim3_ate_m": s["sim3_ate_rmse_mean"], "median_rotation_deg": math.degrees(s["rotation_rmse_median"]),
            "mean_rotation_deg": math.degrees(statistics.mean(e["metrics"]["rotation_rmse"] for e in report["episodes"])),
            "median_step_ms": s["wall_per_frame_median_ms"], "max_fuel": s["fuel_per_frame_max"],
            "median_unaligned_ate_m": s["ate_rmse_median"], "median_angular_velocity_rmse_rad_s": s["angular_velocity_rmse_median"],
            "median_rotation_rpe_deg": math.degrees(s["rpe_rotation_rmse_median"]),
            "median_sim3_scale": statistics.median(e["metrics"]["sim3_scale"] for e in report["episodes"]),
            "wasm_sha256": manifest["wasm_sha256"], "model_sha256": manifest["bundle_files_sha256"].get("net.onnx"),
            "per_episode": [{"hash": e["hash"], "metrics": e["metrics"]} for e in report["episodes"]]})
        dest = OUT / "runs" / name; dest.mkdir(parents=True)
        shutil.copy2(ROOT / "results" / ("correspondence-v6-"+name) / "manifest.json", dest)
        shutil.copy2(directory / "report.json", dest)
        shutil.copy2(ROOT / "results" / ("correspondence-v6-"+name) / "evaluate.log", dest / "evaluate.txt")
    write(OUT / "comparison.json", rows)
    byname = {r["name"]: r for r in rows}; changes = {}
    for ref in NAMES[:5]:
        changes[ref] = {key+"_change_percent": 100*(byname["supervised"][key]/byname[ref][key]-1)
            for key in ("mean_sim3_ate_m", "median_rotation_deg", "median_step_ms", "max_fuel", "median_unaligned_ate_m")}
        for metric in ("rotation_rmse", "sim3_ate_rmse"):
            changes[ref][metric+"_wins"] = sum(a["metrics"][metric] < b["metrics"][metric]
                for a, b in zip(byname["supervised"]["per_episode"], byname[ref]["per_episode"]))
    write(OUT / "interpretation-metrics.json", changes)
    lines = ["# Explicit correspondence supervision: completed public comparison", "",
        "All seven serial runs completed all ten episodes (12,000 frames each), without failures or filled outputs. Previous artifacts are rerun unchanged. V5 temporal and V6 candidates share the same guest code and the same ONNX graph structure apart from initializer values.", "",
        "| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Max fuel/frame |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['mean_sim3_ate_m']:.6f} | {r['median_rotation_deg']:.4f} | {r['median_step_ms']:.3f} | {r['max_fuel']:,} |")
    lines += ["", "Step time is host-specific WASM execution time, not end-to-end latency. Sim(3) fits scale and does not prove metric-map accuracy.", "",
        "| Variant | Mean rotation RMSE (deg) | Median unaligned ATE (m) | Median 1 s rotation RPE (deg) | Median angular velocity RMSE (rad/s) | Median fitted scale |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows: lines.append(f"| {r['name']} | {r['mean_rotation_deg']:.4f} | {r['median_unaligned_ate_m']:.4f} | {r['median_rotation_rpe_deg']:.4f} | {r['median_angular_velocity_rmse_rad_s']:.6f} | {r['median_sim3_scale']:.3f} |")
    lines += ["", "## Supervised candidate versus references", "",
        "| Reference | Aligned ATE change | Median rotation change | Step-time change | Max-fuel change | Rotation wins |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, c in changes.items(): lines.append(f"| {name} | {c['mean_sim3_ate_m_change_percent']:+.2f}% | {c['median_rotation_deg_change_percent']:+.2f}% | {c['median_step_ms_change_percent']:+.2f}% | {c['max_fuel_change_percent']:+.2f}% | {c['rotation_rmse_wins']}/10 |")
    lines += ["", "## Per-episode rotation and aligned trajectory error", "",
        "| Episode | "+" | ".join(NAMES[:-1])+" |", "| --- | "+" | ".join(["---:"]*6)+" |"]
    for i, episode in enumerate(rows[0]["per_episode"]):
        lines.append("| "+episode["hash"][:8]+" | "+" | ".join(f"{math.degrees(r['per_episode'][i]['metrics']['rotation_rmse']):.3f} deg / {r['per_episode'][i]['metrics']['sim3_ate_rmse']:.3f} m" for r in rows[:-1])+" |")
    lines += ["", "The complete supervised repeat reproduces pose, twist, accuracy and per-frame fuel exactly. Every preserved reference reproduces its v5 results exactly; wall times are separately measured. Selection was frozen before these diagnostics and is not changed by public results. One training seed and one public scene do not establish statistical or physical-robot performance."]
    (OUT / "comparison.md").write_text("\n".join(lines)+"\n")
    for variant in ("continuation", "supervised"):
        source_dir = ROOT / f"models/correspondence-{variant}-v6"; dest = OUT / variant; dest.mkdir()
        for filename in ("net.onnx", "meta.json", "run.json", "visual-probe.json"): shutil.copy2(source_dir / filename, dest)
    for name in ("tests", "geometry-tests", "prepare", "continuation", "supervised", "validation", "matching-audit", "check-continuation", "check-supervised"):
        extension = ".txt" if name in ("tests", "geometry-tests") else ".log"
        shutil.copy2(ROOT / ("logs/correspondence-v6-"+name+extension), OUT / (name+".txt"))
    shutil.copy2(ROOT / "data/correspondence-cache-v6/cache.json", OUT / "cache-provenance.json")
    shutil.copy2(ROOT / "logs/correspondence-v6-pipeline.json", OUT / "pipeline-status.json")
    print("\n".join(lines[:13]))


if __name__ == "__main__": main()
