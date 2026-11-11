"""Run the predeclared v5 WASM checks and six serial public evaluations."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection_path = ROOT / "experiments/feature-correlation-v5/selection.json"
    selection = json.loads(selection_path.read_text())
    selection_hash = digest(selection_path)
    if selection["public_data_used"] or selection["protocol_sha256"] != digest(ROOT / "configs/feature-correlation-v5.json"):
        raise ValueError("invalid frozen selection")
    for name, expected in selection["source_sha256"].items():
        if digest(ROOT / name) != expected: raise ValueError("guest source changed after validation")
    status = ROOT / "logs/correlation-v5-pipeline.json"
    if status.exists(): raise ValueError("preserve existing pipeline evidence")
    stages = []
    for variant, directory in (("single-frame", "models/correlation-single-frame-v5-complete"),
                               ("temporal", "models/correlation-temporal-v5")):
        name = "correlation-v5-" + variant
        stages.append(("package-" + variant, [".venv/bin/python", "scripts/package_motion.py", "--model", directory,
                       "--name", name, "--calibrate-gyro"]))
        stages.append(("check-" + variant, [".venv/bin/python", "-u", "-m", "viobench", "check",
                       "artifacts/" + name + ".wasm", "--frames", "40", "--logs", "logs/check-" + name]))
    # Use the exact preserved reference artifacts, not rebuilt approximations.
    variants = [("baseline", "klt_imu"), ("current", "gyro-v4-current"),
                ("calibrated-gyro", "gyro-v4-cal-gyro"), ("single-frame", "correlation-v5-single-frame"),
                ("temporal", "correlation-v5-temporal"), ("temporal-repeat", "correlation-v5-temporal")]
    for variant, name in variants:
        command = [".venv/bin/python", "-u", "scripts/evaluate_public.py", "--wasm", "artifacts/" + name + ".wasm",
                   "--run-dir", "results/correlation-v5-" + variant]
        if variant == "baseline": command.append("--verify-baseline")
        stages.append(("evaluate-" + variant, command))
    records = []
    for stage, command in stages:
        if digest(selection_path) != selection_hash: raise ValueError("selection changed")
        log = ROOT / ("logs/correlation-v5-" + stage + ".log")
        print("START " + stage, flush=True)
        start = time.perf_counter()
        with log.open("x") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        records.append({"stage": stage, "command": command, "exit_code": result.returncode,
                        "elapsed_seconds": time.perf_counter()-start, "log": str(log),
                        "selection_sha256": selection_hash})
        status.write_text(json.dumps(records, indent=2) + "\n")
        print("END " + stage + ": " + str(result.returncode), flush=True)
        if result.returncode: raise SystemExit(result.returncode)


if __name__ == "__main__": main()
