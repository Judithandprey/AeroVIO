"""Frozen v6 packaging, checks and seven serial public runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    selection_path = ROOT / "experiments/correspondence-v6/selection.json"
    selection = json.loads(selection_path.read_text()); selection_hash = digest(selection_path)
    if selection["public_data_used"] or selection["protocol_sha256"] != digest(ROOT / "configs/correspondence-v6.json"):
        raise ValueError("invalid frozen selection")
    for path, expected in selection["source_sha256"].items():
        if digest(ROOT / path) != expected: raise ValueError("guest changed after validation")
    status = ROOT / "logs/correspondence-v6-pipeline.json"
    if status.exists(): raise ValueError("preserve pipeline")
    stages = []
    for variant in ("continuation", "supervised"):
        name = "correspondence-v6-"+variant
        stages.append(("package-"+variant, [".venv/bin/python", "scripts/package_motion.py", "--model",
            "models/correspondence-"+variant+"-v6", "--name", name, "--calibrate-gyro"]))
        stages.append(("check-"+variant, [".venv/bin/python", "-u", "-m", "viobench", "check", "artifacts/"+name+".wasm",
            "--frames", "40", "--logs", "logs/check-"+name]))
    names = [("baseline", "klt_imu"), ("calibrated-gyro", "gyro-v4-cal-gyro"),
             ("v5-single-frame", "correlation-v5-single-frame"), ("v5-temporal", "correlation-v5-temporal"),
             ("continuation", "correspondence-v6-continuation"), ("supervised", "correspondence-v6-supervised"),
             ("supervised-repeat", "correspondence-v6-supervised")]
    for variant, name in names:
        command = [".venv/bin/python", "-u", "scripts/evaluate_public.py", "--wasm", "artifacts/"+name+".wasm",
                   "--run-dir", "results/correspondence-v6-"+variant]
        if variant == "baseline": command.append("--verify-baseline")
        stages.append(("evaluate-"+variant, command))
    records = []
    for stage, command in stages:
        if digest(selection_path) != selection_hash: raise ValueError("selection changed")
        log = ROOT / ("logs/correspondence-v6-"+stage+".log")
        print("START "+stage, flush=True); start = time.perf_counter()
        with log.open("x") as handle: result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        records.append({"stage": stage, "command": command, "exit_code": result.returncode,
            "elapsed_seconds": time.perf_counter()-start, "selection_sha256": selection_hash, "log": str(log)})
        status.write_text(json.dumps(records, indent=2)+"\n")
        print("END "+stage+": "+str(result.returncode), flush=True)
        if result.returncode: raise SystemExit(result.returncode)


if __name__ == "__main__": main()
