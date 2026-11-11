"""Package untouched tiny_cnn guests and run frozen v7 comparisons serially."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol_path = ROOT / "configs/tiny-cnn-v7.json"
    protocol = json.loads(protocol_path.read_text()); protocol_hash = digest(protocol_path)
    status = ROOT / "logs/tiny-cnn-v7-pipeline.json"
    if status.exists(): raise ValueError("preserve previous pipeline")
    stages = []; identities = {}
    for variant in ("upstream", "matched"):
        name = "tiny-cnn-v7-"+variant
        run = json.loads((ROOT / "models" / name / "run.json").read_text())
        if run["protocol_sha256"] != protocol_hash: raise ValueError("training protocol differs")
        stage = ROOT / "artifacts" / (name+"-solution")
        for filename, expected in run["bundle_files_sha256"].items():
            if digest(stage / filename) != expected: raise ValueError("training payload changed")
        stages += [("package-"+variant, [".venv/bin/python", "-m", "viobench", "bundle", str(stage),
            "-o", "artifacts/"+name+".wasm", "--runtime", "workspace/build/runtime/dist/runtime.wasm"]),
            ("check-"+variant, [".venv/bin/python", "-u", "-m", "viobench", "check", "artifacts/"+name+".wasm", "--frames", "40", "--logs", "logs/check-"+name])]
    # Preserve the exact old artifacts. Hashes are frozen before any public evaluation.
    for variant, name in protocol["variants"].items():
        if variant not in ("upstream", "matched", "matched-repeat"):
            identities[variant] = digest(ROOT / "artifacts" / (name+".wasm"))
        stages.append(("evaluate-"+variant, [".venv/bin/python", "-u", "scripts/evaluate_public.py", "--wasm", "artifacts/"+name+".wasm", "--run-dir", "results/tiny-cnn-v7-"+variant]))
    (ROOT / "logs/tiny-cnn-v7-reference-hashes.json").write_text(json.dumps(identities, indent=2)+"\n")
    records = []
    for label, command in stages:
        if digest(protocol_path) != protocol_hash: raise ValueError("protocol changed")
        for variant, expected in identities.items():
            if digest(ROOT / "artifacts" / (protocol["variants"][variant]+".wasm")) != expected: raise ValueError("reference changed")
        log = ROOT / ("logs/tiny-cnn-v7-"+label+".log")
        print("START", label, flush=True); start = time.perf_counter()
        with log.open("x") as handle: result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        records.append({"stage": label, "command": command, "exit_code": result.returncode, "elapsed_seconds": time.perf_counter()-start, "log": str(log), "protocol_sha256": protocol_hash})
        status.write_text(json.dumps(records, indent=2)+"\n")
        print("END", label, result.returncode, flush=True)
        if result.returncode: raise SystemExit(result.returncode)


if __name__ == "__main__": main()
