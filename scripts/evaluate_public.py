"""Run the pinned public set without overwriting evidence or generating videos.

Use the project venv: python scripts/evaluate_public.py --run-dir results/<unique-name>
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wasm", type=Path, default=ROOT / "artifacts/klt_imu.wasm")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, choices=range(1, 11), default=10)
    parser.add_argument("--verify-baseline", action="store_true")
    args = parser.parse_args()
    wasm = args.wasm.resolve(strict=True)
    run_dir = args.run_dir.resolve()
    limits = ROOT / "configs/evaluation-limits.json"
    selection = ROOT / "configs/public-episodes.json"
    episodes = json.loads(selection.read_text())[:args.max_episodes]
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    upstream = ROOT / "external/viobench"
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != lock["commit"]:
        parser.error("upstream revision differs from upstream.lock.json")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        parser.error("upstream checkout must be clean")
    # Ensure the selected Python environment really imports this reviewed checkout.
    import viobench
    if not Path(viobench.__file__).resolve().is_relative_to(upstream.resolve()):
        parser.error("viobench must be installed from external/viobench")
    from viobench.tools.abi.module import embedded_bundle
    from viobench.solution.runtime.python.bundler.embed import strip_bundle
    wasm_bytes = wasm.read_bytes()
    with zipfile.ZipFile(io.BytesIO(embedded_bundle(wasm_bytes))) as archive:
        bundle_files = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
        embedded_meta = json.loads(archive.read("meta.json")) if "meta.json" in archive.namelist() else None
    source = ROOT / "workspace/solution/solution.py"
    if args.verify_baseline:
        baseline = json.loads((ROOT / "experiments/baseline-klt-imu/source.json").read_text())
        if bundle_files.get("solution.py") != baseline["solution_sha256"]:
            parser.error("embedded solution.py is not the unchanged baseline")
    if run_dir.exists():
        parser.error("run directory already exists; choose a new name to preserve evidence")
    run_dir.mkdir(parents=True)
    command = [sys.executable, "-u", "-m", "viobench", "evaluate", str(wasm),
               *["conta:" + entry["hash"] for entry in episodes], "--limits", str(limits)]
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "working_directory": str(run_dir),
        "upstream_commit": revision, "solution_sha256": bundle_files.get("solution.py"),
        "workspace_solution_sha256": sha256(source),
        "bundle_files_sha256": bundle_files, "embedded_model_metadata": embedded_meta,
        "runtime_sha256": hashlib.sha256(strip_bundle(wasm_bytes)).hexdigest(),
        "wasm_sha256": sha256(wasm), "wasm_bytes": wasm.stat().st_size,
        "limits_sha256": sha256(limits), "selection_sha256": sha256(selection),
        "episodes": episodes, "python": sys.version, "host": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["wasmtime", "numpy", "opencv-python-headless", "conta", "viobench"]},
        "baseline_source_verified": args.verify_baseline,
        "evidence_scope": "Local public-set evaluation only; server limits and held-out performance unverified."
    }
    manifest = run_dir / "manifest.json"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    with (run_dir / "evaluate.log").open("w") as log:
        process = subprocess.Popen(command, cwd=run_dir, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        result = process.wait()
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["exit_code"] = result
    report = run_dir / ("report-" + wasm.stem) / "report.json"
    if report.exists():
        record["report_sha256"] = sha256(report)
        record["summary"] = json.loads(report.read_text())["summary"]
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
