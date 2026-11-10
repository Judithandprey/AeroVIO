"""Bundle one trained checkpoint, with optional map correction, without editing it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args()
    if not args.name.replace("-", "").replace("_", "").isalnum():
        parser.error("name must contain only letters, digits, dash or underscore")
    stage = ROOT / "artifacts" / (args.name + "-solution")
    artifact = ROOT / "artifacts" / (args.name + ".wasm")
    if stage.exists() or artifact.exists():
        parser.error("artifact already exists; use a new name")
    meta = json.loads((args.model / "meta.json").read_text())
    model = args.model / "net.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != meta["model_sha256"]:
        raise ValueError("model and metadata hashes differ")
    stage.mkdir(parents=True)
    for source in sorted((ROOT / "workspace/solution").glob("*.py")):
        shutil.copyfile(source, stage / source.name)
    shutil.copyfile(model, stage / "net.onnx")
    meta["map"] = {"enabled": args.map}
    meta["diagnostics_interval"] = 0
    (stage / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    command = [sys.executable, "-m", "viobench", "bundle", str(stage), "-o", str(artifact),
               "--runtime", str(ROOT / "workspace/build/runtime/dist/runtime.wasm")]
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"wasm": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      "model_sha256": meta["model_sha256"], "map_enabled": args.map}, indent=2))


if __name__ == "__main__":
    main()
