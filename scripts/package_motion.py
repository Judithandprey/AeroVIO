"""Bundle a trained checkpoint with explicit runtime ablations, without editing it."""
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
    ablation = parser.add_mutually_exclusive_group()
    ablation.add_argument("--map", action="store_true")
    ablation.add_argument("--disable-rotation-residual", action="store_true",
                          help="use the same predicted translation with gyro-only rotation; map stays off")
    parser.add_argument("--calibrate-gyro", action="store_true",
                        help="enable the fixed causal one-second initial gyro calibration; map stays off")
    args = parser.parse_args()
    if args.calibrate_gyro and args.map:
        parser.error("--calibrate-gyro cannot be combined with --map")
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
    # Preserve the checkpoint's input mode and graph contract. Only this requested
    # output-composition switch changes the fixed-weight rotation ablation.
    if args.disable_rotation_residual:
        meta["rotation_residual_enabled"] = False
    meta["map"] = {"enabled": args.map}
    meta["gyro_bias"] = {"enabled": args.calibrate_gyro}
    sys.path.insert(0, str(ROOT / "workspace/solution"))
    from motion_inference import validate_motion_options
    input_mode, network_inputs, rotation_enabled = validate_motion_options(meta)
    stage.mkdir(parents=True)
    for source in sorted((ROOT / "workspace/solution").glob("*.py")):
        shutil.copyfile(source, stage / source.name)
    shutil.copyfile(model, stage / "net.onnx")
    meta["diagnostics_interval"] = 0
    (stage / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    command = [sys.executable, "-m", "viobench", "bundle", str(stage), "-o", str(artifact),
               "--runtime", str(ROOT / "workspace/build/runtime/dist/runtime.wasm")]
    subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"wasm": str(artifact), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      "model_sha256": meta["model_sha256"], "map_enabled": args.map,
                      "rotation_residual_enabled": rotation_enabled, "input_mode": input_mode,
                      "network_inputs": network_inputs,
                      "gyro_bias_enabled": args.calibrate_gyro}, indent=2))


if __name__ == "__main__":
    main()
