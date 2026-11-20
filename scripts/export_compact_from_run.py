"""Re-run only the compact export of a finished IMU-only train_motion.py run (no retraining)."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from train_motion import export_compact, load_data
from training.motion_model import MotionNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="finished full-graph output directory")
    parser.add_argument("--compact-out", type=Path, required=True)
    parser.add_argument("--compact-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.compact_out.exists():
        parser.error("compact output exists")
    meta = json.loads((args.run / "meta.json").read_text())
    run = json.loads((args.run / "run.json").read_text())
    if meta["input_mode"] != "imu_only":
        parser.error("compact export needs an IMU-only run")
    net = MotionNet()
    net.load_state_dict(torch.load(args.run / "best.pt", weights_only=True))
    net.eval()
    normalization = tuple(torch.tensor(meta[key]) for key in ["state_mean", "state_std", "target_mean", "target_std"])
    validation = load_data(args.cache / "validation.npz")
    export_compact(net, args.compact_out, args.run, meta, run, validation, normalization, args.compact_tolerance)


if __name__ == "__main__":
    main()
