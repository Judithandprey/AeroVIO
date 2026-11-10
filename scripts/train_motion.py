"""Train the same small network with pose-only or pose-plus-warp supervision.

Checkpoint selection uses scene-disjoint validation pose loss, never public-set
scores. Training-only depth and GT visibility never enter the inference bundle.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.motion_model import MotionNet, photometric_loss


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: torch.from_numpy(archive[key].copy()) for key in archive.files if key != "episode_id"}


def evaluate(net, data, normalization, batch_size):
    totals = {"pose_loss": 0., "photo_loss": 0., "translation_sq": 0., "rotation_residual_sq": 0.}
    visible_pairs = 0
    mean, std, target_mean, target_std = normalization
    size = len(data["target"])
    with torch.no_grad():
        for start in range(0, size, batch_size):
            batch = {key: value[start:start + batch_size] for key, value in data.items()}
            count = len(batch["target"])
            prediction = net(batch["images"], (batch["state"] - mean) / std)
            raw = prediction * target_std + target_mean
            totals["pose_loss"] += float(F.smooth_l1_loss(prediction, (batch["target"] - target_mean) / target_std)) * count
            usable = int((batch["visibility"].sum((1, 2, 3)) > 0).sum())
            totals["photo_loss"] += float(photometric_loss(batch["images"], batch["depth"], batch["intrinsics"], raw,
                                                          batch["state"][:, :3], batch["visibility"])) * usable
            visible_pairs += usable
            difference = raw - batch["target"]
            totals["translation_sq"] += float(difference[:, :3].square().sum())
            totals["rotation_residual_sq"] += float(difference[:, 3:].square().sum())
    return {"pose_loss": totals["pose_loss"] / size, "photo_loss": totals["photo_loss"] / max(visible_pairs, 1),
            "visible_pairs": visible_pairs,
            "translation_rmse_m": (totals["translation_sq"] / size) ** .5,
            "rotation_residual_rmse_rad": (totals["rotation_residual_sq"] / size) ** .5}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warp-weight", type=float, default=0.)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists; preserve previous runs")
    args.out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    train, validation = (load_data(args.cache / f"{split}.npz") for split in ["train", "validation"])
    state_mean = train["state"].mean(0)
    state_std = train["state"].std(0, correction=0).clamp_min(1e-6)
    target_mean = train["target"].mean(0)
    target_std = train["target"].std(0, correction=0).clamp_min(torch.tensor([1e-3] * 3 + [1e-4] * 3))
    normalization = state_mean, state_std, target_mean, target_std
    net = MotionNet()
    initial = {name: tensor.detach().clone() for name, tensor in net.state_dict().items()}
    initial_hash = hashlib.sha256(b"".join(value.numpy().tobytes() for value in initial.values())).hexdigest()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    meta = {"schema_version": 1, "input_width": 64, "input_height": 48,
            "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
            "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
            "map": {"enabled": False}, "diagnostics_interval": 0}
    cache_record = json.loads((args.cache / "cache.json").read_text())
    run = {"started_utc": datetime.now(timezone.utc).isoformat(), "seed": args.seed,
           "epochs": args.epochs, "batch": args.batch, "learning_rate": args.lr,
           "warp_weight": args.warp_weight, "threads": args.threads, "device": "cpu",
           "parameters": sum(p.numel() for p in net.parameters()), "initial_weights_sha256": initial_hash,
           "train_pairs": len(train["target"]), "validation_pairs": len(validation["target"]),
           "train_cache_sha256": digest(args.cache / "train.npz"),
           "validation_cache_sha256": digest(args.cache / "validation.npz"),
           "cache_provenance": cache_record, "selection": "minimum validation normalized pose loss",
           "packages": {name: importlib.metadata.version(name) for name in ["torch", "onnx", "numpy", "opencv-python-headless"]},
           "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                             [Path(__file__), ROOT / "training/motion_model.py", ROOT / "workspace/solution/preintegrate.py"]}}
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    best_loss = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        # Identical order in both ablations, independent of other RNG consumption.
        order = torch.randperm(len(train["target"]), generator=torch.Generator().manual_seed(args.seed + epoch + 1000))
        net.train()
        loss_sum = 0.
        for indices in order.split(args.batch):
            batch = {key: value[indices] for key, value in train.items()}
            prediction = net(batch["images"], (batch["state"] - state_mean) / state_std)
            loss = F.smooth_l1_loss(prediction, (batch["target"] - target_mean) / target_std)
            if args.warp_weight:
                raw = prediction * target_std + target_mean
                loss = loss + args.warp_weight * photometric_loss(batch["images"], batch["depth"], batch["intrinsics"],
                    raw, batch["state"][:, :3], batch["visibility"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        scheduler.step()
        net.eval()
        metrics = evaluate(net, validation, normalization, args.batch)
        row = {"epoch": epoch + 1, "train_objective": loss_sum / len(order), **metrics,
               "elapsed_s": time.perf_counter() - epoch_start}
        history.append(row)
        if metrics["pose_loss"] < best_loss:
            best_loss = metrics["pose_loss"]
            best_epoch = epoch + 1
            torch.save(net.state_dict(), args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    net.load_state_dict(torch.load(args.out / "best.pt", weights_only=True))
    net.eval()
    model_path = args.out / "net.onnx"
    example = (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14))
    torch.onnx.export(net, example, str(model_path), input_names=["images", "state"], output_names=["motion"],
                      opset_version=13, dynamo=False)
    cv_net = cv2.dnn.readNetFromONNX(str(model_path))
    cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    maximum_difference = 0.
    for index in range(min(32, len(validation["target"]))):
        image = validation["images"][index:index + 1]
        state = (validation["state"][index:index + 1] - state_mean) / state_std
        with torch.no_grad(): expected = net(image, state).numpy()
        cv_net.setInput(image.numpy(), "images")
        cv_net.setInput(state.numpy(), "state")
        actual = cv_net.forward()
        maximum_difference = max(maximum_difference, float(np.max(np.abs(actual - expected))))
    if maximum_difference > 1e-4:
        raise ValueError(f"ONNX parity failed: {maximum_difference}")
    meta["model_sha256"] = digest(model_path)
    meta["training_stage"] = "warp" if args.warp_weight else "pose"
    meta["training_seed"] = args.seed
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    run.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": time.perf_counter() - started,
                "best_epoch": best_epoch, "best_validation": evaluate(net, validation, normalization, args.batch),
                "model_sha256": digest(model_path), "model_bytes": model_path.stat().st_size,
                "opencv_torch_max_abs_difference": maximum_difference})
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print("FINISHED", json.dumps({key: run[key] for key in ["best_epoch", "best_validation", "elapsed_s", "model_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
