"""Matched continuation versus rotation-supervised auxiliary fine-tuning.

Uses the unchanged MotionNet and frozen starting normalization. Selection uses
real scene-disjoint validation rollouts with two translation guards; synthetic
rotation validation is diagnostic only. Public data is never loaded here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "workspace/solution")]
from motion_inference import validate_motion_options
from training.motion_model import MotionNet, photometric_loss, so3_exp
from training.rotation_focus import (eligible_checkpoint, geodesic_angle,
                                     rollout_metrics, rotation_auxiliary_loss)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_arrays(path, keys):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: torch.from_numpy(archive[key].copy()) for key in keys}
    count = len(data["images"])
    if count == 0 or any(len(value) != count for value in data.values()):
        raise ValueError("cache arrays must be nonempty with equal lengths")
    if data["images"].shape != (count, 2, 48, 64) or data["state"].shape != (count, 14):
        raise ValueError("cache image/state dimensions do not match MotionNet")
    if not all(torch.isfinite(value).all().item() for value in data.values()):
        raise ValueError("cache contains non-finite values")
    if "rotation_gt" in data:
        rotation = data["rotation_gt"]
        if rotation.shape != (count, 3, 3):
            raise ValueError("auxiliary rotations must have shape Nx3x3")
        identity = torch.eye(3, dtype=rotation.dtype)
        if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity.expand_as(rotation), atol=1e-5):
            raise ValueError("auxiliary rotations must be orthonormal")
        if not torch.allclose(torch.linalg.det(rotation), torch.ones(count, dtype=rotation.dtype), atol=1e-5):
            raise ValueError("auxiliary rotations must have positive unit determinant")
    return data


def audit_caches(real_path, aux_path):
    real = json.loads((real_path / "cache.json").read_text())
    aux = json.loads((aux_path / "cache.json").read_text())
    if aux.get("status") != "complete" or aux.get("kind") != "pure_rotation_augmentation":
        raise ValueError("rotation cache must be a completed pure-rotation augmentation")
    source = aux["source_real_cache"]
    for key, path in [("cache_json_sha256", real_path / "cache.json"),
                      ("train_sha256", real_path / "train.npz"),
                      ("validation_sha256", real_path / "validation.npz")]:
        if source[key] != digest(path):
            raise ValueError("auxiliary source real-cache fingerprint mismatch: " + key)
    if aux["source_manifest"]["sha256"] != real["source_manifest_sha256"]:
        raise ValueError("auxiliary source dataset manifest differs")
    scenes = {}
    for split in ("train", "validation"):
        if digest(real_path / (split + ".npz")) != real["splits"][split]["sha256"]:
            raise ValueError("real cache hash mismatch")
        if digest(aux_path / (split + ".npz")) != aux["splits"][split]["sha256"]:
            raise ValueError("auxiliary cache hash mismatch")
        original = {entry["scene"] for entry in real["splits"][split]["episodes"]}
        scenes[split] = set(aux["splits"][split]["scenes"])
        if not scenes[split] or not scenes[split] <= original:
            raise ValueError("auxiliary scenes do not belong to their real split")
    if scenes["train"] & scenes["validation"] or "ProcTHOR-Train-1" in set.union(*scenes.values()):
        raise ValueError("training/validation/public scene separation failed")
    for name, expected in aux["source_sha256"].items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT / path
        if digest(path) != expected:
            raise ValueError("augmentation source changed after generation: " + name)
    return real, aux


def decoded(net, batch, normalization):
    mean, std, target_mean, target_std = normalization
    prediction = net(batch["images"], (batch["state"] - mean) / std)
    return prediction, prediction * target_std + target_mean


def real_objective(net, batch, normalization):
    prediction, raw = decoded(net, batch, normalization)
    _, _, target_mean, target_std = normalization
    pose = F.smooth_l1_loss(prediction, (batch["target"] - target_mean) / target_std)
    photo = photometric_loss(batch["images"], batch["depth"], batch["intrinsics"],
                             raw, batch["state"][:, :3], batch["visibility"])
    return pose + 2.0 * photo, pose, photo, raw


def evaluate_real(net, data, normalization, batch_size):
    raw_values = []
    pose_total = photo_total = 0.0
    visible_pairs = 0
    with torch.no_grad():
        for start in range(0, len(data["images"]), batch_size):
            batch = {key: value[start:start + batch_size] for key, value in data.items()}
            _, pose, photo, raw = real_objective(net, batch, normalization)
            usable = int((batch["visibility"].sum((1, 2, 3)) > 0).sum())
            pose_total += float(pose) * len(raw)
            photo_total += float(photo) * usable
            visible_pairs += usable
            raw_values.append(raw)
    raw = torch.cat(raw_values)
    metrics = rollout_metrics(raw, data["target"], data["state"], data["episode_id"])
    metrics.update({"normalized_pose_loss": pose_total / len(raw),
                    "photo_loss": photo_total / max(visible_pairs, 1),
                    "visible_pairs": visible_pairs,
                    "rotation_residual_rmse_rad": float((raw[:, 3:] - data["target"][:, 3:]).square().sum(1).mean().sqrt())})
    return metrics


def evaluate_aux(net, data, normalization, batch_size, scale):
    angle_sq = loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, len(data["images"]), batch_size):
            batch = {key: value[start:start + batch_size] for key, value in data.items()}
            _, raw = decoded(net, batch, normalization)
            angles = geodesic_angle(so3_exp(batch["state"][:, :3]) @ so3_exp(raw[:, 3:]), batch["rotation_gt"])
            angle_sq += float(angles.square().sum())
            loss_sum += float(rotation_auxiliary_loss(raw, batch["state"], batch["rotation_gt"], scale)) * len(raw)
    return {"rotation_geodesic_rmse_rad": math.sqrt(angle_sq / len(data["images"])),
            "normalized_rotation_huber": loss_sum / len(data["images"]),
            "pairs": len(data["images"])}


def verify_onnx(net, path, validation, normalization):
    cv_net = cv2.dnn.readNetFromONNX(str(path))
    cv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    cv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    maximum = 0.0
    with torch.no_grad():
        for index in range(min(32, len(validation["images"]))):
            image = validation["images"][index:index + 1]
            state = (validation["state"][index:index + 1] - normalization[0]) / normalization[1]
            expected = net(image, state).numpy()
            cv_net.setInput(image.numpy(), "images")
            cv_net.setInput(state.numpy(), "state")
            actual = cv_net.forward()
            if not np.isfinite(actual).all():
                raise ValueError("ONNX output is non-finite")
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    if maximum > 1e-4:
        raise ValueError("checkpoint/ONNX prediction parity failed: " + str(maximum))
    return maximum


class AuxiliarySampler:
    def __init__(self, count, seed=314161):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.int64)
        self.cursor = 0

    def next(self, count):
        pieces = []
        while count:
            if self.cursor == len(self.order):
                self.order = torch.randperm(self.count, generator=self.generator)
                self.cursor = 0
            take = min(count, len(self.order) - self.cursor)
            pieces.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            count -= take
        return torch.cat(pieces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/motion-cache-v2")
    parser.add_argument("--rotation-cache", type=Path, default=ROOT / "data/rotation-aux-v3")
    parser.add_argument("--initial-model", type=Path, default=ROOT / "models/motion-warp-v1")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("control", "focus"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--aux-batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--aux-weight", type=float, default=.1)
    parser.add_argument("--translation-guard", type=float, default=1.05)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; preserve earlier evidence")
    if min(args.epochs, args.batch, args.aux_batch, args.threads) <= 0 or args.lr <= 0 or args.aux_weight < 0:
        parser.error("training counts/rate must be positive and auxiliary weight nonnegative")
    if not np.isfinite(args.translation_guard) or args.translation_guard < 1:
        parser.error("translation guard must be finite and at least one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    real_record, aux_record = audit_caches(args.cache, args.rotation_cache)
    real_keys = ("images", "state", "target", "depth", "intrinsics", "visibility", "episode_id")
    aux_keys = ("images", "state", "rotation_gt")
    train, validation = [load_arrays(args.cache / (split + ".npz"), real_keys) for split in ("train", "validation")]
    aux_train, aux_validation = [load_arrays(args.rotation_cache / (split + ".npz"), aux_keys) for split in ("train", "validation")]
    meta = json.loads((args.initial_model / "meta.json").read_text())
    if validate_motion_options(meta) != ("image_imu", "images_state", True) or meta.get("map", {}).get("enabled", False):
        raise ValueError("starting model must be the full image+IMU network with residual and no map")
    if digest(args.initial_model / "net.onnx") != meta["model_sha256"]:
        raise ValueError("starting ONNX hash differs from metadata")
    normalization = tuple(torch.tensor(meta[key], dtype=torch.float32) for key in
                          ("state_mean", "state_std", "target_mean", "target_std"))
    if any(x.shape != (length,) or not torch.isfinite(x).all() for x, length in zip(normalization, (14, 14, 6, 6))):
        raise ValueError("invalid starting normalization")
    if torch.any(normalization[1] <= 0) or torch.any(normalization[3] <= 0):
        raise ValueError("normalization scales must be positive")
    net = MotionNet()
    net.load_state_dict(torch.load(args.initial_model / "best.pt", weights_only=True, map_location="cpu"))
    net.eval()
    initial_hash = hashlib.sha256(b"".join(value.detach().numpy().tobytes() for value in net.state_dict().values())).hexdigest()
    starting_parity = verify_onnx(net, args.initial_model / "net.onnx", validation, normalization)
    scale = float(torch.linalg.vector_norm(normalization[3][3:]))
    baseline = evaluate_real(net, validation, normalization, args.batch)
    baseline_aux = evaluate_aux(net, aux_validation, normalization, args.batch, scale)
    args.out.mkdir(parents=True)
    torch.save(net.state_dict(), args.out / "best.pt")
    best_epoch, best_metrics = 0, baseline
    history = [{"epoch": 0, "real_validation": baseline, "aux_validation": baseline_aux,
                "eligible": True, "selected": True}]
    source_paths = [Path(__file__), ROOT / "configs/rotation-focus-v3.json",
                    ROOT / "training/rotation_focus.py", ROOT / "training/motion_model.py",
                    ROOT / "workspace/solution/preintegrate.py", ROOT / "workspace/solution/motion_inference.py"]
    steps_per_epoch = math.ceil(len(train["images"]) / args.batch)
    run = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
           "variant": args.variant, "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
           "aux_batch": args.aux_batch, "auxiliary_sampler_seed": 314161, "learning_rate": args.lr,
           "final_learning_rate": args.lr * .1, "optimizer": "fresh Adam, all MotionNet parameters",
           "gradient_clip_norm": 5.0, "threads": args.threads, "device": "cpu",
           "parameters": sum(p.numel() for p in net.parameters()),
           "initial_weights_sha256": initial_hash,
           "initial_checkpoint_sha256": digest(args.initial_model / "best.pt"),
           "initial_model_sha256": meta["model_sha256"], "initial_metadata_sha256": digest(args.initial_model / "meta.json"),
           "initial_checkpoint_onnx_max_abs_difference": starting_parity,
           "normalization": {key: meta[key] for key in ("state_mean", "state_std", "target_mean", "target_std")},
           "real_objective": "normalized pose SmoothL1 + 2 * real-image masked photometric loss",
           "auxiliary_objective": "SmoothL1(SO3 angle / norm(initial target_std[3:]), 0), rotation only",
           "auxiliary_weight": args.aux_weight if args.variant == "focus" else 0.0,
           "auxiliary_rotation_scale_rad": scale,
           "selection": {"objective": "minimum arithmetic mean per-episode accumulated rotation RMSE on real validation",
                         "guards": "pair translation RMSE and mean unaligned trajectory position RMSE each <= ratio * epoch0",
                         "translation_guard_ratio": args.translation_guard, "include_epoch_zero": True,
                         "ties": "keep earlier checkpoint", "auxiliary_validation": "diagnostic only"},
           "planned_optimizer_steps": steps_per_epoch * args.epochs,
           "train_pairs": len(train["images"]), "validation_pairs": len(validation["images"]),
           "aux_train_pairs": len(aux_train["images"]), "aux_validation_pairs": len(aux_validation["images"]),
           "real_cache": real_record, "auxiliary_cache": aux_record,
           "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in source_paths},
           "packages": {name: importlib.metadata.version(name) for name in ("torch", "onnx", "numpy", "opencv-python-headless")}}
    write_json(args.out / "run.json", run)
    write_json(args.out / "history.json", history)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .1)
    sampler = AuxiliarySampler(len(aux_train["images"]))
    real_order_hash, aux_order_hash = hashlib.sha256(), hashlib.sha256()
    optimizer_steps = auxiliary_examples = real_examples = 0
    auxiliary_compute_s = train_step_s = 0.0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        order = torch.randperm(len(train["images"]), generator=torch.Generator().manual_seed(args.seed + epoch + 1000))
        real_order_hash.update(order.numpy().tobytes())
        real_total = auxiliary_total = 0.0
        auxiliary_count = 0
        net.train()
        for indices in order.split(args.batch):
            step_start = time.perf_counter()
            batch = {key: value[indices] for key, value in train.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = real_objective(net, batch, normalization)
            loss.backward()
            real_total += float(loss.detach()) * len(indices)
            real_examples += len(indices)
            if args.variant == "focus":
                aux_start = time.perf_counter()
                aux_indices = sampler.next(args.aux_batch)
                aux_order_hash.update(aux_indices.numpy().tobytes())
                auxiliary = {key: value[aux_indices] for key, value in aux_train.items()}
                _, raw = decoded(net, auxiliary, normalization)
                aux_loss = rotation_auxiliary_loss(raw, auxiliary["state"], auxiliary["rotation_gt"], scale)
                # Separate backward adds the same auxiliary gradient before one
                # shared clipping/optimizer step and measures its extra compute.
                (args.aux_weight * aux_loss).backward()
                auxiliary_compute_s += time.perf_counter() - aux_start
                auxiliary_total += float(aux_loss.detach()) * len(aux_indices)
                auxiliary_count += len(aux_indices)
                auxiliary_examples += len(aux_indices)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
            optimizer.step()
            optimizer_steps += 1
            train_step_s += time.perf_counter() - step_start
        scheduler.step()
        net.eval()
        metrics = evaluate_real(net, validation, normalization, args.batch)
        auxiliary_metrics = evaluate_aux(net, aux_validation, normalization, args.batch, scale)
        eligible = eligible_checkpoint(metrics, baseline, args.translation_guard)
        selected = eligible and metrics["rotation_rmse_mean_rad"] < best_metrics["rotation_rmse_mean_rad"]
        if selected:
            best_epoch, best_metrics = epoch + 1, metrics
            torch.save(net.state_dict(), args.out / "best.pt")
        row = {"epoch": epoch + 1, "real_train_objective": real_total / len(order),
               "aux_train_objective": auxiliary_total / max(auxiliary_count, 1),
               "real_validation": metrics, "aux_validation": auxiliary_metrics,
               "eligible": eligible, "selected": bool(selected), "elapsed_s": time.perf_counter() - epoch_start}
        history.append(row)
        write_json(args.out / "history.json", history)
        print(json.dumps({"epoch": epoch + 1, "variant": args.variant, "eligible": eligible,
                          "selected": bool(selected), "rotation_rmse_rad": metrics["rotation_rmse_mean_rad"],
                          "pair_translation_rmse_m": metrics["pair_translation_rmse_m"],
                          "trajectory_position_rmse_m": metrics["trajectory_position_rmse_mean_m"],
                          "aux_rotation_rmse_rad": auxiliary_metrics["rotation_geodesic_rmse_rad"],
                          "elapsed_s": row["elapsed_s"]}), flush=True)
    net.load_state_dict(torch.load(args.out / "best.pt", weights_only=True, map_location="cpu"))
    net.eval()
    model_path = args.out / "net.onnx"
    torch.onnx.export(net, (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14)), str(model_path),
                      input_names=["images", "state"], output_names=["motion"], opset_version=13, dynamo=False)
    export_parity = verify_onnx(net, model_path, validation, normalization)
    exported_meta = {**meta, "model_sha256": digest(model_path), "training_stage": "rotation-" + args.variant,
                     "training_seed": args.seed, "input_mode": "image_imu", "network_inputs": "images_state",
                     "rotation_residual_enabled": True, "map": {"enabled": False}, "diagnostics_interval": 0}
    write_json(args.out / "meta.json", exported_meta)
    run.update({"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": time.perf_counter() - started, "best_epoch": best_epoch,
                "best_validation": evaluate_real(net, validation, normalization, args.batch),
                "best_auxiliary_validation": evaluate_aux(net, aux_validation, normalization, args.batch, scale),
                "baseline_validation": baseline, "optimizer_steps": optimizer_steps,
                "real_examples_processed": real_examples, "auxiliary_examples_processed": auxiliary_examples,
                "train_step_s": train_step_s, "auxiliary_forward_backward_s": auxiliary_compute_s,
                "extra_compute_scope": "aux sampling, forward, loss and backward; excludes shared optimizer/clip and validation",
                "real_minibatch_order_sha256": real_order_hash.hexdigest(),
                "auxiliary_minibatch_order_sha256": aux_order_hash.hexdigest() if auxiliary_examples else None,
                "checkpoint_sha256": digest(args.out / "best.pt"), "model_sha256": digest(model_path),
                "model_bytes": model_path.stat().st_size, "opencv_torch_max_abs_difference": export_parity})
    write_json(args.out / "run.json", run)
    print("FINISHED", json.dumps({key: run[key] for key in ("variant", "best_epoch", "elapsed_s", "model_sha256")}), flush=True)


if __name__ == "__main__":
    main()
