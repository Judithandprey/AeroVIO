"""Frozen v5 data preparation and two bounded rotation-branch training runs."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
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
from training.motion_model import MotionNet
from training.feature_correlation import CorrelationMotionNet
from preintegrate import exp_so3, log_so3, make_state
from gyro_bias import InitialGyroBias, integrate_debiased_rotation
from solution import quaternion_from_matrix
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices
from viobench.tools.runner.metrics import evaluate_trajectory

PROTOCOL = ROOT / "configs/feature-correlation-v5.json"
CACHE = ROOT / "data/correlation-cache-v5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare():
    if CACHE.exists():
        raise ValueError("preserve existing derived cache")
    protocol = json.loads(PROTOCOL.read_text())
    manifest_path = ROOT / protocol["manifest"]
    manifest = json.loads(manifest_path.read_text())
    original = ROOT / protocol["cache"]
    old_record = json.loads((original / "cache.json").read_text())
    if digest(manifest_path) != old_record["source_manifest_sha256"]:
        raise ValueError("manifest identity differs")
    scenes = [{e["scene"] for e in manifest[s]} for s in ("train", "validation")]
    if scenes[0] & scenes[1] or "ProcTHOR-Train-1" in scenes[0] | scenes[1]:
        raise ValueError("scene leakage")
    CACHE.mkdir(parents=True)
    record = {"protocol_sha256": digest(PROTOCOL), "manifest_sha256": digest(manifest_path),
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                  [Path(__file__), ROOT / "workspace/solution/gyro_bias.py", ROOT / "workspace/solution/preintegrate.py"]},
              "splits": {}}
    for split in ("train", "validation"):
        cache_path = original / (split + ".npz")
        if digest(cache_path) != old_record["splits"][split]["sha256"]:
            raise ValueError("original cache differs")
        with np.load(cache_path) as data:
            states, ids = data["state"], data["episode_id"]
        arrays = {key: [] for key in ("prior", "residual", "truth", "times")}
        episodes = []
        for episode_id, entry in enumerate(manifest[split]):
            episode = read_episode(Path(entry["path"]), frames="lazy")
            config, _ = episode.adapt((320, 240))
            calibration = InitialGyroBias()
            times = episode.frame_times
            truth = episode.ground_truth_state()
            poses = poses_to_matrices(truth[:, :7])
            indices = np.flatnonzero(ids == episode_id)
            if len(indices) != len(times) - 1:
                raise ValueError("cache episode length differs")
            calibration.observe_after_output(float(times[0]), episode.imu_window(0))
            for frame_index, index in enumerate(indices, start=1):
                t0, t1 = map(float, times[frame_index - 1:frame_index + 1])
                imu = episode.imu_window(frame_index)
                state = make_state(imu, config["T_cam_imu"], config, t0, t1)
                if not np.array_equal(state, states[index]):
                    raise ValueError("historical raw state differs")
                prior = exp_so3(state[:3].astype(np.float64))
                if calibration.accepted:
                    prior = integrate_debiased_rotation(imu, config["T_cam_imu"], t0, t1, calibration.bias_imu)
                relative = poses[frame_index - 1, :3, :3].T @ poses[frame_index, :3, :3]
                arrays["prior"].append(log_so3(prior))
                arrays["residual"].append(log_so3(prior.T @ relative))
                # No future/deadline-frame bias may enter the label's prior.
                calibration.observe_after_output(t1, imu)
            arrays["truth"].append(truth)
            arrays["times"].append(times)
            episodes.append({"episode_id": episode_id, "scene": entry["scene"],
                             "episode_json_sha256": digest(Path(entry["path"]) / "episode.json"),
                             "calibration": calibration.diagnostics()})
        out = CACHE / (split + ".npz")
        np.savez_compressed(out, **{key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()})
        record["splits"][split] = {"sha256": digest(out), "original_sha256": digest(cache_path), "episodes": episodes}
        print("prepared " + split, flush=True)
    record["status"] = "complete"
    write_json(CACHE / "cache.json", record)


def load(split, meta):
    record = json.loads((CACHE / "cache.json").read_text())
    original = ROOT / "data/motion-cache-v2" / (split + ".npz")
    derived = CACHE / (split + ".npz")
    if record["protocol_sha256"] != digest(PROTOCOL) or digest(original) != record["splits"][split]["original_sha256"] or digest(derived) != record["splits"][split]["sha256"]:
        raise ValueError("frozen data/protocol differs")
    with np.load(original) as data:
        result = {key: torch.from_numpy(data[key].copy()) for key in ("images", "state", "episode_id")}
    with np.load(derived) as data:
        result.update({key: torch.from_numpy(data[key].copy()) for key in data.files})
    result["normalized_state"] = (result["state"] - torch.tensor(meta["state_mean"])) / torch.tensor(meta["state_std"])
    return result


def score(data, translation, corrections):
    """Same ordered SO(3) composition and finite-interval twist as Solution.step."""
    corrections = np.asarray(corrections, dtype=np.float64)
    prior = data["prior"].numpy()
    ids = data["episode_id"].numpy()
    rows = []
    for episode_id in np.unique(ids):
        indices = np.flatnonzero(ids == episode_id)
        times = data["times"][episode_id].numpy()
        pose = np.eye(4)
        outputs = [np.array([0., 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])]
        for frame_index, index in enumerate(indices, start=1):
            rotation = exp_so3(prior[index]) @ exp_so3(corrections[index])
            new = pose.copy()
            new[:3, :3] = pose[:3, :3] @ rotation
            new[:3, 3] = pose[:3, 3] + pose[:3, :3] @ translation[index]
            dt = times[frame_index] - times[frame_index - 1]
            outputs.append(np.r_[new[:3, 3], quaternion_from_matrix(new[:3, :3]),
                                  new[:3, :3].T @ (new[:3, 3] - pose[:3, 3]) / dt,
                                  log_so3(pose[:3, :3].T @ new[:3, :3]) / dt])
            pose = new
        metrics = evaluate_trajectory(data["truth"][episode_id].numpy(), np.asarray(outputs), times).as_dict()
        if metrics["filled_frames"]:
            raise ValueError("incomplete validation")
        rows.append({"episode_id": int(episode_id), "metrics": metrics})
    return {"rotation_deg": float(np.degrees(np.mean([e["metrics"]["rotation_rmse"] for e in rows]))),
            "sim3_ate_m": float(np.mean([e["metrics"]["sim3_ate_rmse"] for e in rows])),
            "unaligned_ate_m": float(np.mean([e["metrics"]["ate_rmse"] for e in rows])), "episodes": rows}


def eligible(metrics, reference, rules):
    tol = rules["numeric_absolute_tolerance"]
    wins = sum(a["metrics"]["rotation_rmse"] < b["metrics"]["rotation_rmse"] - tol
               for a, b in zip(metrics["episodes"], reference["episodes"]))
    checks = {"rotation": metrics["rotation_deg"] <= reference["rotation_deg"] * (1-rules["min_rotation_improvement_percent"]/100) + np.degrees(tol),
              "aligned_position": metrics["sim3_ate_m"] <= reference["sim3_ate_m"] * rules["max_aligned_ate_ratio"] + tol,
              "unaligned_position": metrics["unaligned_ate_m"] <= reference["unaligned_ate_m"] * rules["max_unaligned_position_ratio"] + tol,
              "rotation_wins": wins >= rules["min_rotation_episode_wins"]}
    return {"eligible": all(checks.values()), "checks": checks, "rotation_wins": wins}


def infer(net, data, batch=64, images=None):
    images = data["images"] if images is None else images
    values = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(images), batch):
            values.append(net.correction(images[start:start+batch], data["normalized_state"][start:start+batch],
                                         data["prior"][start:start+batch].float()).numpy())
    return np.concatenate(values)


def train(variant, out):
    if out.exists():
        raise ValueError("preserve earlier training")
    protocol = json.loads(PROTOCOL.read_text())
    settings = protocol["training"]
    random.seed(settings["seed"]); np.random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"])
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    base_dir = ROOT / protocol["base_model"]
    meta = json.loads((base_dir / "meta.json").read_text())
    if digest(base_dir / "net.onnx") != protocol["base_onnx_sha256"]:
        raise ValueError("base identity differs")
    base = MotionNet()
    base.load_state_dict(torch.load(base_dir / "best.pt", weights_only=True))
    net = CorrelationMotionNet(base, meta["target_mean"], meta["target_std"], variant)
    initial = {key: value.clone() for key, value in net.state_dict().items()}
    initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial.values())).hexdigest()
    training, validation = load("train", meta), load("validation", meta)
    with torch.no_grad():
        translations = []
        for start in range(0, len(validation["images"]), 64):
            raw = base(validation["images"][start:start+64], validation["normalized_state"][start:start+64])
            translations.append((raw * torch.tensor(meta["target_std"]) + torch.tensor(meta["target_mean"]))[:, :3].numpy())
        translations = np.concatenate(translations)
    reference = score(validation, translations, np.zeros((len(translations), 3)))
    out.mkdir(parents=True)
    source_paths = [Path(__file__), PROTOCOL, ROOT / "training/feature_correlation.py", ROOT / "training/motion_model.py"]
    record = {"status": "running", "variant": variant, "started_utc": datetime.now(timezone.utc).isoformat(),
              "settings": settings, "initial_weights_sha256": initial_hash,
              "protocol_sha256": digest(PROTOCOL), "cache_record_sha256": digest(CACHE / "cache.json"),
              "base_checkpoint_sha256": digest(base_dir / "best.pt"), "base_onnx_sha256": meta["model_sha256"],
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "parameters": sum(p.numel() for p in net.parameters()),
              "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
              "reference": reference, "history": [], "torch": torch.__version__, "opencv": cv2.__version__}
    write_json(out / "run.json", record)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"], eta_min=settings["final_lr"])
    generator = torch.Generator().manual_seed(settings["seed"] + 1000)
    best = math.inf
    start_time = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        net.train()
        order = torch.randperm(len(training["images"]), generator=generator)
        total = 0.
        for start in range(0, len(order), settings["batch"]):
            ids = order[start:start+settings["batch"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = net.correction(training["images"][ids], training["normalized_state"][ids], training["prior"][ids].float())
            loss = F.smooth_l1_loss(prediction / .01, training["residual"][ids].float() / .01)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        scheduler.step()
        corrections = infer(net, validation)
        metrics = score(validation, translations, corrections)
        chosen = metrics["rotation_deg"] < best
        if chosen:
            best = metrics["rotation_deg"]
            torch.save(net.state_dict(), out / "best.pt")
            record["best_epoch"], record["best_validation"] = epoch, metrics
        record["history"].append({"epoch": epoch, "train_loss": total/len(order), "validation": metrics,
                                   "selected": chosen, "acceptance": eligible(metrics, reference, protocol["acceptance"]),
                                   "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        write_json(out / "run.json", record)
        print(f"{variant} epoch {epoch:02d} rotation={metrics['rotation_deg']:.5f} ATE={metrics['sim3_ate_m']:.6f} best={record['best_epoch']}", flush=True)
    torch.save(net.state_dict(), out / "last.pt")
    net.load_state_dict(torch.load(out / "best.pt", weights_only=True)); net.eval()
    for name, value in initial.items():
        if name.startswith("base.") and not torch.equal(net.state_dict()[name], value):
            raise ValueError("frozen translation weights changed")
    path = out / "net.onnx"
    torch.onnx.export(net, (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14), torch.zeros(1, 3)),
                      str(path), input_names=["images", "state", "rotation_prior"], output_names=["motion"],
                      opset_version=13, dynamo=False)
    cvnet = cv2.dnn.readNetFromONNX(str(path))
    maximum = translation_diff = 0.
    with torch.no_grad():
        for index in range(len(validation["images"])):
            inputs = (validation["images"][index:index+1], validation["normalized_state"][index:index+1], validation["prior"][index:index+1].float())
            expected = net(*inputs).numpy()
            for name, value in zip(("images", "state", "rotation_prior"), inputs): cvnet.setInput(value.numpy(), name)
            actual = cvnet.forward()
            maximum = max(maximum, float(np.max(np.abs(actual-expected))))
            original = base(inputs[0], inputs[1]).numpy()[:, :3]
            translation_diff = max(translation_diff, float(np.max(np.abs(original-actual[:, :3]))))
    if maximum > 1e-4 or translation_diff > 1e-4:
        raise ValueError("ONNX parity or frozen translation failed")
    shuffled = validation["images"].clone()
    order = torch.randperm(len(shuffled), generator=torch.Generator().manual_seed(314162))
    shuffled[:, 0] = shuffled[order, 0]
    real_prediction, shuffled_prediction = infer(net, validation), infer(net, validation, images=shuffled)
    probe = {"kind": "previous-only permutation, fixed current image/state/causal prior/translation",
             "seed": 314162, "ordinary": score(validation, translations, real_prediction),
             "shuffled_previous": score(validation, translations, shuffled_prediction),
             "prediction_change_rms_rad": float(np.sqrt(np.mean(np.sum((real_prediction-shuffled_prediction)**2, axis=1)))),
             "interpretation": "OOD diagnostic; degradation alone does not prove correspondence accuracy"}
    output_meta = copy.deepcopy(meta)
    output_meta.update(model_sha256=digest(path), rotation_prior_input=True, rotation_residual_enabled=True,
                       gyro_bias={"enabled": True}, map={"enabled": False}, architecture="local-correlation-v5",
                       correlation_variant=variant, selected_epoch=record["best_epoch"],
                       checkpoint_role="best trained validation rotation candidate; eligibility is separate")
    write_json(out / "meta.json", output_meta)
    write_json(out / "visual-probe.json", probe)
    record.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                  elapsed_seconds=time.perf_counter()-start_time, best_checkpoint_sha256=digest(out / "best.pt"),
                  model_sha256=digest(path), frozen_base_exact=True, onnx_max_abs_difference=maximum,
                  onnx_translation_max_abs_difference=translation_diff,
                  acceptance=eligible(record["best_validation"], reference, protocol["acceptance"]),
                  optimizer_steps=settings["epochs"]*math.ceil(len(training["images"])/settings["batch"]))
    for name, expected in record["source_sha256"].items():
        if digest(ROOT / name) != expected: raise ValueError("source changed during fit: " + name)
    write_json(out / "run.json", record)
    print(json.dumps({k: record[k] for k in ("status", "best_epoch", "parameters", "trainable_parameters", "acceptance", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--variant", choices=("temporal", "single-frame"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.variant and args.out: train(args.variant, args.out)
    else: parser.error("use --prepare or --variant with --out")
