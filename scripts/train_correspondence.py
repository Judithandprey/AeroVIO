"""Prepare audited geometric labels and run the fixed v6 matched fine-tunes."""
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "workspace/solution")]
from train_feature_correlation import digest, write_json, load, score, infer, eligible
from training.motion_model import MotionNet
from training.feature_correlation import CorrelationMotionNet
from training.correspondence_supervision import correspondence_targets, correspondence_loss, matching_displacement, volume

PROTOCOL = ROOT / "configs/correspondence-v6.json"


def source_hashes():
    paths = [Path(__file__), PROTOCOL, ROOT / "training/correspondence_supervision.py",
             ROOT / "training/feature_correlation.py", ROOT / "training/motion_model.py",
             ROOT / "scripts/train_feature_correlation.py", ROOT / "training/rotation_focus.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def prepare():
    protocol = json.loads(PROTOCOL.read_text())
    destination = ROOT / protocol["correspondence_cache"]
    if destination.exists(): raise ValueError("preserve existing label cache")
    original = ROOT / protocol["original_cache"]
    record = json.loads((original / "cache.json").read_text())
    manifest = json.loads((ROOT / protocol["manifest"]).read_text())
    scenes = [{e["scene"] for e in manifest[s]} for s in ("train", "validation")]
    if scenes[0] & scenes[1] or "ProcTHOR-Train-1" in scenes[0] | scenes[1]: raise ValueError("scene leakage")
    if digest(ROOT / protocol["manifest"]) != record["source_manifest_sha256"]: raise ValueError("manifest differs")
    destination.mkdir(parents=True)
    torch.set_num_threads(4)
    result = {"protocol_sha256": digest(PROTOCOL), "source_sha256": source_hashes(), "splits": {}}
    for split in ("train", "validation"):
        path = original / (split + ".npz")
        if digest(path) != record["splits"][split]["sha256"]: raise ValueError("original cache identity differs")
        with np.load(path) as data:
            inputs = {key: torch.from_numpy(data[key].copy()) for key in ("depth", "intrinsics", "target", "state", "visibility")}
        pieces = {key: [] for key in ("soft_target", "valid", "displacement")}
        for start in range(0, len(inputs["depth"]), 64):
            batch = {k: v[start:start+64] for k, v in inputs.items()}
            targets = correspondence_targets(batch["depth"], batch["intrinsics"], batch["target"],
                                              batch["state"][:, :3], batch["visibility"])
            for key, value in targets.items(): pieces[key].append(value.numpy())
        arrays = {k: np.concatenate(v) for k, v in pieces.items()}
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError("nonfinite label")
        if not np.allclose(arrays["soft_target"].sum(1), arrays["valid"], atol=1e-6): raise ValueError("target mass differs")
        length = np.linalg.norm(arrays["displacement"], axis=1)*4
        values = length[arrays["valid"]]
        output = destination / (split + ".npz")
        np.savez_compressed(output, **arrays)
        result["splits"][split] = {"sha256": digest(output), "original_sha256": digest(path),
            "pairs": len(length), "valid_points": int(arrays["valid"].sum()),
            "valid_fraction_all_grid_points": float(arrays["valid"].mean()),
            "pairs_with_no_valid_points": int((arrays["valid"].sum((1, 2)) == 0).sum()),
            "displacement_image_pixels_mean": float(values.mean()),
            "displacement_image_pixels_quantiles": np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist(),
            "moving_fraction_valid_points": float((values >= .25).mean()),
            "nearest_centre_fraction_valid_points": float(((np.abs(arrays["displacement"]) < .5).all(1)[arrays["valid"]]).mean()),
            "scenes": sorted(scenes[0 if split == "train" else 1])}
        print(json.dumps({split: result["splits"][split]}), flush=True)
    result["status"] = "complete"
    write_json(destination / "cache.json", result)


def load_labels(split):
    protocol = json.loads(PROTOCOL.read_text())
    directory = ROOT / protocol["correspondence_cache"]
    record = json.loads((directory / "cache.json").read_text())
    if record["status"] != "complete" or record["protocol_sha256"] != digest(PROTOCOL): raise ValueError("invalid labels")
    path = directory / (split + ".npz")
    if digest(path) != record["splits"][split]["sha256"]: raise ValueError("labels changed")
    with np.load(path) as data: return {k: torch.from_numpy(data[k].copy()) for k in data.files}


def matching_metrics(net, data, labels, images=None):
    images = data["images"] if images is None else images
    endpoints, entropies, cross_entropies = [], [], []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            correlation = volume(net, images[start:start+64])
            probabilities = F.softmax(correlation/.1, dim=1)
            predicted = matching_displacement(correlation)
            endpoints.append((torch.linalg.vector_norm(predicted-labels["displacement"][start:start+64], dim=1)*4).numpy())
            entropies.append((-(probabilities*F.log_softmax(correlation/.1, 1)).sum(1)).numpy())
            cross_entropies.append((-(labels["soft_target"][start:start+64]*F.log_softmax(correlation/.1, 1)).sum(1)).numpy())
    error, entropy, cross_entropy = map(np.concatenate, (endpoints, entropies, cross_entropies))
    zero_error = np.linalg.norm(labels["displacement"].numpy(), axis=1)*4
    valid = labels["valid"].numpy()
    ids = data["episode_id"].numpy()
    output = {}
    for name, mask in (("all", valid), ("moving", valid & (zero_error >= .25))):
        episodes = []
        for episode_id in np.unique(ids):
            select = mask & (ids[:, None, None] == episode_id)
            count = int(select.sum())
            if not count: continue
            episodes.append({"episode_id": int(episode_id), "valid_points": count,
                "epe_pixels": float(error[select].mean()), "zero_epe_pixels": float(zero_error[select].mean()),
                "cross_entropy": float(cross_entropy[select].mean()), "entropy": float(entropy[select].mean())})
        if not episodes: raise ValueError("no valid matching evaluation points")
        output[name] = {"mean_episode_epe_pixels": float(np.mean([e["epe_pixels"] for e in episodes])),
                        "mean_episode_zero_epe_pixels": float(np.mean([e["zero_epe_pixels"] for e in episodes])),
                        "episodes": episodes}
    return output


def construct(directory, variant="temporal"):
    meta = json.loads((directory / "meta.json").read_text())
    net = CorrelationMotionNet(MotionNet(), meta["target_mean"], meta["target_std"], variant)
    net.load_state_dict(torch.load(directory / "best.pt", weights_only=True, map_location="cpu"))
    return net.eval(), meta


def translations(net, data, meta):
    values = []
    with torch.no_grad():
        for start in range(0, len(data["images"]), 64):
            raw = net.base(data["images"][start:start+64], data["normalized_state"][start:start+64])
            values.append((raw*torch.tensor(meta["target_std"])+torch.tensor(meta["target_mean"]))[:, :3].numpy())
    return np.concatenate(values)


def verify_export(net, path, validation):
    cvnet = cv2.dnn.readNetFromONNX(str(path))
    max_error = translation_error = 0.
    with torch.no_grad():
        for index in range(len(validation["images"])):
            inputs = (validation["images"][index:index+1], validation["normalized_state"][index:index+1], validation["prior"][index:index+1].float())
            expected = net(*inputs).numpy()
            for name, value in zip(("images", "state", "rotation_prior"), inputs): cvnet.setInput(value.numpy(), name)
            actual = cvnet.forward()
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            translation_error = max(translation_error, float(np.max(np.abs(actual[:, :3]-net.base(inputs[0], inputs[1]).numpy()[:, :3]))))
    if max_error > 1e-4 or translation_error > 1e-4: raise ValueError("ONNX parity failed")
    return {"all_validation_samples": len(validation["images"]), "max_abs_difference": max_error,
            "translation_max_abs_difference": translation_error}


def train(variant, out):
    if out.exists(): raise ValueError("preserve earlier model")
    protocol = json.loads(PROTOCOL.read_text()); settings = protocol["training"]
    random.seed(settings["seed"]); np.random.seed(settings["seed"]); torch.manual_seed(settings["seed"])
    torch.set_num_threads(settings["threads"]); torch.use_deterministic_algorithms(True); cv2.setNumThreads(1)
    initial_directory = ROOT / protocol["initial_model"]
    if digest(initial_directory / "net.onnx") != protocol["initial_onnx_sha256"]: raise ValueError("starting identity differs")
    net, meta = construct(initial_directory)
    training, validation = load("train", meta), load("validation", meta)
    labels_train, labels_validation = load_labels("train"), load_labels("validation")
    initial_weights = {k: v.clone() for k, v in net.state_dict().items()}
    initial_hash = hashlib.sha256(b"".join(v.numpy().tobytes() for v in initial_weights.values())).hexdigest()
    local_translation = translations(net, validation, meta)
    reference = score(validation, local_translation, np.zeros((len(local_translation), 3)))
    start_pose = score(validation, local_translation, infer(net, validation))
    start_matching = matching_metrics(net, validation, labels_validation)
    single, _ = construct(ROOT / "models/correlation-single-frame-v5-complete", "single-frame")
    single_matching = matching_metrics(single, validation, labels_validation)
    out.mkdir(parents=True)
    run = {"status": "running", "variant": variant, "started_utc": datetime.now(timezone.utc).isoformat(),
           "protocol_sha256": digest(PROTOCOL), "source_sha256": source_hashes(), "settings": settings,
           "initial_checkpoint_sha256": digest(initial_directory / "best.pt"), "initial_weights_sha256": initial_hash,
           "initial_model_sha256": digest(initial_directory / "net.onnx"),
           "label_cache_record_sha256": digest(ROOT / protocol["correspondence_cache"] / "cache.json"),
           "initial_onnx_parity": verify_export(net, initial_directory / "net.onnx", validation),
           "parameters": sum(p.numel() for p in net.parameters()),
           "trainable_parameters": sum(p.numel() for p in net.parameters() if p.requires_grad),
           "initial_pose": start_pose, "initial_matching": start_matching, "v5_single_frame_matching": single_matching,
           "reference_calibrated_gyro": reference, "history": []}
    write_json(out / "run.json", run)
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=settings["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings["epochs"], eta_min=settings["final_lr"])
    generator = torch.Generator().manual_seed(settings["seed"]+1000)
    best = math.inf; start_time = time.perf_counter()
    for epoch in range(1, settings["epochs"]+1):
        net.train(); order = torch.randperm(len(training["images"]), generator=generator)
        rotation_total = matching_total = 0.
        for start in range(0, len(order), settings["batch"]):
            indices = order[start:start+settings["batch"]]
            images = training["images"][indices]
            optimizer.zero_grad(set_to_none=True)
            correction = net.correction(images, training["normalized_state"][indices], training["prior"][indices].float())
            rotation_loss = F.smooth_l1_loss(correction/.01, training["residual"][indices].float()/.01)
            if variant == "supervised":
                match_loss = correspondence_loss(volume(net, images), labels_train["soft_target"][indices],
                    labels_train["valid"][indices], settings["temperature"])
                loss = rotation_loss + settings["matching_weight"]*match_loss
            else:
                match_loss = correction.new_zeros(()); loss = rotation_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], settings["gradient_clip"])
            optimizer.step()
            rotation_total += float(rotation_loss.detach())*len(indices)
            matching_total += float(match_loss.detach())*len(indices)
        scheduler.step(); net.eval()
        metrics = score(validation, local_translation, infer(net, validation))
        match = matching_metrics(net, validation, labels_validation)
        selected = metrics["rotation_deg"] < best
        if selected:
            best = metrics["rotation_deg"]; torch.save(net.state_dict(), out / "best.pt")
            run.update(best_epoch=epoch, best_validation=metrics, best_matching=match)
        run["history"].append({"epoch": epoch, "rotation_loss": rotation_total/len(order),
            "matching_loss": matching_total/len(order) if variant == "supervised" else None,
            "validation": metrics, "matching": match, "selected": selected,
            "order_sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
        write_json(out / "run.json", run)
        print(f"{variant} {epoch:02d} rotation={metrics['rotation_deg']:.5f} EPE={match['all']['mean_episode_epe_pixels']:.5f} best={run['best_epoch']}", flush=True)
    torch.save(net.state_dict(), out / "last.pt")
    net.load_state_dict(torch.load(out / "best.pt", weights_only=True)); net.eval()
    if any(not torch.equal(net.state_dict()[k], value) for k, value in initial_weights.items() if k.startswith("base.")):
        raise ValueError("translation network changed")
    path = out / "net.onnx"
    torch.onnx.export(net, (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14), torch.zeros(1, 3)), str(path),
        input_names=["images", "state", "rotation_prior"], output_names=["motion"], opset_version=13, dynamo=False)
    parity = verify_export(net, path, validation)
    output_meta = copy.deepcopy(meta)
    output_meta.update(model_sha256=digest(path), training_stage="correspondence-v6-"+variant,
        source_model_sha256=protocol["initial_onnx_sha256"], selected_epoch=run["best_epoch"],
        checkpoint_role="validation-selected trained diagnostic; acceptance separate", correspondence_variant=variant)
    write_json(out / "meta.json", output_meta)
    shuffled = validation["images"].clone()
    order = torch.randperm(len(shuffled), generator=torch.Generator().manual_seed(314162))
    shuffled[:, 0] = shuffled[order, 0]
    predicted, changed = infer(net, validation), infer(net, validation, images=shuffled)
    write_json(out / "visual-probe.json", {"type": "previous-only shuffle; fixed current/state/prior/labels/translation",
        "seed": 314162, "ordinary_pose": run["best_validation"], "ordinary_matching": run["best_matching"],
        "shuffled_pose": score(validation, local_translation, changed),
        "shuffled_matching": matching_metrics(net, validation, labels_validation, images=shuffled),
        "correction_change_rms_rad": float(np.sqrt(np.mean(np.sum((predicted-changed)**2, axis=1)))),
        "interpretation": "OOD sensitivity diagnostic, not causal proof of useful matching"})
    run.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter()-start_time,
        best_checkpoint_sha256=digest(out / "best.pt"), model_sha256=digest(path), onnx_parity=parity,
        frozen_base_exact=True, optimizer_steps=settings["epochs"]*math.ceil(len(training["images"])/settings["batch"]),
        pose_eligibility=eligible(run["best_validation"], reference, protocol["acceptance"]))
    if source_hashes() != run["source_sha256"]: raise ValueError("source changed during fit")
    write_json(out / "run.json", run)
    print(json.dumps({k: run[k] for k in ("status", "best_epoch", "elapsed_seconds", "pose_eligibility")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--variant", choices=("continuation", "supervised"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prepare: prepare()
    elif args.variant and args.out: train(args.variant, args.out)
    else: parser.error("use --prepare or --variant plus --out")
