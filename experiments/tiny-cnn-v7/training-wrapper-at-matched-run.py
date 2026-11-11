"""Reproduce untouched upstream training and a clearly separate matched-budget fit.

Both use the old training flights, never the public scoring set. Upstream guest
files are copied byte-for-byte, including its original preintegration behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/viobench/viobench/solution/examples/python/tiny_cnn"
PROTOCOL = ROOT / "configs/tiny-cnn-v7.json"


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data): Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


class GuestResolutionEpisode:
    """Use the same 320x240 guest adaptation as AeroVIO's original data cache."""
    def __init__(self, episode): self.episode = episode
    def __getattr__(self, name): return getattr(self.episode, name)
    def adapt(self): return self.episode.adapt((320, 240))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("upstream", "matched"), required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    settings = protocol[args.variant]
    out = ROOT / ("models/tiny-cnn-v7-"+args.variant)
    if out.exists(): raise ValueError("preserve previous fit")
    manifest_path = ROOT / protocol["data_manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    torch.set_num_threads(protocol["threads"])
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location("upstream_tiny_training", SOURCE / "train.py")
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    assert Path(sys.modules["preintegrate"].__file__).resolve() == SOURCE / "solution/preintegrate.py"
    source_hashes = {str(p.relative_to(SOURCE)): digest(p) for p in [SOURCE / "train.py", *sorted((SOURCE / "solution").glob("*.py"))]}
    data_hashes = {}
    for split in ("train", "validation"):
        for entry in manifest[split]:
            archive = ROOT / (entry["path"]+".zip")
            if digest(archive) != entry["archive_sha256"]: raise ValueError("training data changed")
            data_hashes[str(archive.relative_to(ROOT))] = entry["archive_sha256"]
    started = time.perf_counter()
    record = {"variant": args.variant, "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": digest(PROTOCOL), "manifest_sha256": digest(manifest_path),
              "source_sha256": source_hashes, "wrapper_sha256": digest(__file__), "data_sha256": data_hashes,
              "server_weights_reproduced": False, "settings": settings, "threads": protocol["threads"],
              "parameters": sum(p.numel() for p in upstream.TinyNet().parameters()),
              "public_data_used_for_training_or_selection": False}
    if args.variant == "upstream":
        # Run the original main verbatim. Only choose output/data and cap host threads.
        sys.argv = [str(SOURCE / "train.py"), *[str(ROOT / e["path"]) for e in manifest["train"]], "--out", str(out)]
        upstream.main()
        record["command_argv"] = sys.argv
        record["selected_epoch"] = 10
    else:
        arrays = {}
        for split in ("train", "validation"):
            rows = [upstream.samples_from_episode(GuestResolutionEpisode(upstream.read_episode(ROOT / e["path"]))) for e in manifest[split]]
            arrays[split] = [np.concatenate([row[k] for row in rows]) for k in range(3)]
            # Exact image parity with the prior training cache; IMU/targets intentionally
            # retain the upstream algorithm and therefore are not assumed identical.
            with np.load(ROOT / f"data/motion-cache-v2/{split}.npz") as old:
                if not np.array_equal(arrays[split][0], old["images"]): raise ValueError("image pairs differ")
            print(split, len(arrays[split][0]), "image pairs match original cache", flush=True)
        out.mkdir(parents=True)
        record["training_frame_adaptation"] = "320x240 RGB before upstream gray/64x48; matches AeroVIO cache exactly, unlike untouched upstream native 640x480 training"
        images, imus, targets = arrays["train"]
        mean, std, target_std = imus.mean(0), imus.std(0)+1e-6, targets.std(0)+1e-6
        meta = {"input_width": 64, "input_height": 48, "imu_mean": mean.tolist(), "imu_std": std.tolist(), "target_std": target_std.tolist()}
        tensors = {s: (torch.from_numpy(v[0]), torch.from_numpy((v[1]-mean)/std), torch.from_numpy(v[2]/target_std)) for s, v in arrays.items()}
        torch.manual_seed(settings["seed"])
        net = upstream.TinyNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=settings["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"], eta_min=settings["lr"]*.1)
        history = []; order_hash = hashlib.sha256(); best = float("inf")
        x, m, y = tensors["train"]; vx, vm, vy = tensors["validation"]
        for epoch in range(settings["epochs"]):
            order = torch.randperm(len(y), generator=torch.Generator().manual_seed(settings["seed"]+epoch+1000))
            order_hash.update(order.numpy().tobytes())
            net.train(); total = 0.
            for ids in order.split(settings["batch"]):
                loss = F.smooth_l1_loss(net(x[ids], m[ids]), y[ids])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), settings["gradient_norm_cap"])
                optimizer.step(); total += float(loss.detach())*len(ids)
            scheduler.step(); net.eval()
            with torch.no_grad():
                validation = sum(float(F.smooth_l1_loss(net(vx[i:i+64], vm[i:i+64]), vy[i:i+64]))*len(vy[i:i+64]) for i in range(0, len(vy), 64))/len(vy)
            row = {"epoch": epoch+1, "train_loss": total/len(y), "validation_pose_loss": validation}
            history.append(row); print(json.dumps(row), flush=True)
            if validation < best:
                best = validation; record["selected_epoch"] = epoch+1
                torch.save(net.state_dict(), out / "net.pt")
            write(out / "history.json", history)
        net.load_state_dict(torch.load(out / "net.pt", weights_only=True)); net.eval()
        torch.onnx.export(net, (x[:1], m[:1]), str(out / "net.onnx"), input_names=["images", "imu"], output_names=["motion"], opset_version=13, dynamo=False)
        np.savez(out / "weights.npz", **{k: v.detach().numpy() for k, v in net.state_dict().items()})
        write(out / "meta.json", meta)
        record.update({"train_pairs": len(y), "validation_pairs": len(vy), "minibatch_order_sha256": order_hash.hexdigest(), "best_validation_pose_loss": best, "image_pairs_exactly_match_v1_cache": True})
        old = json.loads((ROOT / "experiments/motion-ablation-v2/imu-only/run.json").read_text())
        if record["minibatch_order_sha256"] != old["minibatch_order_sha256"]: raise ValueError("training order differs")
        record["order_hash_reference"] = "experiments/motion-ablation-v2/imu-only/run.json; same seed+epoch+1000 ordering used by v1"
    # Check the actual ONNX against PyTorch on all original validation pairs.
    net = upstream.TinyNet(); net.load_state_dict(torch.load(out / "net.pt", weights_only=True)); net.eval()
    meta = json.loads((out / "meta.json").read_text())
    cv_net = cv2.dnn.readNetFromONNX(str(out / "net.onnx"))
    maximum = 0.; count = 0
    for entry in manifest["validation"]:
        episode = upstream.read_episode(ROOT / entry["path"])
        if args.variant == "matched": episode = GuestResolutionEpisode(episode)
        ims, imu, _ = upstream.samples_from_episode(episode)
        imu = ((imu-np.asarray(meta["imu_mean"], dtype=np.float32))/np.asarray(meta["imu_std"], dtype=np.float32)).astype(np.float32)
        for i in range(len(ims)):
            with torch.no_grad(): expected = net(torch.from_numpy(ims[i:i+1]), torch.from_numpy(imu[i:i+1])).numpy()
            cv_net.setInput(ims[i:i+1], "images"); cv_net.setInput(imu[i:i+1], "imu")
            maximum = max(maximum, float(np.abs(cv_net.forward()-expected).max())); count += 1
    if maximum > 1e-4: raise ValueError("ONNX export mismatch")
    record["export_check"] = {"validation_pairs": count, "max_normalized_output_error": maximum, "tolerance": 1e-4}
    stage = ROOT / ("artifacts/tiny-cnn-v7-"+args.variant+"-solution")
    stage.mkdir(parents=True)
    for p in sorted((SOURCE / "solution").glob("*.py")): shutil.copyfile(p, stage / p.name)
    for name in ("net.onnx", "weights.npz", "meta.json"): shutil.copyfile(out / name, stage / name)
    record.update({"finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-started,
                   "bundle_files_sha256": {p.name: digest(p) for p in sorted(stage.iterdir())}, "model_sha256": digest(out / "net.onnx")})
    write(out / "run.json", record)
    print("FINISHED", json.dumps(record["export_check"]), flush=True)


if __name__ == "__main__": main()
