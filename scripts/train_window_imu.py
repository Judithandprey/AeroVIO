"""Train a windowed IMU motion network (stage 2) and export it as a C++ header.

The deployed compact network sees only the IMU samples between two frames, so it cannot know how fast the
drone already moves and regresses to the training set's mean speed. This network sees the last WINDOW frames
of preintegrated IMU (velocity change and rotation per frame), rotated into the previous camera frame with the
gyro chain, and predicts the camera translation of the current frame in that frame (the same target and frame
convention as the compact network, so the native solution can swap it in).

Episodes are standard viobench archives; images are not needed (stripped archives work).

    python scripts/train_window_imu.py --train data/s2-imu --val data/s2-val --out experiments/window-imu-v1
"""
import argparse
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "workspace/solution")]
from preintegrate import preintegrate, exp_so3  # noqa: E402
from viobench.tools.datasets.storage import read_episode  # noqa: E402
from viobench.tools.geometry import poses_to_matrices, invert  # noqa: E402

FEATURES = 6  # per frame: velocity change (3), rotation vector (3)


def episode_arrays(path, window):
    episode = read_episode(path)
    T_cam_imu = np.asarray(episode.config["imu"]["T_cam_imu"], dtype=np.float64).reshape(4, 4)
    poses = poses_to_matrices(episode.ground_truth_relative())
    n = episode.num_frames
    per_frame = np.zeros((n, FEATURES))  # in camera k-1 axes
    world = np.zeros((n, FEATURES))      # in camera 0 axes (gyro chain)
    chain = [np.eye(3)]                  # chain[k] = camera k in camera 0 (gyro only)
    targets = np.zeros((n, 3))
    dts = np.zeros(n)
    for k in range(1, n):
        t0, t1 = episode.frame_times[k - 1:k + 1]
        f = preintegrate(episode.imu_window(k), T_cam_imu, t0, t1).astype(np.float64)
        rotvec, dv = f[:3], f[3:6]
        per_frame[k] = np.concatenate([dv, rotvec])
        R_prev = chain[-1]
        world[k] = np.concatenate([R_prev @ dv, R_prev @ rotvec])
        chain.append(R_prev @ exp_so3(rotvec))
        targets[k] = (invert(poses[k - 1]) @ poses[k])[:3, 3]
        dts[k] = t1 - t0
    inputs = np.zeros((n - 1, window * FEATURES + 1), dtype=np.float32)
    for k in range(1, n):
        ref = chain[k - 1].T  # camera k-1 axes
        lo = max(1, k - window + 1)
        block = world[lo:k + 1]
        rotated = np.concatenate([block[:, :3] @ ref.T, block[:, 3:] @ ref.T], axis=1)
        flat = np.zeros((window, FEATURES))
        flat[window - len(rotated):] = rotated
        inputs[k - 1, :-1] = flat.reshape(-1)
        inputs[k - 1, -1] = dts[k]
    gt_rot = np.stack([poses[k - 1][:3, :3] for k in range(1, n)]).astype(np.float32)  # camera k-1 in camera 0
    return inputs, targets[1:].astype(np.float32), gt_rot


def load(paths, window):
    xs, ys, ids, rs = [], [], [], []
    for i, path in enumerate(paths):
        x, y, r = episode_arrays(path, window)
        xs.append(x); ys.append(y); ids.append(np.full(len(y), i)); rs.append(r)
        print(f"  {Path(path).name}: {len(y)} frames, path {np.linalg.norm(y, axis=1).sum():.1f} m", flush=True)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids), np.concatenate(rs)


class WindowNet(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def chain_error(net, x, y, ids, rot, x_mean, x_std, y_mean, y_std):
    """Unaligned position RMSE per episode (mean over episodes) when chaining predicted increments with
    ground-truth rotation, plus the same for a constant mean-increment predictor and the step RMSE."""
    with torch.no_grad():
        pred = net((x - x_mean) / x_std) * y_std + y_mean
    def ate(increments):
        out = []
        for i in torch.unique(ids):
            m = ids == i
            world_est = torch.einsum("nij,nj->ni", rot[m], increments[m]).cumsum(0)
            world_gt = torch.einsum("nij,nj->ni", rot[m], y[m]).cumsum(0)
            out.append(float((world_est - world_gt).norm(dim=1).pow(2).mean().sqrt()))
        return float(np.mean(out))
    return ate(pred), ate(y_mean.expand_as(y)), float((pred - y).pow(2).sum(1).mean().sqrt())


def export_header(net, path, meta):
    shape = {k: meta[k] for k in ("window", "inputs", "hidden")}
    lines = ["#pragma once", "#include <array>", f"// window IMU network: {json.dumps(shape)}",
             f"constexpr int window_frames = {meta['window']};", f"constexpr int window_inputs = {meta['inputs']};",
             f"constexpr int window_hidden = {meta['hidden']};"]

    def arr(name, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        body = ", ".join(float(v).hex() + "f" for v in values)
        lines.append(f"constexpr std::array<float, {len(values)}> {name} = {{{body}}};")

    arr("window_x_mean", meta["x_mean"]); arr("window_x_std", meta["x_std"])
    arr("window_y_mean", meta["y_mean"]); arr("window_y_std", meta["y_std"])
    layers = [m for m in net.net if isinstance(m, nn.Linear)]
    for i, layer in enumerate(layers, 1):
        arr(f"window_w{i}", layer.weight.detach().cpu().numpy())  # row-major [out][in]
        arr(f"window_b{i}", layer.bias.detach().cpu().numpy())
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", required=True, help="directories or archives of training episodes")
    parser.add_argument("--val", nargs="+", required=True, help="directories or archives of validation episodes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    def expand(items):
        out = []
        for item in items:
            p = Path(item)
            out += sorted(glob.glob(str(p / "*.zip"))) if p.is_dir() else [str(p)]
        return out

    train_paths, val_paths = expand(args.train), expand(args.val)
    print(f"train {len(train_paths)} episodes"); xt, yt, it, _ = load(train_paths, args.window)
    print(f"val {len(val_paths)} episodes"); xv, yv, iv, rv = load(val_paths, args.window)
    x_mean, x_std = xt.mean(0), xt.std(0) + 1e-6
    y_mean, y_std = yt.mean(0), yt.std(0) + 1e-6
    T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    xt_t, yt_t, xv_t, yv_t, iv_t, rv_t = T(xt), T(yt), T(xv), T(yv), torch.from_numpy(iv), T(rv)
    xm, xs, ym, ys = T(x_mean), T(x_std), T(y_mean), T(y_std)
    xn, yn = (xt_t - xm) / xs, (yt_t - ym) / ys
    net = WindowNet(xt.shape[1], args.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state, log = float("inf"), None, []
    for epoch in range(args.epochs):
        started = time.time(); net.train()
        order = torch.randperm(len(xn)); total = 0.
        for s in range(0, len(order), args.batch):
            idx = order[s:s + args.batch]
            loss = (net(xn[idx]) - yn[idx]).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss.detach()) * len(idx)
        sched.step(); net.eval()
        final, constant, step_rmse = chain_error(net, xv_t, yv_t, iv_t, rv_t, xm, xs, ym, ys)
        row = {"epoch": epoch + 1, "train_mse": round(total / len(xn), 4), "val_step_rmse_m": round(step_rmse, 5),
               "val_ate_gt_rotation_m": round(final, 3), "val_ate_constant_speed_m": round(constant, 3),
               "seconds": round(time.time() - started, 1)}
        log.append(row); print(row, flush=True)
        if final < best:
            best, best_state = final, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"window": args.window, "inputs": int(xt.shape[1]), "hidden": args.hidden,
            "x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist()}
    torch.save(net.state_dict(), args.out / "net.pt")
    (args.out / "meta.json").write_text(json.dumps({k: v for k, v in meta.items() if not k.startswith(("x_", "y_"))}
                                                   | {"train": train_paths, "val": val_paths, "log": log,
                                                      "best_val_ate_gt_rotation_m": best}, indent=1))
    export_header(net, args.out / "window_weights.h",
                  {k: meta[k] for k in ("window", "inputs", "hidden")} | {"x_mean": meta["x_mean"],
                   "x_std": meta["x_std"], "y_mean": meta["y_mean"], "y_std": meta["y_std"]})
    print(f"best val ATE (ground-truth rotation) {best:.3f} m -> {args.out}")


if __name__ == "__main__":
    main()
