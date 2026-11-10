"""Generate independent HyperDrone RGB/IMU/pose/depth training episodes on CPU.

The public evaluation scene (ProcTHOR-Train-1) is explicitly excluded. Output
episodes use viobench's standard 640x480 format. A camera-z depth sidecar at
64x48 shares the RGB capture timestamps and uses point sampling, not averaging.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def world_header(output):
    """Same X500 IMU/autopilot as official generator, capture only at 20 Hz."""
    from hyperdrone.camera import camera_config_header, ImageObservation
    from hyperlux.camera import CameraSpec, RGB, Depth

    upstream = ROOT / "external/viobench/viobench/tools/datasets/generate/hyperdrone_world.h.in"
    text = upstream.read_text()
    for key, value in {"INSTANCES": "1", "WIDTH": "640", "HEIGHT": "480",
                       "FOV_RANGE": "15.0", "HOLD_STEPS": "200", "WAYPOINTS": "48"}.items():
        text = text.replace("@" + key + "@", value)
    rig = camera_config_header((CameraSpec("front", (RGB(format="float32"), Depth())),),
                               ImageObservation(cameras=("front",)))
    rig = rig.replace("namespace hyperlux_camera_config", "namespace aerovio_camera_config")
    text = text.replace("namespace hyperdrone_user {", rig + "\nnamespace hyperdrone_user {")
    text = text.replace("struct SPEC: hyperdrone::presets::X500FPVIMU<T, TI> {",
                        "struct SPEC: hyperdrone::presets::X500FPVIMU<T, TI> {\n"
                        "        using CAMERA_OUTPUTS = aerovio_camera_config::CAMERA_OUTPUTS;\n"
                        "        using OBSERVATION = aerovio_camera_config::OBSERVATION;")
    text = text.replace("static constexpr TI FRAME_STRIDE = 1;  // render at the IMU rate; frames are subsampled in Python",
                        "static constexpr TI FRAME_STRIDE = 10;  // training captures at 20 Hz; IMU remains 200 Hz")
    text = text.replace("rendering every IMU step", "rendering every tenth IMU step")
    path = output / "hyperdrone_training_world.h"
    if path.exists() and path.read_text() != text:
        raise ValueError("Existing generated world differs; choose a new output directory")
    path.write_text(text)
    return path


def capture(env, duration, seed, scene, name, output):
    import cv2
    import numpy as np
    from hyperlux.camera import Depth
    from viobench.tools.abi.config import make_config, pinhole_intrinsics
    from viobench.tools.geometry import matrix_to_quaternion
    from viobench.tools.datasets.episode import Episode
    from viobench.tools.datasets.generate.simulation import mount_to_body_camera
    from viobench.tools.datasets.storage import write_episode, read_episode

    started = time.perf_counter()
    env.reset()
    dt = float(env.dt)
    assert abs(dt - 0.005) < 1e-8, dt
    parameters = env.instance_parameters()[0]
    camera = parameters["cameras"].get("camera0") or parameters["cameras"].get("front")
    fov = float(camera["fov"])
    extrinsic = mount_to_body_camera(camera["mount"])
    state = env.observe_privileged()[0]
    gt = [np.concatenate(([0.0], state[:3].astype(np.float64),
                          matrix_to_quaternion(state[3:12].reshape(3, 3)).astype(np.float64)))]
    frames, depth_frames, frame_times, imu = [], [], [], []
    terminated = False
    # Match the nearest point to each training-resolution pixel centre. Pixel
    # coordinates and their subpixel discrepancy are recorded in the manifest.
    x_samples = np.floor((np.arange(64) + 0.5) * 10.0).astype(int)
    y_samples = np.floor((np.arange(48) + 0.5) * 10.0).astype(int)
    for step in range(1, int(round(duration / dt)) + 1):
        env.step()
        t = step * dt
        row = env.observe_imu()[0]
        imu.append(np.concatenate(([t], row[:6].astype(np.float64))))
        state = env.observe_privileged()[0]
        gt.append(np.concatenate(([t], state[:3].astype(np.float64),
                                  matrix_to_quaternion(state[3:12].reshape(3, 3)).astype(np.float64))))
        terminated = terminated or bool(env.terminated()[0])
        if step % 10 == 0:
            frame = np.asarray(env.frames())[0]
            frames.append(np.clip(np.rint(frame * 255), 0, 255).astype(np.uint8))
            depth = np.asarray(env.outputs[Depth()].read())[0]
            depth = np.squeeze(depth)
            assert depth.shape == (480, 640), depth.shape
            sampled = depth[np.ix_(y_samples, x_samples)].astype(np.float32)
            sampled[~np.isfinite(sampled) | (sampled <= 0) | (sampled > 100)] = 0
            depth_frames.append(sampled)
            frame_times.append(t)
    if terminated:
        return None, {"scene": scene, "seed": seed, "crashed": True, "elapsed_s": time.perf_counter() - started}
    intrinsics = pinhole_intrinsics(640, 480, fov)
    config = make_config(width=640, height=480, channels=3, frame_rate_hz=20.0,
                         num_frames=len(frames), hover_seconds=1.0,
                         T_body_camera=extrinsic, intrinsics=intrinsics, episode_name=name)
    config["camera"]["horizontal_fov_degrees"] = fov
    config["episode"]["exposure_seconds"] = 0.005
    episode = Episode(name=name, config=config, frame_times=np.asarray(frame_times),
                      frames=np.stack(frames), imu=np.asarray(imu), ground_truth=np.asarray(gt),
                      T_body_camera=extrinsic,
                      meta={"source": "hyperdrone", "scene": scene, "seed": seed,
                            "training_only": True, "render_backend": env.render_backend,
                            "frame_capture": "20 Hz at post-step states; timestamps in seconds",
                            "nominal_exposure_s": 0.005, "depth_kind": "camera_z_metres"})
    archive = output / (name + ".zip")
    if archive.exists() or (output / name).exists():
        raise FileExistsError(name)
    write_episode(episode, archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            target = (output / member).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError("Unexpected archive member outside dataset")
        z.extractall(output)
    path = output / name
    depth_path = path / "depth_64x48.npy"
    np.save(depth_path, np.stack(depth_frames), allow_pickle=False)
    checked = read_episode(path)
    checked.check_consistency()
    assert len(checked.frame_times) == len(depth_frames)
    assert np.allclose(checked.frame_times, frame_times, atol=1e-8)
    stats = {"scene": scene, "seed": seed, "crashed": False,
             "frames": len(frames), "elapsed_s": time.perf_counter() - started}
    item = {"path": str(path.relative_to(ROOT)), "scene": scene,
            "depth_path": str(depth_path.relative_to(ROOT)), "seed": seed,
            "frames": len(frames), "archive_sha256": digest(archive),
            "depth_sha256": digest(depth_path), "camera_fov_degrees": fov,
            "camera_parameter_keys": list(parameters["cameras"]), "upstream_consistency_check": True}
    return item, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training-data-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/hyperdrone-motion-v1")
    parser.add_argument("--smoke", action="store_true", help="One 2-second training flight only")
    args = parser.parse_args()
    os.environ.setdefault("HYPERLUX_BACKEND", "generic")
    os.environ.setdefault("HYPERLUX_BUILD_JOBS", "4")
    import hyperdrone
    import hyperlux

    cfg = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Dataset manifest already exists; preserve it and choose a new output")
    header = world_header(output)
    plan = {"train": cfg["train"], "validation": cfg["validation"]}
    if args.smoke:
        plan = {"train": [{"scene": cfg["train"][0]["scene"], "episodes": 1}], "validation": []}
    train_scenes = {s["scene"] for s in plan["train"]}
    validation_scenes = {s["scene"] for s in plan["validation"]}
    if train_scenes & validation_scenes or "ProcTHOR-Train-1" in train_scenes | validation_scenes:
        raise ValueError("Scene split overlaps validation or public evaluation scene")
    manifest = {"schema_version": 1, "seed": cfg["seed"], "train": [], "validation": [],
                "created_utc": datetime.now(timezone.utc).isoformat(), "status": "generating",
                "generator_sha256": digest(Path(__file__)), "config_sha256": digest(args.config),
                "world_header_sha256": digest(header), "host": platform.platform(),
                "packages": {k: importlib.metadata.version(k) for k in ["hyperdrone", "hyperlux", "viobench"]},
                "depth": {"kind": "camera_z_metres", "shape_per_frame": [48, 64],
                          "invalid_value": 0, "sample_source": "640x480 RGB-aligned depth",
                          "sample_x": "floor((u+0.5)*10)", "sample_y": "floor((v+0.5)*10)",
                          "pixel_centre_note": "Nearest source pixel is +0.5 source pixels from ideal resized pixel centre (= +0.05 small-image pixels). No depth averaging across surfaces.",
                          "valid_range_m": [0, 100]},
                "training_only_ground_truth": True,
                "public_evaluation_scene_excluded": "ProcTHOR-Train-1", "attempts": []}
    write_json(manifest_path, manifest)
    catalog = hyperlux.datasets.procthor.Train200()
    seed = int(cfg["seed"])
    for split, scenes in plan.items():
        for specification in scenes:
            scene = specification["scene"]
            reference = catalog[scene].reference
            env = None
            try:
                completed = 0
                for attempt in range(3 * specification["episodes"]):
                    if env is not None:
                        env.close()
                    env = hyperdrone.HyperDrone([reference], config=hyperdrone.Config(spec_header=str(header), device="cpu"),
                                               seed=seed, parameters={"cameras": {"camera0": {"fov": cfg["fov_degrees"]}}})
                    name = f"{split}-{scene}-seed{seed}"
                    print(f"Generating {name}", flush=True)
                    item, stats = capture(env, 2.0 if args.smoke else cfg["duration_seconds"], seed, scene, name, output)
                    manifest["attempts"].append(stats)
                    seed += 1
                    print(json.dumps(stats), flush=True)
                    if item is not None:
                        manifest[split].append(item)
                        completed += 1
                    write_json(manifest_path, manifest)
                    if completed == specification["episodes"]:
                        break
                if completed != specification["episodes"]:
                    raise RuntimeError(f"Not enough non-crashing flights for {scene}")
            finally:
                if env is not None:
                    env.close()
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print(f"Dataset manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
