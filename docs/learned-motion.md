# Small motion network and controlled geometry ablations

This experiment implements the proposed inverse inference: estimate camera
motion from consecutive observations. Warping supplies a training constraint;
it does not require a full image generator at inference time.

The first full experiment is complete. See the
[measured results and limitations](../experiments/learned-motion-v1/README.md)
and [four-way comparison](../experiments/learned-motion-v1/comparison.md).

## Variants

- **B0:** immutable official KLT + IMU baseline, tag `baseline-klt-imu-v1`.
- **N1:** small paired-image CNN + IMU, trained with supervised relative-motion loss.
- **N2:** identical architecture, initialization, batches, data split and schedule,
  with an additional masked photometric warping loss.
- **N3:** exactly N2's weights with bounded sparse-map correction enabled.

Each learned checkpoint is selected by the lowest normalized motion loss on
scene-disjoint validation episodes. Public-set scores never select epochs,
hyperparameters, training scenes, or map gates. These are pilot ablations from
one seed and a small generated dataset, not a statistical superiority claim.

## Network and output convention

The network takes two 64x48 grayscale frames and a 14-value state vector:
camera-frame IMU preintegration (rotation, delta velocity, delta position),
four normalized intrinsics, and frame interval. It has three stride-2 Conv/ReLU
layers with 16/32/64 channels, global average pooling and a small MLP.

The six outputs are translation in previous camera axes and a rotation residual.
The relative camera rotation is `exp(IMU_rotation) @ exp(rotation_residual)`.
The pose is composed into the first-camera frame. Twist is the finite-interval
mean computed from the final pose, expressed at the current camera centre in
current camera axes; it is an approximation to the instantaneous requested twist.

Training and inference share `workspace/solution/preintegrate.py`. The first IMU
sample interval is included. Its 1 microsecond timestamp tolerance matches the
upstream episode reader and only clips rounding at the frame endpoint. This
intentional adaptation differs from the upstream tiny-CNN example. The original
KLT baseline remains unchanged in its tagged artifact.

## Training data

`scripts/generate_training_data.py` uses the pinned HyperDrone/HyperLux stack,
CPU generic rendering, static ProcTHOR scenes, an X500 camera/IMU/autopilot, and
the official full 640x480 rendering resolution. The training set uses scenes
2/3/4; validation uses scenes 5/6. Public evaluation scene 1 is excluded.
The completed pilot has 12 training and 4 validation flights, each 10 seconds.

RGB and camera-z depth share each capture. After a stride boundary, the renderer
captures the current post-step state; the recorded timestamp, IMU window and
interpolated camera GT match that state. Timestamps are explicitly float64.
The standard episode readback passes upstream consistency checks.

Training depth is point-sampled to 64x48. Nearest source pixel centres differ
from resized RGB pixel centres by 0.05 low-resolution pixels; this approximation
is recorded instead of hiding it. No depth averaging across different surfaces
is used. The model receives neither depth nor reference poses at inference.

## Warping objective

The supervised target is translation plus IMU rotation residual. Its six
components are standardized using training-only statistics, and optimized with
Smooth L1 loss. N2 adds a fixed-weight photometric L1 loss.

For each target/current pixel, camera-z depth back-projects it into 3D. Predicted
current-to-previous camera motion maps it into the previous camera. Its image
projection samples the previous grayscale frame using differentiable bilinear
sampling (`align_corners=False`, explicit pixel-centre convention).

The visibility mask is fixed from true **training** geometry, including image
bounds and source-depth agreement. Disoccluded pixels have no photometric target.
The network cannot lower this loss by predicting zero confidence or masking
everything. Empty masks produce a finite zero photometric term and retain the
supervised pose term. Pose-only and warp variants share checkpoint selection.

## Sparse map

N3 keeps at most 160 persistent KLT tracks and four anchor keyframes. It uses
forward/backward tracking checks, rotation-compensated parallax, positive depth
and two-view reprojection gates before triangulation. Newly triangulated points
cannot correct their own creation frame. PnP has a 40-iteration RANSAC cap and a
5-iteration refinement cap, with support/geometry/innovation rejection gates.

The map contains currently tracked local landmarks; it discards lost tracks
without re-identification after occlusion. It does not infer unseen surfaces,
access ground truth, perform loop closure or revisit already emitted outputs.
All state resets per episode. If an update is rejected, the learned prior is
retained. Accepted corrections update both pose and the reported interval twist.

## Reproduction

Use the same Linux/WSL prerequisites and pinned upstream checkout as `setup.md`.
Keep the evaluation venv unchanged. Create a separate training venv, install the
CPU PyTorch wheel from its official index and then the pinned training packages.

```bash
python3 -m venv .train-venv
.train-venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r configs/requirements-training.lock.txt
.train-venv/bin/python -m pip install --no-deps -e external/viobench
.venv/bin/python scripts/generate_training_data.py --output data/hyperdrone-motion-v2
.train-venv/bin/python scripts/prepare_motion_data.py --manifest data/hyperdrone-motion-v2/data_manifest.json --out data/motion-cache-v2
.train-venv/bin/python scripts/train_motion.py --cache data/motion-cache-v2 --out models/motion-pose-v1 --warp-weight 0
.train-venv/bin/python scripts/train_motion.py --cache data/motion-cache-v2 --out models/motion-warp-v1 --warp-weight 2
.venv/bin/python scripts/package_motion.py --model models/motion-pose-v1 --name motion-pose-v1
.venv/bin/python scripts/package_motion.py --model models/motion-warp-v1 --name motion-warp-v1
.venv/bin/python scripts/package_motion.py --model models/motion-warp-v1 --name motion-warp-map-v1 --map
```

Run synthetic `viobench check` on each artifact, then use `evaluate_public.py`
with a new results directory for every run. Run measurements sequentially after
generation/training stops. The comparison helper rejects mismatched runtime,
episode selection, limits, runner revision or Wasmtime version. Source and model
hashes come from the actual WASM payload, not the mutable working tree.

All training-only operations remain outside the WASM solution. The deployed
network uses ONNX opset 13 and OpenCV DNN; PyTorch, depth GT, data loaders and
photometric loss do not ship in the inference artifact.

## Attribution and evidence boundary

The small backbone and IMU-residual design adapt viobench's `tiny_cnn` teaching
example. The baseline, runtime and original example remain attributed in
`THIRD_PARTY_NOTICES.md`. New geometry, temporal handling, calibration features,
data separation, training losses and measurements are documented separately.

Local public results, host unit tests and WASM execution are separate evidence.
No course-server upload, held-out score, robot control or WAM training is implied.
