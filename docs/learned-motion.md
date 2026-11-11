# Small motion network and controlled geometry ablations

The completed [feature-correlation follow-up](feature-correlation.md) freezes this
translation model and adds a local matching rotation branch. Both the temporal
branch and a matched single-frame control fail original-flight validation; their
public diagnostic results do not replace the prior model. The implementation,
extra causal prior input and measured compute are recorded separately.

This experiment implements the proposed inverse inference: estimate camera
motion from consecutive observations. Warping supplies a training constraint;
it does not require a full image generator at inference time.

The first full experiment is preserved in the
[measured results and limitations](../experiments/learned-motion-v1/README.md)
and [four-way comparison](../experiments/learned-motion-v1/comparison.md).
The follow-up [rotation/input experiment](../experiments/motion-ablation-v2/README.md)
isolates learned rotation correction and visual information. The
[v3 rotation-focused experiment](../experiments/rotation-focus-v3/README.md)
adds calibrated rotational view changes during training; both fine-tunes retained
their starting checkpoint under the fixed real-validation rule. Read the
[warping and world-model guide](warping-and-world-models.md) for the conceptual
scope of this estimator.

The [v4 startup-calibration experiment](gyro-bias-calibration.md) changes only
inference-time rotation integration, using the same v1 warp checkpoint. Original
simulated-flight validation retains `current`: its calibrated-gyro alternative
improves rotation but fails the strict aligned-ATE guard. All seven public runs
are complete, including exact selected/current and diagnostic repeats; no new
model was trained or promoted.

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

## Rotation and visual-input controls

The runtime accepts optional metadata fields with backward-compatible defaults:

| Field | Default | Supported ablation |
| --- | --- | --- |
| `rotation_residual_enabled` | `true` | `false` composes `exp(IMU_rotation)` alone; sparse map must be disabled |
| `input_mode` | `image_imu` | `imu_only` replaces the network's image input with zeros at training and inference |
| `network_inputs` | `images_state` | `state` selects the equivalent compact graph and requires `input_mode=imu_only` |
| `gyro_bias` | `{"enabled": false}` | `{"enabled": true}` enables the fixed causal startup calibrator; sparse map must be disabled |

Rotation on/off packages use identical ONNX weights, image inputs and
normalization. Disabling the residual leaves each predicted local translation
unchanged. World-axis translations on later frames can change because they are
composed using a different orientation. Twist is recomputed from the resulting
pose. See the [rotation diagnostics](../experiments/motion-ablation-v2/rotation-diagnostics.md)
for checks on the saved trajectories and limits on causal interpretation.

The primary IMU-only control retrains the same 29,398-parameter model with zero
images. Its 14 state values still contain nine IMU features, four normalized
camera-intrinsic values, and dt. It retains the same data/cache, seed 17,
initialization, batch order, 40-epoch schedule, warping weight 2, and validation
checkpoint-selection rule as the image+IMU warping run. A different selected
epoch is allowed by that shared rule.

Real RGB pairs remain **training supervision**: the photometric loss warps the
observed source image and compares it with the observed target image. Depth,
visibility and reference poses are also training labels. Only the prediction
network receives zero images; replacing the photometric target with zeros would
change the loss and invalidate this input-information comparison. Inference has
no access to those labels.

An optional compact export evaluates the trained visual branch on zero images,
then folds its constant feature contribution into the first head-layer bias.
The remaining two-layer MLP has 1,350 parameters and a 5,949-byte ONNX graph.
It uses the same trained checkpoint without additional optimization. Its
`source_model_sha256` records the full model, and the export validates full versus
compact predictions in both PyTorch and OpenCV on all validation states plus
seeded synthetic states. Floating-point roundoff is recorded explicitly. Treat
this as a separate compute experiment; the full graph remains the primary
control for whether images contribute useful information.

## Fixed startup-calibration control

V4 crosses bias calibration on/off with learned rotation residual on/off. Every
cell uses the same ONNX and raw 14-value network state; correction never changes
the predictor inputs or its camera-local translation output. After the complete
initial one-second window passes the fixed coverage, variation and prior gates,
a separate path subtracts the frozen IMU-coordinate gyro mean before every
sample exponential, then transforms the integrated rotation into camera axes.
Correction begins after the closing frame has been emitted. The residual is
composed afterward when enabled. No output is replayed or reset.

On the four original simulated validation flights, calibrated-gyro lowers mean
rotation RMSE from 6.948942° to 2.170470° (68.77%), while mean Sim(3) ATE increases
from 0.265557 m to 0.265630 m (0.02752%). The fixed nonregression rule therefore
retains the uncalibrated `current` reference. Calibration with the residual also
fails the aligned-ATE guard. Exact repeated validation outputs establish
reproduction, not statistical significance. The completed public calibrated-gyro
diagnostic improves aligned ATE and rotation versus current at additional
compute and a small unaligned-position regression; the selection remains fixed.

The [calibration guide](gyro-bias-calibration.md) explains why the simulator's
documented initial hover is an assumption: its pose still moves, and a low gyro
standard deviation cannot rule out slow constant rotation. The experiment
contains no physical-sensor measurements. See the
[validation decision](../experiments/gyro-bias-v4/validation.md) and
[reproduction commands](setup.md#reproduce-startup-gyro-calibration).

## Training data

The [rotation-training guide](rotation-focused-training.md) explains v3's
additional training-only examples. `prepare_rotation_data.py` uses each existing
source frame to create two virtual views through the calibrated SO(3) homography.
Both views follow the same 640→320 adaptation, symmetric perspective reprojection,
fixed 0.85 crop and shared 64x48 grayscale preprocessing. Source-support checks
reject unsupported rays instead of exposing angle-dependent padded borders.

The auxiliary cache contains 2,400 training and 800 validation pairs, retaining
the original scene split. Camera x/y/z/mixed axes are balanced, with independent
signed rotations of 0–3 degrees over 0.05 seconds. Ten synthetic IMU samples
describe the same stationary-camera-centre rotation, using source gravity,
camera/IMU extrinsics, centripetal lever-arm acceleration and configured
noise/bias/random-walk priors. The updated intrinsics enter the state vector.

`train_rotation_focus.py` continues the original warp model with frozen
normalization. The control and focus runs share real minibatches, 760 optimizer
steps and the 20-epoch schedule; focus additionally uses rotation-only SO(3)
supervision. Auxiliary samples do not teach zero translation. Their validation
split is diagnostic only. Selection minimizes accumulated rotation error on
original simulated-flight validation, retaining epoch zero unless both pair-translation and integrated
position guards permit an improvement.

Both recorded runs selected epoch zero, so this trial established no deployed
gain. A retention-only rerun preserves the unselected epoch-20 candidate under
`diagnostic-last/`, separately from the selected top-level model. The
[pre-public addendum](../configs/rotation-focus-v3-public-diagnostics.json)
fixes a six-way comparison: baseline, starting warp model, two selected models
and two last-epoch diagnostics. All six public runs completed. The focused final
candidate slightly improves aligned ATE but worsens rotation RMSE; see the
[v3 record](../experiments/rotation-focus-v3/README.md).
See [setup](setup.md#reproduce-rotation-focused-training) for reproduction commands.

The [input examples](../experiments/rotation-focus-v3/rotation-examples.png) show
actual auxiliary validation inputs with exact signed labels. They illustrate
perspective changes and provide no model-accuracy evidence by themselves.

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

To reproduce the IMU-only training and its optional compact export from the same
cache, use new output directories. `--compact-out` requires `--input-mode
imu_only`; it exports after checkpoint selection, without another training run.

```bash
.train-venv/bin/python scripts/train_motion.py --cache data/motion-cache-v2 --out models/motion-imu-only-v2-repro --input-mode imu_only --warp-weight 2 --epochs 40 --batch 64 --seed 17 --lr .001 --threads 4 --compact-out models/motion-imu-only-compact-v2-repro
.venv/bin/python scripts/package_motion.py --model models/motion-imu-only-v2-repro --name motion-imu-only-trained-repro
.venv/bin/python scripts/package_motion.py --model models/motion-imu-only-compact-v2-repro --name motion-imu-compact-trained-repro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name motion-rotation-off-trained-repro --disable-rotation-residual
```

The archived v2 models are in `experiments/motion-ablation-v2/imu-only/` and
`imu-only-compact/`; [setup](setup.md) shows how to package them without retraining.
The rotation control reuses the archived v1 warping ONNX file. Refer to the
[v2 experiment record](../experiments/motion-ablation-v2/README.md) for training
provenance, compact equivalence checks, and completed evaluation status.

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
