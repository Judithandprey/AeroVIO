# Rotation and image-input ablations

Completed locally on 2026-11-10 on `experiment/perspective-map`, starting from
`bd340cfe9c90d564d10bbd1b6ce23c95d127e001`. The question was whether the learned
rotation residual causes accumulated orientation error, and whether image input
helps the small motion model. Settings were fixed in
[`configs/motion-ablation-v2.json`](../../configs/motion-ablation-v2.json).

## Results

Each variant completed the same 10 public episodes and 12,000 frames, without
failures or filled outputs. Timing runs were serial after training stopped.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Maximum fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| Official KLT + IMU baseline | 2.041635 | 13.245 | 3.211 | 141,910,846 |
| Current image + IMU, warping-trained | 0.966622 | 21.593 | 1.064 | 14,078,994 |
| Same weights, rotation residual disabled | 0.963882 | 26.786 | 1.053 | 13,994,827 |
| IMU-only, matched full graph | 1.101235 | 15.427 | 1.067 | 14,162,855 |
| Same IMU-only model, compact graph | 1.101236 | 15.427 | 0.756 | 6,296,120 |

Here **current** means the v1 warp-trained checkpoint, not a newly tuned model.
The working tree's default checkpoint remains the v1 pose-only model. Every
experimental artifact is packaged explicitly. See [all metrics and episodes](comparison.md).
Sim(3) fits scale/rotation/translation; these errors do not establish correct
metric scale. Wall time is local guest-step computation, not end-to-end latency.

## What the rotation toggle establishes

The two image+IMU variants have identical ONNX weights, Python sources, inputs,
and metadata except `rotation_residual_enabled`. Both have the map disabled.
Removing the learned correction changes `R_imu @ R_residual` into `R_imu` while
preserving predicted translation in the previous-camera axes. World translations
can still diverge because those increments are composed through different poses.

Keeping the residual reduces median orientation RMSE by **19.39%** relative to
disabling it, and improves orientation RMSE in **8/10 episodes**. It also improves
rotation RPE and angular-velocity RMSE. Disabling it lowers aligned ATE by only
0.28%, while worsening orientation. The evidence contradicts the initial idea
that the residual is the overall cause of the orientation regression.

The [offline rotation analysis](rotation-diagnostics.md) reproduces report
orientation errors from actual saved WASM poses. Paired camera-local translation
increments agree within 2.35e-8 m, consistent with finite-precision serialization.
The learned residual is positively aligned with the required correction in all
ten episodes, but leaves systematic components and harms two episodes overall.

The current model has lower mean per-step rotational error-rate RMS than the
baseline (1.853 versus 2.443 deg/s), but more directionally consistent errors:
median fixed-frame directional coherence is 0.358 versus 0.175. This is evidence
consistent with small persistent errors accumulating, rather than simply larger
instantaneous noise. It is not proof of a particular training-data bias. Signed
vector sums are first-order diagnostics, not exact SO(3) integration.

The official baseline additionally attempts gyro-bias calibration during its
first 0.5 s, applying it when its rest test passes, and initializes its pose
reference there. The learned runtime has no equivalent initialization. The present toggle does
not isolate that difference or prove it causes the remaining baseline gap.
Calibration/state initialization should be tested separately on validation data.

## What the image-input control establishes

The matched IMU-only run retains the same 29,398-parameter architecture,
initialization, minibatch order, training/validation data, normalization,
40-epoch budget, optimizer and warping weight 2. Images are replaced by constants
both during training and inference. The predictor receives the same IMU9,
intrinsics4 and frame interval. Real RGB/depth still provide the **training-only
photometric labels** in both runs; this tests image information as a predictor
input, not whether visual supervision played any role in learning.

Both models select checkpoints by the same validation pose-loss rule: epoch 4
for image+IMU and epoch 14 for IMU-only. Re-running image+IMU with the extended
trainer reproduces the original ONNX byte hash and all 40 epochs' metrics exactly.

Adding image input reduces mean aligned ATE by **12.22%** relative to IMU-only;
removing it increases that error by 13.93% relative to current. Image+IMU wins
6/10 episodes. However, one episode (`9b6a9279`) accounts for **88.50% of the net
aggregate improvement**. The other nine average 1.0119 m versus 1.0291 m. All ten
remain in the headline result; this sensitivity analysis is not a reason to
discard an episode.

Image+IMU improves median unaligned ATE (2.936 versus 3.122 m), while IMU-only
has better accumulated rotation, translation RPE and velocity errors. Thus image
input helps some trajectory estimates in this pilot, but is not a uniform gain
across tasks or episodes. The IMU-only validation loss was lower despite worse
public aligned ATE, which further cautions against generalizing from one metric.

## Compact inference is a separate compute experiment

The matched control deliberately still runs the full CNN with constant images.
After training, its visual feature vector is constant and can be folded into the
first dense layer's bias. The compact model has **1,350 parameters** and a
**5,949-byte ONNX**, compared with 29,398 parameters / 119,770 bytes for the full
graph. It is derived from the same checkpoint without retraining.

Host PyTorch/OpenCV equivalence passed on 796 validation and 64 synthetic inputs
(maximum checked difference 1.43e-6 in normalized output). On actual complete
WASM runs, maximum differences were 1.30e-6 m in position, 1.20e-7 rad in rotation,
and 7.50e-8 in a twist component. These are numerical equivalents, not byte-
identical trajectories.

Compaction lowers median step time by **29.10%** and maximum frame fuel by
**55.54%** relative to matched full-graph IMU-only. The existing ABI still supplies
images and the shared wrapper retains image preprocessing; this is not a claim
of minimal sensor-to-output inertial-odometry latency.

## Verification and evidence

- **62 tests passed** across runtime, training, warping, map, output conventions,
  comparison guards and synthetic rotation diagnostics.
- All four newly packaged artifacts passed the upstream 20-frame WASM check.
- Fresh baseline and current image+IMU runs reproduced the previous experiment's
  pose/twist files byte-for-byte and all per-episode accuracy metrics exactly.
- All variants share the pinned upstream runner, stripped runtime, environment,
  limits and public episode order. Learned variants share identical Python files.
- Raw reports/manifests/logs are in `runs/`; training records in `control/`,
  `imu-only/`, and `imu-only-compact/`. The original current weights remain in
  `../learned-motion-v1/warp/`.
- `training-audit.json`, `output-verification.json`, `rotation-diagnostics.json`,
  `interpretation-metrics.json`, and `source-inventory.json` retain the checks,
  derived quantities and hashes. `pipeline-status.json` records run commands.
- `windows-artifacts.json` records the copied, SHA-256-verified local WASM files.

Raw trajectory files, cached episodes, full PyTorch checkpoints and large WASM
bundles remain local and ignored. No course-server submission, private score,
real-robot result or additional seed experiment is claimed.

## Reproduce and continue

See [setup](../../docs/setup.md) and [training/ablation commands](../../docs/learned-motion.md).
The saved reports can be compared without regenerating data:

```bash
.venv/bin/python scripts/compare_motion_runs.py \
  --run baseline experiments/motion-ablation-v2/runs/baseline \
  --run current experiments/motion-ablation-v2/runs/current \
  --run no-rotation experiments/motion-ablation-v2/runs/no-rotation \
  --run imu-only experiments/motion-ablation-v2/runs/imu-only \
  --run imu-compact experiments/motion-ablation-v2/runs/imu-compact \
  --out results/reproduced-ablation-v2.md
```

The rotation analysis additionally needs the local raw trajectory files and
cached episode metadata; its recorded command is in `rotation-diagnostics.json`.
`scripts/verify_ablation_outputs.py` checks old/new control identity and compact
output parity against those raw outputs.

Next justified experiments are a controlled gyro-bias initialization comparison,
longer/more diverse training trajectories with short-window accumulated rotation
loss, and multiple training seeds. The tradeoff between image-assisted translation
and rotation also motivates separately testing how the two prediction heads
share visual features. None of these follow-up changes was fitted in this run.

For the conceptual distinction requested alongside this experiment, read
[Warping, motion estimation and world models](../../docs/warping-and-world-models.md).
