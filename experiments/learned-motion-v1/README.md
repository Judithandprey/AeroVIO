# Learned motion pilot: measured results

Completed locally on 2026-11-10. This is a four-way controlled experiment on
`experiment/perspective-map`; the official estimator remains preserved at
`baseline-klt-imu-v1`.

| Variant | Mean Sim(3) ATE (m), lower is better | Median step (ms) | Maximum fuel/frame |
| --- | ---: | ---: | ---: |
| Official KLT + IMU | 2.041635 | 3.342 | 141,910,846 |
| Small motion CNN + IMU | 0.991676 | 1.058 | 14,073,824 |
| CNN + warping training loss | 0.966622 | 1.062 | 14,073,824 |
| Same warping model + sparse 3D map | 1.122139 | 5.139 | 157,635,808 |

All four runs completed the same 10 public episodes and 12,000 frames with zero
failures or filled frames. They used the same WASM runtime, runner revision,
limits, Wasmtime version and episode order. Evaluation ran sequentially after
generation and training finished. Wall time measures local step execution only,
excluding sensor acquisition, communications and other end-to-end work.

The [generated comparison](comparison.md) includes every episode and additional
accuracy metrics. Raw reports and their manifests are archived byte-for-byte in
`runs/`; each manifest binds the report hash and the actual embedded WASM files.

## Interpretation

- The motion CNN lowers mean aligned trajectory error by **51.43%** relative to
  the official baseline, wins all 10 episodes on that metric, and uses **90.08%
  less maximum fuel/frame**. Its measured median step is about **3.16x faster**.
- Adding warping lowers mean aligned error by a further **2.53%**, winning 7/10
  episodes. Its inference graph has the same size and architecture; warping is
  a training loss, so its roughly equal inference cost is expected. This modest
  improvement from one training seed is not a statistical significance result.
- The current map increases mean aligned error by **16.09%** relative to warping
  alone, wins only 2/10 episodes, and costs about **4.84x** as much median step
  time. Preserve it as a negative ablation; do not present it as an improvement.

There are accuracy tradeoffs. Median accumulated rotation RMSE is 13.245 degrees
for the baseline, 21.177 for the CNN, 21.593 with warping, and 17.390 with the map.
The map helps this accumulated orientation metric while worsening rotation RPE
and angular-velocity error. Warping slightly worsens translation RPE and velocity
RMSE despite reducing aligned trajectory error. These results do not establish
an across-the-board pose/twist improvement.

The sparse map triangulates from predicted poses, so it inherits their scale
bias and does not provide an independent metric-scale measurement. Timed runs
disable diagnostic logging; accepted-correction counts and the exact reason for
the map's regression have not yet been established.

Sim(3) alignment fits scale as well as rotation and translation. Learned runs
need median scale factors around 2.00 and 2.16, so metric scale remains wrong.
Their median unaligned ATE improves substantially over the baseline (41.429 m
to 2.985/2.936 m), but still needs work. All public episodes share one scene;
performance on unseen scenes, new motion distributions, real cameras and the
course server remains unverified.

## Training and validation

The network has **29,398 parameters** and a **119,770-byte ONNX** artifact.
It consumes consecutive 64x48 grayscale images, camera-frame IMU preintegration,
normalized intrinsics and elapsed time. It estimates relative translation and
an IMU rotation residual. No depth or reference pose enters inference.

Data comprises 12 training flights (120 seconds) in scenes 2/3/4 and four
validation flights (40 seconds) in scenes 5/6. Scene 1, used by public evaluation,
is excluded from both. There are 2,388 training and 796 validation adjacent pairs.
Each flight is only 10 seconds, and typical adjacent translation is small; this
pilot is insufficient to establish robustness on long trajectories.

Both models use seed 17, the same initialization and minibatch order, and 40
epochs with the same optimizer and learning-rate schedule. The only training
change is photometric-loss weight 0 versus 2. Both select epoch 4 by the lowest
scene-disjoint validation pose loss, before public evaluation. No public result
was used to tune the models, map gates or checkpoint choice in this experiment.

A separate complete pose-only training repetition reproduced the exact ONNX
hash, selected epoch and validation metrics. PyTorch/OpenCV export differences
were below 5e-7 on the checked validation examples. Host tests cover geometry,
occlusion masks, differentiability, runtime conventions, ONNX parity, map gates,
and comparison provenance: **39 tests passed**. Each trained WASM additionally
passed the upstream 20-frame check, followed by the full public run above.

## Evidence index

- `pose/`, `warp/`: deployable ONNX, normalization metadata, 40-epoch histories
  and training manifests with source/cache hashes.
- `dataset-manifest.json`, `dataset-audit.json`, `cache-provenance.json`: scene
  split, seeds, generated-episode consistency and training-cache identities.
- `data-repeatability.json`, `generator-alignment-audit.json`: deterministic
  generation and RGB/depth/IMU/reference timing audit.
- `training-repeatability.json`: exact complete-training repetition checks.
- `runs/{baseline,motion-cnn,cnn-warp,cnn-warp-map}/`: raw report, manifest and log.
- `tests.txt`, `wasm-check-*.txt`, `map-geometry-wasm-check.txt`: host and WASM checks.
- `pipeline-status.json`: exact commands, completion status and elapsed time.
- `source-inventory.json`: source/config hashes at final evidence archival.

The early `data-warp-smoke.json` was a geometry/debug check on a separate short
smoke sequence, not a model evaluation. The full training data uses corrected
float64 timestamps. The interrupted first data-generation attempt was preserved
locally and excluded from both models.

## Reproduce the saved comparison

From the repository root, without downloading raw training data:

```bash
.venv/bin/python scripts/compare_motion_runs.py \
  --run baseline experiments/learned-motion-v1/runs/baseline \
  --run motion-cnn experiments/learned-motion-v1/runs/motion-cnn \
  --run cnn-warp experiments/learned-motion-v1/runs/cnn-warp \
  --run cnn-warp-map experiments/learned-motion-v1/runs/cnn-warp-map \
  --out results/reproduced-comparison.md
```

See [the design and training commands](../../docs/learned-motion.md) for training
from scratch, and [setup](../../docs/setup.md) for building WASM from the archived
small models. Raw datasets, full checkpoints and the ~50 MB WASM bundles remain
local and ignored. The repository's default `workspace/solution` model is the
pose-only CNN, keeping the first learned stage explicit; package `warp/` to use
the best aligned-trajectory result observed in this pilot.

Next experiments should add longer and more varied independently generated
trajectories, multiple training seeds, and image/IMU-only ablations to measure
how much the network uses vision versus a motion prior. Diagnose map scale,
landmark quality and correction timing on validation trajectories before any
further public-set comparison.
