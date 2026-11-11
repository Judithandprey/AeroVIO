# Lightweight feature correspondence for rotation

V5 tests whether explicit local matching helps rotation after the fixed causal
gyro calibration. It preserves the official baseline, v4 current model and v4
calibrated-gyro artifact. The frozen protocol is
[`feature-correlation-v5.json`](../configs/feature-correlation-v5.json).
Completed measurements belong in the [experiment record](../experiments/feature-correlation-v5/README.md).

## Architecture

The original 29,398-parameter MotionNet is frozen and still supplies camera-local
translation from both images and its original raw 14-value IMU/calibration state.
Its old rotation output is replaced by a new branch with 6,243 trainable parameters
(35,641 stored parameters in total).

The shared encoder processes each 64x48 grayscale frame separately:
`Conv(1,8,3,stride=2) -> ReLU -> Conv(8,16,3,stride=2) -> ReLU`.
At each cell of the resulting 12x16 grid, the branch compares current features
with previous features at nine offsets in a 3x3 neighborhood. Features are
channel-normalized with a finite epsilon; their dot product is the correlation.
For current `(y,x)`, offset `(dy,dx)` denotes previous `(y+dy,x+dx)`.
Out-of-image support is zero, including its contribution to edge pooling.

Average pooling with kernel 3x4 preserves a 4x4 spatial grid for every offset:
144 matching values. The head also receives the actual camera-axis rotation
prior (three radians divided by 0.01) and five normalized intrinsics/dt values.
`Linear(152,32) -> ReLU -> Linear(32,3)` predicts a correction scaled by 0.01 rad.
The final layer starts at zero, so the initial branch applies no correction.

Rotation composition remains `R_prior @ exp(correction)`. The new
`rotation_prior_input: true` metadata field asks the runtime to pass the actual
prior it will compose: raw gyro before activation, calibrated gyro after the
causal calibration decision. The original translation path continues to receive
raw state. Missing or malformed required prior input fails visibly. Existing
models omit the new field and retain their old two-input contract.

This is a small local matching volume, not a RAFT implementation. RAFT's
[paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php)
provides background on feature correlation; its all-pairs multiscale volume and
iterative flow updates are not used here. No optical-flow labels, correspondence
accuracy claim, 3D map, new warping loss, hidden-surface prediction or robot action
model is added in this round.

## Controlled training

The temporal branch compares previous/current frames. A matched single-frame
branch compares current/current features, withholding previous-frame information
from rotation while preserving the original paired-frame translation predictor.
Both branches have identical parameter counts, initial state tensors, real
minibatch orders, labels, optimizer steps and schedule. Their compute need not be
identical: the single-frame graph may reuse an encoder result.

Each fit uses 40 epochs, seed 17, Adam on the new branch only, batch 64, four CPU
threads, cosine learning rate 0.001 to 0.0001 and gradient clipping at 5.
The original 2,388 pairs from scenes 2/3/4 train the branch; 796 pairs from four
scene-disjoint flights in scenes 5/6 select checkpoints. Public scene 1 is absent.
There is no new synthetic auxiliary data or hyperparameter search.

Derived labels are `log(R_causal_prior.T @ R_reference_relative)`, supervised
with Smooth L1 after division by fixed 0.01 radians. The causal prior is replayed
from IMU alone, including the original deadline-frame activation boundary.
Reference poses label training and score validation; they never enter runtime
inputs or the calibration decision. The base translation model remains frozen,
so no additional photometric or translation objective is required in this round.

For each branch, the trained epoch with lowest equal-episode accumulated
validation rotation RMSE is retained as a diagnostic. Earlier epochs win ties.
Eligibility is separate: at least 5% lower mean rotation error than zero
correction, improvement on at least three of four flights, and at most 1% increase
in mean aligned or unaligned position error, with absolute numeric tolerance
1e-6. This is a new v5 engineering budget; it does not revise v4's stricter guard.
Public data never chooses an epoch or changes these thresholds.

## Verification and interpretation

Tests check known signed displacements, finite zero-feature correlation, boundary
support, frozen translation, gradients from both frames, the single-frame
information restriction, ONNX/OpenCV parity and causal runtime prior input.
Original-flight actual `Solution.step` rollouts check raw-state identity, prior
identity, translation invariance and agreement with training-side scoring.
Both trained exports pass a 40-frame check inside the pinned WASM runtime.

A previous-image permutation probe keeps the current image, state, causal prior
and translation fixed. This is an out-of-distribution sensitivity diagnostic;
degradation does not by itself prove useful correspondence estimation.

The six public runs are predeclared: baseline, original current, original
calibrated-gyro, single-frame branch, temporal branch and temporal repeat.
They run serially after training stops, using the same 10 episodes, runtime and
limits. Report both local wall time and deterministic fuel. Sim(3) alignment
removes scale; unaligned position, rotation and angular velocity remain relevant.
Repeated identical outputs establish reproducibility, not new-scene validity.

## Reproduction

Use WSL `/root/projects/AeroVIO`, the pinned `.venv` and separate `.train-venv`.
The following are the recorded commands; existing output directories are refused.
For another reproduction, update a copied protocol/script to use new cache/model
and report directories, preserving the original evidence.

```bash
.train-venv/bin/python scripts/train_feature_correlation.py --prepare
.train-venv/bin/python scripts/train_feature_correlation.py --variant single-frame --out models/correlation-single-frame-v5-complete
.train-venv/bin/python scripts/train_feature_correlation.py --variant temporal --out models/correlation-temporal-v5
.train-venv/bin/python scripts/validate_feature_correlation.py
.venv/bin/python scripts/run_feature_correlation_evaluation.py
.venv/bin/python scripts/report_feature_correlation.py
```

Package archived candidates explicitly, using new artifact names:

```bash
.venv/bin/python scripts/package_motion.py --model experiments/feature-correlation-v5/temporal --name correlation-temporal-repro --calibrate-gyro
.venv/bin/python scripts/package_motion.py --model experiments/feature-correlation-v5/single-frame --name correlation-control-repro --calibrate-gyro
```

These exports are diagnostics until the recorded eligibility rule accepts a
change. Packaging is separate from course-server submission. The working
`workspace/solution/net.onnx` remains the original pose-only default.
