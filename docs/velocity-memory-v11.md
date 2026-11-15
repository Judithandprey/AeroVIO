# Explicit velocity memory V11

This is a fixed contrast on `experiment/velocity-memory-v11`, starting from the
V10 checkpoint in main `d1d0e6b`. Existing workspaces, checkpoints, and defaults
are preserved. The proposed change is an explicit three-value recurrent state,
not a GRU, map, or change to visual/rotation fusion.

## Runtime contract

The compact network grows from 14 -> 64 -> 6 to 18 -> 64 -> 6. Added inputs are
previous predicted velocity in the current interval's starting camera axes and
a validity flag. Initial/new-episode history is invalid. Invalid or >0.2 s
intervals disable memory. Native gyro rotation transports the pre-visual network
displacement/dt into endpoint-camera axes for the next call. Components are
clipped to +/-2 m/s in both training and inference. Velocity is not derived from
the emitted twist: that twist also contains delayed visual position correction.

Teacher C++ is copied from `0c4f778`; only the memory fields/update, 18-input
inference signature, weight export shape and checkpoint build paths change.
Calibrated gyro, KLT and direction fusion remain unchanged. Source snapshots and
hashes are archived. A separately compiled logging guest checks each 18-value
input, 6-value output and coordinate transport against ONNX; it is not timed.

The deployed network has 1,606 parameters (+256), 1,536 semantic dense MACs
(+256; excludes padded SIMD lanes), and 1 KiB extra FP32 weights. The history
uses three float32 values plus a flag. This is not a prediction of whole-system
latency or WASM file size; the final native artifacts are measured directly.

## Matched training

Both arms start with the same V10 compact weights and zero new input columns.
Both have 18 inputs and identical preprocessing, target/state normalization,
loss (normalized SmoothL1 + 2x photometric warping), Adam, cosine schedule, seed
17, 40 epochs, batch 64 targets, and 1,520 optimizer steps. The no-history arm
sets all four extra inputs to zero. The feedback arm uses its own predictions,
never ground-truth velocity. Extra features have mean zero/std one in m/s.

For each V10 joint-sampled target frame, use up to 19 preceding frames from the
same episode. Left padding is masked; no history crosses episodes. Different
sequences may be batched or shuffled. The entire context is differentiable, but
pose/photometric loss is applied only at the target. V10's exact 2,388-target
epoch plan retains 25% low-speed target exposure. Context frames have a different
exposure distribution and are not described as independent loss-bearing targets.

Gyro transports are prepared by replaying the original causal calibration from
episode start. Ground truth supplies supervised targets/visibility and sampling
labels only. Checkpoint selection rolls out all four original validation
episodes without teacher forcing and includes epoch zero as a fallback.

Both arms fine-tune the compact network directly. This differs from V10's
full zero-image training graph; the matched no-history arm controls for this
and continued optimization. Do not attribute changes versus V10 solely to memory.

## Selection and diagnostic replay

All three original fits (control, feedback, feedback repeat) selected epoch zero.
Thus the primary selection retains V10. Before reading any new long-flight
scores, retain fixed terminal epoch 40 for diagnosis. An exact replay adds only
terminal-checkpoint saving and distinct output paths; every epoch metric and the
selected ONNX must match the original fit. This is not a hyperparameter sweep or
a long-score-based checkpoint choice. Final control/feedback and their repeat
are explicitly marked diagnostic.

## Evaluation

Use the six V10 long flights as regression evidence and six separately seeded
60-second flights (430000 onward) as confirmation. Same generator, scenes 5/6/7,
successful-flight policy and 320x240 preprocessing; preserve all attempts.
The new set is excluded from training and checkpoint selection. Verify distinct
ground-truth prefixes against training, prior validation and each other.

Time teacher, V10 and the two fixed-final diagnostics in all 24 execution orders,
shuffled in 24-frame blocks, each model in each position 300 times per flight.
Run selected-epoch equivalence and full final-feedback repetition separately,
outside the timed interleave. Regression accuracy may overlap generation, so
its wall times are excluded. Generation/training finish before confirmation timing.
Check all rotations/angular twists remain exact, linear twist matches final
position increments, no missing frames, and repeated outputs/fuel are identical.

Report pooled one-second distance RMSE/bias, aligned and unaligned ATE, small-net-
motion error, true interval-speed regimes, stopping errors, deterministic fuel
and host wall time. Small net displacement can include curved/returning motion
and is not equivalent to standing still. Empty regimes stay missing. Stops
require >=0.5 s below 0.02 m/s after recent motion >=0.1 m/s; measure up to the
first second of that hold, and report settling failures rather than omit them.
True motion classifications are evaluation labels, never inference gates.

No course-server upload or deployment default change is part of this experiment.
One training seed and these generated flights do not establish generalization.

## Reproduction

Use a fresh output namespace rather than overwrite any retained evidence.

```bash
.train-venv/bin/python scripts/prepare_velocity_memory.py
.train-venv/bin/python scripts/train_velocity_memory.py --arm no-history
.train-venv/bin/python scripts/train_velocity_memory.py --arm feedback
.train-venv/bin/python scripts/train_velocity_memory.py --arm feedback-repeat
.train-venv/bin/python scripts/replay_velocity_diagnostics.py --arm no-history
.train-venv/bin/python scripts/replay_velocity_diagnostics.py --arm feedback
.train-venv/bin/python scripts/replay_velocity_diagnostics.py --arm feedback-repeat
.venv/bin/python scripts/build_velocity_memory.py
.venv/bin/python scripts/verify_velocity_runtime.py
.venv/bin/python scripts/generate_memory_validation.py
.venv/bin/python scripts/evaluate_velocity_memory.py --phase regression
# Wait for fresh generation to finish before the timing/confirmation pass.
.venv/bin/python scripts/evaluate_velocity_memory.py --phase confirmation
.venv/bin/python scripts/report_velocity_memory.py
```
