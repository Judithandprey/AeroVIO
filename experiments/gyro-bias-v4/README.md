# Causal gyro calibration and residual interaction

Initial gyro calibration substantially improves orientation on the original
simulated validation flights. The best orientation variant reduces mean episode
rotation RMSE by **68.77%**, but increases aligned ATE by **0.02752%**. It therefore
fails the frozen strict nonregression rule. **The selected variant remains
`current`; no acceptance threshold was relaxed.**

This is a fixed 2×2 runtime ablation, with no retraining or architecture change.
The [protocol](../../configs/gyro-bias-v4.json) was frozen after the source/startup
audit and before candidate rollouts. All four cells use the original warp-trained
ONNX, SHA-256
`0271431fc621b160cabad4b448cedc91f9ea4e355f080a366823273041afdbb7`.

## Method and source assumptions

The [startup audit](startup-audit.json) and its
[source snapshots](startup-audit-sources) identify a 200-step initial hover at
200 Hz in the pinned simulator. The experiment uses that complete one-second
window. The audit also shows that an initial hover command does not mean
perfectly zero physical rotation: startup transients can contaminate a gyro
mean. Its ground-truth comparisons are offline diagnostics, not estimator inputs.

Calibration uses only observations delivered so far. Its anchor is the first
IMU timestamp minus the first positive sample interval observed in the first
frame. It collects unique samples inside the next one-second window, with the
existing 1e-6 timestamp tolerance. Constant-memory online moments estimate the
mean gyro in **IMU coordinates**.

Acceptance requires at least 180 samples spanning 0.9 seconds, acceleration
standard-deviation norm below 1 m/s², gyro standard-deviation norm below
0.2 rad/s, and each absolute gyro-mean component no greater than
`0.02 + 3 × 0.07 / sqrt(N)` rad/s. The variance limits follow the frozen baseline
gate; the additional mean check uses the audited simulator bias/noise priors.
These are low-variation/prior checks, not proof that the camera is stationary.

The decision is finalized **after** emitting the first frame at the window
deadline. Accepted bias is subtracted from each future IMU gyro sample before
its rotation exponential; sample rotations are composed in order, then
transformed through the camera/IMU extrinsics. There is no retrospective
correction, pose reset, output hold, or gravity realignment. Rejection is final:
the window is never extended or retried.

All four validation flights accept 200 samples. Their decision occurs after
frame index 19 at 0.999999978 s; frame index 20 is the first corrected interval.
The [independent runtime review](independent-runtime-review.json) checks the
window, online mean, sample ordering, extrinsic conversion, and causal timing.

## Four controlled cells

| Variant | Initial calibration | Learned rotation residual |
| --- | --- | --- |
| `current` | Off | On |
| `gyro-only` | Off | Off |
| `calibrated-residual` | On | On |
| `calibrated-gyro` | On | Off |

“Gyro-only” describes the **orientation composition**, not the whole estimator.
Every cell still runs the same image+IMU network and uses its predicted
translation. Raw paired images, raw IMU-derived predictor state, ONNX predictions,
and camera-local translation increments stay identical. Calibration changes
only the orientation prior used for output composition. Mapping is disabled.

This separation matters because the residual was trained to correct the raw
gyro prior. Combining it unchanged with a calibrated prior can overlap with
the calibration correction. The 2×2 design measures that interaction while
holding the predictor inputs fixed.

## Original-flight validation and frozen selection

“Original-flight validation” means four **simulated HyperDrone flights** with
translation and rotation in ProcTHOR scenes 5/6. It is not physical robot or
real-sensor data. These are the same ten-second development-validation flights
used in earlier experiments; they are not a newly untouched test set. Public
scene 1 is excluded from selection.

The [validator](../../scripts/validate_gyro_bias.py) runs actual host
`Solution.step` on 320×240 adapted RGB and original IMU windows. It runs all four
cells twice: 32 complete episode executions and 6,400 frame outputs. Reference
camera poses and twist enter only subsequent scoring, using the pinned upstream
metric implementations. All table values below are arithmetic means across the
four episode metrics, not pooled-frame scores or episode medians.

| Variant | Rotation RMSE (deg) | Sim(3) ATE (m) | Unaligned position RMSE (m) | Rotation wins versus current |
| --- | ---: | ---: | ---: | ---: |
| `current` | 6.948942 | 0.265557318 | 0.518700462 | — |
| `gyro-only` | 7.639686 | 0.265584785 | 0.521585485 | 3/4 |
| `calibrated-residual` | 4.857039 | 0.265755264 | 0.509897097 | 3/4 |
| `calibrated-gyro` | 2.170470 | 0.265630405 | 0.511331897 | 4/4 |

The frozen eligibility rule requires at least 20% lower mean orientation RMSE,
no regression in either mean Sim(3) ATE or mean unaligned position RMSE beyond
the fixed 1e-6 numerical tolerance, and orientation improvement in at least
three flights. Eligible candidates are ordered by aligned ATE, then orientation
RMSE, then protocol order. Otherwise the current model is retained.

Both calibrated variants satisfy the orientation and unaligned-position
requirements but fail **only the aligned-ATE guard**:

- `calibrated-residual`: orientation improves 30.10%, unaligned position improves
  1.70%, and aligned ATE increases 0.000197946 m, or 0.07454%.
- `calibrated-gyro`: orientation improves 68.77%, unaligned position improves
  1.42%, and aligned ATE increases 0.000073087 m, or 0.02752%.

These are meaningful orientation gains with very small aligned-position
tradeoffs. Rejection under this strict rule does not mean calibration has no
value. The [selection record](selection.json), frozen before public evaluation,
preserves `selected_variant: current`, every eligibility check, and protocol,
model, source, and validation hashes. See [full validation results](validation.md)
and [machine-readable evidence](validation.json).

## Interaction and causal limits

With raw gyro, enabling the learned residual improves mean orientation RMSE
from 7.63969° to 6.94894°. After calibration, enabling that same residual worsens
it from 2.17047° to 4.85704°. This reversal is consistent with compensation
overlap or a residual being applied to a different orientation prior from the
one it was trained around. It does not establish that learned residuals are
generally harmful, nor identify the true sensor bias exactly.

Within these fixed observations and weights, the runtime toggles isolate their
effects. They do not prove that all earlier orientation error came from gyro
bias, that the low-variation gate recognizes every moving startup, or that
calibration will work on arbitrary cameras. Small constant true angular motion
can pass the checks and be subtracted as apparent bias. The official KLT baseline
also uses a different initialization procedure; this experiment preserves the
first-camera reference and is not a reproduction of that full initialization.

World positions can change despite identical camera-local translations because
those translations are composed through different orientations. Sim(3) alignment
also fits scale, rotation, and translation, so aligned ATE alone cannot establish
correct metric position or explain the orientation tradeoff.

## Repetition and verification

The complete validation repeats reproduce outputs, raw predictions, and every
calibration state exactly. Actual raw predictor states and image pairs match
the historical original-flight cache. Camera-local translation differs across
cells by at most 2.08e-16 m. Every calibrated output through the deadline frame
matches its corresponding uncalibrated cell exactly.

The current cell reproduces the historical original-flight audit within
6.84e-10 rad in aggregate rotation RMSE and 2.91e-9 m in position RMSE. Raw NPZ
outputs and complete bias traces remain under ignored
`results/gyro-bias-v4-validation/`; their hashes are retained in the compact
validation evidence. [Tests](tests.txt), the runtime review, and the source
snapshots cover the supporting contracts.

These repetitions establish deterministic reproduction on the same inputs,
not statistical significance or robustness. There is no new training seed in
this no-training ablation. Stronger generalization evidence would require
independent simulated or physical episodes with varied startup motion and
sensor conditions, not merely another execution of these flights.

## Completed public comparison

All seven serial public runs completed 10/10 episodes and 12,000 frames each,
without failures or filled outputs: the unchanged official baseline, four
factorial cells, and complete repeats of `current` and `calibrated-gyro`.
All runs use the same pinned runner/runtime, environment, limits and episode
order. All learned cells use identical Python sources and ONNX weights.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Maximum fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| Official KLT + IMU baseline | 2.041635 | 13.245 | 3.212 | 141,910,846 |
| Current: raw gyro + learned residual | 0.966622 | 21.593 | 1.063 | 14,090,709 |
| Raw gyro, no learned residual | 0.963882 | 26.786 | 1.061 | 13,998,029 |
| Calibrated gyro + learned residual | 0.904211 | 18.756 | 1.231 | 14,830,095 |
| Calibrated gyro, no learned residual | 0.881818 | 12.196 | 1.241 | 14,757,111 |

The calibrated-gyro diagnostic reduces mean aligned ATE by **56.81%** and
maximum per-frame fuel by **89.60%** versus the official baseline; local median
step time falls by **61.37%**. Median rotation RMSE falls by **7.92%**. Aligned
ATE improves in all ten episodes; orientation improves in six. These are local
simulated public-set results against the teaching baseline, not production VIO
or physical-robot performance.

Relative to the existing warp model, the same candidate improves mean aligned
ATE by **8.77%** and median rotation RMSE by **43.52%**, at **4.73%** higher maximum
fuel and **16.69%** higher median step time. Its median unaligned position error
increases **1.55%** (2.9359 to 2.9814 m), and angular-velocity RMSE increases
**25.56%** (0.03226 to 0.04050 rad/s). Thus it has a reproducible tradeoff, not
uniform superiority to the previous model. Its median fitted scale remains
2.147, so low Sim(3)-aligned error does not establish accurate metric scale.

Calibration lowers orientation RMSE in 9/10 episodes when the learned residual
is disabled, and in 7/10 when it is enabled. The paired effects and interaction
are in [output verification](output-verification.md). They support the effect
of this fixed causal toggle while leaving the exact compensation mechanism and
broader startup robustness uncertain. See [all metrics and episodes](comparison.md)
and [derived percentages and win counts](interpretation-metrics.json).

The [replication addendum](../../configs/gyro-bias-v4-public-replication.json)
was recorded after original-flight validation and before any v4 public run.
It adds the calibrated-gyro repeat to verify the promising but unselected
orientation tradeoff. It does not change selection, relax thresholds, or permit
public-driven tuning. A better public score cannot retroactively make a candidate
pass the frozen original-flight validation rule.

Both complete repeats reproduce pose/twist files byte-for-byte, every per-episode
accuracy metric, and every frame's fuel exactly. The three uncalibrated controls
also reproduce earlier experiments' poses/twists and accuracy exactly. Across
the four cells, camera-local translations agree within 2.35e-8 m of saved-text
precision. Wall time is reported separately and is not bitwise deterministic.
Local public measurements are not course-server acceptance.

## Evidence and decision boundary

- [Frozen experiment protocol](../../configs/gyro-bias-v4.json) and
  [public replication addendum](../../configs/gyro-bias-v4-public-replication.json).
- [Startup/source audit](startup-audit.json), [window statistics](startup-window-statistics.json),
  and [offline bias-estimator diagnostic](startup-bias-estimator-diagnostic.json).
- [Independent runtime review](independent-runtime-review.json),
  [validation](validation.json), and [frozen selection](selection.json).
- [Comparison](comparison.md), [output verification](output-verification.json),
  [pipeline commands](pipeline-status.json), `runs/` and `checks/` retain all
  reports, manifests, packaging checks and repeat evidence.
- [Windows artifact records](windows-artifacts.json) verify the current fallback
  and calibrated-gyro diagnostic copies by SHA-256. The labels remain distinct.
- [Public calibration audit](public-calibration-audit.md) separates host-replayed
  calibration states from actual WASM outputs: all ten public prefixes are
  identical through frame 19 (0.995 s), and the first corrected interval matches
  independent integration within 4.46e-11 Frobenius error. Its public timestamp
  phase differs from the original generated validation flights.
- [Comparison figure](comparison.png) shows all five primary cells; its caption
  and input hashes are preserved alongside it.
- All **104 host tests** and four upstream **40-frame WASM checks** pass; the
  checks cover the startup-calibration activation period.

The original-flight validation choice remains **`current`**. The public
calibrated-gyro result is a measured, repeated experimental alternative; it has
not been promoted by changing the original acceptance rule. Any later adoption
needs an explicitly stated tolerance/tradeoff decision and fresh validation,
including startup patterns that differ from the deterministic generated hold.

The protocol specifies one fixed calibration experiment with no public-driven
tuning. No additional fitting or threshold search is authorized by a favorable
diagnostic result. No course-server submission, private held-out score, paid
compute, or physical-robot result is claimed.
