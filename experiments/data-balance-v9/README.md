# Data balance V9: completed matched sampling contrast

Teacher reference: `0c4f778`. Branch: `experiment/native-fusion-v8`. Same inference code, network, normalization, losses and training budget. Only training example exposure changes. No server submission.

## Fixed treatment

Original training counts: startup/transition 1346, cruise 749, turn 293. Balanced exposure is 796/796/796 per epoch, still 2,388 examples and 38 optimizer steps. Forty epochs give 1,520 steps in both arms.

Measured low-speed (<0.1 m/s) exposure changes from 49.7% to 51.5%. Median exposed true speed changes from 0.102 to 0.086 m/s. See `exposure-distribution.json`.

First second is startup. Afterwards angular speed >=15 deg/s takes precedence as turn; speed <0.1 m/s or absolute tangential acceleration >0.2 m/s2 is startup/transition; the remainder is approximate cruise. Acceleration uses a fixed five-frame speed average. Cruise is not a claim of perfectly constant 3D velocity. These labels only choose training examples.

Training uses the original zero-image graph and folds it into the same 1,350-parameter state-only graph. Same seed 17, batch 64, Adam 0.001, cosine schedule, warping weight 2, CPU four threads. Normalization is computed from the unchanged original cache, not the resampled exposures. Validation retains its original distribution and normalized pose-loss checkpoint criterion.

Control chooses epoch 14 and reproduces the teacher ONNX exactly. Balanced chooses epoch 7; a complete same-seed rerun reproduces its ONNX and validation metrics exactly. This repetition establishes determinism, not across-seed statistical significance.

## Validation decision before public evaluation

Selected reference: **teacher**. Decision flags: `{"distance_improvement": false, "aligned_ate_nonregression": false, "rotation_unchanged": true, "fuel": true}`.

| Variant | Mean Sim3 ATE (m) | Mean 1s length RMSE (m) | Mean rotation RMSE (deg) |
| --- | ---: | ---: | ---: |
| teacher | 0.221909 | 0.188307 | 2.1705 |
| control | 0.221909 | 0.188307 | 2.1705 |
| balanced | 0.222567 | 0.186707 | 2.1705 |

## Full public comparison: 10 flights x 60 seconds

| Variant | Sim3 ATE (m) | 1s length RMSE (m) | Predicted/true distance median | Rotation RMSE median (deg) | Max fuel/frame | Median step (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0.486329 | 0.229366 | 0.4006 | 12.1963 | 2,232,749 | 0.2916 |
| control | 0.486329 | 0.229366 | 0.4006 | 12.1963 | 2,232,749 | 0.1481 |
| balanced | 0.475930 | 0.231712 | 0.4000 | 12.1963 | 2,232,749 | 0.1480 |
| balanced-repeat | 0.475930 | 0.231712 | 0.4000 | 12.1963 | 2,232,749 | 0.2361 |
| teacher-repeat | 0.486329 | 0.229366 | 0.4006 | 12.1963 | 2,232,749 | 0.3215 |

Balanced changes 1s distance-length RMSE by +1.02%, aligned ATE by -2.14%, mean unaligned ATE by -0.40%, mean fuel by +0.00%, and peak fuel by +0.00%. Negative error changes are improvements.

## Speed-conditioned distance

| True net speed (m/s) | Teacher length RMSE (m) | Balanced length RMSE (m) | Teacher distance ratio | Balanced distance ratio |
| --- | ---: | ---: | ---: | ---: |
| below_0.1 | 0.079811 | 0.081841 | 1.4827 | 1.6095 |
| 0.1_to_0.25 | 0.095132 | 0.088543 | 0.5714 | 0.6024 |
| 0.25_to_0.5 | 0.236902 | 0.237765 | 0.3835 | 0.3779 |
| above_0.5 | 0.398661 | 0.408420 | 0.3268 | 0.3043 |

## Interpretation of this contrast

The balanced arm does not establish an improvement in displacement magnitude. Its public one-second length RMSE rises while mean aligned trajectory ATE falls slightly. The validation distance gain is below the predeclared 2% threshold and aligned ATE slightly regresses, so teacher remains the selected reference.

Motion-category balance did not balance speed: many turn examples are themselves slow. Low-speed exposure rose rather than fell. Therefore this experiment does not test a successful removal of the low-speed bias, and does not refute the broader data-distribution hypothesis. A future speed-by-turn stratification would be a new experiment, with fixed thresholds and fresh independent long-flight validation before another public comparison.

## Verification and interpretation

- All five runs complete 10/10 flights, 12,000 frames each, without filled frames. Two new WASMs pass 80-frame checks. The original teacher and the rebuilt control have identical poses, twists, accuracy and per-frame fuel. The original archived teacher run also matches.
- Both complete public repeats reproduce accuracy and per-frame fuel exactly. The balanced model preserves every teacher orientation and angular twist; learned rotation residual remains disabled. Geometry and tracking behavior may still react to the changed translation predictions.
- Distance RMSE measures error in one-second net displacement lengths, without alignment. Direction and length-ratio summaries exclude true displacements below 2 cm and degenerate predicted directions. Windows overlap; they are descriptive, not 11,800 independent trials.
- Sim3 scoring removes one global scale per episode. Distance and trajectory scores can move in different directions; a constant scale change cannot improve Sim3 ATE.
- Timing was measured with all training stopped and public evaluations running serially. Wall time remains host-dependent; compare repeated-run ranges and deterministic fuel instead of claiming a hardware-independent speedup.
- Rebalancing repeated existing examples; it did not add scenes, longer trajectories, temporal memory or new motion information. The previously inspected one-scene public set is diagnostic, not fresh independent generalization evidence. No threshold, checkpoint rule or default was changed after seeing public results.

See `sampling-audit.json`, `training-verification.json`, `validation.json`, `selection.json`, `distance-diagnostics.json`, `runs/`, and `verification.json` for exact values. Raw traces and build sources remain in `results/data-balance-v9/` in WSL.

## Paired timing follow-up

The byte-identical teacher/control binary had substantially different serial wall times. To avoid interpreting host variability as a model speedup, a timing-only follow-up alternated which fixed artifact stepped first on each identical frame. All 12,000 pairs preserve prior outputs and per-frame fuel.

Pooled median step: teacher 0.1403 ms; balanced 0.1395 ms. Median per-episode balanced/teacher timing ratio: 0.9910. This is a warm-process host measurement; peak fuel remains the stronger reproducible compute comparison. See `paired-timing.json`.
