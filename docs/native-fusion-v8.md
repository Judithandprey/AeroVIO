# Native fusion V8: confidence and visual rotation ablation

This experiment extends teacher commit `0c4f778` on
`experiment/native-fusion-v8`. The teacher's Native 160 WASM is preserved as
the reference. Both new CMake switches default to `OFF`; prior Python models,
weights, experimental selections and upload artifacts remain unchanged.

## Estimators

All variants share the 1,350-parameter compact IMU checkpoint, startup gyro
calibration, 160x120 tracking, 48-corner budget, eight-frame capture interval,
and forward/backward Lucas-Kanade checks. There is no new training.

| Variant | Confidence weighting | Visual rotation correction |
| --- | --- | --- |
| Teacher reference | Original implementation | Original implementation |
| Control rebuild | Off | Off |
| Confidence | On | Off |
| Rotation | Off | On |
| Combined | On | On |

The control rebuild must reproduce all teacher poses and twists exactly. Its
fuel need not be identical because the experimental binary includes additional
state and a read-only diagnostics export.

### Confidence weighting

Accepted direction fits receive a bounded heuristic quality score based on
inlier fraction, two-dimensional feature spread, epipolar residual, backward
tracking error, photometric error, and derotated parallax. The teacher's weight
is multiplied by `0.35 + 0.65 * quality`. A carried direction also retains the
preceding fit's weight. This score is not a calibrated probability. Existing
fit rejection gates remain active.

### Visual rotation correction

After a successful direction fit with at least 12 tracks, optimize a small
three-axis rotation and two tangent coordinates of translation direction
together. The objective uses robust epipolar distances and a 0.01 rad gyro
prior. Four Gauss-Newton iterations use numerical Jacobians. The solver
rejects poorly observable rotations using the translation-marginalized
information matrix, excessive corrections, insufficient inliers, and less
than 5% improvement in the robust visual objective. The application gain is
0.25, with a maximum accepted pre-gain residual of 0.015 rad.

This is a bounded two-view residual experiment, not an MSCKF, joint online
IMU-bias estimator, or a full VIO optimizer. Translation flow is explicitly
modeled. The method still requires an accepted translation-direction fit;
pure rotation and weak-parallax periods do not independently trigger it.

The tracker and learned prior retain the original gyro coordinate system.
Accepted rotation increments are transformed from the stored anchor camera
axes into that fixed frame, accumulated in a separate visual alignment, and
applied causally to emitted orientation and displacement increments. Stored
images are not relabeled with a newer orientation. Pose-derived linear and
angular twist are recomputed after correction.

Capture remains every eight frames, fitting ends four frames after capture,
and correction is spread over eight frames. At 20 Hz these are 0.4 s, 0.2 s,
and 0.4 s. A sub-millisecond execution time does not remove this delay.

## Fixed evaluation protocol

`configs/native-fusion-v8.json` fixes the four arms and selection rules before
validation. All four original 10-second validation flights from scenes 5/6
run through the actual WASM. They are disjoint from training scenes 2/3/4 and
public scene 1, but they have been used in earlier experiments. This is a
small development validation set, not a new untouched generalization set.

Candidate eligibility requires nonregression of mean Sim(3) ATE and mean
rotation RMSE, at most 2% unaligned ATE regression, at most 50% greater peak
fuel, at least three episode ATE wins, and at least 2% improvement in either
mean aligned ATE or mean rotation. Eligible variants are ranked by aligned
ATE, then rotation. These are engineering guards, not statistical significance
tests. There is no parameter sweep or public-score-driven tuning in this round.

Selection is frozen before the ten public 60-second episodes. All candidates,
including rejected diagnostics, use the same upstream revision, data order,
resource limits, image adaptation, and serial execution. The combined variant
is repeated on both sets; the validation-selected candidate additionally
receives a full public repeat when it is a different variant.

The new C++ geometry tests execute in the same Wasmtime engine. They cover
known three-axis gyro errors with translation and varying depth, translation
without rotation, degenerate rays, track count, focal validity, degraded
tracking quality, and bounded behavior with outliers. Existing host tests run
in the training environment because they import PyTorch and ONNX.

## Reproduction

Run from the WSL checkout with its existing pinned dependencies, preserved
teacher WASM, and dataset. Choose a fresh name to retain the original evidence:

```bash
.venv/bin/python scripts/reproduce_native_fusion.py --name native-fusion-v8-repro1
```

The runner refuses to overwrite evidence. Original raw traces and WASMs are in
`results/native-fusion-v8/`; compact records are in
`experiments/native-fusion-v8/`. Reproductions use the chosen name in both
directories. The source and protocol hashes bind validation
to the subsequent public evaluation. Windows test artifacts are copied only
after complete report checks, then verified by SHA-256.

Diagnostics count accepted fits and summed weights. The legacy field name
`applied_rotation_norm_sum_rad` in validation JSON counts the norms scheduled
by accepted fits, before the eight-frame spread; it is not net orientation
change or proof that all scheduled increments completed before episode end.

Interpret Sim(3) ATE with rotation, unaligned position, velocity and scale.
Sim(3) removes global scale. Local public results do not establish the course
server's private-set score. Packaging is not submission.
