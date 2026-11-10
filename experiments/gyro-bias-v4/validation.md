# Gyro-bias factorial validation

Original-flight validation uses four simulated HyperDrone flights with translation and rotation; it is not physical robot/sensor data. Every cell runs actual host Solution.step twice. This is not a WASM speed or fuel measurement.

| Variant | Mean episode rotation RMSE (deg) | Mean Sim(3) ATE (m) | Mean unaligned position RMSE (m) | Rotation wins | Eligible change |
| --- | ---: | ---: | ---: | ---: | --- |
| current | 6.948942 | 0.265557 | 0.518700 | 0/4 | False |
| gyro-only | 7.639686 | 0.265585 | 0.521585 | 3/4 | False |
| calibrated-residual | 4.857039 | 0.265755 | 0.509897 | 3/4 | False |
| calibrated-gyro | 2.170470 | 0.265630 | 0.511332 | 4/4 | False |

Frozen selected variant: **current** (retain_current).

Acceptance requires at least 20% lower mean orientation RMSE, no increase in aligned ATE or unaligned position RMSE beyond the fixed 1e-6 tolerance, and orientation improvement in at least three flights. Eligible candidates are ordered by aligned ATE, then orientation error, then protocol order. The current model remains the fallback.

## Per-episode metrics

| Episode | Variant | Rotation RMSE (deg) | Sim(3) ATE (m) | Unaligned RMSE (m) | Calibration |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | current | 8.076363 | 0.174181 | 0.650602 | disabled |
| 1 | current | 7.724986 | 0.293704 | 0.421988 | disabled |
| 2 | current | 8.424469 | 0.230705 | 0.499815 | disabled |
| 3 | current | 3.569947 | 0.363640 | 0.502397 | disabled |
| 0 | gyro-only | 8.012925 | 0.174366 | 0.657886 | disabled |
| 1 | gyro-only | 7.058771 | 0.293465 | 0.419081 | disabled |
| 2 | gyro-only | 7.255550 | 0.230645 | 0.504357 | disabled |
| 3 | gyro-only | 8.231499 | 0.363862 | 0.505018 | disabled |
| 0 | calibrated-residual | 2.629552 | 0.175228 | 0.617977 | accepted |
| 1 | calibrated-residual | 5.641616 | 0.293262 | 0.426570 | accepted |
| 2 | calibrated-residual | 3.669240 | 0.230578 | 0.488159 | accepted |
| 3 | calibrated-residual | 7.487747 | 0.363953 | 0.506883 | accepted |
| 0 | calibrated-gyro | 1.790869 | 0.175529 | 0.625708 | accepted |
| 1 | calibrated-gyro | 2.555673 | 0.293008 | 0.422385 | accepted |
| 2 | calibrated-gyro | 2.167140 | 0.230524 | 0.493115 | accepted |
| 3 | calibrated-gyro | 2.168198 | 0.363460 | 0.504120 | accepted |

## Verification and interpretation

All four cells use the same ONNX, raw paired images, raw IMU-derived predictor state, and network outputs. Camera-local translation agrees to numerical tolerance; world positions can differ through rotation composition. Ground truth is read for scoring after the rollouts and never enters Solution.step.

All four complete rollouts repeat exactly in outputs, predictions and calibration states. Calibrated cells preserve every output through the activation frame; rejected calibration preserves the entire corresponding uncalibrated trajectory. The current cell also matches the historical original-flight orientation/position audit within the frozen numerical tolerance.

Low-variation/prior checks are not proof of rest. Initial calibration assumes the documented one-second hover; small true motion can contaminate the estimate. The factorial effects and their interaction are diagnostic for this fixed estimator, not proof of a sensor-bias mechanism or robustness to other startups.

Selection is frozen before any public comparison. Repetition establishes deterministic reproduction, not statistical significance. The JSON stores complete per-episode metrics, bias decisions, invariants, and hashes; raw full trajectories/traces are retained under the ignored results directory.
