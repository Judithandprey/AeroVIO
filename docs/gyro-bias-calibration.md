# Causal startup gyro calibration

The v4 experiment tests one fixed startup calibration rule with the existing
warp-trained motion model. It changes neither the network nor its training.
The rule and validation acceptance criteria were fixed in
[`gyro-bias-v4.json`](../configs/gyro-bias-v4.json) before candidate evaluation.
Original simulated-flight validation retains `current`. Calibrated gyro-only
rotation improves mean orientation RMSE by 68.77% across all four flights, but
increases aligned ATE by 0.02752%, failing the frozen nonregression guard. The
[validation report](../experiments/gyro-bias-v4/validation.md) records the complete
four-cell comparison and exact repeats. All seven public runs completed, and
current/calibrated-gyro reproduce complete poses, twists, accuracy and fuel.
The [public report](../experiments/gyro-bias-v4/README.md) presents the improved
diagnostic result together with the unchanged validation decision and tradeoffs.

## What a gyro mean can and cannot tell us

A measured angular rate contains true rotation, sensor bias and noise:

```text
measured_gyro = true_angular_rate + bias + noise
```

Taking its mean estimates bias only when the average true angular rate is
negligible. A low standard deviation does not establish this: slow constant
rotation also has low variation. A mean guard based on the documented bias/noise
range rejects some implausibly large offsets, but cannot separate a small
constant rotation from bias. This calibration therefore assumes the generator's
documented initial hover. It is not a general stationary-state detector.

Chinese hints: bias = 传感器零偏; hover = 悬停，不等于完全静止;
causal = 只使用当前和过去观测.

## Why the full initial second is used

The pinned HyperDrone task holds the first waypoint for 200 steps at 200 Hz.
It still runs its controller and physical dynamics during that hold; it does
not freeze the pose. The startup audit inspected only the original simulated
training and validation flights from scenes 2–6. It did not inspect public
episodes or measure a physical sensor.

The observed startup contains an oscillatory settling motion. Ground truth was
used only to diagnose that motion, never as an online calibration input:

| Diagnostic window, measured from simulator time zero | IMU samples | Net orientation change | Accumulated angular path |
| --- | ---: | ---: | ---: |
| First 0.5 s | 100 | 0.3264° | 2.9007° |
| Original baseline's actual calibration window, about 0–0.55 s | 110 | 0.4172° | 3.0088° |
| Complete documented hold, about 0–1.0 s | 200 | 0.0927° | 3.6097° |

All 16 generated flights share this deterministic startup motion; different
scenes and sensor-noise seeds do not make these 16 independent motion patterns.
The complete hold averages more of the transient rather than assuming that its
second half is perfectly stationary. Sources, timing and per-flight statistics
are preserved in [the startup audit](../experiments/gyro-bias-v4/startup-audit.json).
These diagnostics motivate the fixed window; they are not VIO accuracy results.

## Fixed online rule

The guest receives frame dimensions, camera intrinsics and `T_cam_imu`. It does
not receive an IMU rate, frame rate, hover duration or sensor-noise metadata.
The one-second window and sensor priors are therefore explicit experimental
constants derived from the pinned generator, not hidden values read at runtime.

The first call must contain at least two distinct IMU timestamps. The anchor is
the first timestamp minus the first positive observed sample interval. The
calibrator collects unique observations in that initial one-second interval,
using the shared 1 microsecond boundary tolerance. It requires at least 180
samples spanning at least 0.9 seconds, then applies these fixed gates:

- Accelerometer standard-deviation vector norm is below 1.0 m/s².
- Gyroscope standard-deviation vector norm is below 0.2 rad/s.
- Each body/IMU-axis gyro mean satisfies
  `abs(mean) <= 0.02 + 3 * 0.07 / sqrt(sample_count)` rad/s.

The first two limits are the official baseline's low-variation checks. The last
uses the stored simulator's initial gyro-bias range and per-sample noise standard
deviation. At 200 samples its limit is approximately 0.03485 rad/s per axis.
This is a bounded engineering guard, not a proof of rest or a universal sensor
confidence interval. Failure freezes rejection; the window is not extended or
retried. Success freezes the sample-mean estimate for the remaining episode.

On the audited data, timestamps start at 0.005 s and their first interval is
0.005 s, giving anchor zero. Frame index 19 arrives at 0.999999978 s and closes
the window with 200 samples under the existing tolerance. That frame is emitted
with the previous calibration state. Correction starts on the next complete
interval, whose first IMU sample is at 1.004999978 s and whose frame is at
1.049999977 s. There is no future-data access or retrospective pose correction.

## What changes in rotation composition

The network still receives the original paired images and raw IMU-derived
14-value state. Its ONNX predictions, including camera-local translation,
remain identical across the four experimental cells. A separate integration
path subtracts the frozen bias in IMU coordinates before each sample exponential:

```text
R_imu_corrected = product(exp((gyro_i - estimated_bias) * dt_i))
R_camera = R_cam_imu @ R_imu_corrected @ transpose(R_cam_imu)
R_relative = R_camera @ exp(predicted_rotation_residual)  # residual enabled
```

The companion variant omits the learned residual. This four-cell comparison
tests bias correction on/off against learned residual on/off. The residual was
trained with raw IMU inputs, so applying both corrections could compensate the
same error twice; their interaction must be measured. World positions can also
change because identical local translations are composed using different
orientations. The sparse map stays disabled.

## Difference from the original KLT initialization

The frozen KLT baseline accumulates about half a second of samples and emits an
identity, zero-motion state while waiting. At initialization it estimates
gravity direction from mean acceleration, initializes its world orientation,
sets its reference camera pose and, if its low-variation gate passes, installs
the mean gyro as bias. With the stored frame/sample timing, this closes near
0.55 seconds rather than exactly 0.5 seconds.

The new optional calibrator continuously emits the learned estimator's normal
states. It does not hold output, realign gravity, reset pose, restart the model,
or replay earlier observations. It changes only future rotation integration
after the collection window closes. Calibration remains disabled by default,
and the original KLT source and earlier experiment records remain unchanged.

This is one bounded experiment. A fixed bias cannot follow later sensor random
walk, and the initialization assumption may fail in other datasets or physical
systems. The protocol requires original simulated-flight validation and an
exact repeat before selecting a candidate; public scores cannot retune the rule.

## Sources

- [Frozen protocol](../configs/gyro-bias-v4.json).
- [Archived baseline calibration implementation](../experiments/gyro-bias-v4/startup-audit-sources/baseline_solution.py).
- [Guest ABI and stored sensor priors](../experiments/gyro-bias-v4/startup-audit-sources/abi_config.py).
- [Generator hold specification](../experiments/gyro-bias-v4/startup-audit-sources/generated_training_world.h).
- [Controller and hold progression](../experiments/gyro-bias-v4/startup-audit-sources/tracking_operations_cpu.h).
- [Source snapshot hashes](../experiments/gyro-bias-v4/startup-audit-sources/manifest.json).
