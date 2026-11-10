# Offline rotation ablation diagnostics

Computed from saved WASM output poses and reference camera poses, after evaluation. Ground truth never enters the estimator.

| Variant | Median rotation RMSE (deg) | Median final error (deg) | Mean step-error rate RMS (deg/s) |
| --- | ---: | ---: | ---: |
| baseline | 13.24494 | 22.01097 | 2.44299 |
| current | 21.59307 | 36.37196 | 1.85306 |
| no-rotation | 26.78616 | 37.14321 | 2.48420 |
| imu-only | 15.42729 | 27.93957 | 1.72187 |
| imu-compact | 15.42729 | 27.93957 | 1.72187 |

## Accumulation over time

Median across episodes of orientation RMSE within each elapsed-time bucket. These are measured absolute errors, not sums of rotation vectors.

| Seconds | baseline | current | no-rotation | imu-only | imu-compact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–10 | 3.4768 | 4.8901 | 6.6461 | 3.7030 | 3.7030 |
| 10–20 | 8.5581 | 12.2878 | 15.6543 | 9.6885 | 9.6885 |
| 20–30 | 10.0096 | 16.6170 | 21.8979 | 11.2415 | 11.2415 |
| 30–40 | 13.4469 | 21.6942 | 29.3477 | 14.8105 | 14.8105 |
| 40–50 | 18.4054 | 28.5547 | 33.7390 | 17.8780 | 17.8780 |
| 50–60 | 20.5526 | 33.7721 | 35.0785 | 25.7251 | 25.7251 |

## Controlled residual toggle

Keeping the learned rotation residual gives median orientation RMSE 21.59307 degrees versus 26.78616 with it disabled. Residual ON wins 8/10 episodes; OFF wins 2/10. The bundle/model/source/metadata checks isolate this deployment toggle, and paired camera-local translations match within serialization tolerance. These results must determine whether the residual helps; a larger error than a different baseline does not establish that the residual caused it.

| Episode | On RMSE (deg) | Off RMSE (deg) | On final (deg) | Off final (deg) | Residual mean camera channels (deg/s) | Required mean camera channels (deg/s) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 0386e5c9 | 22.64316 | 25.74109 | 34.98393 | 35.04186 | +0.2166, +0.0600, +0.2396 | +0.7006, -0.4851, +0.2942 |
| 9b6a9279 | 32.87695 | 47.68293 | 37.75998 | 63.88469 | -0.0068, -0.4598, +0.2713 | +0.9064, -0.8119, +0.7836 |
| 6e8e3766 | 30.84355 | 30.87433 | 54.97090 | 53.53263 | -0.1045, -0.0143, -0.2404 | +0.9122, +0.8362, -0.1075 |
| f8fb89ff | 30.03538 | 39.49743 | 50.47501 | 59.24105 | -0.2576, -0.1474, -0.5980 | +0.2030, -0.9764, -0.9998 |
| 489ad518 | 16.09495 | 19.58121 | 25.78064 | 33.27411 | +0.2758, -0.0277, -0.0731 | +0.8668, -0.3736, -0.1790 |
| 51ea6311 | 20.54298 | 14.17478 | 38.10108 | 27.22383 | -0.4524, -0.2544, +0.0124 | -0.6113, +0.3706, -0.1148 |
| 60ed4323 | 11.97627 | 27.83122 | 10.35860 | 39.24457 | -0.2575, -0.6084, +0.5289 | -0.0180, -0.4979, +1.1544 |
| 8fff5c8d | 3.56064 | 6.62732 | 6.44106 | 10.41415 | +0.0167, -0.0442, +0.3848 | +0.0516, -0.1491, +0.2986 |
| 41e11443 | 44.29392 | 34.73339 | 73.45482 | 57.48367 | -0.6242, +0.2783, -0.5062 | -0.4706, -0.9496, -0.7855 |
| 8321da1f | 3.39527 | 18.79790 | 4.78127 | 23.09392 | -1.0584, +0.2041, -0.5114 | -1.0702, +0.2659, -0.7621 |

## Interpretation boundaries

- The uncentered cosine between recovered residual and reference-required correction ranges from 0.5423 to 0.7561. This measures per-frame directional agreement, not unbiasedness or successful cancellation of all drift components.
- The learned residual and the correction required by the reference are recovered from actual relative rotations. Sustained signed errors and their time evolution can diagnose a drift mechanism, but do not prove which training-distribution or label mechanism caused it.
- The pinned KLT baseline estimates gyro bias during an initial 0.5-second calibration when its rest test passes, and initializes its pose reference there. The learned runtime does neither. These are additional differences that may contribute to the remaining baseline gap; no matched calibration toggle was performed, so causation is unproven.
- Signed averages in fixed axes use the first camera separately for each episode. Camera-channel averages describe sensor-axis bias; they are not fixed-world cumulative rotations. No cross-episode vector is labelled as a universal world bias.
- Vector sums/coherence are first-order diagnostics, not exact composition on SO(3). Absolute pose errors and the controlled toggle establish the measured accumulated effect.
- World-position differences are expected when identical camera-local translations are composed through different rotations. This diagnostic does not attribute all trajectory error to rotation.
- All ten public episodes are post-hoc diagnostics of this fixed model, not new validation data for selecting correction parameters. No bias subtraction or estimator change is fitted here.

The JSON contains fixed-first-camera signed vectors, camera-channel vectors, 10-second buckets, per-file hashes and complete per-episode statistics.
