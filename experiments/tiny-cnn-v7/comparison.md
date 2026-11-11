# Tiny CNN reproduction: local comparison

All eight runs completed 10/10 public episodes and 12,000 frames with no filled outputs. The runner, runtime, episode hashes/order, input adaptation, limits and host environment match. The new guests retain the upstream implementation byte-for-byte; old WASMs remain unchanged. These local scores do not reproduce the server's different 20-episode evaluation.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median WASM step (ms) | Max fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| upstream | 1.095712 | 16.0765 | 1.171 | 14,680,072 |
| matched | 1.171630 | 17.5217 | 1.163 | 14,685,424 |
| v1-pose | 0.991676 | 21.1772 | 1.163 | 14,073,824 |
| v4-current | 0.966622 | 21.5931 | 1.156 | 14,090,709 |
| calibrated-gyro | 0.881818 | 12.1963 | 1.285 | 14,757,111 |
| v5-single | 0.855547 | 9.0634 | 1.531 | 20,141,128 |
| v6-supervised | 0.868606 | 10.6784 | 1.620 | 21,874,382 |
| matched-repeat | 1.171630 | 17.5217 | 1.123 | 14,685,424 |

`upstream`: original 10-epoch, seed-0 trainer, final checkpoint, on our twelve training flights. `matched`: same flights and exact image pairs as AeroVIO, seed 17, 40 epochs, batch 64, Adam/cosine schedule, clipping and validation-selection policy matched to v1. It retains upstream 9-value IMU features, normalization, targets and inference. The latter is a controlled-budget reproduction, not the unmodified training recipe. V5/V6 received additional later training; this is not a one-factor architecture ablation.

Wall time measures host-specific WASM step execution, not end-to-end sensor latency. Fuel is the portable scoring measure. Sim(3) fits global scale; it does not establish metric-map accuracy.

| Variant | Median unaligned ATE (m) | Mean rotation RMSE (deg) | Median velocity RMSE (m/s) | Median alignment scale |
| --- | ---: | ---: | ---: | ---: |
| upstream | 3.1506 | 18.6049 | 0.2916 | 2.5227 |
| matched | 3.4105 | 20.4856 | 0.3015 | 2.6890 |
| v1-pose | 2.9847 | 21.3641 | 0.2601 | 2.0004 |
| v4-current | 2.9359 | 21.6263 | 0.2625 | 2.1563 |
| calibrated-gyro | 2.9814 | 14.1203 | 0.2625 | 2.1471 |
| v5-single | 2.9263 | 12.0750 | 0.2625 | 2.1339 |
| v6-supervised | 2.9018 | 12.8759 | 0.2625 | 2.1387 |
| matched-repeat | 3.4105 | 20.4856 | 0.3015 | 2.6890 |

## Per-episode aligned ATE

| Episode | upstream | matched | v1-pose | v4-current | calibrated-gyro | v5-single | v6-supervised | matched-repeat |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0386e5c9 | 0.814583 | 0.859263 | 0.663516 | 0.668308 | 0.593050 | 0.578559 | 0.580133 | 0.859263 |
| 9b6a9279 | 1.328235 | 1.533872 | 0.507827 | 0.558813 | 0.737880 | 0.659513 | 0.681364 | 1.533872 |
| 6e8e3766 | 1.567105 | 1.624262 | 1.070139 | 0.983021 | 0.793569 | 0.802454 | 0.817203 | 1.624262 |
| f8fb89ff | 0.525120 | 0.626765 | 0.601327 | 0.607244 | 0.461704 | 0.453516 | 0.455774 | 0.626765 |
| 489ad518 | 0.480201 | 0.456856 | 0.672651 | 0.636033 | 0.696202 | 0.692597 | 0.692093 | 0.456856 |
| 51ea6311 | 1.490067 | 1.501544 | 1.551510 | 1.521455 | 1.460322 | 1.457674 | 1.459821 | 1.501544 |
| 60ed4323 | 1.839802 | 1.937168 | 1.593067 | 1.552419 | 1.576406 | 1.499400 | 1.525425 | 1.937168 |
| 8fff5c8d | 0.801545 | 0.879295 | 1.095699 | 1.058315 | 1.066013 | 1.038361 | 1.064731 | 0.879295 |
| 41e11443 | 1.453678 | 1.659185 | 1.627516 | 1.589927 | 0.933556 | 0.878205 | 0.887877 | 1.659185 |
| 8321da1f | 0.656782 | 0.638090 | 0.533511 | 0.490689 | 0.499479 | 0.495189 | 0.521639 | 0.638090 |

The full matched repetition and all preserved-reference reruns match trajectory, twist, accuracy and per-frame fuel exactly. No default or past acceptance decision changes. One seed per recipe and one public scene do not establish generalization or explain the server performance gap.
