# Local public-set comparison

All runs use the same ten public episodes, pinned runner/runtime and saved limits.
Wall time is local step execution time, not end-to-end latency or the grading score.

| Variant | Completed | Mean Sim(3) ATE (m) | Max fuel/frame | Median step (ms) | Filled frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10/10 | 2.041635 | 141,910,846 | 3.212 | 0 |
| current | 10/10 | 0.966622 | 14,090,709 | 1.063 | 0 |
| gyro-only | 10/10 | 0.963882 | 13,998,029 | 1.061 | 0 |
| calibrated-residual | 10/10 | 0.904211 | 14,830,095 | 1.231 | 0 |
| calibrated-gyro | 10/10 | 0.881818 | 14,757,111 | 1.241 | 0 |

## Additional accuracy metrics

Sim(3) alignment fits scale, rotation and translation. It does not measure metric-scale accuracy.
Values below are medians across episodes; rotation and unaligned ATE expose tradeoffs hidden by alignment.

| Variant | Unaligned ATE (m) | Rotation RMSE (deg) | 1 s translation RPE (m) | Velocity RMSE (m/s) | Alignment scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 41.4288 | 13.245 | 2.3801 | 2.5566 | 0.064 |
| current | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| gyro-only | 2.9381 | 26.786 | 0.2594 | 0.2625 | 2.182 |
| calibrated-residual | 3.0170 | 18.756 | 0.2593 | 0.2625 | 2.126 |
| calibrated-gyro | 2.9814 | 12.196 | 0.2594 | 0.2625 | 2.147 |

| Variant | Rotation RMSE (deg) | 1 s rotation RPE (deg) | Angular velocity RMSE (rad/s) |
| --- | ---: | ---: | ---: |
| baseline | 13.245 | 0.889 | 0.12198 |
| current | 21.593 | 0.904 | 0.03226 |
| gyro-only | 26.786 | 1.351 | 0.04463 |
| calibrated-residual | 18.756 | 0.898 | 0.03212 |
| calibrated-gyro | 12.196 | 0.717 | 0.04050 |

## Per-episode Sim(3) ATE

| Episode | baseline | current | gyro-only | calibrated-residual | calibrated-gyro |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0386e5c9` | 2.099235 | 0.668308 | 0.657448 | 0.605121 | 0.593050 |
| `9b6a9279` | 1.914120 | 0.558813 | 0.664757 | 0.592461 | 0.737880 |
| `6e8e3766` | 2.499868 | 0.983021 | 0.987945 | 0.765671 | 0.793569 |
| `f8fb89ff` | 1.928537 | 0.607244 | 0.605984 | 0.505070 | 0.461704 |
| `489ad518` | 1.074172 | 0.636033 | 0.640682 | 0.724474 | 0.696202 |
| `51ea6311` | 2.330615 | 1.521455 | 1.493789 | 1.492181 | 1.460322 |
| `60ed4323` | 2.197604 | 1.552419 | 1.641413 | 1.535209 | 1.576406 |
| `8fff5c8d` | 2.156139 | 1.058315 | 1.028794 | 1.097589 | 1.066013 |
| `41e11443` | 2.619881 | 1.589927 | 1.385136 | 1.158540 | 0.933556 |
| `8321da1f` | 1.596178 | 0.490689 | 0.532873 | 0.565792 | 0.499479 |

These are pilot results from one training seed and a small independent synthetic training set.
They do not establish statistical significance, unseen-scene generalization, or course-server performance.
