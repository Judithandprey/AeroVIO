# Local public-set comparison

All runs use the same ten public episodes, pinned runner/runtime and saved limits.
Wall time is local step execution time, not end-to-end latency or the grading score.

| Variant | Completed | Mean Sim(3) ATE (m) | Max fuel/frame | Median step (ms) | Filled frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10/10 | 2.041635 | 141,910,846 | 3.211 | 0 |
| current | 10/10 | 0.966622 | 14,078,994 | 1.064 | 0 |
| no-rotation | 10/10 | 0.963882 | 13,994,827 | 1.053 | 0 |
| imu-only | 10/10 | 1.101235 | 14,162,855 | 1.067 | 0 |
| imu-compact | 10/10 | 1.101236 | 6,296,120 | 0.756 | 0 |

## Additional accuracy metrics

Sim(3) alignment fits scale, rotation and translation. It does not measure metric-scale accuracy.
Values below are medians across episodes; rotation and unaligned ATE expose tradeoffs hidden by alignment.

| Variant | Unaligned ATE (m) | Rotation RMSE (deg) | 1 s translation RPE (m) | Velocity RMSE (m/s) | Alignment scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 41.4288 | 13.245 | 2.3801 | 2.5566 | 0.064 |
| current | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| no-rotation | 2.9381 | 26.786 | 0.2594 | 0.2625 | 2.182 |
| imu-only | 3.1217 | 15.427 | 0.2491 | 0.2540 | 1.955 |
| imu-compact | 3.1217 | 15.427 | 0.2491 | 0.2540 | 1.955 |

| Variant | Rotation RMSE (deg) | 1 s rotation RPE (deg) | Angular velocity RMSE (rad/s) |
| --- | ---: | ---: | ---: |
| baseline | 13.245 | 0.889 | 0.12198 |
| current | 21.593 | 0.904 | 0.03226 |
| no-rotation | 26.786 | 1.351 | 0.04463 |
| imu-only | 15.427 | 0.917 | 0.03079 |
| imu-compact | 15.427 | 0.917 | 0.03079 |

## Per-episode Sim(3) ATE

| Episode | baseline | current | no-rotation | imu-only | imu-compact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0386e5c9` | 2.099235 | 0.668308 | 0.657448 | 0.632363 | 0.632363 |
| `9b6a9279` | 1.914120 | 0.558813 | 0.664757 | 1.750075 | 1.750075 |
| `6e8e3766` | 2.499868 | 0.983021 | 0.987945 | 0.834038 | 0.834038 |
| `f8fb89ff` | 1.928537 | 0.607244 | 0.605984 | 0.590022 | 0.590022 |
| `489ad518` | 1.074172 | 0.636033 | 0.640682 | 0.592216 | 0.592216 |
| `51ea6311` | 2.330615 | 1.521455 | 1.493789 | 1.575851 | 1.575851 |
| `60ed4323` | 2.197604 | 1.552419 | 1.641413 | 1.729789 | 1.729789 |
| `8fff5c8d` | 2.156139 | 1.058315 | 1.028794 | 1.114688 | 1.114688 |
| `41e11443` | 2.619881 | 1.589927 | 1.385136 | 1.693015 | 1.693015 |
| `8321da1f` | 1.596178 | 0.490689 | 0.532873 | 0.500298 | 0.500298 |

These are pilot results from one training seed and a small independent synthetic training set.
They do not establish statistical significance, unseen-scene generalization, or course-server performance.
