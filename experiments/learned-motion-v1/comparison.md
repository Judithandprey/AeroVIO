# Local public-set comparison

All runs use the same ten public episodes, pinned runner/runtime and saved limits.
Wall time is local step execution time, not end-to-end latency or the grading score.

| Variant | Completed | Mean Sim(3) ATE (m) | Max fuel/frame | Median step (ms) | Filled frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10/10 | 2.041635 | 141,910,846 | 3.342 | 0 |
| motion-cnn | 10/10 | 0.991676 | 14,073,824 | 1.058 | 0 |
| cnn-warp | 10/10 | 0.966622 | 14,073,824 | 1.062 | 0 |
| cnn-warp-map | 10/10 | 1.122139 | 157,635,808 | 5.139 | 0 |

## Additional accuracy metrics

Sim(3) alignment fits scale, rotation and translation. It does not measure metric-scale accuracy.
Values below are medians across episodes; rotation and unaligned ATE expose tradeoffs hidden by alignment.

| Variant | Unaligned ATE (m) | Rotation RMSE (deg) | 1 s translation RPE (m) | Velocity RMSE (m/s) | Alignment scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 41.4288 | 13.245 | 2.3801 | 2.5566 | 0.064 |
| motion-cnn | 2.9847 | 21.177 | 0.2567 | 0.2601 | 2.000 |
| cnn-warp | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| cnn-warp-map | 3.1319 | 17.390 | 0.2624 | 0.2743 | 2.295 |

## Per-episode Sim(3) ATE

| Episode | baseline | motion-cnn | cnn-warp | cnn-warp-map |
| --- | ---: | ---: | ---: | ---: |
| `0386e5c9` | 2.099235 | 0.663516 | 0.668308 | 0.893825 |
| `9b6a9279` | 1.914120 | 0.507827 | 0.558813 | 0.673021 |
| `6e8e3766` | 2.499868 | 1.070139 | 0.983021 | 0.987704 |
| `f8fb89ff` | 1.928537 | 0.601327 | 0.607244 | 0.651127 |
| `489ad518` | 1.074172 | 0.672651 | 0.636033 | 0.575297 |
| `51ea6311` | 2.330615 | 1.551510 | 1.521455 | 2.435783 |
| `60ed4323` | 2.197604 | 1.593067 | 1.552419 | 1.697088 |
| `8fff5c8d` | 2.156139 | 1.095699 | 1.058315 | 1.167344 |
| `41e11443` | 2.619881 | 1.627516 | 1.589927 | 1.094467 |
| `8321da1f` | 1.596178 | 0.533511 | 0.490689 | 1.045730 |

These are pilot results from one training seed and a small independent synthetic training set.
They do not establish statistical significance, unseen-scene generalization, or course-server performance.
