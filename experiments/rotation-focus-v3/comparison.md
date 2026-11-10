# Local public-set comparison

All runs use the same ten public episodes, pinned runner/runtime and saved limits.
Wall time is local step execution time, not end-to-end latency or the grading score.

| Variant | Completed | Mean Sim(3) ATE (m) | Max fuel/frame | Median step (ms) | Filled frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 10/10 | 2.041635 | 141,910,846 | 3.295 | 0 |
| current | 10/10 | 0.966622 | 14,078,994 | 1.075 | 0 |
| control-selected | 10/10 | 0.966622 | 14,083,612 | 1.072 | 0 |
| focus-selected | 10/10 | 0.966622 | 14,080,100 | 1.076 | 0 |
| control-final | 10/10 | 0.969556 | 14,079,972 | 1.070 | 0 |
| focus-final | 10/10 | 0.962323 | 14,081,340 | 1.077 | 0 |

## Additional accuracy metrics

Sim(3) alignment fits scale, rotation and translation. It does not measure metric-scale accuracy.
Values below are medians across episodes; rotation and unaligned ATE expose tradeoffs hidden by alignment.

| Variant | Unaligned ATE (m) | Rotation RMSE (deg) | 1 s translation RPE (m) | Velocity RMSE (m/s) | Alignment scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 41.4288 | 13.245 | 2.3801 | 2.5566 | 0.064 |
| current | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| control-selected | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| focus-selected | 2.9359 | 21.593 | 0.2593 | 0.2625 | 2.156 |
| control-final | 2.9138 | 21.483 | 0.2637 | 0.2671 | 2.365 |
| focus-final | 2.8968 | 24.084 | 0.2604 | 0.2632 | 2.387 |

| Variant | Rotation RMSE (deg) | 1 s rotation RPE (deg) | Angular velocity RMSE (rad/s) |
| --- | ---: | ---: | ---: |
| baseline | 13.245 | 0.889 | 0.12198 |
| current | 21.593 | 0.904 | 0.03226 |
| control-selected | 21.593 | 0.904 | 0.03226 |
| focus-selected | 21.593 | 0.904 | 0.03226 |
| control-final | 21.483 | 0.917 | 0.03124 |
| focus-final | 24.084 | 1.063 | 0.03695 |

## Per-episode Sim(3) ATE

| Episode | baseline | current | control-selected | focus-selected | control-final | focus-final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0386e5c9` | 2.099235 | 0.668308 | 0.668308 | 0.668308 | 0.728102 | 0.702387 |
| `9b6a9279` | 1.914120 | 0.558813 | 0.558813 | 0.558813 | 1.215707 | 1.277320 |
| `6e8e3766` | 2.499868 | 0.983021 | 0.983021 | 0.983021 | 0.825293 | 0.860485 |
| `f8fb89ff` | 1.928537 | 0.607244 | 0.607244 | 0.607244 | 0.700237 | 0.688452 |
| `489ad518` | 1.074172 | 0.636033 | 0.636033 | 0.636033 | 0.558091 | 0.565760 |
| `51ea6311` | 2.330615 | 1.521455 | 1.521455 | 1.521455 | 1.405684 | 1.429312 |
| `60ed4323` | 2.197604 | 1.552419 | 1.552419 | 1.552419 | 1.280476 | 1.239505 |
| `8fff5c8d` | 2.156139 | 1.058315 | 1.058315 | 1.058315 | 0.969443 | 0.966053 |
| `41e11443` | 2.619881 | 1.589927 | 1.589927 | 1.589927 | 1.576396 | 1.478634 |
| `8321da1f` | 1.596178 | 0.490689 | 0.490689 | 0.490689 | 0.436130 | 0.415321 |

These are pilot results from one training seed and a small independent synthetic training set.
They do not establish statistical significance, unseen-scene generalization, or course-server performance.
