# Feature-correlation v5: completed public comparison

All six serial runs completed all ten public episodes (12,000 frames each), without failures or filled outputs. The preserved baseline/current/calibrated-gyro WASM files were rerun unchanged. Both new branches use the same new guest source and pinned runtime/limits. Reference payloads retain their original source; the extension's default compatibility was verified separately.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Max fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.041635 | 13.2449 | 3.340 | 141,910,846 |
| current | 0.966622 | 21.5931 | 1.084 | 14,090,709 |
| calibrated-gyro | 0.881818 | 12.1963 | 1.199 | 14,757,111 |
| single-frame | 0.855547 | 9.0634 | 1.440 | 20,141,128 |
| temporal | 0.870938 | 9.7654 | 1.522 | 21,854,083 |
| temporal-repeat | 0.870938 | 9.7654 | 1.543 | 21,854,083 |

Step time is host-specific WASM execution time, not complete sensor-to-output latency. Fuel is reported separately. Sim(3) alignment removes scale and does not establish metric-map accuracy.

| Variant | Mean rotation RMSE (deg) | Median unaligned ATE (m) | Median 1 s rotation RPE (deg) | Median angular-velocity RMSE (rad/s) | Median fitted scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 19.4864 | 41.4288 | 0.8886 | 0.121979 | 0.064 |
| current | 21.6263 | 2.9359 | 0.9039 | 0.032257 | 2.156 |
| calibrated-gyro | 14.1203 | 2.9814 | 0.7174 | 0.040502 | 2.147 |
| single-frame | 12.0750 | 2.9263 | 0.7416 | 0.030983 | 2.134 |
| temporal | 12.7369 | 2.9265 | 0.6878 | 0.034834 | 2.135 |
| temporal-repeat | 12.7369 | 2.9265 | 0.6878 | 0.034834 | 2.135 |

## Temporal candidate versus references

| Reference | Aligned ATE change | Median rotation change | Step-time change | Max-fuel change | Rotation wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | -57.34% | -26.27% | -54.43% | -84.60% | 6/10 |
| current | -9.90% | -54.78% | +40.36% | +55.10% | 6/10 |
| calibrated-gyro | -1.23% | -19.93% | +26.91% | +48.09% | 5/10 |
| single-frame | +1.80% | +7.75% | +5.68% | +8.50% | 3/10 |

## Per-episode rotation and aligned position

| Episode | baseline | current | calibrated-gyro | single-frame | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0386e5c9 | 26.980 deg / 2.099 m | 22.643 deg / 0.668 m | 10.106 deg / 0.593 m | 6.663 deg / 0.579 m | 9.704 deg / 0.579 m |
| 9b6a9279 | 62.598 deg / 1.914 m | 32.877 deg / 0.559 m | 39.435 deg / 0.738 m | 31.420 deg / 0.660 m | 34.946 deg / 0.680 m |
| 6e8e3766 | 27.978 deg / 2.500 m | 30.844 deg / 0.983 m | 7.763 deg / 0.794 m | 11.669 deg / 0.802 m | 9.144 deg / 0.821 m |
| f8fb89ff | 8.255 deg / 1.929 m | 30.035 deg / 0.607 m | 12.441 deg / 0.462 m | 7.270 deg / 0.454 m | 4.197 deg / 0.456 m |
| 489ad518 | 22.166 deg / 1.074 m | 16.095 deg / 0.636 m | 7.969 deg / 0.696 m | 7.197 deg / 0.693 m | 9.826 deg / 0.722 m |
| 51ea6311 | 8.101 deg / 2.331 m | 20.543 deg / 1.521 m | 5.868 deg / 1.460 m | 7.163 deg / 1.458 m | 9.596 deg / 1.458 m |
| 60ed4323 | 6.692 deg / 2.198 m | 11.976 deg / 1.552 m | 15.004 deg / 1.576 m | 16.664 deg / 1.499 m | 17.671 deg / 1.537 m |
| 8fff5c8d | 7.021 deg / 2.156 m | 3.561 deg / 1.058 m | 17.781 deg / 1.066 m | 14.577 deg / 1.038 m | 11.329 deg / 1.055 m |
| 41e11443 | 6.839 deg / 2.620 m | 44.294 deg / 1.590 m | 11.951 deg / 0.934 m | 7.907 deg / 0.878 m | 8.015 deg / 0.886 m |
| 8321da1f | 18.235 deg / 1.596 m | 3.395 deg / 0.491 m | 12.886 deg / 0.499 m | 10.220 deg / 0.495 m | 12.939 deg / 0.515 m |

The temporal repeat reproduces all trajectories, twists, accuracy and per-frame fuel exactly. The three preserved references also reproduce their v4 outputs/fuel exactly. Timing is measured separately and need not match.

Validation had already rejected both trained branches; public diagnostics do not override that decision. The v5 reference remains calibrated-gyro, itself an unselected v4 diagnostic; v4's selected current model and workspace pose-only default are unchanged. One training seed, four short validation flights and one public scene do not establish generalization or robot performance.
