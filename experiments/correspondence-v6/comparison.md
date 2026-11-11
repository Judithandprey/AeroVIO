# Explicit correspondence supervision: completed public comparison

All seven serial runs completed all ten episodes (12,000 frames each), without failures or filled outputs. Previous artifacts are rerun unchanged. V5 temporal and V6 candidates share the same guest code and the same ONNX graph structure apart from initializer values.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Max fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.041635 | 13.2449 | 3.102 | 141,910,846 |
| calibrated-gyro | 0.881818 | 12.1963 | 1.212 | 14,757,111 |
| v5-single-frame | 0.855547 | 9.0634 | 1.456 | 20,141,128 |
| v5-temporal | 0.870938 | 9.7654 | 1.561 | 21,854,083 |
| continuation | 0.864435 | 10.6038 | 1.516 | 21,874,382 |
| supervised | 0.868606 | 10.6784 | 1.506 | 21,874,382 |
| supervised-repeat | 0.868606 | 10.6784 | 1.517 | 21,874,382 |

Step time is host-specific WASM execution time, not end-to-end latency. Sim(3) fits scale and does not prove metric-map accuracy.

| Variant | Mean rotation RMSE (deg) | Median unaligned ATE (m) | Median 1 s rotation RPE (deg) | Median angular velocity RMSE (rad/s) | Median fitted scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 19.4864 | 41.4288 | 0.8886 | 0.121979 | 0.064 |
| calibrated-gyro | 14.1203 | 2.9814 | 0.7174 | 0.040502 | 2.147 |
| v5-single-frame | 12.0750 | 2.9263 | 0.7416 | 0.030983 | 2.134 |
| v5-temporal | 12.7369 | 2.9265 | 0.6878 | 0.034834 | 2.135 |
| continuation | 12.5112 | 2.9424 | 0.6884 | 0.033679 | 2.142 |
| supervised | 12.8759 | 2.9018 | 0.7246 | 0.032787 | 2.139 |
| supervised-repeat | 12.8759 | 2.9018 | 0.7246 | 0.032787 | 2.139 |

## Supervised candidate versus references

| Reference | Aligned ATE change | Median rotation change | Step-time change | Max-fuel change | Rotation wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | -57.46% | -19.38% | -51.45% | -84.59% | 6/10 |
| calibrated-gyro | -1.50% | -12.45% | +24.25% | +48.23% | 6/10 |
| v5-single-frame | +1.53% | +17.82% | +3.46% | +8.61% | 3/10 |
| v5-temporal | -0.27% | +9.35% | -3.52% | +0.09% | 6/10 |
| continuation | +0.48% | +0.70% | -0.63% | +0.00% | 4/10 |

## Per-episode rotation and aligned trajectory error

| Episode | baseline | calibrated-gyro | v5-single-frame | v5-temporal | continuation | supervised |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0386e5c9 | 26.980 deg / 2.099 m | 10.106 deg / 0.593 m | 6.663 deg / 0.579 m | 9.704 deg / 0.579 m | 7.523 deg / 0.577 m | 9.330 deg / 0.580 m |
| 9b6a9279 | 62.598 deg / 1.914 m | 39.435 deg / 0.738 m | 31.420 deg / 0.660 m | 34.946 deg / 0.680 m | 30.901 deg / 0.635 m | 33.352 deg / 0.681 m |
| 6e8e3766 | 27.978 deg / 2.500 m | 7.763 deg / 0.794 m | 11.669 deg / 0.802 m | 9.144 deg / 0.821 m | 11.085 deg / 0.818 m | 14.223 deg / 0.817 m |
| f8fb89ff | 8.255 deg / 1.929 m | 12.441 deg / 0.462 m | 7.270 deg / 0.454 m | 4.197 deg / 0.456 m | 7.187 deg / 0.457 m | 6.499 deg / 0.456 m |
| 489ad518 | 22.166 deg / 1.074 m | 7.969 deg / 0.696 m | 7.197 deg / 0.693 m | 9.826 deg / 0.722 m | 10.836 deg / 0.734 m | 8.167 deg / 0.692 m |
| 51ea6311 | 8.101 deg / 2.331 m | 5.868 deg / 1.460 m | 7.163 deg / 1.458 m | 9.596 deg / 1.458 m | 10.582 deg / 1.463 m | 8.467 deg / 1.460 m |
| 60ed4323 | 6.692 deg / 2.198 m | 15.004 deg / 1.576 m | 16.664 deg / 1.499 m | 17.671 deg / 1.537 m | 19.709 deg / 1.552 m | 15.810 deg / 1.525 m |
| 8fff5c8d | 7.021 deg / 2.156 m | 17.781 deg / 1.066 m | 14.577 deg / 1.038 m | 11.329 deg / 1.055 m | 10.392 deg / 1.052 m | 12.860 deg / 1.065 m |
| 41e11443 | 6.839 deg / 2.620 m | 11.951 deg / 0.934 m | 7.907 deg / 0.878 m | 8.015 deg / 0.886 m | 6.271 deg / 0.857 m | 8.024 deg / 0.888 m |
| 8321da1f | 18.235 deg / 1.596 m | 12.886 deg / 0.499 m | 10.220 deg / 0.495 m | 12.939 deg / 0.515 m | 10.626 deg / 0.499 m | 12.026 deg / 0.522 m |

The complete supervised repeat reproduces pose, twist, accuracy and per-frame fuel exactly. Every preserved reference reproduces its v5 results exactly; wall times are separately measured. Selection was frozen before these diagnostics and is not changed by public results. One training seed and one public scene do not establish statistical or physical-robot performance.
