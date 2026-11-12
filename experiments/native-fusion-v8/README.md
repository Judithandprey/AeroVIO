# Native fusion V8: completed fixed ablation

Teacher reference: `0c4f778`. Branch: `experiment/native-fusion-v8`. No retraining, map, online bias estimator, or server upload.

Validation selected **rotation** before public evaluation. Source defaults and previous artifacts remain unchanged.

## Validation: 4 scene-disjoint 10-second flights

| Variant | Mean Sim(3) ATE (m) | Mean rotation RMSE (deg) | Max fuel/frame |
| --- | ---: | ---: | ---: |
| teacher | 0.221909 | 2.1705 | 2,587,891 |
| confidence | 0.235164 | 2.1705 | 2,588,854 |
| rotation | 0.221891 | 2.0031 | 2,589,216 |
| combined | 0.235149 | 2.0031 | 2,589,266 |

## Public: 10 fixed 60-second episodes

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Max fuel/frame | Median step (ms) |
| --- | ---: | ---: | ---: | ---: |
| teacher | 0.486329 | 12.1963 | 2,232,749 | 0.179 |
| control | 0.486329 | 12.1963 | 2,232,986 | 0.183 |
| confidence | 0.629407 | 12.1963 | 2,233,106 | 0.180 |
| rotation | 0.479744 | 11.4083 | 2,233,398 | 0.166 |
| combined | 0.624094 | 11.4083 | 2,233,518 | 0.131 |
| combined-repeat | 0.624094 | 11.4083 | 2,233,518 | 0.177 |
| rotation-repeat | 0.479744 | 11.4083 | 2,233,398 | 0.183 |

## Interpretation and boundaries

The validation-selected rotation changes public mean aligned ATE by -1.354%, median rotation error by -6.461%, peak fuel by +0.029%, and mean fuel by +1.385% relative to the preserved teacher artifact. Negative error changes indicate improvement.

Confidence weighting alone changes public mean aligned ATE by +29.420% while leaving orientation identical. It was rejected on validation before these scores were observed.

All runs finish completely with zero filled frames. The frozen selection is not recomputed from public scores. The confidence weighting is an uncalibrated heuristic, and its failure does not establish that uncertainty-aware fusion is generally ineffective.

The rotation correction jointly refines translation direction and a bounded rotation residual, but is gated on a successful translation-direction fit. It does not independently solve pure-rotation or weak-parallax periods. The learned translation magnitudes, original gyro integration and startup calibration remain frozen.

Capture interval / fitting delay / correction spread remain 8 / 4 / 8 frames (0.4 / 0.2 / 0.4 seconds at 20 Hz). Step time is local execution time and can vary with host load. Max fuel is deterministic computation, not milliseconds.

Sim(3) alignment removes global scale. The following diagnostics must accompany any mapping or physical-robot claims:

| Variant | Median unaligned ATE (m) | Median velocity RMSE (m/s) | Median Sim(3) scale |
| --- | ---: | ---: | ---: |
| teacher | 2.6188 | 0.2298 | 2.3760 |
| control | 2.6188 | 0.2298 | 2.3760 |
| confidence | 2.6695 | 0.2331 | 2.3668 |
| rotation | 2.6184 | 0.2298 | 2.3755 |
| combined | 2.6664 | 0.2331 | 2.3661 |

## Verification

- 119 existing host tests and 11 new C++ geometry checks pass; the geometry checks execute inside Wasmtime.
- All four new modules pass 80-frame ABI checks.
- Control rebuild has exactly the teacher poses, twists and metrics on validation and public data.
- Teacher public rerun matches its prior trajectories, twists, metrics and per-frame fuel exactly.
- Combined and validation-selected public repeats reproduce poses, twists, metrics and every frame fuel exactly.
- Validation confirms unchanged confidence-arm orientation, no changes before the first causal correction, and pose/twist consistency.

An initial preflight invoked training tests in the evaluation environment and stopped on missing PyTorch/ONNX. The runner was corrected to use the existing training environment before validation; no dependency was installed or test skipped. That failed preflight is retained under `results/native-fusion-v8-preflight-env/`.

See `../../docs/native-fusion-v8.md`, `selection.json`, `validation.json`, `comparison.json`, `verification.json`, `runs/`, and `windows-artifacts.json`. Raw traces and logs remain in ignored `results/native-fusion-v8/`.
