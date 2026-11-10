# Public calibration replay and observable output audit

Calibration state is inferred by host replay of the exact deployed helper. Actual WASM internal diagnostics were not logged. Saved output prefixes and an independent first-correction calculation verify observable behavior separately.

The replay reads no RGB and uses no ground-truth fields. The upstream frames='none' storage reader loads metadata/IMU/reference arrays, but this script consults only timestamps, the initial IMU windows, and camera/IMU extrinsics.

| Episode | Host replay decision | Samples | Finalized frame | First permitted corrected frame | Largest corrected-rotation check error (Frobenius) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0386e5c9 | accepted | 200 | 19 | 20 | 3.96e-11 |
| 9b6a9279 | accepted | 200 | 19 | 20 | 3.49e-11 |
| 6e8e3766 | accepted | 200 | 19 | 20 | 3.78e-11 |
| f8fb89ff | accepted | 200 | 19 | 20 | 4.46e-11 |
| 489ad518 | accepted | 200 | 19 | 20 | 2.29e-11 |
| 51ea6311 | accepted | 200 | 19 | 20 | 2.67e-11 |
| 60ed4323 | accepted | 200 | 19 | 20 | 3.48e-11 |
| 8fff5c8d | accepted | 200 | 19 | 20 | 2.68e-11 |
| 41e11443 | accepted | 200 | 19 | 20 | 2.87e-11 |
| 8321da1f | accepted | 200 | 19 | 20 | 3.79e-11 |

Frame indices are zero-based. Both calibrated variants exactly reproduce their corresponding uncalibrated saved pose/twist prefixes through the inferred decision frame. For a rejection, the full saved output-file hashes must also match.

The first later rotation is checked against an independent ordered product of gyro-minus-bias exponentials and camera-frame conjugation. The unchanged learned residual is recovered from the actual current/gyro-only saved relative rotations. Comparisons use a fixed 2e-6 Frobenius tolerance for serialized quaternion precision; the JSON also states whether the difference from no correction is large enough to resolve.

This is a post-hoc behavior and provenance check. It does not directly observe a WASM internal bias variable, prove the rest assumption, establish bias accuracy, or select/tune a model.
