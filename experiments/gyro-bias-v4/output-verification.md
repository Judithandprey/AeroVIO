# Paired calibration effects and output verification

All three uncalibrated controls reproduce earlier pose/twist files and accuracy exactly.
The selected `current` artifact reproduces all ten pose/twist files, accuracy and per-frame fuel exactly.
The unselected calibrated-gyro diagnostic also reproduces all ten pose/twist files, accuracy and per-frame fuel exactly.
Wall time is measured separately and need not repeat exactly.

| Episode | Rotation change with residual ON (deg) | Rotation change with residual OFF (deg) | Difference of changes (deg) |
| --- | ---: | ---: | ---: |
| 0386e5c9 | -10.806723 | -15.635012 | 4.828289 |
| 9b6a9279 | -8.733704 | -8.247998 | -0.485706 |
| 6e8e3766 | -23.753132 | -23.111452 | -0.641679 |
| f8fb89ff | -8.183249 | -27.055959 | 18.872710 |
| 489ad518 | -8.659540 | -11.612362 | 2.952822 |
| 51ea6311 | -8.834650 | -8.307094 | -0.527556 |
| 60ed4323 | 17.077183 | -12.827539 | 29.904722 |
| 8fff5c8d | 12.098338 | 11.153274 | 0.945064 |
| 41e11443 | -20.241923 | -22.782312 | 2.540389 |
| 8321da1f | 19.419110 | -5.912336 | 25.331446 |

Effects are calibrated minus uncalibrated error; negative helps. Interaction is descriptive, not proof of a double-compensation mechanism.

Full repeated public outputs establish deterministic reproduction, not independent-startup or physical-robot generalization.

Maximum camera-local translation discrepancy across four cells is 2.35e-08 m, within saved-pose serialization tolerance.
