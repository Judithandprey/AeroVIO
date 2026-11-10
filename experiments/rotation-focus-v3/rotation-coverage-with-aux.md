# Cached rotation coverage

Computed from cached ground-truth rotations, not from public evaluation results.

| Split | Pairs | Rotation median / p95 (deg) | Axis-wise p95 absolute rotation (deg) | Residual median / p95 (deg) |
| --- | ---: | --- | --- | --- |
| real_train | 2388 | 0.0702 / 1.8296 | 0.1521, 1.8294, 0.1026 | 0.1117 / 0.2002 |
| real_validation | 796 | 0.0609 / 1.5892 | 0.1515, 1.5887, 0.0856 | 0.1124 / 0.2115 |
| aux_validation | 800 | 1.4849 / 2.8520 | 2.4721, 2.4820, 2.4385 | 0.1132 / 0.1986 |
| aux_train | 2400 | 1.4921 / 2.8311 | 2.4828, 2.4204, 2.3785 | 0.1084 / 0.1985 |

Axis order is camera x, y, z. Real rotations are reconstructed as Exp(IMU rotation) Exp(target residual). Synthetic rotations use their explicit labels and are checked against that residual convention.
Coverage is a dataset description; it does not establish that a particular data bias caused deployed drift.
