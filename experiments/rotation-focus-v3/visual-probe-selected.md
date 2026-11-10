# Rotation image-pair shuffle probe

Post-training, out-of-distribution image correspondence sensitivity; not standalone proof of visual usefulness and never checkpoint selection.

Whole image pairs are independently permuted with fixed seed 314162 while states and labels stay fixed. The two channels remain together.

| Dataset | Model | Paired rotation RMSE (deg) | Shuffled rotation RMSE (deg) | Shuffled minus paired (deg) | Prediction change RMS (deg) |
| --- | --- | ---: | ---: | ---: | ---: |
| real_validation | original | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | control | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | focus | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | imu-only | 0.105397 | 0.105397 | +0.000000 | 0.000000 |
| aux_validation | original | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | control | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | focus | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | imu-only | 0.581412 | 0.581412 | +0.000000 | 0.000000 |

## real_validation: original

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.096749 | 0.096018 | -0.000732 |
| camera_y | 455 | 0.105486 | 0.105260 | -0.000226 |
| camera_z | 68 | 0.136569 | 0.136243 | -0.000325 |
| [0,0.1) | 467 | 0.098756 | 0.098297 | -0.000458 |
| [0.1,0.5) | 200 | 0.116639 | 0.115938 | -0.000701 |
| [0.5,1) | 53 | 0.123282 | 0.123455 | +0.000172 |
| [1,2) | 76 | 0.102958 | 0.103361 | +0.000403 |

## real_validation: control

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.096749 | 0.096018 | -0.000732 |
| camera_y | 455 | 0.105486 | 0.105260 | -0.000226 |
| camera_z | 68 | 0.136569 | 0.136243 | -0.000325 |
| [0,0.1) | 467 | 0.098756 | 0.098297 | -0.000458 |
| [0.1,0.5) | 200 | 0.116639 | 0.115938 | -0.000701 |
| [0.5,1) | 53 | 0.123282 | 0.123455 | +0.000172 |
| [1,2) | 76 | 0.102958 | 0.103361 | +0.000403 |

## real_validation: focus

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.096749 | 0.096018 | -0.000732 |
| camera_y | 455 | 0.105486 | 0.105260 | -0.000226 |
| camera_z | 68 | 0.136569 | 0.136243 | -0.000325 |
| [0,0.1) | 467 | 0.098756 | 0.098297 | -0.000458 |
| [0.1,0.5) | 200 | 0.116639 | 0.115938 | -0.000701 |
| [0.5,1) | 53 | 0.123282 | 0.123455 | +0.000172 |
| [1,2) | 76 | 0.102958 | 0.103361 | +0.000403 |

## real_validation: imu-only

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.091655 | 0.091655 | +0.000000 |
| camera_y | 455 | 0.105716 | 0.105716 | +0.000000 |
| camera_z | 68 | 0.146729 | 0.146729 | +0.000000 |
| [0,0.1) | 467 | 0.095509 | 0.095509 | +0.000000 |
| [0.1,0.5) | 200 | 0.116027 | 0.116027 | +0.000000 |
| [0.5,1) | 53 | 0.131875 | 0.131875 | +0.000000 |
| [1,2) | 76 | 0.112876 | 0.112876 | +0.000000 |

## aux_validation: original

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.588455 | 0.587446 | -0.001009 |
| camera_y | 200 | 0.095982 | 0.096048 | +0.000066 |
| camera_z | 200 | 0.592786 | 0.592949 | +0.000163 |
| mixed | 200 | 0.494168 | 0.494332 | +0.000164 |
| [0,0.1) | 25 | 0.084123 | 0.083043 | -0.001080 |
| [0.1,0.5) | 114 | 0.161876 | 0.162605 | +0.000729 |
| [0.5,1) | 132 | 0.280656 | 0.280632 | -0.000024 |
| [1,2) | 280 | 0.455481 | 0.455507 | +0.000027 |
| [2,3) | 249 | 0.690052 | 0.689496 | -0.000557 |

## aux_validation: control

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.588455 | 0.587446 | -0.001009 |
| camera_y | 200 | 0.095982 | 0.096048 | +0.000066 |
| camera_z | 200 | 0.592786 | 0.592949 | +0.000163 |
| mixed | 200 | 0.494168 | 0.494332 | +0.000164 |
| [0,0.1) | 25 | 0.084123 | 0.083043 | -0.001080 |
| [0.1,0.5) | 114 | 0.161876 | 0.162605 | +0.000729 |
| [0.5,1) | 132 | 0.280656 | 0.280632 | -0.000024 |
| [1,2) | 280 | 0.455481 | 0.455507 | +0.000027 |
| [2,3) | 249 | 0.690052 | 0.689496 | -0.000557 |

## aux_validation: focus

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.588455 | 0.587446 | -0.001009 |
| camera_y | 200 | 0.095982 | 0.096048 | +0.000066 |
| camera_z | 200 | 0.592786 | 0.592949 | +0.000163 |
| mixed | 200 | 0.494168 | 0.494332 | +0.000164 |
| [0,0.1) | 25 | 0.084123 | 0.083043 | -0.001080 |
| [0.1,0.5) | 114 | 0.161876 | 0.162605 | +0.000729 |
| [0.5,1) | 132 | 0.280656 | 0.280632 | -0.000024 |
| [1,2) | 280 | 0.455481 | 0.455507 | +0.000027 |
| [2,3) | 249 | 0.690052 | 0.689496 | -0.000557 |

## aux_validation: imu-only

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.669354 | 0.669354 | +0.000000 |
| camera_y | 200 | 0.108036 | 0.108036 | +0.000000 |
| camera_z | 200 | 0.736637 | 0.736637 | +0.000000 |
| mixed | 200 | 0.591455 | 0.591455 | +0.000000 |
| [0,0.1) | 25 | 0.083494 | 0.083494 | +0.000000 |
| [0.1,0.5) | 114 | 0.177923 | 0.177923 | +0.000000 |
| [0.5,1) | 132 | 0.326381 | 0.326381 | +0.000000 |
| [1,2) | 280 | 0.540266 | 0.540266 | +0.000000 |
| [2,3) | 249 | 0.828361 | 0.828361 | +0.000000 |

Positive task-error changes indicate an advantage for correct image/state correspondence under this artificial intervention. Prediction change measures output sensitivity separately: unchanged task RMSE does not imply unchanged predictions. Shuffling can also mismatch camera intrinsics across episodes. Negative or small changes do not establish that vision is useless. This probe does not measure real-world generalization, trajectory drift, or image usefulness under a matched retrained control. No checkpoint or hyperparameter is selected here.
Error components in JSON are Log(R_gt.T R_pred) in current-camera channels. Source/cache/model/permutation hashes are retained there.
