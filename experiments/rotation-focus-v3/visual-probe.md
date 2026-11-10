# Rotation image-pair shuffle probe

Post-training, out-of-distribution image correspondence sensitivity; not standalone proof of visual usefulness and never checkpoint selection.

Whole image pairs are independently permuted with fixed seed 314162 while states and labels stay fixed. The two channels remain together.

| Dataset | Model | Paired rotation RMSE (deg) | Shuffled rotation RMSE (deg) | Shuffled minus paired (deg) | Prediction change RMS (deg) |
| --- | --- | ---: | ---: | ---: | ---: |
| real_validation | original | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | control-selected | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | focus-selected | 0.105660 | 0.105266 | -0.000394 | 0.007393 |
| real_validation | control-epoch20-unselected | 0.106131 | 0.105874 | -0.000257 | 0.004569 |
| real_validation | focus-epoch20-unselected | 0.114949 | 0.114302 | -0.000647 | 0.005138 |
| real_validation | imu-only | 0.105397 | 0.105397 | +0.000000 | 0.000000 |
| aux_validation | original | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | control-selected | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | focus-selected | 0.487619 | 0.487409 | -0.000210 | 0.007075 |
| aux_validation | control-epoch20-unselected | 0.567904 | 0.567621 | -0.000283 | 0.008256 |
| aux_validation | focus-epoch20-unselected | 0.237649 | 0.237563 | -0.000086 | 0.005331 |
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

## real_validation: control-selected

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

## real_validation: focus-selected

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

## real_validation: control-epoch20-unselected

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.095157 | 0.094653 | -0.000504 |
| camera_y | 455 | 0.104229 | 0.103969 | -0.000260 |
| camera_z | 68 | 0.151024 | 0.151384 | +0.000360 |
| [0,0.1) | 467 | 0.096648 | 0.096302 | -0.000346 |
| [0.1,0.5) | 200 | 0.120737 | 0.120559 | -0.000177 |
| [0.5,1) | 53 | 0.129617 | 0.129518 | -0.000099 |
| [1,2) | 76 | 0.102462 | 0.102315 | -0.000147 |

## real_validation: focus-epoch20-unselected

dominant ground-truth camera-axis component; real motion may be mixed

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 273 | 0.108377 | 0.107383 | -0.000994 |
| camera_y | 455 | 0.117235 | 0.116825 | -0.000410 |
| camera_z | 68 | 0.124717 | 0.123777 | -0.000940 |
| [0,0.1) | 467 | 0.113772 | 0.113014 | -0.000758 |
| [0.1,0.5) | 200 | 0.115938 | 0.115360 | -0.000578 |
| [0.5,1) | 53 | 0.123092 | 0.122507 | -0.000585 |
| [1,2) | 76 | 0.113641 | 0.113441 | -0.000200 |

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

## aux_validation: control-selected

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

## aux_validation: focus-selected

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

## aux_validation: control-epoch20-unselected

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.650209 | 0.648804 | -0.001405 |
| camera_y | 200 | 0.091657 | 0.091353 | -0.000304 |
| camera_z | 200 | 0.726417 | 0.726586 | +0.000169 |
| mixed | 200 | 0.575503 | 0.575807 | +0.000304 |
| [0,0.1) | 25 | 0.082807 | 0.082499 | -0.000308 |
| [0.1,0.5) | 114 | 0.178710 | 0.179037 | +0.000326 |
| [0.5,1) | 132 | 0.322084 | 0.322463 | +0.000378 |
| [1,2) | 280 | 0.529648 | 0.529860 | +0.000213 |
| [2,3) | 249 | 0.806496 | 0.805588 | -0.000908 |

## aux_validation: focus-epoch20-unselected

synthetic labelled axis

| Group | Pairs | Paired RMSE (deg) | Shuffled RMSE (deg) | Change (deg) |
| --- | ---: | ---: | ---: | ---: |
| camera_x | 200 | 0.271193 | 0.270613 | -0.000580 |
| camera_y | 200 | 0.111554 | 0.111449 | -0.000105 |
| camera_z | 200 | 0.283500 | 0.283561 | +0.000061 |
| mixed | 200 | 0.244020 | 0.244305 | +0.000285 |
| [0,0.1) | 25 | 0.094832 | 0.093739 | -0.001092 |
| [0.1,0.5) | 114 | 0.124807 | 0.125521 | +0.000713 |
| [0.5,1) | 132 | 0.160657 | 0.160853 | +0.000196 |
| [1,2) | 280 | 0.224964 | 0.225000 | +0.000036 |
| [2,3) | 249 | 0.320664 | 0.320283 | -0.000381 |

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
