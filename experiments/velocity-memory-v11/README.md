# Explicit velocity memory V11

**Primary result: retain V10.** Both matched 18-input arms chose epoch zero under the frozen full original-validation pose-loss rule. The trained terminal epoch 40 models below are diagnostics, not selected replacements. No deployment default changes.

The network adds previous pre-visual predicted velocity and a validity flag: 18 -> 64 -> 6, 1,606 parameters (+256), 1,536 semantic dense MACs (+256), and 1 KiB additional FP32 weights. Teacher calibrated gyro and visual direction fusion are retained. The velocity is transported into the next camera frame; it excludes delayed visual correction.

Both arms use identical V10 initialization, zero added columns, 20-frame causal contexts, target-only supervision, original V10 joint target plan, 40 epochs and 1,520 optimizer steps. Feedback uses self-predictions, never true previous velocity. No-history sets all added inputs to zero. Compact fine-tuning and sequence training are common changes versus frozen V10; only feedback versus matched control isolates memory.

A full deterministic training repeat and exact replay reproduce all epoch metrics. The replay only retains the fixed terminal checkpoint after both primary selectors retained epoch zero; this diagnostic decision happened before long-flight scores, without a new fit setting or checkpoint sweep.

## Original validation selection

| Arm | Selected epoch | Initial pose loss | Fixed final pose loss | Final translation RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| no-history | 0 | 0.277686 | 0.316048 | 0.011651 |
| feedback | 0 | 0.277686 | 0.332766 | 0.012346 |

## Regression long flights

Six 60-second flights, 7,200 frames per variant. Regression reuses the V10 set; confirmation uses fresh seeds starting at 430000. Models and diagnostic epoch were fixed before scores. All variants receive identical inputs.

| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Low net-motion RMSE (m) | Mean fuel/frame | Peak fuel/frame | Median step (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0.519165 | 0.186428 | 0.087477 | 1232725.52 | 3124076 | excluded |
| v10 | 0.480920 | 0.141485 | 0.164825 | 1232739.61 | 3124076 | excluded |
| no-history-final | 0.507205 | 0.152487 | 0.156942 | 1235072.13 | 3126413 | excluded |
| feedback-final | 0.522959 | 0.156914 | 0.154937 | 1235105.95 | 3126463 | excluded |

Regression accuracy runs overlapped fresh-data generation. Their wall times are excluded; deterministic fuel and accuracy remain comparable. Final confirmation timings start after generation finishes.

Feedback final versus no-history-final (% change; negative error change means improvement): `{"mean_sim3_ate_m": 3.106, "pooled_length_rmse_m": 2.903, "low_net_motion_rmse_m": -1.277, "mean_fuel": 0.003, "peak_fuel": 0.002}`.

Feedback final versus v10 (% change; negative error change means improvement): `{"mean_sim3_ate_m": 8.741, "pooled_length_rmse_m": 10.906, "low_net_motion_rmse_m": -5.999, "mean_fuel": 0.192, "peak_fuel": 0.076}`.

Feedback final versus teacher (% change; negative error change means improvement): `{"mean_sim3_ate_m": 0.731, "pooled_length_rmse_m": -15.831, "low_net_motion_rmse_m": 77.117, "mean_fuel": 0.193, "peak_fuel": 0.076}`.

### True motion regimes

| Regime | Windows | Teacher RMSE | V10 RMSE | Final no-history RMSE | Final feedback RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary | 254 | 0.099107 | 0.191450 | 0.162442 | 0.160537 |
| slow_straight | 102 | 0.104435 | 0.169521 | 0.193567 | 0.189944 |
| decelerating | 206 | 0.052882 | 0.107942 | 0.108786 | 0.101894 |
| turning | 1521 | 0.127866 | 0.095065 | 0.116019 | 0.123210 |
| small_net_but_moving | 8 | 0.048975 | 0.098897 | 0.019834 | 0.036817 |

Small-net-motion means true one-second net displacement <0.1 m. It is not synonymous with stationary or low instantaneous speed. Regime counts describe this dataset, not independent trials; windows overlap.

| Variant | Stop events | Settled events | Mean post-stop displacement error (m) | Mean settling time among settled (s) |
| --- | ---: | ---: | ---: | ---: |
| teacher | 0 | 0 | n/a | n/a |
| v10 | 0 | 0 | n/a | n/a |
| no-history-final | 0 | 0 | n/a | n/a |
| feedback-final | 0 | 0 | n/a | n/a |

Missing regimes/events remain n/a; no conclusion about stopping recovery is supported when there are no stops. A stop requires at least 0.5 seconds below 0.02 m/s after recent >=0.1 m/s motion; displacement error covers up to the first second of the observed hold.

### Per-flight errors

| Scene / seed | Teacher length | V10 length | No-history final length | Feedback final length |
| --- | ---: | ---: | ---: | ---: |
| ProcTHOR-Train-5 / 420000 | 0.080894 | 0.113952 | 0.151149 | 0.169488 |
| ProcTHOR-Train-5 / 420001 | 0.260372 | 0.192192 | 0.169169 | 0.168430 |
| ProcTHOR-Train-6 / 420002 | 0.177153 | 0.121879 | 0.139460 | 0.138366 |
| ProcTHOR-Train-6 / 420003 | 0.201640 | 0.155545 | 0.188127 | 0.191650 |
| ProcTHOR-Train-7 / 420004 | 0.202705 | 0.142679 | 0.149333 | 0.151321 |
| ProcTHOR-Train-7 / 420005 | 0.145134 | 0.103820 | 0.104442 | 0.108925 |

## Confirmation long flights

Six 60-second flights, 7,200 frames per variant. Regression reuses the V10 set; confirmation uses fresh seeds starting at 430000. Models and diagnostic epoch were fixed before scores. All variants receive identical inputs.

| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Low net-motion RMSE (m) | Mean fuel/frame | Peak fuel/frame | Median step (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0.571272 | 0.220737 | 0.075371 | 1210236.51 | 3366112 | 0.170562 |
| v10 | 0.600712 | 0.167795 | 0.157649 | 1210238.55 | 3366112 | 0.171959 |
| no-history-final | 0.579938 | 0.160293 | 0.136720 | 1212564.17 | 3368449 | 0.180160 |
| feedback-final | 0.692283 | 0.168791 | 0.128124 | 1212621.11 | 3368499 | 0.181052 |

Feedback final versus no-history-final (% change; negative error change means improvement): `{"mean_sim3_ate_m": 19.372, "pooled_length_rmse_m": 5.302, "low_net_motion_rmse_m": -6.287, "mean_fuel": 0.005, "peak_fuel": 0.001}`.

Feedback final versus v10 (% change; negative error change means improvement): `{"mean_sim3_ate_m": 15.244, "pooled_length_rmse_m": 0.593, "low_net_motion_rmse_m": -18.728, "mean_fuel": 0.197, "peak_fuel": 0.071}`.

Feedback final versus teacher (% change; negative error change means improvement): `{"mean_sim3_ate_m": 21.183, "pooled_length_rmse_m": -23.533, "low_net_motion_rmse_m": 69.992, "mean_fuel": 0.197, "peak_fuel": 0.071}`.

### True motion regimes

| Regime | Windows | Teacher RMSE | V10 RMSE | Final no-history RMSE | Final feedback RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| stationary | 206 | 0.107168 | 0.183573 | 0.171171 | 0.155824 |
| slow_straight | 108 | 0.092718 | 0.162175 | 0.173404 | 0.144529 |
| decelerating | 443 | 0.051568 | 0.121253 | 0.101142 | 0.082191 |
| turning | 2113 | 0.156067 | 0.115551 | 0.125944 | 0.130753 |
| small_net_but_moving | 13 | 0.048999 | 0.103014 | 0.050030 | 0.066228 |

Small-net-motion means true one-second net displacement <0.1 m. It is not synonymous with stationary or low instantaneous speed. Regime counts describe this dataset, not independent trials; windows overlap.

| Variant | Stop events | Settled events | Mean post-stop displacement error (m) | Mean settling time among settled (s) |
| --- | ---: | ---: | ---: | ---: |
| teacher | 2 | 0 | 0.063939 | n/a |
| v10 | 2 | 0 | 0.095571 | n/a |
| no-history-final | 2 | 0 | 0.118099 | n/a |
| feedback-final | 2 | 0 | 0.123080 | n/a |

Missing regimes/events remain n/a; no conclusion about stopping recovery is supported when there are no stops. A stop requires at least 0.5 seconds below 0.02 m/s after recent >=0.1 m/s motion; displacement error covers up to the first second of the observed hold.

### Per-flight errors

| Scene / seed | Teacher length | V10 length | No-history final length | Feedback final length |
| --- | ---: | ---: | ---: | ---: |
| ProcTHOR-Train-5 / 430000 | 0.249633 | 0.214795 | 0.181842 | 0.198258 |
| ProcTHOR-Train-5 / 430001 | 0.216344 | 0.154592 | 0.139822 | 0.136798 |
| ProcTHOR-Train-6 / 430002 | 0.265527 | 0.176336 | 0.220656 | 0.242648 |
| ProcTHOR-Train-6 / 430003 | 0.243298 | 0.192960 | 0.163460 | 0.164854 |
| ProcTHOR-Train-7 / 430004 | 0.199268 | 0.139562 | 0.137327 | 0.131147 |
| ProcTHOR-Train-7 / 430005 | 0.117560 | 0.105310 | 0.085315 | 0.098326 |

## Selection and limits

```json
{
  "selected_variant": "v10",
  "reason": "Both full original-validation selectors retained epoch zero; terminal-epoch models remain diagnostic regardless of long scores.",
  "fixed_final_epoch_confirmation_guards": {
    "low_motion_vs_control": true,
    "low_motion_vs_v10": true,
    "low_motion_vs_teacher": false,
    "ate": false,
    "distance": true,
    "mean_fuel": true,
    "peak_fuel": true
  },
  "no_long_scores_used_for_checkpoint_selection": true
}
```

136 host tests pass. Diagnostic guest/ONNX motion error and causal memory checks are in runtime-parity.json. Selected epoch-zero guests match V10 on every evaluated frame. A separate complete feedback-final run reproduces all poses, twists and fuel. All orientations and angular twists equal teacher; linear twist agrees with final position increments.

Timing uses four distinct artifacts, all 24 execution orders, each artifact in each position 300 times per flight. Duplicate/repeat and selected-epoch checks run outside timing. Wall time is host-specific, while fuel is deterministic. No simulated sensor/render time is included in guest step time.

One training seed, short training trajectories, 20-frame training contexts and six fresh confirmation flights limit generalization. Scenes 5/6/7 are excluded from training but are not new scene identities relative to V10. Full validation and long evaluations are closed loop, rather than ground-truth-history evaluation. Training context truncation versus much longer deployment is a remaining possible source of drift.

Read docs/velocity-memory-v11.md for method, replay policy and reproduction. Raw logs/traces are retained in WSL results/velocity-memory-v11; raw fresh data is retained on D:. Teacher, V9, V10 and all prior experiments are preserved.
