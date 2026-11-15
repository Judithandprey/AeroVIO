# Frozen-predictor memory gate V12

**Reference retained: V10. No main merge or deployment-default change.** This experiment adds only a learned scalar gate between frozen V10 and V11 predictors. The comparison uses independently selected updated gate candidates, not a claim that either is a validated replacement.

The adaptive 24 -> 8 -> 1 sigmoid gate has 209 trainable parameters. Its inputs are normalized IMU/intrinsics/dt/history plus both frozen candidate velocities. It blends pre-visual translation and feeds back the resulting velocity in the correct camera axes. Both predictor weights, calibrated gyro, visual tracking/fusion and memory validity/clip/reset rules remain unchanged. Total network parameters are 3,165, with 3,016 semantic dense MACs. No zero-velocity update, stationary classifier, new visual input or new training data is added.

The constant-mixture control trains only one sigmoid bias. Both arms start at g=0.5 and use the same target sampling plan, 20-frame causal contexts, self-predicted history, loss, 40 epochs and 1,520 updates. Only gate parameters enter the optimizer. The raw rotation residual used by the training warp stays V10 for all mixtures; deployed orientation always uses the unchanged gyro.

## Original validation and gate behavior

| Gate | Selected updated epoch | Initial pose loss | Best updated pose loss | Mean gate | Gate standard deviation |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 0.278171 | 0.278540 | 0.509373 | 0.000000 |
| adaptive | 1 | 0.278171 | 0.279215 | 0.526078 | 0.004975 |

Hard V10 endpoint validation pose loss: 0.277686. The trained adaptive candidate beats V10 on this selection metric: False; beats its initial 50/50 mixture: False.

As predeclared before training, candidate checkpoints are selected among updated epochs 1..40. Initial and endpoint references are reported separately and V10 remains available. A best updated checkpoint is not necessarily better than initialization. Gate statistics above describe original validation, not independent evidence of turn/stop recognition. No epoch is selected using long-flight metrics.

## Regression set

Six 60-second flights, 7,200 frames per artifact. Regression uses the six earlier V11 flights; confirmation uses fresh seeds starting at 440000 in the same scene identities 5/6/7. All variants receive identical data and settings.

| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Turning length RMSE (m) | Cruise length RMSE (m) | Stop displacement error (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.600712 | 0.167795 | 0.115551 | 0.207946 | 0.095571 |
| v11 | 0.692283 | 0.168791 | 0.130753 | 0.202840 | 0.123080 |
| constant | 0.545626 | 0.165608 | 0.119655 | 0.201485 | 0.081228 |
| adaptive | 0.545311 | 0.165952 | 0.121176 | 0.201501 | 0.081858 |

Coverage: 2113 turning windows, 862 cruise windows, 206 stationary windows, 2 stop events. Windows overlap and are not independent trials.

| Variant | Stops settled | False-stop runs while moving | Cruise length bias (m) | Stationary length RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| v10 | 0 / 2 | 0 | -0.118700 | 0.183573 |
| v11 | 0 / 2 | 0 | -0.157037 | 0.155824 |
| constant | 0 / 2 | 0 | -0.137224 | 0.156490 |
| adaptive | 0 / 2 | 0 | -0.137973 | 0.155844 |

Cruise means one-second true-motion windows with mean speed >=0.1 m/s, CV <=0.1, speed range <=0.05 m/s, path efficiency >=0.95 and mean angular rate <15 deg/s. False-stop runs are tested on all true-moving intervals >=0.1 m/s, requiring five consecutive predicted increments below 0.02 m/s. These are evaluation-only truth labels.

| Variant | Mean fuel/frame | Peak fuel/frame | Median guest step (ms) |
| --- | ---: | ---: | ---: |
| v10 | 1210238.55 | 3366112 | excluded |
| v11 | 1212621.11 | 3368499 | excluded |
| constant | 1217542.94 | 3373433 | excluded |
| adaptive | 1217742.02 | 3373627 | excluded |

Regression overlapped simulation; its wall times are excluded. Fuel and outputs remain deterministic.

Adaptive versus v10, percentage change (negative error change is better): `{"mean_sim3_ate_m": -9.223, "pooled_length_rmse_m": -1.099, "mean_fuel": 0.62, "turning_error": 4.868, "cruise_error": -3.099, "stop_error": -14.349}`.

Adaptive versus v11, percentage change (negative error change is better): `{"mean_sim3_ate_m": -21.23, "pooled_length_rmse_m": -1.682, "mean_fuel": 0.422, "turning_error": -7.325, "cruise_error": -0.66, "stop_error": -33.492}`.

Adaptive versus constant, percentage change (negative error change is better): `{"mean_sim3_ate_m": -0.058, "pooled_length_rmse_m": 0.207, "mean_fuel": 0.016, "turning_error": 1.271, "cruise_error": 0.008, "stop_error": 0.776}`.

### Individual stops

| Seed / start frame | V10 | V11 | Constant | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| 430004 / 744 | 0.131749 | 0.124002 | 0.127550 | 0.127301 |
| 430004 / 1105 | 0.059394 | 0.122157 | 0.034905 | 0.036414 |

Stop errors are net displacement errors over up to one second of the observed hold. Stops require >=0.5 seconds below 0.02 m/s after recent >=0.1 m/s movement. No settled event means no settling-time estimate; do not call a reduced drift error successful stopping.

## Confirmation set

Six 60-second flights, 7,200 frames per artifact. Regression uses the six earlier V11 flights; confirmation uses fresh seeds starting at 440000 in the same scene identities 5/6/7. All variants receive identical data and settings.

| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Turning length RMSE (m) | Cruise length RMSE (m) | Stop displacement error (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v10 | 0.493972 | 0.141926 | 0.093519 | 0.154836 | n/a |
| v11 | 0.652062 | 0.168497 | 0.123020 | 0.211336 | n/a |
| constant | 0.467923 | 0.151463 | 0.115388 | 0.175185 | n/a |
| adaptive | 0.469452 | 0.152184 | 0.116942 | 0.176222 | n/a |

Coverage: 1920 turning windows, 1558 cruise windows, 247 stationary windows, 0 stop events. Windows overlap and are not independent trials.

| Variant | Stops settled | False-stop runs while moving | Cruise length bias (m) | Stationary length RMSE (m) |
| --- | ---: | ---: | ---: | ---: |
| v10 | 0 / 0 | 0 | -0.112685 | 0.193587 |
| v11 | 0 / 0 | 0 | -0.175712 | 0.131095 |
| constant | 0 / 0 | 0 | -0.146213 | 0.155034 |
| adaptive | 0 / 0 | 0 | -0.147422 | 0.153629 |

Cruise means one-second true-motion windows with mean speed >=0.1 m/s, CV <=0.1, speed range <=0.05 m/s, path efficiency >=0.95 and mean angular rate <15 deg/s. False-stop runs are tested on all true-moving intervals >=0.1 m/s, requiring five consecutive predicted increments below 0.02 m/s. These are evaluation-only truth labels.

| Variant | Mean fuel/frame | Peak fuel/frame | Median guest step (ms) |
| --- | ---: | ---: | ---: |
| v10 | 1159508.99 | 3152311 | 0.296443 |
| v11 | 1161892.93 | 3154698 | 0.302235 |
| constant | 1166822.58 | 3159632 | 0.302249 |
| adaptive | 1167016.24 | 3159826 | 0.301182 |

Adaptive versus v10, percentage change (negative error change is better): `{"mean_sim3_ate_m": -4.964, "pooled_length_rmse_m": 7.228, "mean_fuel": 0.647, "turning_error": 25.046, "cruise_error": 13.812, "stop_error": null}`.

Adaptive versus v11, percentage change (negative error change is better): `{"mean_sim3_ate_m": -28.005, "pooled_length_rmse_m": -9.682, "mean_fuel": 0.441, "turning_error": -4.941, "cruise_error": -16.615, "stop_error": null}`.

Adaptive versus constant, percentage change (negative error change is better): `{"mean_sim3_ate_m": 0.327, "pooled_length_rmse_m": 0.476, "mean_fuel": 0.017, "turning_error": 1.347, "cruise_error": 0.592, "stop_error": null}`.

### Individual stops

| Seed / start frame | V10 | V11 | Constant | Adaptive |
| --- | ---: | ---: | ---: | ---: |
| No qualifying events | n/a | n/a | n/a | n/a |

Stop errors are net displacement errors over up to one second of the observed hold. Stops require >=0.5 seconds below 0.02 m/s after recent >=0.1 m/s movement. No settled event means no settling-time estimate; do not call a reduced drift error successful stopping.

## Frozen confirmation guards

```json
{
  "reference_retained": "v10",
  "main_or_default_changed": false,
  "confirmation_guards": {
    "turning": false,
    "stopping": null,
    "cruise_vs_v10": false,
    "cruise_vs_v11": true,
    "ate": true,
    "overall_distance": false,
    "mean_fuel": true,
    "peak_fuel": true
  },
  "stop_events": 0,
  "minimum_stop_events": 5,
  "candidate_original_validation_beats_v10": false,
  "candidate_original_validation_beats_initial_mixture": false,
  "no_long_scores_used_for_checkpoint_selection": true
}
```

Fewer than five new stop events makes that guard inconclusive, even if the measured mean improves. Empty bins are not scored as zero. The earlier stopping audit found only one qualifying transition in the training set; this gate-only experiment deliberately keeps that data unchanged.

## Verification and limits

143 host tests pass, including frozen-parameter optimization, endpoint blending, history gradients, resets, and cruise/false-stop definitions. Five 80-frame WASM checks and independent C++/ONNX gate/input/memory parity pass. A complete same-seed training repeat matches weights and every epoch metric. Native hard-zero and hard-one endpoint guests reproduce V10 and V11 outputs on a full flight. Complete adaptive repeats match every output and per-frame fuel on all twelve long flights. Forty-eight primary trace hashes and all frozen sources/artifacts are verified.

Fresh wall timing begins only after generation, training, compilation and regression complete. Every model occupies every execution position 300 times per flight. Endpoint/repeat/diagnostic runs are excluded from timing. Reported milliseconds are host-specific guest-step times and exclude simulation/rendering. No course-server submission or real-robot test occurred.

One training seed, the original short training data, and these simulated flights limit generalization. Both mixture candidates have independently optimized validation-selected weights; differences versus the constant control are not a perfectly mean-matched gate ablation. Additional predictors raise neural MACs, while vision dominates much of the system cost. No independent stationary detector or new visual confidence feature was added.

See `docs/memory-gate-v12.md` for the fixed protocol, `runtime-parity.json` for numerical checks, and `artifacts.json` for exact WASM identities. Raw data is retained on D: and logs/traces in WSL `results/memory-gate-v12`. All prior models and branches remain preserved.
