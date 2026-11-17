# Structural-line rotation fallback V13

**Frozen reference: selected rotation-only V8.** Only the structural-line module changes; original checkpoint, IMU integration, point tracker and V8 correction remain fixed. No memory model, retraining, absolute room direction, metric width or trajectory-straightening prior is introduced.

Confirmation engineering guards all pass: **False**; rotation wins: **2/4**. This is a local simulated comparison, not automatic model promotion.

The three primary arms are the original frozen V8 artifact, detector-only, and gated bounded line correction. Detector-only must reproduce every baseline pose and twist. All correction repeats reproduce output, fuel and diagnostics exactly. Each 160x120 image uses the original green-channel downsampling and pixel-center calibration. Capture is every eight frames; solve is phase six, with bounded correction spread across two frames.

## Development

Four flights. Development uses original 10-second room 5/6 data; confirmation uses new 60-second room 8/9 data after the development freeze.

| Variant | Mean Sim3 ATE m | Mean raw ATE m | Mean rotation RMSE deg | Mean 1s rotation RPE deg | Max fuel/frame | Median step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.221891 | 0.429587 | 2.003087 | 0.629161 | 2589216 | excluded |
| detect | 0.221891 | 0.429587 | 2.003087 | 0.629161 | 16760164 | excluded |
| correct | 0.221891 | 0.429587 | 2.003087 | 0.629161 | 16760164 | excluded |

detect versus frozen V8, percent changes (negative errors are better): `{"mean_sim3_ate_m": 0.0, "mean_raw_ate_m": 0.0, "mean_rotation_rmse_deg": 0.0, "mean_rpe_rotation_1s_deg": 0.0, "mean_fuel": 144.5832699903896, "max_fuel": 547.3065205838369}`.

correct versus frozen V8, percent changes (negative errors are better): `{"mean_sim3_ate_m": 0.0, "mean_raw_ate_m": 0.0, "mean_rotation_rmse_deg": 0.0, "mean_rpe_rotation_1s_deg": 0.0, "mean_fuel": 144.5832699903896, "max_fuel": 547.3065205838369}`.

### Every trajectory

| Room / seed | Variant | Sim3 ATE m | Raw ATE m | Rotation RMSE deg | 1s rotation RPE deg | Max fuel | Peak frame / phase | Accepted / attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ProcTHOR-Train-5 / 271840 | baseline | 0.188695 | 0.489397 | 1.753202 | 0.597932 | 1858254 | 120 / 0 | n/a |
| ProcTHOR-Train-5 / 271840 | detect | 0.188695 | 0.489397 | 1.753202 | 0.597932 | 13345212 | 6 / 6 | 0/25 |
| ProcTHOR-Train-5 / 271840 | correct | 0.188695 | 0.489397 | 1.753202 | 0.597932 | 13345212 | 6 / 6 | 0/25 |
| ProcTHOR-Train-5 / 271841 | baseline | 0.258229 | 0.406300 | 2.382503 | 0.637785 | 2208969 | 186 / 2 | n/a |
| ProcTHOR-Train-5 / 271841 | detect | 0.258229 | 0.406300 | 2.382503 | 0.637785 | 14020264 | 182 / 6 | 0/25 |
| ProcTHOR-Train-5 / 271841 | correct | 0.258229 | 0.406300 | 2.382503 | 0.637785 | 14020264 | 182 / 6 | 0/25 |
| ProcTHOR-Train-6 / 271842 | baseline | 0.197184 | 0.292171 | 1.947856 | 0.559163 | 2589216 | 32 / 0 | n/a |
| ProcTHOR-Train-6 / 271842 | detect | 0.197184 | 0.292171 | 1.947856 | 0.559163 | 16760164 | 198 / 6 | 0/25 |
| ProcTHOR-Train-6 / 271842 | correct | 0.197184 | 0.292171 | 1.947856 | 0.559163 | 16760164 | 198 / 6 | 0/25 |
| ProcTHOR-Train-6 / 271843 | baseline | 0.243456 | 0.530479 | 1.928788 | 0.721762 | 2515738 | 170 / 2 | n/a |
| ProcTHOR-Train-6 / 271843 | detect | 0.243456 | 0.530479 | 1.928788 | 0.721762 | 16423149 | 46 / 6 | 0/25 |
| ProcTHOR-Train-6 / 271843 | correct | 0.243456 | 0.530479 | 1.928788 | 0.721762 | 16423149 | 46 / 6 | 0/25 |

### Gate and detection coverage

| Variant | Usable two-direction images / sampled | Accepted / attempted | Applied steps | Scheduled angle sum rad | Applied angle sum rad |
| --- | --- | --- | ---: | ---: | ---: |
| detect | 7/100 | 0/100 | 0 | 0.000000 | 0.000000 |

detect rejection reasons: `{"reject_weak": {"count": 28.0, "rate_per_decision": 0.28}, "reject_match": {"count": 0.0, "rate_per_decision": 0.0}, "reject_ambiguous": {"count": 0.0, "rate_per_decision": 0.0}, "reject_shape": {"count": 0.0, "rate_per_decision": 0.0}, "reject_residual": {"count": 0.0, "rate_per_decision": 0.0}, "reject_temporal": {"count": 2.0, "rate_per_decision": 0.02}, "reject_point_overlap": {"count": 70.0, "rate_per_decision": 0.7}, "reject_gap": {"count": 0.0, "rate_per_decision": 0.0}}`.
| correct | 7/100 | 0/100 | 0 | 0.000000 | 0.000000 |

correct rejection reasons: `{"reject_weak": {"count": 28.0, "rate_per_decision": 0.28}, "reject_match": {"count": 0.0, "rate_per_decision": 0.0}, "reject_ambiguous": {"count": 0.0, "rate_per_decision": 0.0}, "reject_shape": {"count": 0.0, "rate_per_decision": 0.0}, "reject_residual": {"count": 0.0, "rate_per_decision": 0.0}, "reject_temporal": {"count": 2.0, "rate_per_decision": 0.02}, "reject_point_overlap": {"count": 70.0, "rate_per_decision": 0.7}, "reject_gap": {"count": 0.0, "rate_per_decision": 0.0}}`.

Accepted/attempted counts are conditional on the sampling and gate sequence; samples overlap in scene content and are not statistically independent. A detector-only accepted opportunity is not an applied correction. Scheduled/applied angle sums are sums of magnitudes, not net orientation change.

### Negative cases and descriptive event checks

- ProcTHOR-Train-5 seed 271840: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +618.159%.
- ProcTHOR-Train-5 seed 271841: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +534.697%.
- ProcTHOR-Train-6 seed 271842: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +547.307%.
- ProcTHOR-Train-6 seed 271843: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +552.816%.

No accepted correction events occurred. This establishes rejection behavior and cost, not successful drift correction.

These post-hoc event windows are descriptive, not isolated causal proof. Full counters, phase fuel, all trajectory changes and accepted windows are retained in JSON; negative results are not filtered.

## Confirmation

Four flights. Development uses original 10-second room 5/6 data; confirmation uses new 60-second room 8/9 data after the development freeze.

| Variant | Mean Sim3 ATE m | Mean raw ATE m | Mean rotation RMSE deg | Mean 1s rotation RPE deg | Max fuel/frame | Median step ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.555830 | 1.601593 | 11.023601 | 0.795742 | 3215473 | 0.417089 |
| detect | 0.555830 | 1.601593 | 11.023601 | 0.795742 | 18505582 | 0.489067 |
| correct | 0.555803 | 1.601620 | 11.019874 | 0.795649 | 18505582 | 0.482811 |

detect versus frozen V8, percent changes (negative errors are better): `{"mean_sim3_ate_m": 0.0, "mean_raw_ate_m": 0.0, "mean_rotation_rmse_deg": 0.0, "mean_rpe_rotation_1s_deg": 0.0, "mean_fuel": 135.81441621310836, "max_fuel": 475.51663472216995}`.

correct versus frozen V8, percent changes (negative errors are better): `{"mean_sim3_ate_m": -0.0047589145450444015, "mean_raw_ate_m": 0.0017228248926803857, "mean_rotation_rmse_deg": -0.033808194232431976, "mean_rpe_rotation_1s_deg": -0.01174582437386551, "mean_fuel": 135.81493148844044, "max_fuel": 475.51663472216995}`.

### Every trajectory

| Room / seed | Variant | Sim3 ATE m | Raw ATE m | Rotation RMSE deg | 1s rotation RPE deg | Max fuel | Peak frame / phase | Accepted / attempts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ProcTHOR-Train-8 / 450000 | baseline | 0.579097 | 1.485974 | 9.488915 | 1.051859 | 3215473 | 746 / 2 | n/a |
| ProcTHOR-Train-8 / 450000 | detect | 0.579097 | 1.485974 | 9.488915 | 1.051859 | 18073482 | 814 / 6 | 0/150 |
| ProcTHOR-Train-8 / 450000 | correct | 0.579097 | 1.485974 | 9.488915 | 1.051859 | 18073482 | 814 / 6 | 0/150 |
| ProcTHOR-Train-8 / 450001 | baseline | 0.411542 | 1.091324 | 16.158511 | 0.677557 | 3175189 | 923 / 3 | n/a |
| ProcTHOR-Train-8 / 450001 | detect | 0.411542 | 1.091324 | 16.158511 | 0.677557 | 18505582 | 1174 / 6 | 0/150 |
| ProcTHOR-Train-8 / 450001 | correct | 0.411542 | 1.091324 | 16.158511 | 0.677557 | 18505582 | 1174 / 6 | 0/150 |
| ProcTHOR-Train-9 / 450002 | baseline | 0.429723 | 2.201691 | 10.963344 | 0.763579 | 1903761 | 360 / 0 | n/a |
| ProcTHOR-Train-9 / 450002 | detect | 0.429723 | 2.201691 | 10.963344 | 0.763579 | 14221665 | 174 / 6 | 1/150 |
| ProcTHOR-Train-9 / 450002 | correct | 0.429674 | 2.201820 | 10.960083 | 0.763589 | 14221665 | 174 / 6 | 1/150 |
| ProcTHOR-Train-9 / 450003 | baseline | 0.802957 | 1.627382 | 7.483633 | 0.689975 | 3127562 | 547 / 3 | n/a |
| ProcTHOR-Train-9 / 450003 | detect | 0.802957 | 1.627382 | 7.483633 | 0.689975 | 15098647 | 582 / 6 | 1/150 |
| ProcTHOR-Train-9 / 450003 | correct | 0.802900 | 1.627363 | 7.471986 | 0.689591 | 15098647 | 582 / 6 | 1/150 |

### Gate and detection coverage

| Variant | Usable two-direction images / sampled | Accepted / attempted | Applied steps | Scheduled angle sum rad | Applied angle sum rad |
| --- | --- | --- | ---: | ---: | ---: |
| detect | 96/600 | 2/600 | 0 | 0.000764 | 0.000000 |

detect rejection reasons: `{"reject_weak": {"count": 169.0, "rate_per_decision": 0.2816666666666667}, "reject_match": {"count": 3.0, "rate_per_decision": 0.005}, "reject_ambiguous": {"count": 0.0, "rate_per_decision": 0.0}, "reject_shape": {"count": 0.0, "rate_per_decision": 0.0}, "reject_residual": {"count": 1.0, "rate_per_decision": 0.0016666666666666668}, "reject_temporal": {"count": 29.0, "rate_per_decision": 0.04833333333333333}, "reject_point_overlap": {"count": 396.0, "rate_per_decision": 0.66}, "reject_gap": {"count": 0.0, "rate_per_decision": 0.0}}`.
| correct | 96/600 | 2/600 | 4 | 0.000764 | 0.000764 |

correct rejection reasons: `{"reject_weak": {"count": 169.0, "rate_per_decision": 0.2816666666666667}, "reject_match": {"count": 3.0, "rate_per_decision": 0.005}, "reject_ambiguous": {"count": 0.0, "rate_per_decision": 0.0}, "reject_shape": {"count": 0.0, "rate_per_decision": 0.0}, "reject_residual": {"count": 1.0, "rate_per_decision": 0.0016666666666666668}, "reject_temporal": {"count": 29.0, "rate_per_decision": 0.04833333333333333}, "reject_point_overlap": {"count": 396.0, "rate_per_decision": 0.66}, "reject_gap": {"count": 0.0, "rate_per_decision": 0.0}}`.

Accepted/attempted counts are conditional on the sampling and gate sequence; samples overlap in scene content and are not statistically independent. A detector-only accepted opportunity is not an applied correction. Scheduled/applied angle sums are sums of magnitudes, not net orientation change.

### Negative cases and descriptive event checks

- ProcTHOR-Train-8 seed 450000: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +462.078%.
- ProcTHOR-Train-8 seed 450001: rotation win=False, rotation change +0.000000 deg, 1s RPE change +0.000000 deg, raw ATE change +0.000000 m, peak fuel change +482.818%.
- ProcTHOR-Train-9 seed 450002: rotation win=True, rotation change -0.003261 deg, 1s RPE change +0.000010 deg, raw ATE change +0.000129 m, peak fuel change +647.030%.
- ProcTHOR-Train-9 seed 450003: rotation win=True, rotation change -0.011647 deg, 1s RPE change -0.000384 deg, raw ATE change -0.000019 m, peak fuel change +382.761%.

Accepted two-frame windows with the largest changes in corrected-minus-baseline absolute rotation error (positive means worse):
- Seed 450002, decision frame 902: error-gap change -0.004324 deg; 8 lines / 2 directions, residual 0.004630 rad.
- Seed 450003, decision frame 518: error-gap change -0.012483 deg; 10 lines / 2 directions, residual 0.003013 rad.

These post-hoc event windows are descriptive, not isolated causal proof. Full counters, phase fuel, all trajectory changes and accepted windows are retained in JSON; negative results are not filtered.

## Fixed guards

```json
{
  "reference_retained": "frozen V8 rotation-only",
  "main_or_default_changed": false,
  "server_uploaded": false,
  "engineering_guards": {
    "rotation": false,
    "raw_ate": true,
    "sim3_ate": true,
    "peak_fuel": false,
    "rotation_episode_wins": true
  },
  "all_guards_pass": false,
  "rotation_episode_wins": 2,
  "evaluated_confirmation_episodes": 4,
  "rule": "No automatic promotion; fixed engineering guards are not significance tests."
}
```

## Verification and interpretation limits

Twenty-four primary trace and diagnostic-log hashes verify. All eight complete correction repeats match output, per-frame fuel and diagnostic values. Detector-only output exactly matches frozen V8 on all flights. An optional disabled-module control is checked on the first development flight. Twist consistency is verified from emitted poses. Original weights and frozen inference/evaluation sources match their recorded hashes.

All six execution orders are balanced; 200-frame development flights differ by at most one execution-position count and 1200-frame confirmation flights are exactly balanced. Diagnostic reads, repeated flights and control checks do not enter step timing. Development wall times are excluded. Confirmation timing is valid only after data generation and builds have ended; milliseconds are host-specific WASM step costs, not camera latency. Maximum per-frame fuel, rather than average CPU time, is the compute guard.

Rotation RMSE and 1-second relative rotation errors are unaligned SO(3) metrics; the report converts radians to degrees. Sim3 ATE aligns positions with global rotation, translation and scale, so it cannot stand alone as evidence of corrected yaw, metric scale or raw trajectory improvement. Raw ATE and relative rotation are reported alongside it.

Structural directions are local and sign-ambiguous. Vertical lines alone do not constrain yaw. The module does not know a global room axis, wall width or metric scale, and does not force a straight flight path. Static furniture edges can form convincing direction groups; there is no semantic wall classifier. Oblique walls, axis swaps, room changes, low texture and repeated structure remain failure modes. Pure-rotation/weak-parallax benefit must be measured rather than assumed.

Only four fresh flights in two new rooms are used. Development data were used by earlier experiments. Guard thresholds and acceptance counts are engineering checks, not significance tests or real-robot validation. No main/default replacement or course-server upload occurs.

## Reproduction

From the WSL checkout with pinned dependencies and fresh output directories:

```bash
.venv/bin/python scripts/evaluate_structural_lines.py --phase development
# Finish new data generation/builds; preserve frozen sources and artifacts.
.venv/bin/python scripts/evaluate_structural_lines.py --phase confirmation
.venv/bin/python scripts/report_structural_lines.py
```

The evaluator refuses to overwrite a phase. Preserve this completed evidence and use a new experiment namespace for a rerun. `artifacts.json` identifies the exact three WASMs; the configuration, manifests, source freeze, traces and decision logs bind the comparison to its inputs. Raw traces/logs remain in `results/structural-lines-v13/`; compact evidence is archived here.
