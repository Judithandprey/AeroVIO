# Feature correlation v5

The small temporal matching branch is implemented, trained and evaluated, but
does **not** qualify to replace the preserved calibrated-gyro reference. Its
public rotation/position metrics improve over that reference, while the matched
single-frame control performs better on those aggregate metrics. Both branches
worsen original-flight validation rotation. This trial does not establish an
incremental accuracy benefit from temporal correspondence.

See the [complete comparison](comparison.md), [figure](comparison.png),
[architecture and reproduction](../../docs/feature-correlation.md),
[frozen protocol](../../configs/feature-correlation-v5.json),
[validation](validation.json) and [pre-public decision](selection.json).

## Measured public results

All six serial runs completed 10/10 episodes, 12,000 frames each, without failures
or filled outputs. These are local simulated public-set results.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median WASM step (ms) | Max fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| Official KLT + IMU | 2.041635 | 13.2449 | 3.340 | 141,910,846 |
| Preserved current learning model | 0.966622 | 21.5931 | 1.084 | 14,090,709 |
| Preserved calibrated gyro | 0.881818 | 12.1963 | 1.199 | 14,757,111 |
| Single-frame control | 0.855547 | 9.0634 | 1.440 | 20,141,128 |
| Temporal correlation | 0.870938 | 9.7654 | 1.522 | 21,854,083 |
| Temporal repeat | 0.870938 | 9.7654 | 1.543 | 21,854,083 |

Against calibrated gyro, temporal correlation reduces aligned ATE by 1.23% and
median rotation RMSE by 19.93%, while increasing step time by 26.91% and max fuel
by 48.09%. Rotation improves in 5/10 episodes. Its mean episode rotation RMSE
also decreases, from 14.1203 to 12.7369 degrees. Median unaligned ATE improves
1.84%, while fitted scale remains inaccurate (see the full comparison).

Against the single-frame control, temporal correlation has 1.80% higher aligned
ATE and 7.75% higher median rotation RMSE, and takes 5.68% longer. It wins rotation
in 3/10 episodes and aligned ATE in 0/10. Its median 1-second rotation RPE is
better (0.6878 versus 0.7416 degrees), so the variants are not identical across
all motion metrics. The aggregate pose results do not support attributing the
public improvement to previous/current matching.

Both new models remain faster than the official baseline, but that overall
advantage includes the preserved small translation network and gyro calibration;
it is not a benefit created entirely by this new branch. Step time is local WASM
execution time, not end-to-end latency. Sim(3) removes global scale.

## Training and validation

The old 29,398-parameter network stays frozen. A 6,243-parameter Siamese encoder,
local 3x3 correlation volume and spatially pooled rotation head bring the model
to 35,641 parameters. The new head corrects the actual causal calibrated IMU
prior; old raw IMU inputs and camera-local translation remain unchanged. The
single-frame control has the same parameter tensors and schedule but compares
current/current features. Compute can differ because identical encoder calls
may be reused in the exported single-frame graph.

Both fits use seed 17, 40 epochs, 1,520 optimizer steps, identical minibatch
orders and 95,520 pair exposures. Data remains 2,388 training pairs from scenes
2/3/4 and 796 validation pairs from scenes 5/6. Public scene 1 is excluded.
Only the new branch is trained; no new warping or multi-frame objective is added.

| Variant | Selected trained epoch | Mean validation rotation RMSE (deg) | Mean Sim(3) ATE (m) | Mean unaligned ATE (m) | Rotation wins vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated gyro, zero correction | 0 | 2.170470 | 0.265630 | 0.511332 | reference |
| Single-frame | 14 | 2.453287 | 0.265527 | 0.512153 | 0/4 |
| Temporal correlation | 7 | 2.780753 | 0.265555 | 0.513904 | 1/4 |

Each branch retains its lowest-rotation trained epoch as a diagnostic. The
predeclared v5 acceptance rule requires at least 5% lower mean rotation error,
at least 3/4 episode rotation wins and no more than 1% aligned/unaligned position
degradation. Both fail the rotation conditions. The v5 reference stays
calibrated-gyro; v4's stricter selection still stays `current`. Neither decision
changes because of the public results, and the workspace pose-only default is
unchanged.

The previous-image shuffle probe changes temporal validation rotation from
2.780753 to 2.829766 degrees; correction RMS changes by 0.00010643 rad
(about 0.0061 degrees). The branch is sensitive to the previous image, but this
does not establish useful geometric matching. The single-frame branch is exactly
invariant to that intervention. See each model's `visual-probe.json`.

## Verification, artifacts and provenance

- 111 host tests pass, including seven new matching/runtime tests.
- All 796 validation samples pass PyTorch/OpenCV export parity. Frozen translation
  weights are exact; exported translation and actual runtime inputs are checked.
- Actual `Solution.step` rollouts on all four validation flights agree with the
  training-side metrics and causal IMU prior. Old-reference host outputs match
  the archived v4 outputs exactly.
- Both new WASM artifacts pass 40-frame execution checks.
- The full temporal repeat has byte-identical pose, twist and per-frame fuel files
  and identical accuracy. All three preserved reference artifacts reproduce
  their v4 accuracy, output and fuel records exactly. See
  [output verification](output-verification.json).
- `single-frame/` and `temporal/` archive the ONNX, metadata, training history and
  visual probes. `runs/` retains evaluation manifests, reports and console records.
  Large raw outputs and checkpoints remain in ignored `results/` and `models/`.
- `meta.json` retains the base model's legacy `training_stage: warp` field;
  `architecture`, `correlation_variant`, `selected_epoch` and the v5 `run.json`
  identify the new rotation-only fit. No new warping loss was used.
- An initial single-frame attempt stopped after epoch 1 due to JSON serialization
  of a NumPy Boolean in the report, before any checkpoint choice could be
  published. The Boolean conversion was fixed and the full fixed-seed fit restarted
  in a new directory. Its failed logs/checkpoint remain preserved; no experiment
  settings were changed in response to accuracy.

The upload-ready Windows files are
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/AeroVIO-correlation-temporal-v5.wasm`
and `AeroVIO-correlation-control-v5.wasm` in the same directory. They are
**unselected diagnostic candidates**. `windows-artifacts.json` records verified
copies and SHA-256 sidecars. Existing v4 and baseline files remain unchanged.
No course-server or physical-robot test was performed.

## What this suggests next

The experiment demonstrates deployable matching computation, not a proven
correspondence-based rotation improvement. A useful next isolated test would
supervise visual alignment/displacement explicitly, then evaluate new motion and
scene conditions before changing deployment selection. Accumulated multi-frame
rotation supervision is another separate hypothesis. Neither has been trained
or claimed as an improvement here. One seed, short validation flights, a shared
startup pattern and one public scene limit generalization claims.
