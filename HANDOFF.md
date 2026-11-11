# AeroVIO - Handoff

## Baseline correction (November 10, 2026)

The server's Exact values table identifies the similar-compute frontier point
as `tiny_cnn`, not KLT + IMU. Its current 20-episode score is 0.6684122639764059 m
mean Sim(3) ATE and 14,684,708 max fuel/frame; the user's latest upload
is 0.9155300067970294 m / 14,757,611. All v1-v6 references to "the official
baseline" should be read as the chosen upstream KLT + IMU sample on our local
ten-episode set. Those experiments do not establish superiority over the full
server baseline frontier. See `experiments/tiny-cnn-v7/` and its source/weights
limitations. Preserve the existing defaults and prior acceptance decisions.
Details / Raw JSON subsequently confirmed the upload is v4 calibrated-gyro
(SHA-256 matches the Windows artifact), the server set is `public+private-10`,
and the six saved resource limits match. Private data and official Tiny CNN
weights were not obtained; this is a local algorithm reproduction.

V7 is complete: original trainer (10 epochs/seed 0/final checkpoint) and a
matched-budget fit (40 epochs/seed 17/validation epoch 7) retain byte-identical
upstream inference code. Their local public mean ATE is 1.095712/1.171630 m at
14,680,072/14,685,424 max fuel/frame. All eight serial runs finished 10/10, the
matched full repeat is exact, and all five preserved-reference reruns reproduce
trajectory/twist/fuel/accuracy exactly. Both 40-frame WASM checks and 796-pair
ONNX checks passed. The public part of the user's existing server result also
matches our calibrated-gyro ATE and max fuel exactly for every episode.
Reports, source snapshots, weights, provenance and Windows artifact hashes are
in `experiments/tiny-cnn-v7/`. No experiment or training run is pending.


The v6 explicit-correspondence supervision experiment is complete and rejected
by the frozen acceptance rule. V5 temporal architecture and all guest source are
unchanged; two 40-epoch fine-tunes share initial weights and minibatches. The
supervised arm adds geometric bilinear soft-label matching cross-entropy; the
continuation arm retains rotation-only supervision. Trained epochs 13 and 4,
respectively, are archived as diagnostics. Matching validation EPE worsens from
0.336266 (V5) to 1.067577 image pixels (supervised), versus 0.306630 for continuation
and 0.236517 for zero motion. Cross-entropy decreases but displacement does not
improve. Independent OpenCV projection and all 118 host tests pass.

All seven v6 public runs completed 10/10 episodes. Supervised achieves 0.868606 m
aligned ATE, 10.6784 degrees median rotation error and 1.506 ms median WASM step,
versus V5 temporal's 0.870938 m, 9.7654 degrees and 1.561 ms in this round. Its
rotation median worsens 9.35% while ATE improves only 0.27%; the V5 single-frame
control remains better on both. Max fuel is 21,874,382 (+0.093% versus V5).
Both new 40-frame WASM checks pass. Full supervised repetition and all four
preserved-reference reruns match outputs, accuracy and per-frame fuel exactly.
See `docs/correspondence-supervision.md`, `experiments/correspondence-v6/README.md`,
`selection.json`, `matching-audit.json` and `windows-artifacts.json` in that record.
All previous selections/defaults and artifacts remain unchanged. No new run is
pending; future continuous-displacement supervision is only a hypothesis.

The v5 lightweight feature-correlation experiment is complete. The old translation
network is frozen; a 6,243-parameter local matching branch corrects the actual
causal calibrated gyro prior. A matched current/current-image control isolates
previous-frame information. Both seed-17 fits ran 40 epochs. Validation chose
trained epochs 7 (temporal) and 14 (single-frame) as diagnostics, but neither
qualified: mean rotation RMSE is 2.780753 and 2.453287 degrees versus 2.170470
for calibrated gyro. V5 retains its calibrated-gyro reference, while v4's selected
`current` and the workspace pose-only default remain unchanged.

All six v5 public runs completed 10/10 episodes. Temporal correlation achieves
0.870938 m mean Sim(3) ATE, 9.7654 degrees median rotation RMSE and 1.522 ms median
step time. Against calibrated gyro, these are -1.23%, -19.93% and +26.91%.
The single-frame control is better on the first two metrics (0.855547 m, 9.0634
degrees) and faster (1.440 ms), so this round does not establish an incremental
benefit from temporal matching. 111 tests and both 40-frame WASM checks pass.
Full temporal repetition reproduces outputs, accuracy and frame fuel exactly;
the three preserved references reproduce v4 exactly. See
`docs/feature-correlation.md`, `experiments/feature-correlation-v5/README.md`,
`comparison.md`, `selection.json` and `windows-artifacts.json` in that record.
No new training or default change is pending; proposed next hypotheses are not
authorization to start more experiments.

The v4 startup gyro-calibration experiment has completed original simulated-flight
validation. Selection remains `current`: the unchanged v1 warp model with bias
calibration off and its learned residual on. Calibrated gyro-only rotation lowers
mean orientation RMSE by 68.77% across all four validation flights, but aligned
ATE rises by 0.02752%, violating the frozen nonregression guard. Calibration with
the residual also fails that guard (ATE +0.07454%). No threshold was relaxed.
All 104 host tests and the four new 40-frame WASM checks pass. All seven public
runs completed 10/10 episodes. Calibrated-gyro achieves 0.881818 m mean Sim(3)
ATE, 12.196 degrees median rotation RMSE and 14,757,111 max fuel/frame: 56.81%,
7.92% and 89.60% lower than the official baseline, respectively. Its full repeat
matches poses, twists, accuracy and every frame's fuel exactly. Against current,
it costs 16.69% more median step time and increases median unaligned ATE 1.55%.
The current and candidate Windows WASM copies are SHA-256 verified. Public
results do not override the earlier strict validation decision.
See `docs/gyro-bias-calibration.md`, `experiments/gyro-bias-v4/validation.md`, and
the frozen `selection.json`. No new training or default-model change occurred.

The v3 rotation-focused training run is complete. Both the matched continuation
control and the rotation-auxiliary fine-tune selected epoch zero: neither supplied
a qualifying improvement on original simulated-flight validation within the fixed translation guards.
Their selected model is the original v1 warp checkpoint. Epoch-20 candidates are
retained separately as unselected diagnostics. All six public variants completed
10/10 episodes; 84 tests and all four new WASM checks pass. Final focused training
lowered aligned ATE by 0.44% but worsened rotation RMSE by 11.54% versus the original
model, so it remains unselected. Four Windows WASM copies are hash-verified.
See `experiments/rotation-focus-v3/README.md` and
`docs/rotation-focused-training.md`.

The previous rotation/image ablation remains complete: all five v2 variants
finished 10/10 public episodes (12,000 frames each), 62 tests passed, and all four
new Windows WASM copies were SHA-256 verified. Its findings remain in
`experiments/motion-ablation-v2/README.md`: learned rotation correction helps
overall; visual-input benefits vary by episode; compact IMU-only trades compute.

Read `README.md`, `docs/requirements.md`, `docs/setup.md`,
`docs/learned-motion.md`, `docs/gyro-bias-calibration.md`,
`experiments/gyro-bias-v4/validation.md`, `docs/rotation-focused-training.md`, and
`experiments/rotation-focus-v3/README.md` first.
Use `experiments/motion-ablation-v2/comparison.md` and `rotation-diagnostics.md`
for the earlier input/rotation investigation; preserve `experiments/learned-motion-v1/` as the
completed first experiment. See `docs/warping-and-world-models.md` for the
conceptual distinction between geometric warping and a learned world model.

## Workspace and branch

- Private repository: https://github.com/Judithandprey/AeroVIO.
- Branch: `experiment/perspective-map`.
- Build checkout: WSL Ubuntu `/root/projects/AeroVIO`.
- Windows Git mirror: `C:/Users/ROG/Documents/GitHub/AeroVIO`.
- WSL owns build products, datasets and caches; source changes must be synchronized
  deliberately with the Windows mirror before publishing.
- Upstream runner/runtime revision: `2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`.
- Keep the evaluation `.venv` and training `.train-venv` separate. Host PyTorch
  training uses CPU; guest inference uses ONNX through OpenCV DNN in WASM.

## First experiment: preserved implementation and evidence

- B0: original KLT + IMU, frozen at commit
  `8ea22f52b2083ee53d75b9419651c2505f949a67`, tag `baseline-klt-imu-v1`.
  Its 10-episode mean Sim(3) ATE is 2.04163486385224 m and max frame fuel is
  141,910,846. First-episode repeat matched pose/twist files and fuel exactly.
- N1: 29,398-parameter paired-image CNN + IMU, supervised relative motion.
- N2: identical architecture, seed, initialization, batch order, data and schedule,
  with masked training-only photometric warping loss.
- N3: exactly N2's network weights, with bounded sparse-map correction enabled.
- Both learned models trained for 40 epochs, seed 17, on 2,388 pairs from 12
  generated flights; 796 pairs from four scene-disjoint flights select checkpoints.
  Training scenes 2/3/4, validation scenes 5/6; public comparison scene 1 excluded.
- Both saved checkpoints were selected at epoch 4 by validation motion loss.
  Public evaluation scores did not select checkpoints or tune map gates.
- A complete same-seed pose-only training repeat produced an identical ONNX hash
  and best validation metrics; see the archived training-repeatability record.
- Current `workspace/solution/net.onnx` and `meta.json` select N1 (pose-only).
  For N2/N3, package the recorded checkpoint explicitly; see `docs/setup.md`.
- The v1 evidence archive records 39 passing host tests and a 20-frame WASM
  check for each learned variant.
  These checks establish execution/geometry contracts, not accuracy by themselves.
- Completed public-run reports and the accuracy/compute comparison belong in
  `experiments/learned-motion-v1/`; refer there rather than copying provisional
  numbers into new documents. All four comparison runs completed; inspect the
  positional gains together with rotation/scale trade-offs and the map regression.
  Raw trajectories/logs remain in ignored `results/`.

Windows upload-ready copies have been hash-verified in
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/`:
`AeroVIO-motion-pose-v1.wasm`, `AeroVIO-motion-warp-v1.wasm`, and
`AeroVIO-motion-warp-map-v1.wasm`. Packaging/copying is not a server submission.

## Rotation and input ablations

`experiments/motion-ablation-v2/` records the fixed-weight rotation toggle,
matched IMU-only training, and optional algebraically compacted export. Consult
its reports and checks for completed evaluation status and metrics.

- Rotation on/off packages reuse `experiments/learned-motion-v1/warp/net.onnx`.
  `rotation_residual_enabled` defaults to `true`; `false` composes only the IMU
  rotation. The predicted relative translation is unchanged, while subsequent
  world-axis translation can change because the orientation has changed.
  This ablation requires the sparse map to be disabled.
- `input_mode` defaults to `image_imu`. The `imu_only` model is trained with
  zero image inputs and receives zero images at inference. It retains all 14
  state values, including normalized calibration and elapsed time. Real RGB,
  depth and reference motion still supervise its training losses.
- The primary input ablation retains the original 29,398-parameter graph,
  initialization, cache, seed 17, 40 epochs, batch order and warping weight 2.
  Checkpoint selection follows the same validation rule; the IMU-only run
  selected epoch 14.
- `network_inputs` defaults to `images_state`. The optional `state` export
  requires `input_mode=imu_only`; it folds constant zero-image features into
  the first head-layer bias. It has 1,350 parameters and a 5,949-byte ONNX file,
  derived from the same trained checkpoint without retraining. Numerical
  equivalence evidence belongs with `imu-only-compact/run.json`.
- All compared v2 packages use the same guest Python source. The current
  workspace default remains N1, the pose-only image+IMU model.

The v2 Windows artifact filenames under
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/` are:
`AeroVIO-motion-current-v2.wasm`, `AeroVIO-motion-no-rotation-v2.wasm`,
`AeroVIO-motion-imu-only-v2.wasm`, and `AeroVIO-motion-imu-compact-v2.wasm`.
Check the v2 experiment record for copy/hash verification. Packaging and copying
remain separate from a course-server submission.

## Startup gyro calibration

`configs/gyro-bias-v4.json` fixes a two-by-two experiment: calibration on/off
and learned rotation residual on/off. All four cells share the original warp
ONNX, raw image/IMU predictor inputs and camera-local translation predictions.
The optional calibrator collects the first complete one-second hover window,
applies the fixed count, span, variation and simulator-prior gates, then either
freezes rejection or applies its IMU-axis bias estimate to future integration.
The closing frame is emitted before activation. There is no reset, replay,
gravity realignment, output hold, map correction or ground-truth input.

Startup evidence comes only from original simulated training/validation flights.
They share a deterministic startup motion; hover does not mean a frozen pose.
The method assumes that documented startup, and low variation cannot distinguish
small constant true rotation from sensor bias. Keep this limitation with claims.

Validation repeats all four complete flights per cell, checks unchanged raw
network inputs/outputs and freezes selection before public evaluation. The
pre-public `configs/gyro-bias-v4-public-replication.json` adds a full repeat of
the promising but unselected calibrated-gyro candidate. Seven public runs cover
the official baseline, four cells, the selected-current repeat and that diagnostic
repeat. Reproducibility cannot retroactively make a candidate eligible.
Four new packages cover `current`, `gyro-only`, `calibrated-residual`, and
`calibrated-gyro`; see `docs/setup.md` for explicit packaging flags.

## Important implementation details

The v3 auxiliary cache is `data/rotation-aux-v3/`: 2,400 training pairs from
scenes 2/3/4 and 800 validation pairs from scenes 5/6. It reprojects existing
320x240 adapted images into symmetric virtual rotations, then uses the shared
64x48 preprocessing. Camera x/y/z/mixed axes are balanced over 0–3 degrees;
the fixed crop and source-support check avoid padded borders. Synthetic IMU
samples include the actual extrinsics, gravity, lever-arm acceleration and
configured independent sensor noise. Only rotation is supervised by this cache.

The fixed protocol is `configs/rotation-focus-v3.json`. Both 20-epoch fine-tunes
start from `models/motion-warp-v1/best.pt`, share 760 optimizer steps and real
minibatches, and keep the original normalization and deployed architecture.
Selection minimizes accumulated real-validation rotation RMSE subject to both
translation guards; synthetic validation and public scores do not select weights.

The retention-only reruns live in `models/rotation-{control|focus}-v3-retained/`,
with compact records archived under `experiments/rotation-focus-v3/control/`
and `focus/`. Their top-level `net.onnx` is the selected epoch-zero model.
`diagnostic-last/net.onnx` is the unselected epoch-20 candidate, explicitly marked
as such in metadata. Keep these roles distinct in packaging, reporting and any
public comparison. The original v3 run records remain preserved as well.
The pre-public addendum `configs/rotation-focus-v3-public-diagnostics.json`
adds the fixed epoch-20 candidates without changing checkpoint selection.
The six compared variants are the original baseline, the starting warp model,
both selected v3 models and both last-epoch diagnostics. Four new v3 packages
cover the selected/control, selected/focus and two diagnostic roles. Public
completion and metrics must come from the finished reports, not package creation.

See `experiments/rotation-focus-v3/rotation-examples.png` for calibrated input
examples and its caption/JSON for exact sample identities and display settings.
The illustration is not evidence that the network estimates those rotations well.

The network receives two 64x48 grayscale frames and 14 values: camera-frame
IMU rotation/delta-velocity/delta-position, normalized intrinsics, and frame dt.
It predicts translation and an IMU rotation residual. `preintegrate.py` is shared
with training and includes the first sample interval. Its 1 microsecond endpoint
tolerance matches the upstream reader; truly future samples are rejected.

Pose maps the current camera into the first-camera frame. Twist is a finite-
interval mean in current camera axes, recomputed after map correction; it is
an approximation to instantaneous twist. All state resets per episode.

Training-only camera-z depth and fixed visibility masks never enter the deployed
network. New/disoccluded pixels do not supervise image reconstruction. Sparse
map points come only from causal observations and estimated poses. Rejected map
updates retain the learned prior. No unseen-surface generation is implemented.

## Next development decisions

Inspect the complete comparison and failure/per-episode evidence before deciding
which variant to extend. This is a small, single-seed pilot. Additional seeds and
scene diversity are needed before making broad superiority/generalization claims.
Keep architecture, data, runtime and limits fixed when isolating a loss or map
change. Do not tune on the public set and then call it independent validation.

## Boundaries

- No course-server upload, held-out evaluation, robot control or WAM training
  has been performed. No team token is stored here.
- November 9 course quota: 2 submissions/team/day. Current authenticated server
  scoring, limits, quota and deadline still require verification before submission.
- First-stage PDF omits a timezone; America/Los_Angeles is the course context.
- Sim(3) alignment removes global scale; inspect unaligned ATE, rotation and
  velocity errors as well. A low aligned error alone does not prove metric maps.
- Course PDFs retain their authors' rights; keep the repository private.
- Keep credentials, generated data, large logs, raw checkpoints and WASM binaries
  outside tracked source; compact experiment model snapshots are intentional.
