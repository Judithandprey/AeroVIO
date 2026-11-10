# AeroVIO - Handoff

Read `README.md`, `docs/requirements.md`, `docs/setup.md`,
`docs/learned-motion.md`, and `experiments/learned-motion-v1/README.md` first.
Use `experiments/learned-motion-v1/comparison.md` as the current result summary.

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

## Verified implementation and training

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
- All 39 host tests pass. Each learned variant passed a 20-frame WASM check.
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

## Important implementation details

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
