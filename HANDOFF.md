# AeroVIO - Handoff

Read `README.md`, `docs/requirements.md`, `docs/setup.md`, and
`experiments/baseline-klt-imu/README.md` first.

## Verified state, November 10, 2026

- Private repository: https://github.com/Judithandprey/AeroVIO.
- Build checkout: WSL Ubuntu `/root/projects/AeroVIO`; Windows Git mirror:
  `C:/Users/ROG/Documents/GitHub/AeroVIO`. The WSL checkout owns the build/cache.
- Python KLT + IMU copied byte-for-byte from viobench
  `2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`; attribution retained.
- Generated runtime recipes are tracked; compiled binaries/data remain ignored.
- WASM build and 20-frame smoke check passed. The embedded solution matches
  tracked and upstream source exactly.
- All 10 public episodes completed, 12,000 frames, no failed/filled frames.
  Mean Sim(3) ATE: 2.04163486385224 m; max fuel/frame: 141,910,846.
- The first episode was repeated: metrics, per-frame fuel, pose/twist files
  matched exactly. Full ten-episode repeat was not performed.
- Compact evidence is in `experiments/baseline-klt-imu/`; raw evidence in
  `results/baseline-public/`, `results/baseline-repeat/`, and `logs/` in WSL.
- No neural-network training, map-based estimator, server submission, or
  held-out evaluation has been performed. No team token is stored here.

## Next development step

Freeze this baseline before creating `experiment/perspective-map`. Design a
bounded, causal landmark map and reprojection/PnP correction. Preserve source,
runner, runtime, input selection and limits for comparisons. Distinguish planned
architecture from implemented and measured behavior.

The baseline has substantial unaligned drift on several trajectories. Sim(3)
ATE alone does not establish metric scale accuracy. Validate initialization and
triangulation before trusting a map built from estimated poses.

## Boundaries

- Latest course quota is 2 submissions/team/day, unlike older upstream defaults.
- The first-stage deadline lacks an explicit timezone in its PDF;
  America/Los_Angeles is the course-context interpretation.
- The course figure says median ATE, while the pinned runner aggregates mean
  Sim(3) ATE. Current authenticated server scoring/limits remain unverified.
- Preserve causal inputs and per-episode resets; never feed ground truth to the
  estimator. Keep public-set tuning separate from held-out performance claims.
- Course PDFs retain their authors' rights. Keep this repository private.
- Keep credentials, data, generated binaries, and large logs out of Git.
