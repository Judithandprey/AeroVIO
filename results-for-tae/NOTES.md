# Window-net run for Tae (run by Leo's machine, 2026-11-20)

Branch `origin/tae_testbranch` at `7bd90aa`, untouched (detached worktree, nothing committed).

## What was trained
`scripts/train_window_imu.py --hidden 128` (window 40, 30 epochs, batch 512, lr 2e-3, seed 0; CPU, ~1 min).
Output: `experiments/window-imu-h128/` (net.pt, meta.json, window_weights.h). Best epoch 13,
val ATE with GT rotation 1.325 m (a constant-speed predictor gets 1.354 m); it overfits after that.

**Data (your `data/s2-imu` / `data/s2-val` were not on this machine, so this is a substitute):**
train = 62 no-hover 60 s flights (`viobench simulate --hover 0`, ProcTHOR-Train-7,13,...,193 without 121,
seeds 7000+), val = 8 flights from scenes 10/70/110/150 (seeds 8000+). None of the evaluation sets below
was used for training or checkpoint selection.

## Build
V8 rotation + `GRAVITY_TILT_GAIN=0.5` + `ONLINE_GYRO_BIAS=ON` (gain 1.0, sign -1) + `WINDOW_IMU=ON` with the
h128 header (your best documented configuration; EKF/KF off).
`tae-v8-tilt0.5-bias1.0-window-h128.wasm`
SHA-256 `bcf77cbbfc0a9405122a6a002173ddb787965b970c7b2131aa282e4669176c19`

## Results
See `results.md`. Compared with Leo's `aerovio-v8-tilt-bias-g0.2-fixedcam.wasm` (4.66 m on the server).
val-narrow / val-fast are new fast 60 s sets (36-50 m per flight; val-narrow has FOV 77-95 deg).

## Narrow-FOV check
The window net does not take camera intrinsics, so the runaway Leo found does not happen: on val-narrow the
drift is 3.5 m and Sim(3) scale 0.83 (the old compact build: 10.5 m, 0.36); `viobench check` (80 deg FOV,
stationary IMU) moves 0.05 m in 2 s. It does take dt with std ~1e-6 (constant 20 Hz in training), so a
different frame interval would push it far out of distribution in the same way.
