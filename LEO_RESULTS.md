# Leo's stage 1 results (branch `leo/stage1-final`)

Local evaluation on the **10 public episodes** (`viobench evaluate … --limits configs/evaluation-limits.json`),
viobench pinned at `2072fa9`, wasi-sdk 34, Ubuntu host, run on 2026-11-15.
The course server also uses private episodes, so its numbers will differ.

| Variant | Mean Sim(3) ATE (m) | Max fuel / frame | Mean fuel / frame | Notes |
| --- | ---: | ---: | ---: | --- |
| **Teacher "Native 160"** | 0.486329 | 2,232,986 | 1,035,272 | Default build. ATE identical to the recorded 0.486329. Max fuel equals the team's own rebuild (`control`) exactly, +0.011% vs her original binary (build-path difference). |
| **V8 rotation** | **0.479744** | 2,233,398 | 1,049,383 | `-DVISUAL_ROTATION_CORRECTION=ON`. Reproduces the team's record exactly (ATE, max fuel). Best accuracy. |
| **6a: IMU only** (new) | 1.061794 | **13,852** | 13,749 | `-DLEO_IMU_ONLY=ON`. No visual tracker at all. About 161× cheaper than the teacher; ATE about 2.2× worse. Worst episode 1.79 m. |
| **6b: downsample 4, every 16 frames** (new) | 0.693135 | 716,501 | 155,391 | `-DVISUAL_DOWNSAMPLE=4 -DVISUAL_INTERVAL=16`. 3.1× lower max fuel and 6.7× lower mean fuel than the teacher, ATE +43%. Worst episode 1.09 m. |

All four complete 10/10 episodes with no failures; init fuel is about 220k for each.
For reference (from HANDOFF): server `tiny_cnn` is 0.668 m / 14.68 M, and the team's current upload is
0.916 m / 14.76 M on the server (0.882 m locally).

## Stage 2: gravity tilt correction (2026-11-19)

viobench updated to `9b1b079` (stage 2 headline = **unaligned ATE**). All runs use
`--limits configs/evaluation-limits.json`. K is in 1/s: each IMU sample adds `K * e * dt` to the
rotation, so the gain does not depend on the IMU rate.

Three local sets:

- **public**: the 10 public episodes.
- **val-agile**: 12 × 60 s flights in `data/val-agile` (ProcTHOR-Train-40/60/80/100/120/140, 2 each,
  seed 2000).
- **val-nohover**: 8 × 60 s flights in `data/val-nohover` (ProcTHOR-Train-50/90/130/170, 2 each,
  seed 3000, `--hover 0`).

`viobench simulate` has no agility option, so val-agile is **not** like the server's agile flights.
On val-agile V8 gets ATE 1.35 m and 13.8° rotation error, against 5.22 m and about 27° on the server.
Its rotation rates (13–21°/s RMS) are close to the public set's (9–15°/s).

Means over the episodes of each set. "Rot" is the rotation RMSE in degrees.

| Variant | Set | ATE (m) | Sim(3) ATE (m) | Sim(3) scale | Rot (°) | Max fuel |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V8 | public | 3.007 | 0.480 | 2.59 | 13.3 | 2,233,398 |
| V8 + tilt K=0.002 | public | 3.005 | 0.480 | 2.59 | 13.2 | 2,235,625 |
| V8 + tilt K=0.005 | public | 3.003 | 0.481 | 2.59 | 13.0 | 2,235,625 |
| V8 + tilt K=0.01 | public | 2.994 | 0.484 | 2.59 | 12.8 | 2,243,405 |
| V8 + tilt K=0.02 | public | 2.987 | 0.486 | 2.58 | 12.5 | 2,243,405 |
| V8 + tilt K=0.05 | public | 2.985 | 0.489 | 2.59 | 11.9 | 2,251,185 |
| **V8 + tilt K=0.2** | public | **2.971** | **0.470** | 2.59 | **11.0** | 2,258,974 |
| V8 + tilt K=1.0 | public | 2.960 | 0.462 | 2.61 | 10.8 | 2,290,061 |
| V8 + tilt K=3.0 | public | 2.963 | 0.468 | 2.62 | 10.8 | 2,305,692 |
| V8 + tilt K=0.2, sign flipped | public | 4.009 | 1.026 | 2.74 | 126.9 | 2,280,062 |
| V8 | val-agile | 1.353 | 0.593 | 1.96 | 13.8 | 2,994,389 |
| V8 + tilt K=0.002 | val-agile | 1.354 | 0.593 | 1.96 | 13.8 | 2,996,556 |
| V8 + tilt K=0.005 | val-agile | 1.352 | 0.595 | 1.95 | 13.7 | 2,996,529 |
| V8 + tilt K=0.01 | val-agile | 1.352 | 0.595 | 1.96 | 13.5 | 2,996,540 |
| V8 + tilt K=0.02 | val-agile | 1.351 | 0.593 | 1.96 | 13.4 | 2,996,549 |
| V8 + tilt K=0.05 | val-agile | 1.348 | 0.584 | 1.96 | 13.0 | 2,949,831 |
| **V8 + tilt K=0.2** | val-agile | **1.340** | **0.561** | 1.98 | **12.6** | 2,949,813 |
| V8 + tilt K=1.0 | val-agile | 1.338 | 0.535 | 2.00 | 12.2 | 2,996,468 |
| V8 + tilt K=3.0 | val-agile | 1.357 | 0.546 | 2.00 | 12.2 | 2,996,531 |
| V8 | val-nohover | 1.827 | 0.742 | 2.21 | 20.1 | 3,499,256 |
| **V8 + tilt K=0.2** | val-nohover | **1.756** | **0.692** | 2.25 | **18.9** | 3,513,144 |
| V8 + tilt K=1.0 | val-nohover | 1.731 | 0.685 | 2.26 | 18.5 | 3,489,672 |

All runs complete every episode. Mean fuel per frame rises by 0.1–0.4% with the tilt filter.

**Sign check.** With the specific-force convention (`e = measured × predicted`, g0 from the raw
accelerometer), the correction lowers the rotation error. Flipping the sign raises it to 127°, so
the convention is right.

**Where the rotation error comes from.** The error splits into tilt (angle between the true and the
estimated up direction) and the rest, which is yaw.

- V8 tilt error per episode: 3–12° on public, 1–8° on val-agile.
- With the filter (K=0.2 / 1.0): about 1–3° everywhere.
- The large remaining errors are yaw, which this filter cannot see. Example: public `9b6a9279` has
  36.7° total but only 2.0° tilt with K=0.2.

**Startup calibration (step 2).**

- Hovering starts are fine. On public and val-agile every flight has a 1 s hover (max speed 0.07 m/s
  in the first second), and all 22 calibrations are accepted.
- Moving starts are already rejected. With `--hover 0` the drone already turns at 4–28°/s at
  startup, and the existing check in `gyro_bias.h` rejects 7 of 8 episodes.
- No false acceptances. In every accepted episode, the true rotation that ends up in the bias is
  0.08°/s.
- So "skip calibration when moving" is already the current behaviour, and a stricter option would
  change no episode. No option was added.
- The cost of a rejected calibration is real. The gyro bias (up to about 1°/s per axis) is then never
  removed, and V8's rotation error rises to 20° on val-nohover. If the server's agile flights start
  without a hover, this would explain part of their 27°.

**Recommendation.** K=0.2 (`artifacts/aerovio-v8-tilt-k0.2.wasm`).

- Rotation improves on 21 of 22 public + val-agile episodes. Sim(3) ATE: −2% public, −5% val-agile.
  Unaligned ATE: −1% public, −1% val-agile, −4% val-nohover. No set gets worse on average.
- Individual episodes can still lose up to +0.06 m of ATE.
- K=1.0 is marginally better locally. K=0.2 (5 s time constant instead of 1 s) is preferred for
  agile flights: a quadrotor's accelerometer reads thrust direction, so sustained tilts up to about
  18° pass the 0.5 m/s² gate and pull the estimate toward level.
- The requested K values (0.002–0.02) are too weak to matter within 60 s.
- This is a small, safe improvement, **not** a fix for the server's 5.22 m. Most of that error is yaw
  drift and scale (Sim(3) scale about 2–2.6 everywhere).

## Stage 2: online gyro bias from the visual rotation (2026-11-19)

**Why.** On val-nohover the startup calibration is rejected in 7 of 8 flights. The gyro bias (about
1°/s) is then never removed, and rotation error reaches about 20°.

**How.** V8's visual refinement already estimates, every 8 frames (0.4 s), the small rotation the gyro
got wrong over that interval. A remaining bias `b` shows up there as about `-R b Δt` (R =
camera_from_imu).

- Option `LEO_ONLINE_BIAS` (default OFF) uses the unscaled estimate. `refine_rotation` returns 0.25×
  its estimate, so the tracker multiplies by 4 again.
- Per accepted fit, the bias moves a fraction **g** (`LEO_ONLINE_BIAS_GAIN`) of the measured rate
  error. This is a slow integral loop that also sees yaw.
- Fits that imply more than 0.03 rad/s are rejected as outliers (`LEO_ONLINE_BIAS_REJECT`), and each
  axis is clamped to 0.05 rad/s.
- Updates start once the startup calibration has finished. They add to its bias when it is accepted
  and replace it when it is rejected.
- Cost: a few vector operations per accepted fit. Mean fuel changes by less than 0.3%.

**Results.** All variants below are on top of V8 + tilt K=0.2, except V8, V8 + tilt and "bias only".
Means over the episodes of each set. Rot = rotation RMSE. Tilt is the error of the up direction; yaw
is the rest of the rotation error, about the up axis.

| Variant | Set | ATE (m) | Sim(3) ATE (m) | Sim(3) scale | Rot (°) | Tilt (°) | Yaw (°) | Max fuel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V8 | val-nohover | 1.827 | 0.742 | 2.21 | 20.1 | 6.9 | 18.1 | 3,499,256 |
| V8 + tilt K=0.2 | val-nohover | 1.756 | 0.692 | 2.25 | 18.9 | 3.0 | 18.3 | 3,513,144 |
| V8 + bias g=0.1 (bias only, no tilt) | val-nohover | 1.747 | 0.658 | 2.27 | 12.7 | 5.6 | 10.5 | 3,519,175 |
| + bias g=0.02 | val-nohover | 1.739 | 0.662 | 2.27 | 16.6 | 2.8 | 16.0 | 3,505,444 |
| + bias g=0.05 | val-nohover | 1.721 | 0.645 | 2.26 | 14.3 | 2.7 | 13.8 | 3,513,251 |
| + bias g=0.1 | val-nohover | 1.709 | 0.612 | 2.28 | 10.8 | 2.5 | 10.2 | 3,513,215 |
| **+ bias g=0.2** | val-nohover | **1.704** | 0.633 | 2.27 | **9.3** | 2.7 | **8.4** | 3,497,671 |
| + bias g=0.5 | val-nohover | 1.707 | 0.711 | 2.17 | 12.4 | 3.9 | 11.3 | 3,693,265 |
| + bias g=1.0 | val-nohover | 1.694 | 0.711 | 2.15 | 13.5 | 5.2 | 12.2 | 3,705,704 |
| + bias g=0.1, sign flipped | val-nohover | 1.921 | 0.997 | 1.96 | 41.8 | 6.9 | 40.8 | 3,738,664 |
| V8 | val-agile | 1.353 | 0.593 | 1.96 | 13.8 | 4.4 | 12.0 | 2,994,389 |
| V8 + tilt K=0.2 | val-agile | 1.340 | 0.561 | 1.98 | 12.6 | 2.0 | 12.0 | 2,949,813 |
| + bias g=0.1 | val-agile | 1.329 | 0.541 | 1.99 | 10.3 | 1.9 | 9.8 | 2,988,822 |
| **+ bias g=0.2** | val-agile | 1.332 | 0.531 | 2.00 | 10.4 | 2.2 | 10.0 | 2,996,582 |
| + bias g=0.5 | val-agile | 1.344 | 0.519 | 2.02 | 10.8 | 2.9 | 10.2 | 2,934,333 |
| V8 | public | 3.007 | 0.480 | 2.59 | 13.3 | 6.1 | 10.6 | 2,233,398 |
| V8 + tilt K=0.2 | public | 2.971 | 0.470 | 2.59 | 11.0 | 2.0 | 10.6 | 2,258,974 |
| + bias g=0.1 | public | 2.960 | 0.443 | 2.59 | 7.5 | 1.9 | 7.2 | 2,259,054 |
| **+ bias g=0.2** | public | 2.967 | 0.438 | 2.59 | 6.7 | 2.1 | 6.2 | 2,266,834 |
| + bias g=0.5 | public | 2.973 | 0.459 | 2.60 | 6.2 | 2.7 | 5.3 | 2,274,632 |

All runs complete every episode.

**Sign check.** Flipping the update sign more than doubles the yaw error (40.8°), so the convention
is right.

**Recommendation.** g=0.2 on top of tilt K=0.2 (`artifacts/aerovio-v8-tilt-bias-g0.2.wasm`). It
clearly improves val-nohover.

- val-nohover: rotation error 18.9° → 9.3°, better on all 8 episodes. Yaw is more than halved
  (18.3° → 8.4°). Sim(3) ATE −9%. Unaligned ATE −3% (better on 5 of 8 episodes, worst +0.05 m).
- val-agile and public: rotation 12.6° → 10.4° and 11.0° → 6.7°. Sim(3) ATE −5% and −7%. Unaligned
  ATE changes by less than 1%.
- Gains of 0.5 and above get noisy: yaw rises again and max fuel goes up to 3.7M.
- Not every episode gains. Public `9b6a9279` drops from 35.7° to 3.4°, but `51ea6311` rises from 3.5°
  to 9.2° and `f8fb89ff` from 2.7° to 6.6°. The visual residual probably has a scene-dependent
  systematic bias there.
- Halving the rotation error barely moves the unaligned ATE. Scale is now the main error (Sim(3) scale
  2.0–2.6 on every set): positions are about 2× too short. That should be the next target.

## Stage 2: MLP retrained on no-hover flights (2026-11-20)

**Why.** Sim(3) scale is 2.0–2.6 on every local set: positions are about 2× too short.
- The MLP sees one 50 ms IMU window and never its own velocity, so its translation is roughly the
  average speed of its training data.
- That data was 12 × 10 s flights starting with a 1 s hover (median 0.10 m/s).
- On no-hover flights (0.28 m/s) the old MLP predicts 0.15 m/s.

**What.** The same 14→64→6 network, recipe and fuel, retrained on the original data plus 62 new
`viobench simulate --hover 0 --no-video` flights (74k pairs).
- The new scenes and seeds are disjoint from public, the original split, val-agile and val-nohover.
- The new pairs have no depth, so they get the pose loss only. The original pairs keep pose + warp.
- Details: `experiments/nohover-v11/README.md`.
- New option `LEO_WEIGHTS_NOHOVER` (default OFF) builds with `weights_nohover.h`.

**Results.** All rows are V8 + tilt K=0.2 + online bias g=0.2. "old" means the old weights
(`aerovio-v8-tilt-bias-g0.2.wasm`). "uniform" mixes all pairs as they are. "balanced" draws 25%
of every epoch from the original pairs.

| Weights | Set | ATE (m) | Sim(3) ATE (m) | Sim(3) scale | Rot (°) | Tilt (°) | Yaw (°) | Max fuel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| old | val-nohover | 1.704 | 0.633 | 2.27 | 9.3 | 2.7 | 8.4 | 3,497,671 |
| **uniform** | val-nohover | **1.194** | **0.495** | 1.27 | 9.3 | 2.7 | 8.4 | 3,497,671 |
| balanced | val-nohover | 1.246 | 0.500 | 1.41 | 9.3 | 2.7 | 8.4 | 3,497,671 |
| old | val-agile | 1.332 | 0.531 | 2.00 | 10.4 | 2.2 | 10.0 | 2,996,582 |
| **uniform** | val-agile | 1.051 | 0.454 | 1.10 | 10.4 | 2.1 | 10.0 | 2,996,582 |
| balanced | val-agile | 0.963 | 0.448 | 1.26 | 10.4 | 2.1 | 10.0 | 2,996,582 |
| old | public | 2.967 | 0.438 | 2.59 | 6.7 | 2.1 | 6.2 | 2,266,834 |
| **uniform** | public | **1.456** | **0.387** | 1.28 | 6.7 | 2.1 | 6.2 | 2,266,834 |
| balanced | public | 1.697 | 0.392 | 1.47 | 6.7 | 2.1 | 6.2 | 2,266,834 |

All runs complete every episode.
- Rotation is unchanged, as expected: the rotation comes from the gyro, and the learned rotation
  residual is not used.
- Max fuel is identical (same network size).

**Per episode (uniform vs old).**
- val-nohover: ATE better on 7/8 and Sim(3) ATE better on 8/8. One flight (`f3cc5e0d`) loses
  +0.57 m of unaligned ATE although its scale (1.60 → 1.19) and Sim(3) ATE (0.98 → 0.77) improve.
  A longer, more correct path makes its heading error count more.
- val-agile: ATE better on 9/12 (worst +0.13 m).
- public: ATE better on 10/10 (every episode at least −0.39 m). Sim(3) ATE better on 7/10
  (worst +0.01 m).
- Scale ranges move from 1.35–4.03 to 0.88–1.66.

**Recommendation.** Use uniform (`artifacts/aerovio-v8-tilt-bias-g0.2-nohover.wasm`).
- It clearly improves val-nohover: ATE −30%, Sim(3) ATE −22%, scale 2.27 → 1.27.
- It also improves public: ATE −51%, Sim(3) ATE −12%.
- Balanced is better on val-agile only, and worse on val-nohover and public.

Caveats:
- This shifts the speed prior; it does not measure speed. The model still cannot see velocity,
  so episodes much faster or slower than the training flights will stay mis-scaled.
- The server's per-episode scale spread (0.09–3.07) points to that. Measuring scale needs velocity
  in the state or a visual scale estimate.
- On hover-start validation pairs the new model over-predicts speed (1.42×). Public, which has a
  1 s hover, still improves.
- Not checked on the server yet.

## Server check of the no-hover weights, and the runaway fix (2026-11-20)

**Server.** The no-hover weights scored **11.62 m**, against 5.04 m for the same build with the
old weights (12 private episodes, none of them in our local sets).
- Three episodes (`cd6dc7`, `e7aa5a`, `214944`) are the worst with **both** weights: 46% of the
  summed ATE with the old weights, 62% with the new.
- They drift away in a near-straight line: final drift 18 / 11 / 12 m with the old weights,
  57 / 32 / 37 m with the new.
- Their median fuel is 713k–836k, against 1.4–1.9M elsewhere, so the tracker is mostly idle.
- Sim(3) scale < 1 on 4/12 (old) and 9/12 (new) episodes. On a runaway this mostly means "wrong
  shape": the fit shrinks an uncorrelated path toward a point (scale 0.03–0.09 on `cd6dc7`).
- Server flights cover 39–80 m per 60 s. The stock `viobench simulate` covers 14–22 m.

| Episode | Path (m) | ATE old / new | Sim(3) scale old / new | Rot (°) | Final drift old / new | Median fuel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cd6dc7 | 59 | 10.25 / 32.28 | 0.09 / 0.03 | 24.4 | 18.3 / 57.0 | 713,200 |
| e7aa5a | 74 | 10.04 / 29.26 | 1.33 / 0.36 | 18.9 | 10.9 / 31.9 | 830,854 |
| 214944 | 80 | 7.73 / 24.34 | 0.61 / 0.14 | 16.4 | 12.1 / 37.3 | 836,123 |
| 78ed49 | 59 | 4.26 / 9.16 | 1.44 / 0.16 | 20.1 | 5.2 / 10.5 | 1,433,024 |
| 004c58 | 73 | 3.81 / 9.06 | 1.17 / 0.41 | 31.9 | 5.5 / 17.6 | 1,508,698 |
| 1c6e73 | 57 | 3.19 / 6.15 | 0.92 / 0.19 | 10.6 | 3.8 / 7.0 | 1,624,766 |
| dd7a73 | 62 | 3.73 / 6.06 | 0.78 / 0.33 | 9.7 | 4.3 / 7.6 | 1,544,087 |
| 89f8c9 | 63 | 2.78 / 6.00 | 1.17 / 0.18 | 14.8 | 2.1 / 11.4 | 1,782,324 |
| f7c254 | 42 | 4.99 / 5.19 | 1.77 / 1.15 | 33.5 | 2.8 / 2.7 | 713,548 |
| 726458 | 48 | 3.35 / 4.64 | 1.78 / 1.35 | 6.2 | 3.5 / 5.1 | 1,949,424 |
| a61ad0 | 60 | 3.92 / 4.41 | 2.33 / 0.87 | 24.9 | 6.2 / 8.2 | 1,405,730 |
| 895039 | 39 | 2.38 / 2.92 | 3.56 / 1.67 | 11.9 | 1.9 / 4.3 | 1,861,166 |

**Cause found locally: the MLP's camera inputs.**
- Besides the IMU, the network gets the normalized intrinsics (fx/w, fy/h, cx, cy) and the frame
  interval. Training only had FOV 105–135° and constant cx, cy and dt (their std is clamped to 1e-6).
- Changing *only* the FOV input on val-fast IMU gives this net drift of the learned translation
  over 60 s (old weights):

  | FOV input | true FOV (108–115°) | 100° | 90° | 80° | 60° |
  | --- | --- | --- | --- | --- | --- |
  | Net drift | 1–3 m | 7–9 m | 15–17 m | 24–27 m | 52–55 m |

- `viobench check` (80° FOV, stationary IMU) shows the same: the estimate moves 0.42 m/s with the
  old weights and 2.7 m/s with the no-hover weights, while the drone is still.
- Things I ruled out:
  - Speed alone: 52 m flights at the stock FOV drift 2–5 m.
  - Accelerometer bias up to 0.6 m/s²: drift up to 6.5 m.
  - The MLP never exceeds 2 m/s on any local flight, so speed caps never bind.
- I could not check the server's FOV (all episodes are private), so this is the best-supported
  explanation, not a proven one.
- Not explained: the no-hover weights drift about 3× more on the server, but not more than the old
  weights at low FOV locally.

**New test sets** (`scripts/simulate_fast.py` raises the autopilot limits that `simulate` doesn't
expose: cruise 2.5–4 m/s, acceleration 4 m/s², jerk 12, yaw rate 2.5 rad/s):
- **val-fast:** 12 × 60 s, scenes 20/30/45/65/95/135, seeds 4000+, no hover, 36–47 m per flight,
  FOV 108–135°.
- **val-narrow:** the same, but FOV 85 ± 10° (77–95°), scenes 15/35/75/125/155/185, seeds 4500+,
  23–50 m per flight. Its baseline reproduces the server signature: drift 10.5 m, Sim(3) scale
  0.36, median fuel 714k.

**Variants.** All on top of V8 + tilt K=0.2 + online bias g=0.2 with the old weights. Each is
behind its own CMake option, default OFF.
- `LEO_SPEED_CAP` (`LEO_SPEED_CAP_MPS` 2 / 3): limits the learned per-frame translation.
- `LEO_VISUAL_DAMPING`: shrinks it by 0.7× per visual cycle without a direct heading fit, after one
  cycle of grace, down to 0.2×.
- `LEO_OUTPUT_SCALE` (`LEO_OUTPUT_SCALE_FACTOR` 0.7).
- `LEO_MLP_FIXED_CAMERA`: feeds the MLP its training-mean intrinsics and frame interval.

| Variant | Set | Done | ATE (m) | Sim(3) ATE (m) | Sim(3) scale | Final drift (m) | Max fuel | ATE better / worse (worst) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| V8 + tilt + bias (old weights) | val-narrow | 12/12 | 6.417 | 1.724 | 0.36 | 10.49 | 4,238,210 |  |
| + fixedcam | val-narrow | 12/12 | 2.195 | 1.250 | 2.16 | 2.24 | 4,238,203 | 12 / 0 (-0.43) |
| + visdamp | val-narrow | 12/12 | 3.932 | 1.621 | 0.67 | 5.74 | 4,238,219 | 12 / 0 (-0.06) |
| + scale0.7 | val-narrow | 12/12 | 4.908 | 1.724 | 0.52 | 7.57 | 4,238,216 | 11 / 1 (+0.05) |
| + speedcap2 | val-narrow | 12/12 | 6.417 | 1.724 | 0.36 | 10.49 | 4,238,234 | 0 / 0 (+0.00) |
| V8 + tilt + bias (old weights) | val-fast | 12/12 | 2.415 | 1.527 | 2.51 | 2.29 | 3,405,718 |  |
| + fixedcam | val-fast | 12/12 | 2.346 | 1.426 | 2.74 | 2.19 | 3,405,711 | 8 / 4 (+0.06) |
| + visdamp | val-fast | 12/12 | 2.406 | 1.513 | 2.78 | 2.12 | 3,405,727 | 6 / 6 (+0.12) |
| + scale0.7 | val-fast | 12/12 | 2.555 | 1.527 | 3.59 | 2.24 | 3,405,724 | 2 / 10 (+0.37) |
| + speedcap2 | val-fast | 12/12 | 2.415 | 1.527 | 2.51 | 2.29 | 3,405,742 | 0 / 0 (+0.00) |
| + speedcap3 | val-fast | 12/12 | 2.415 | 1.527 | 2.51 | 2.29 | 3,405,741 | 0 / 0 (+0.00) |
| V8 + tilt + bias (old weights) | val-nohover | 8/8 | 1.704 | 0.633 | 2.27 | 1.53 | 3,497,671 |  |
| + fixedcam | val-nohover | 8/8 | 1.646 | 0.592 | 2.15 | 1.40 | 3,497,664 | 5 / 3 (+0.39) |
| + visdamp | val-nohover | 8/8 | 1.775 | 0.625 | 2.58 | 1.61 | 3,497,680 | 1 / 7 (+0.37) |
| + scale0.7 | val-nohover | 8/8 | 1.959 | 0.632 | 3.24 | 1.82 | 3,505,441 | 0 / 8 (+0.32) |
| + speedcap2 | val-nohover | 8/8 | 1.704 | 0.633 | 2.27 | 1.53 | 3,497,695 | 0 / 0 (+0.00) |
| + speedcap3 | val-nohover | 8/8 | 1.704 | 0.633 | 2.27 | 1.53 | 3,497,694 | 0 / 0 (+0.00) |
| V8 + tilt + bias (old weights) | val-agile | 12/12 | 1.332 | 0.531 | 2.00 | 1.84 | 2,996,582 |  |
| + fixedcam | val-agile | 12/12 | 1.237 | 0.536 | 1.86 | 1.71 | 2,996,575 | 9 / 3 (+0.10) |
| + visdamp | val-agile | 12/12 | 1.426 | 0.536 | 2.16 | 1.87 | 2,996,591 | 4 / 8 (+0.39) |
| + scale0.7 | val-agile | 12/12 | 1.656 | 0.531 | 2.86 | 2.31 | 2,996,588 | 0 / 12 (+0.60) |
| + speedcap2 | val-agile | 12/12 | 1.332 | 0.531 | 2.00 | 1.84 | 2,996,606 | 0 / 0 (+0.00) |
| + speedcap3 | val-agile | 12/12 | 1.332 | 0.531 | 2.00 | 1.84 | 2,996,605 | 0 / 0 (+0.00) |
| V8 + tilt + bias (old weights) | public | 10/10 | 2.967 | 0.438 | 2.59 | 3.53 | 2,266,834 |  |
| + fixedcam | public | 10/10 | 2.737 | 0.452 | 2.17 | 3.27 | 2,266,827 | 8 / 2 (+0.27) |
| + visdamp | public | 10/10 | 3.006 | 0.446 | 2.67 | 3.58 | 2,266,843 | 4 / 6 (+0.18) |
| + scale0.7 | public | 10/10 | 3.523 | 0.438 | 3.69 | 4.11 | 2,266,840 | 0 / 10 (+1.03) |
| + speedcap2 | public | 10/10 | 2.967 | 0.438 | 2.59 | 3.53 | 2,266,858 | 0 / 0 (+0.00) |
| + speedcap3 | public | 10/10 | 2.967 | 0.438 | 2.59 | 3.53 | 2,266,857 | 0 / 0 (+0.00) |

All runs complete every episode.
- The speed caps are exact no-ops locally (same ATE everywhere).
- ×0.7 makes every normal set worse (local paths are already 0.3–0.6× too short).
- Damping helps val-narrow (−39%) but costs 0–7% elsewhere.

**Recommendation for the next upload: `artifacts/aerovio-v8-tilt-bias-g0.2-fixedcam.wasm`
(`01365bce…`).**
- It is the only variant that is better on every set: val-narrow −66% (12/12 episodes better),
  val-agile −7%, public −8%, val-nohover −3%, val-fast −3%.
- Max fuel is unchanged.
- It removes an input that cannot carry real information for an IMU-only network. If the server
  cameras use a narrower FOV, an off-centre principal point or another frame rate, this is exactly
  the failure it prevents.

## Server results (2026-11-15)

Official course-server scores (20 episodes: public + private), as reported by Leo from the server:

| Variant | Mean Sim(3) ATE (m) | Max fuel / frame |
| --- | ---: | ---: |
| 6a IMU only (`aerovio-leo-6a-imu-only.wasm`) | 0.967 | 13.9k |
| V8 rotation (`aerovio-v8-rotation.wasm`) | 0.514 | 3.5M |

Team claimed area went from **2.55% to 4.89%** after these two uploads.

Compared with the local 10-episode numbers above, 6a's server ATE is lower (0.967 vs 1.062 m) and its
fuel matches, because its per-frame cost hardly depends on the images. V8's server ATE (0.514 vs 0.480 m) and max fuel
(3.5M vs 2.23M) are both higher: the private episodes contain at least one frame where image tracking costs more.

## What each variant does

- **Teacher (Native 160).** A tiny network (1,350 numbers) turns IMU readings into motion, gyro calibration
  fixes rotation drift, and every 8 frames a sparse optical-flow tracker on a 160×120 image corrects the
  direction of travel. Most of its compute is the image work.
- **V8 rotation.** Same as the teacher, plus the tracker also gently corrects rotation. Slightly more
  accurate at almost the same cost (+412 fuel at peak).
- **6a IMU only.** Same network and gyro calibration but the camera image is never looked at. Very cheap
  (the cost is basically the IMU integration and the small network), but position drifts more because
  nothing visual pulls it back. Meets the goal: under 100k fuel and well below 2.7 m.
- **6b middle.** Keeps the tracker but on an 80×60 image and only every 16 frames. Max fuel is set by the
  single heaviest frame (corner detection / image pyramid); the smaller image cuts that, and the longer
  interval mainly cuts *mean* fuel (most frames now do no image work: median fuel is 14k).

## Which two to upload

1. **`artifacts/aerovio-v8-rotation.wasm`**: the most accurate point. Locally it beats `tiny_cnn` on both
   ATE and max fuel.
2. **`artifacts/aerovio-leo-6a-imu-only.wasm`**: the cheap end of the Pareto curve (about 14k fuel, about 1,000×
   cheaper than `tiny_cnn`).

These are the two ends of the trade-off. Check the server Rules for how the "best" submission is
chosen: if it is ranked by ATE alone, use slot 2 for 6b instead, since 6a can't win on ATE.
Stage 1 is due **2026-11-15, 11:59 pm (America/Los_Angeles)**; quota is 2 submissions per team per day.

## Artifacts

| File | SHA-256 |
| --- | --- |
| `artifacts/aerovio-teacher-native160.wasm` | `b385923eb8759fce0a7c416d7cab0969e10ec30557fead61edaf6f2b632c895a` |
| `artifacts/aerovio-v8-rotation.wasm` | `108cb0c4ff7b14a913ed9637a8c658551e4608b8f670f7bc5197a4f1c389f065` |
| `artifacts/aerovio-leo-6a-imu-only.wasm` | `70512ad7a583053d1b5f8fbfa9307f0211d86b2ed8653019fb7bc90834ca13cb` |
| `artifacts/aerovio-leo-6b-ds4-every16.wasm` | `cf4bb1005094a6a3acc25a2bcbd1e450ab17444672e470bca985e8ddb60b419f` |
| `artifacts/aerovio-v8-tilt-k0.2.wasm` (stage 2, tilt only) | `619c16264dca0881b62689d8369cca5dcf733025e8309b32567d39ad24df577a` |
| `artifacts/aerovio-v8-tilt-k1.0.wasm` (stage 2, alternative) | `4bdb001c5b33f571d5d4749cf66999ffa748ed4754a7a666b5ec86026a5d8e2c` |
| `artifacts/aerovio-v8-tilt-bias-g0.2.wasm` (stage 2, old weights; server 5.04 m) | `cdc07749e198d9bc48b8c20c1ba82115e967bc19d7af8618c0858b577bedf307` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-nohover.wasm` (stage 2, no-hover weights; server 11.62 m, superseded) | `3bb0cee07238cce3a99f43630378e8bb60130f4befad48c559422463690dacc3` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-nohover-balanced.wasm` (stage 2, alternative) | `25bae31220b9970ea045ad1d1fd43827022d2ac7ae233169da8a2cb78a07e5e0` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-fixedcam.wasm` (**recommended next upload**) | `01365bce01db7b1205d337cc99b7617da742f7312f38c2bc7b09c51e667e75cc` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-visdamp.wasm` | `3f3262b0c07daf459ba56d20601d2cd93a24686e6ef9e5f3874868b99b558d02` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-scale0.7.wasm` | `eb526281c1f252a3fc9d87f03b561e57ff8e75ff18e0ac99fbf3bded54066a19` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-speedcap2.wasm` | `5b6fbecb4653ab2b29939a3adb7c874ce89d926ebb7ef4bc402b6b6437f1e6d4` |
| `artifacts/aerovio-v8-tilt-bias-g0.2-speedcap3.wasm` | `2d1a6f5dcc73f7455229598c656a14dc8a848a85dfa8252205637b3842024b4d` |

`artifacts/` and `results/` are gitignored. Raw reports are in `results/leo/<variant>/report.json`.

## Code changes (defaults unchanged)

- `workspace/visual_native/CMakeLists.txt`: new cache variable `VISUAL_INTERVAL` (default 8, must be >= 8;
  previously hard-coded 8) and new option `LEO_IMU_ONLY` (default OFF; refuses to combine with the V8
  visual options).
- `workspace/visual_native/solution.cpp`: under `#if LEO_IMU_ONLY`, the two `tracker.step` calls are
  skipped and the visual correction is zero.
- Check: the default build after these changes is **byte-identical** to the teacher bundle
  (`b385923e…`), and V8 rotation is byte-identical to its pre-change build (`108cb0c4…`).
- Stage 2: new option `LEO_GRAVITY_TILT` (default OFF) with cache variables `LEO_GRAVITY_TILT_GAIN`
  (K in 1/s, default 0.01), `LEO_GRAVITY_TILT_GATE` (m/s², default 0.5) and `LEO_GRAVITY_TILT_SIGN`
  (diagnostic, 1 or -1). The new header `gravity_tilt.h` takes the startup up direction from the mean
  accelerometer reading over the first 0.5 s, in frame-0 camera axes. Per IMU sample, when
  `| |a| - 9.81 | < gate`, it adds `K * (a/|a| × predicted up)` to the calibrated gyro rate. The error
  is perpendicular to up, so only roll and pitch change. The predicted up comes from the last emitted
  orientation, which includes V8's visual alignment. The definitions are only added when the option is
  ON. After this change the default build is still `b385923e…` and V8 is still `108cb0c4…`.
- Stage 2: new option `LEO_ONLINE_BIAS` (default OFF; needs `VISUAL_ROTATION_CORRECTION=ON`) with
  `LEO_ONLINE_BIAS_GAIN` (default 0.05), `LEO_ONLINE_BIAS_REJECT` (rad/s, default 0.03) and
  `LEO_ONLINE_BIAS_SIGN` (diagnostic). The new header `online_bias.h` holds the estimator.
  `visual_direction.h` keeps the unscaled refinement residual under `#if LEO_ONLINE_BIAS`, and
  `solution.cpp` subtracts the online bias from the gyro rate. Checked after the change: the default
  build is `b385923e…`, V8 is `108cb0c4…` and V8 + tilt K=0.2 is `619c1626…`, all unchanged.
- Tests: 90 host tests pass (5 test files need PyTorch, which is only in the training venv);
  the 11 native geometry checks pass.
- Stage 2: new option `LEO_WEIGHTS_NOHOVER` (default OFF) and cache path `LEO_WEIGHTS_NOHOVER_CHECKPOINT`
  (default `experiments/nohover-v11/compact`). `export_weights.py` takes `--checkpoint` (default
  unchanged) and writes `weights_nohover.h`. `inference.h` includes it under `#if LEO_WEIGHTS_NOHOVER`.
  `train_motion.py` gets `--compact-tolerance` (default 1e-5, unchanged). Checked after the change:
  default `b385923e…` and V8 + tilt + bias `cdc07749…` are unchanged. 128 host tests pass.
- Stage 2 runaway experiments: options `LEO_SPEED_CAP`, `LEO_VISUAL_DAMPING`, `LEO_OUTPUT_SCALE`
  (`translation_limits.h`, applied to the learned translation in `solution.cpp`; the tracker counts
  cycles without a heading fit) and `LEO_MLP_FIXED_CAMERA` (`solution.cpp` feeds `state_mean[9..13]`).
  All default OFF and compiled in only when ON. Checked: default `b385923e…` and V8 + tilt + bias
  `cdc07749…` are unchanged. 128 host tests pass.
- Side effect: the team's V9/V10 reproduction scripts check the teacher's exact source hashes, so they
  will refuse to run on this branch. Run them from `main`.

## Reproduce

```bash
# one-time setup
bash workspace/toolchain/download_dependencies.sh
# build the wasm runtime/OpenCV outside CMake: CMake 4 passes its wasm CC to child processes,
# which breaks CPython's native build step if the runtime is built from inside the configure
env -u CC -u CXX BUILD="$PWD/workspace/build/runtime" bash workspace/runtime/python/build.sh

TC="$PWD/workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake"
build() { d=workspace/visual_native/build/$1; shift
  cmake -S workspace/visual_native -B $d -DCMAKE_TOOLCHAIN_FILE="$TC" \
    -DVIOBENCH_NATIVE_DIR="$PWD/workspace/runtime/native" -DPython3_EXECUTABLE="$(command -v python)" "$@"
  cmake --build $d; }

python -m viobench bundle workspace/visual_native -o artifacts/aerovio-teacher-native160.wasm
build v8-rotation  -DVISUAL_ROTATION_CORRECTION=ON
build leo-imu-only -DLEO_IMU_ONLY=ON
build leo-ds4-i16  -DVISUAL_DOWNSAMPLE=4 -DVISUAL_INTERVAL=16
build tilt-k0.2    -DVISUAL_ROTATION_CORRECTION=ON -DLEO_GRAVITY_TILT=ON -DLEO_GRAVITY_TILT_GAIN=0.2
build tilt-bias-g0.2 -DVISUAL_ROTATION_CORRECTION=ON -DLEO_GRAVITY_TILT=ON -DLEO_GRAVITY_TILT_GAIN=0.2 \
  -DLEO_ONLINE_BIAS=ON -DLEO_ONLINE_BIAS_GAIN=0.2
# stage 2 validation sets
viobench simulate -o data/val-agile --scene-names ProcTHOR-Train-{40,60,80,100,120,140} --episodes 2 --seed 2000 --no-video
viobench simulate -o data/val-nohover --scene-names ProcTHOR-Train-{50,90,130,170} --episodes 2 --seed 3000 --hover 0 --no-video
# no-hover MLP retrain (data generation, caches, training; then the 3-set evaluation)
scripts/generate_nohover_data.sh
python scripts/generate_training_data.py --config configs/training-data-v1.json --output data/hyperdrone-motion-v2
scripts/run_nohover_training.sh   # needs .train-venv (configs/requirements-training.lock.txt)
build tilt-bias-nohover -DVISUAL_ROTATION_CORRECTION=ON -DLEO_GRAVITY_TILT=ON -DLEO_GRAVITY_TILT_GAIN=0.2 \
  -DLEO_ONLINE_BIAS=ON -DLEO_ONLINE_BIAS_GAIN=0.2 -DLEO_WEIGHTS_NOHOVER=ON
# runaway experiments (val-fast / val-narrow generation, then all evaluations)
scripts/run_runaway_eval.sh; scripts/run_runaway_eval2.sh
build tilt-bias-fixedcam -DVISUAL_ROTATION_CORRECTION=ON -DLEO_GRAVITY_TILT=ON -DLEO_GRAVITY_TILT_GAIN=0.2 \
  -DLEO_ONLINE_BIAS=ON -DLEO_ONLINE_BIAS_GAIN=0.2 -DLEO_MLP_FIXED_CAMERA=ON
# copy each build/<dir>/solution.wasm into artifacts/, then for each artifact:
python -m viobench check artifacts/<file>.wasm --frames 40
python -m viobench evaluate artifacts/<file>.wasm --limits configs/evaluation-limits.json -o results/leo/<name>
```

No server upload has been made from this branch.
