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
| `artifacts/aerovio-v8-tilt-k0.2.wasm` (stage 2, recommended) | `619c16264dca0881b62689d8369cca5dcf733025e8309b32567d39ad24df577a` |
| `artifacts/aerovio-v8-tilt-k1.0.wasm` (stage 2, alternative) | `4bdb001c5b33f571d5d4749cf66999ffa748ed4754a7a666b5ec86026a5d8e2c` |

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
- Tests: 90 host tests pass (5 test files need PyTorch, which is only in the training venv);
  the 11 native geometry checks pass.
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
# stage 2 validation sets
viobench simulate -o data/val-agile --scene-names ProcTHOR-Train-{40,60,80,100,120,140} --episodes 2 --seed 2000 --no-video
viobench simulate -o data/val-nohover --scene-names ProcTHOR-Train-{50,90,130,170} --episodes 2 --seed 3000 --hover 0 --no-video
# copy each build/<dir>/solution.wasm into artifacts/, then for each artifact:
python -m viobench check artifacts/<file>.wasm --frames 40
python -m viobench evaluate artifacts/<file>.wasm --limits configs/evaluation-limits.json -o results/leo/<name>
```

No server upload has been made from this branch.
