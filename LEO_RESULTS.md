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

`artifacts/` and `results/` are gitignored. Raw reports are in `results/leo/<variant>/report.json`.

## Code changes (defaults unchanged)

- `workspace/visual_native/CMakeLists.txt`: new cache variable `VISUAL_INTERVAL` (default 8, must be >= 8;
  previously hard-coded 8) and new option `LEO_IMU_ONLY` (default OFF; refuses to combine with the V8
  visual options).
- `workspace/visual_native/solution.cpp`: under `#if LEO_IMU_ONLY`, the two `tracker.step` calls are
  skipped and the visual correction is zero.
- Check: the default build after these changes is **byte-identical** to the teacher bundle
  (`b385923e…`), and V8 rotation is byte-identical to its pre-change build (`108cb0c4…`).
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
# copy each build/<dir>/solution.wasm into artifacts/, then for each artifact:
python -m viobench check artifacts/<file>.wasm --frames 40
python -m viobench evaluate artifacts/<file>.wasm --limits configs/evaluation-limits.json -o results/leo/<name>
```

No server upload has been made from this branch.
