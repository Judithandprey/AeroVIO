# Stage 2: position scale (branch `tae_testbranch`)

Stage 2 scores **unaligned ATE** (no Sim(3) alignment). On the public set the Sim(3) fit needed a scale of
1.8 to 4.0 (median 2.4) for our trajectories, so the network under-estimates path length.
`-DPOSITION_SCALE=<k>` multiplies the learned translation increment (default 1.0 keeps the old build).

Local public set (10 episodes), viobench `9b1b079`, macOS arm64, wasi-sdk from the lean-vio workspace:

| Variant | k | Raw ATE mean (m) | Max fuel/frame |
| --- | ---: | ---: | ---: |
| V8 rotation | 1.0 | 3.007 | 2,233,398 |
| V8 rotation | 1.5 | 2.196 | 2,233,404 |
| **V8 rotation** | **2.0** | **1.803** | 2,233,407 |
| V8 rotation | 2.5 | 2.072 | 2,233,404 |
| V8 rotation | 3.0 | 2.656 | 2,233,404 |
| IMU only | 1.0 | 3.524 | 13,852 |
| IMU only | 1.5 | 3.404 | 13,858 |
| IMU only | 2.0 | 3.842 | 13,861 |

IMU only gets worse past 1.5: heading drift dominates, and a longer path amplifies it.
Upload candidate: V8 rotation with k = 2.0 (`artifacts/aerovio-v8-scale2.wasm`,
SHA-256 `95edff6d27f4eae8acaeea6bfc6efc00bd13fef38b23c0e2ddd728eb6e7db4a4`).

```bash
cmake -S workspace/visual_native -B build/v8-s2 -DCMAKE_TOOLCHAIN_FILE="$TC" \
  -DVIOBENCH_NATIVE_DIR="$PWD/workspace/runtime/native" -DVISUAL_ROTATION_CORRECTION=ON -DPOSITION_SCALE=2.0
cmake --build build/v8-s2
viobench evaluate build/v8-s2/solution.wasm --limits configs/evaluation-limits.json
```

## Gravity tilt correction (`-DGRAVITY_TILT_GAIN=<1/s>`, default 0 = off)

Stage 2 flights likely start without the 1 s hover, so `gyro_bias.h` rejects calibration and raw gyro bias
(up to 0.02 rad/s in the simulator) drifts the rotation. To test this locally, `nohover` copies of the 10 public
episodes drop the first 2 s. `gravity_tilt.h` anchors the mean specific force in the camera-0 frame over the
first 5 s and nudges roll/pitch so the 3 s running mean stays aligned with it. Yaw is not observable this way.

| Build | Set | Raw ATE mean (m) | Median tilt error | End tilt error | End total rotation error |
| --- | --- | ---: | ---: | ---: | ---: |
| V8 rotation, gain 0 | nohover | 3.198 | 11.4° | 13.3° | 49.6° |
| V8 rotation, gain 0.5 | nohover | 3.179 | 3.2° | 2.3° | 46.7° |
| IMU only, gain 0 | nohover | 3.815 | 12.9° | 15.3° | 51.9° |
| IMU only, gain 0.5 | nohover | 3.748 | 3.4° | 3.0° | 48.7° |
| V8 rotation, gain 0.5 | public | 2.992 (was 3.007) | | | |
| IMU only, gain 0.5 | public | 3.477 (was 3.524) | | | |

Tilt is fixed, but raw ATE barely moves: the remaining ~45° end error is yaw, which needs vision.
