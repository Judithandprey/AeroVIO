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

## Online gyro bias from the visual rotation residual (`-DONLINE_GYRO_BIAS=ON`)

Each accepted visual rotation fit (`refine_rotation`, anchor vs current frame, one visual interval apart) leaves a
residual rotation that is roughly the remaining gyro bias times the interval. We add
`gain * sign * residual / interval` (mapped to the IMU frame) to the gyro bias, clamped to ±0.05 rad/s.
The residual sign convention is `ONLINE_GYRO_BIAS_SIGN=-1` (+1 made things worse).
All rows below use V8 rotation and `GRAVITY_TILT_GAIN=0.5`.

| Online bias | Scale | nohover raw ATE (m) | public raw ATE (m) | nohover end rotation error |
| --- | ---: | ---: | ---: | ---: |
| off | 1.0 | 3.179 | 2.992 | 46.7° |
| gain 0.3 | 1.0 | 3.060 | | 11.2° |
| **gain 1.0** | 1.0 | **3.023** | **2.976** | **9.9°** |
| gain 2.5 | 1.0 | 3.128 | | |
| gain 1.0 | 1.3 | 2.479 | | |
| gain 1.0 | 1.6 | 1.986 | 1.986 | |
| gain 1.0 | 2.0 | 1.633 | | |

Rotation is now mostly fixed (end error 47° → 10°). Locally the remaining raw ATE is path length (median Sim(3)
scale 2.5 at k = 1). The server's stage 2 Sim(3) scales suggested much less (about 1.4 for k = 1), so the safe
uploads are k = 1.0 (`aerovio-v8-rotfix.wasm`) and k = 1.3 (`aerovio-v8-rotfix-scale1.3.wasm`).

## Position/velocity Kalman filter (`-DVIO_KF=ON`, experimental, not better yet)

`vio_kf.h`: linear KF over p, v, a world-frame acceleration correction, and two capture-position clones.
Rotation comes from the existing gyro chain. Prediction integrates the preintegrated specific force with a gravity
estimate (mean specific force over the first 5 s); updates are the tracker's visual direction between capture
frames (perpendicular components, soft-gated) and the network velocity as a weak pseudo-measurement.

| Build (V8 + tilt + online bias) | s2-val raw ATE (m) |
| --- | ---: |
| no KF | 2.915 |
| KF, network sigma 0.3 m/s | 2.927 |
| KF, network sigma 1.0 m/s | 4.059 |
| KF, network sigma 1.0, heading sigma 0.1 | 3.676 |

Why it does not help yet: the remaining tilt error on s2-val is 3.5–6° (median), which leaks 0.6–1 m/s² of gravity
into the integrated acceleration, faster than the visual direction updates (every 0.4 s, many inconsistent) can
correct. Making accelerometer scale usable needs orientation in the filter (full error-state EKF), not an external
rotation.

## Error-state EKF (`-DESEKF=ON`, experimental)

`esekf.h`: 30-state error-state EKF (orientation, position, velocity, gyro/accel bias, gravity, two pose clones).
Updates: visual translation direction and visual rotation residual between capture frames, plus the network
velocity as a pseudo-measurement. On s2-val (window net h128, 48 episodes):

| Build | raw ATE (m) | Sim(3) ATE (m) | rotation RMSE | max fuel |
| --- | ---: | ---: | ---: | ---: |
| V8 + tilt + online bias + window net (no EKF) | 1.268 | 0.725 | 20.0° | 4.9M |
| ESEKF + window net, heading sigma 0.15 | 1.272 | 0.749 | 18.4° | 8.3M |
| ESEKF + window net, heading 0.15, net sigma 0.5 | 1.274 | 0.740 | 17.7° | 8.3M |
| ESEKF without the network velocity | 44.6 (one episode) | | | |

The filter slightly reduces rotation error but not position error, and without the network it diverges:
direction-only constraints between consecutive capture pairs are too weak to observe metric scale. A real gain
would need multi-frame feature constraints (MSCKF-style), which is a larger build.
