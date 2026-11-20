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
