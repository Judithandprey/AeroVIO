# Native visual translation

This viobench solution builds the original compact IMU checkpoint into a native WASM module. It applies calibrated gyro rotation and updates translation using bidirectional sparse optical flow every eight frames. The image pyramid is built directly in padded buffers. The fixed-size Lucas–Kanade kernel uses WASM SIMD and reuses gradient data across iterations and rearranged pixel pairs across adjacent rows. The tracker spreads the work across four groups of corners before applying the correction four frames after the new keyframe. Corner scoring and selection prepare the next round of tracks.

Sobel combines neighboring products in registers before writing the horizontal sum. Scharr reuses horizontal taps across three output rows. The Gaussian downsample reuses its even and odd source columns between adjacent SIMD blocks. At 160×120, the tracker counts the early forward iterations and photometric checks at capture to decide how many coarse gradient rows to build. The first tracking batch finishes the coarse gradient before reverse tracking. The tracker starts one coarse forward estimate at capture and finishes that track in its usual batch. Forward tracks must pass the image boundary and parallax checks before the tracker evaluates the final photometric error.

Both pyramids allocate their backing storage during configuration. Scharr stores the x and y gradients in separate padded planes. The Sobel product and response passes unroll their fixed-width loops. At 160×120, the scoring frames collect local maxima once their neighboring rows have been scored. The selection frame filters candidates with the global threshold, sorts score bins only as needed, and uses a bitset to enforce spacing. Equal scores still sort by descending row, then descending column.

LK interpolates the last column for two rows together and adds each contribution in its original order. Patch coordinates, bilinear weights and displacement stay in SIMD through each iteration. The convergence checks still use double precision. The photometric check handles the last column for four rows together. At 160×120, the backward pass reuses those finest-level pixel samples when preparing its source window. Its first coarse iteration can sample integer and half-pixel origins directly with integer sums. Direction fitting scores each pair once in its original trial order and drops candidates that cannot improve the current fit.

Activate the project's evaluation environment so the build can use Python, NumPy, and OpenCV to export the checkpoint weights.

```bash
mkdir -p artifacts
python -m viobench bundle workspace/visual_native -o artifacts/imu-compact-visual-native.wasm
python -m viobench check artifacts/imu-compact-visual-native.wasm --frames 40
python -m viobench evaluate artifacts/imu-compact-visual-native.wasm --limits configs/evaluation-limits.json
```

The default tracks at 160×120. A fit with only eight or nine valid tracks must pass stricter geometric and depth checks and receives a lower correction weight. Set `-DVISUAL_SPARSE_DIRECTION=OFF` in a separate CMake build directory to require at least ten tracks for every direct fit.

Use a separate CMake build directory for the 80×60 variant, which requires at least ten tracks.

```bash
cmake -S workspace/visual_native -B workspace/visual_native/build/80 \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake" \
  -DPython3_EXECUTABLE="$(command -v python)" -DVISUAL_DOWNSAMPLE=4
cmake --build workspace/visual_native/build/80
cp workspace/visual_native/build/80/solution.wasm artifacts/imu-compact-visual-native-80.wasm
```
