# Native visual translation

This viobench solution builds the original compact IMU checkpoint into a native WASM module. It applies calibrated gyro rotation and updates translation using bidirectional sparse optical flow every eight frames. A fixed-size Shi–Tomasi kernel computes corner scores in two halves. The correction still begins four frames after the new keyframe.

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
