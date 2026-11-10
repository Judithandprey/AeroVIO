# Upstream attribution

AeroVIO builds on [jonas-eschmann/viobench](https://github.com/jonas-eschmann/viobench),
pinned to revision `2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`.
The upstream `pyproject.toml` declares the MIT license; that checkout has no root
LICENSE file. Original notices in copied files are preserved.

## Frozen official baseline

At AeroVIO tag `baseline-klt-imu-v1`, `workspace/solution/solution.py` is the
byte-identical upstream example at
`viobench/solution/examples/python/klt_imu/solution.py`.
The baseline artifact and evidence retain this identity. The working experimental
branch has since replaced that file with a learned estimator; it must not be
presented as the unchanged KLT example.

## Learned-motion implementation

The three-layer 16/32/64-channel convolutional backbone and the idea of predicting
camera translation plus an IMU rotation residual adapt upstream
`viobench/solution/examples/python/tiny_cnn/train.py` and its `solution/` code.
The IMU preintegration, pose composition, camera-axis interval-twist convention,
and matrix-to-quaternion helpers also draw on those examples.

Project-specific changes include normalized camera-intrinsic/time inputs,
including the first IMU interval, explicit normalization/ONNX metadata contracts,
causality and numerical checks, separate generated train/validation scenes,
training-only masked warping loss, bounded sparse-map correction, and controlled
measurement/provenance tooling. Their behavior and evidence limits are documented
in `docs/learned-motion.md` and `experiments/learned-motion-v1/`.
These changes do not claim invention of CNN odometry, image warping, KLT,
triangulation, PnP, or the upstream estimator.

## Runtime, simulation and dependencies

The generated `workspace/runtime/`, `workspace/toolchain/` and workspace ignore
rules originate from the pinned upstream revision; runtime recipes remain
unmodified. Synthetic training data is generated with HyperDrone/HyperLux using
ProcTHOR scene assets and the upstream drone/camera/IMU configuration.

CPython, NumPy, OpenCV, WASI SDK, Wasmtime, PyTorch, ONNX, HyperDrone/HyperLux,
scene assets and their dependencies retain their respective upstream licenses
and attribution. Training-only libraries are not included in the deployed
solution bundle.

Course PDFs in `docs/course/` retain their authors' rights and are stored in this
private repository for coursework.
