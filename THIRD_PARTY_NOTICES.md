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
training-only masked warping loss, bounded sparse-map correction, controlled
rotation/image-input ablations, constant-feature folding for the IMU-only export,
and measurement/provenance tooling. Their behavior and evidence limits are
documented in `docs/learned-motion.md`, `experiments/learned-motion-v1/`, and
`experiments/motion-ablation-v2/`. The compact graph derives from the same
upstream-inspired trained architecture; it is not a separately trained model.
These changes do not claim invention of CNN odometry, image warping, KLT,
triangulation, PnP, or the upstream estimator.

The rotation-focused experiment adds calibrated pure-rotation view synthesis,
consistent synthetic IMU samples, and matched continued-training controls. Its
homography is standard projective geometry, documented in
[OpenCV's tutorial](https://docs.opencv.org/4.11.0/d9/dab/tutorial_homography.html).
Illustrative image pairs derive from the same simulated ProcTHOR observations;
they do not represent a newly invented renderer or learned world model. See
`experiments/rotation-focus-v3/` for the negative transfer result and provenance.

## Startup gyro calibration

The optional v4 calibrator adapts the gyro-mean bias estimate and the
accelerometer/gyro low-variation thresholds from upstream's official `klt_imu`
example. Its one-second collection window and bias/noise prior guard use the
pinned HyperDrone task's documented initial hold and stored sensor configuration.
The baseline implementation and generator source snapshots are retained with
hashes under `experiments/gyro-bias-v4/startup-audit-sources/`.

Project-specific changes are bounded causal collection, explicit acceptance and
rejection states, next-interval activation without resetting pose, and a separate
debiased integration path that preserves the trained model's raw inputs. The
four-cell comparison and provenance distinguish calibration from the learned
residual. This does not claim invention of gyro-bias estimation or modify the
frozen KLT baseline's initialization. See `docs/gyro-bias-calibration.md` for
assumptions and `experiments/gyro-bias-v4/validation.md` for the retained-current
decision; public performance is pending.

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
