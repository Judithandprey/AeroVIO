# Development setup and artifact reproduction

The experimental branch contains a trained **small motion CNN**, a version with
training-only warping supervision, optional sparse-map correction, and explicit
rotation/input ablations. The original official KLT + IMU baseline is frozen at
`baseline-klt-imu-v1`.
The working `workspace/solution/` currently selects the pose-only learned model.

The runner and runtime are pinned to
[viobench 2072fa9](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca).
Read [architecture/training](learned-motion.md), the
[current rotation/input experiment](../experiments/motion-ablation-v2/README.md),
and [earlier geometry evidence](../experiments/learned-motion-v1/README.md)
alongside these commands. The [warping/world-model guide](warping-and-world-models.md)
explains the scope of the implemented geometry.

## Environment

Use Linux, macOS, or WSL2. On Windows, keep builds in the Linux filesystem. The
verified build checkout is WSL Ubuntu `/root/projects/AeroVIO`; the Windows Git
mirror is `C:/Users/ROG/Documents/GitHub/AeroVIO`.

```bash
sudo apt update
sudo apt install build-essential cmake curl git python3-venv python3-dev xz-utils pkg-config
git clone --branch experiment/perspective-map https://github.com/Judithandprey/AeroVIO.git
cd AeroVIO
python3 -m venv .venv
.venv/bin/python -m pip install -r configs/requirements-eval.lock.txt
mkdir -p external
git clone https://github.com/jonas-eschmann/viobench.git external/viobench
git -C external/viobench checkout --detach 2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca
.venv/bin/python -m pip install --no-deps -e ./external/viobench
```

The host snapshot was recorded on Linux/Python 3.14.4; availability on other
Python/platform combinations is not guaranteed. Guest versions are separately
pinned by runtime recipes: CPython 3.14.4, NumPy 2.3.4, OpenCV 4.12.0, WASI SDK 34.
Host OpenCV is not the OpenCV version executing inside WASM. Keep `.venv` for
evaluation; training dependencies belong in the separate `.train-venv` described
in `learned-motion.md`.

`workspace/` already exists. Do not run `viobench init` over it. Upstream runtime
recipes remain unchanged; the learned solution is an attributed adaptation.

## Build the runtime and current default model

```bash
mkdir -p artifacts
.venv/bin/python -m viobench bundle workspace/solution -o artifacts/motion-default.wasm
.venv/bin/python -m viobench check artifacts/motion-default.wasm --frames 20 --logs logs/motion-default-check
```

The first bundle compiles the Python/NumPy/OpenCV WASM runtime; subsequent bundles
reuse `workspace/build/runtime/dist/runtime.wasm`. This command builds the
**pose-only learned model**, not KLT + IMU. The 20-frame synthetic check verifies
basic ABI execution; trajectory accuracy requires the public evaluation below.

## Package the recorded variants

The compact model snapshots under `experiments/learned-motion-v1/pose/` and
`warp/` include `net.onnx`, `meta.json`, training history and provenance. Packaging
checks the model hash and stages copies without modifying those snapshots.
Choose unused artifact names; the helper refuses to overwrite earlier artifacts.

```bash
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/pose --name motion-pose-repro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name motion-warp-repro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name motion-warp-map-repro --map
.venv/bin/python -m viobench check artifacts/motion-pose-repro.wasm --frames 20
.venv/bin/python -m viobench check artifacts/motion-warp-repro.wasm --frames 20
.venv/bin/python -m viobench check artifacts/motion-warp-map-repro.wasm --frames 20
```

The map variant uses exactly the warp model's weights; only the geometric map
switch changes. The runtime must exist before `package_motion.py` is called.
Model inference is OpenCV DNN/ONNX. PyTorch, training depth and warping loss are
not included in the deployed solution.

## Package the rotation and input ablations

The on/off pair uses the same v1 warping checkpoint and current guest source.
Only `rotation_residual_enabled` changes; it defaults to `true` when absent.
The `--disable-rotation-residual` switch uses gyro-only rotation while preserving
predicted local translation. It cannot be combined with `--map`.

```bash
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name motion-current-v2-repro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name motion-no-rotation-v2-repro --disable-rotation-residual
.venv/bin/python scripts/package_motion.py --model experiments/motion-ablation-v2/imu-only --name motion-imu-only-v2-repro
.venv/bin/python scripts/package_motion.py --model experiments/motion-ablation-v2/imu-only-compact --name motion-imu-compact-v2-repro
.venv/bin/python -m viobench check artifacts/motion-current-v2-repro.wasm --frames 20
.venv/bin/python -m viobench check artifacts/motion-no-rotation-v2-repro.wasm --frames 20
.venv/bin/python -m viobench check artifacts/motion-imu-only-v2-repro.wasm --frames 20
.venv/bin/python -m viobench check artifacts/motion-imu-compact-v2-repro.wasm --frames 20
```

Packaging inherits the checkpoint's `input_mode` and `network_inputs` metadata.
The primary `imu-only` control retains the complete CNN and receives zero images;
it still uses the 14-value state, including intrinsics and dt. The compact export
folds its constant visual features into the head and takes only normalized state.
It is a separate compute comparison using the same trained function, with
floating-point equivalence checks archived alongside it. Its 5,949-byte ONNX size
does not describe the complete WASM bundle, which includes the shared runtime.

Run all compared variants from the same guest source revision and existing
runtime. Use the new filenames in the evaluation commands below, give each run
its own directory, and compare the rotation-on and rotation-off pair explicitly:

```bash
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-current-v2-repro.wasm --run-dir results/motion-current-v2-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-no-rotation-v2-repro.wasm --run-dir results/motion-no-rotation-v2-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-imu-only-v2-repro.wasm --run-dir results/motion-imu-only-v2-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-imu-compact-v2-repro.wasm --run-dir results/motion-imu-compact-v2-repro
.venv/bin/python scripts/compare_motion_runs.py --run rotation-on results/motion-current-v2-repro --run rotation-off results/motion-no-rotation-v2-repro --run imu-only results/motion-imu-only-v2-repro --run imu-compact results/motion-imu-compact-v2-repro --out reports/ablation-repro-comparison.md
```

See the [v2 comparison](../experiments/motion-ablation-v2/comparison.md) and
[rotation diagnostics](../experiments/motion-ablation-v2/rotation-diagnostics.md)
for recorded results and interpretation. All v1 evidence remains in its original
experiment directory; the workspace's default model is still pose-only.

The Windows copy names are `AeroVIO-motion-current-v2.wasm`,
`AeroVIO-motion-no-rotation-v2.wasm`, `AeroVIO-motion-imu-only-v2.wasm`, and
`AeroVIO-motion-imu-compact-v2.wasm` under
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/`. Copy/hash verification belongs
in the v2 record; these filenames alone do not prove evaluation or submission.

## Reproduce the frozen baseline

Use the original evaluated `artifacts/klt_imu.wasm` if available, or create a
separate tagged worktree and bundle its source with the same runtime:

```bash
git worktree add --detach ../AeroVIO-baseline baseline-klt-imu-v1
.venv/bin/python -m viobench bundle ../AeroVIO-baseline/workspace/solution --runtime workspace/build/runtime/dist/runtime.wasm -o artifacts/klt-imu-repro.wasm
.venv/bin/python -m viobench check artifacts/klt-imu-repro.wasm --frames 20
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/klt-imu-repro.wasm --run-dir results/baseline-repro --verify-baseline
```

Do not copy the tagged baseline over the experiment's working solution. The
verification flag checks the **embedded** source against the recorded baseline
hash, independently of the current working tree.

## Evaluate and compare

Run measurements sequentially after data generation and training stop, so
competing work does not distort wall time. Use a new run directory each time.

```bash
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-pose-repro.wasm --run-dir results/motion-pose-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-warp-repro.wasm --run-dir results/motion-warp-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/motion-warp-map-repro.wasm --run-dir results/motion-warp-map-repro
.venv/bin/python scripts/compare_motion_runs.py --run B0 results/baseline-repro --run N1 results/motion-pose-repro --run N2 results/motion-warp-repro --run N3 results/motion-warp-map-repro --out reports/repro-comparison.md
```

The evaluator uses `configs/public-episodes.json` and
`configs/evaluation-limits.json`, checks the clean pinned upstream checkout,
and records commands, environment, exit status and hashes from the actual WASM
payload. Public archives download once into conta's cache. Videos are disabled.
Each run writes `manifest.json`, `evaluate.log`, and
`report-<artifact-name>/report.json` with trajectories/fuel arrays under its logs.

The comparison helper rejects incompatible episode selections, limits, runner,
Wasmtime or runtime identities. Compare all completion/failure counts, individual
episodes, equal-episode mean Sim(3) ATE, maximum fuel/frame, unaligned ATE,
rotation and twist errors. Wall time is host-specific and separate from fuel.
Sim(3) alignment can hide global scale error. These are local public-set results,
not held-out generalization or robot control evidence.

## Train again

Follow the complete commands in [learned-motion.md](learned-motion.md). Training
uses separately generated HyperDrone data and scene-disjoint validation. Preserve
the public comparison episodes from training and checkpoint selection. The
recorded experiment fixes seeds, initialization, batch order and schedule across
the pose-only and warp losses. Use new dataset/cache/model output directories to
keep the earlier evidence intact.

Host tests require `.train-venv` because the warping tests use PyTorch:

```bash
.train-venv/bin/python -m unittest discover -s tests -v
```

## Before a course submission

These commands do not submit to the course server. Verify active Rules and
remaining quota at [the server](https://viobench.rl.tools/) before uploading.
The November 9 course update states 2 submissions/team/day. Saved local limits
are not confirmed current authenticated server settings. Keep team tokens and
account credentials outside Git.
