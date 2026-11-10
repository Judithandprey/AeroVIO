# Development setup and artifact reproduction

The experimental branch contains a trained **small motion CNN**, a version with
training-only warping supervision, optional sparse-map correction, and explicit
rotation/input ablations, a rotation-focused auxiliary-training experiment, and
an optional fixed startup gyro-calibration experiment.
The original official KLT + IMU baseline is frozen at
`baseline-klt-imu-v1`.
The working `workspace/solution/` currently selects the pose-only learned model.

The runner and runtime are pinned to
[viobench 2072fa9](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca).
Read [architecture/training](learned-motion.md), the
[calibration guide](gyro-bias-calibration.md), the
[v4 validation decision](../experiments/gyro-bias-v4/validation.md), the
[rotation-training guide](rotation-focused-training.md), the
[completed v3 record](../experiments/rotation-focus-v3/README.md), the
[completed rotation/input experiment](../experiments/motion-ablation-v2/README.md),
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

## Reproduce rotation-focused training

The v3 experiment reuses the generated raw episodes, original real-motion cache,
and the v1 warp model's `best.pt`. Run the first training reproduction in
[learned-motion.md](learned-motion.md) if those ignored local files are absent.
Do not regenerate or overwrite an earlier cache/model directory. The example
commands below use new `-repro` directories.

```bash
.train-venv/bin/python scripts/prepare_rotation_data.py --manifest data/hyperdrone-motion-v2/data_manifest.json --real-cache data/motion-cache-v2 --out data/rotation-aux-v3-repro
.train-venv/bin/python scripts/train_rotation_focus.py --cache data/motion-cache-v2 --rotation-cache data/rotation-aux-v3-repro --initial-model models/motion-warp-v1 --variant control --out models/rotation-control-v3-repro --epochs 20 --batch 64 --aux-batch 32 --lr 0.0001 --seed 17 --threads 4 --translation-guard 1.05
.train-venv/bin/python scripts/train_rotation_focus.py --cache data/motion-cache-v2 --rotation-cache data/rotation-aux-v3-repro --initial-model models/motion-warp-v1 --variant focus --out models/rotation-focus-v3-repro --epochs 20 --batch 64 --aux-batch 32 --lr 0.0001 --seed 17 --threads 4 --aux-weight 0.1 --translation-guard 1.05
```

The auxiliary data is only for rotation supervision; original real-data pose
and photometric losses remain active. Both runs use the fixed protocol in
[`rotation-focus-v3.json`](../configs/rotation-focus-v3.json). The selected
checkpoint is top-level `net.onnx`; `diagnostic-last/net.onnx` preserves the final
epoch whether or not it qualified for selection. In the recorded run, **both
selected epoch zero**. No deployed improvement was established.

Package all four roles explicitly when reproducing the public diagnostics:

```bash
.venv/bin/python scripts/package_motion.py --model models/rotation-control-v3-repro --name rotation-control-selected-v3-repro
.venv/bin/python scripts/package_motion.py --model models/rotation-focus-v3-repro --name rotation-focus-selected-v3-repro
.venv/bin/python scripts/package_motion.py --model models/rotation-control-v3-repro/diagnostic-last --name rotation-control-last-v3-repro
.venv/bin/python scripts/package_motion.py --model models/rotation-focus-v3-repro/diagnostic-last --name rotation-focus-last-v3-repro
```

The selected and unselected artifacts have different scientific roles, even when
a filename is similar. The [pre-public addendum](../configs/rotation-focus-v3-public-diagnostics.json)
includes the two fixed epoch-20 candidates without changing the original
selection rule. The six-way comparison also includes the unchanged official
baseline and starting warp model. All six public runs completed 10/10 episodes;
consult the [v3 record](../experiments/rotation-focus-v3/README.md) for verification
and metrics. Bundle checks and full evaluations must run serially after training
and other compute stop. Preserve earlier v1/v2 reports.

The recorded retention-only reruns are in
`models/rotation-control-v3-retained/` and `models/rotation-focus-v3-retained/`.
Their compact selected-model records are archived under
`experiments/rotation-focus-v3/control/` and `focus/`; their `diagnostic-last/`
subdirectories identify the unselected epoch-20 exports. See the
[rotation-training guide](rotation-focused-training.md) and
[input illustration](../experiments/rotation-focus-v3/rotation-examples.png).

## Reproduce startup gyro calibration

V4 needs no retraining. It uses the archived v1 warp model and the original
simulated validation flights/cache already described in [learned-motion.md](learned-motion.md).
The [frozen protocol](../configs/gyro-bias-v4.json) defines four cells and the
strict selection rule. Validate on scenes 5/6 before any public comparison,
using new output directories to preserve the recorded evidence:

```bash
.venv/bin/python scripts/validate_gyro_bias.py --protocol configs/gyro-bias-v4.json --raw-dir results/gyro-bias-v4-validation-repro --out-dir reports/gyro-bias-v4-validation-repro
```

This runs actual host `Solution.step` twice per cell and flight. It checks
predictor-input/output identity, local translation, causal activation and exact
repeats, then uses reference trajectories for scoring. It is not a WASM speed
measurement. The recorded validation retained `current`: calibrated-gyro improved
mean rotation RMSE by 68.77% but increased aligned ATE by 0.02752%, which fails
the fixed nonregression guard. Neither that guard nor the selection changed.

Package the same four cells from one guest source/runtime revision:

```bash
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name gyro-v4-current-repro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name gyro-v4-raw-repro --disable-rotation-residual
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name gyro-v4-cal-residual-repro --calibrate-gyro
.venv/bin/python scripts/package_motion.py --model experiments/learned-motion-v1/warp --name gyro-v4-cal-gyro-repro --calibrate-gyro --disable-rotation-residual
for name in current raw cal-residual cal-gyro; do
  .venv/bin/python -m viobench check "artifacts/gyro-v4-${name}-repro.wasm" --frames 40
done
```

`--calibrate-gyro` enables the fixed rule and cannot be combined with `--map`.
The predictor still receives raw IMU features; only a separate future-rotation
integration path uses the accepted bias estimate. Calibration is disabled by
default. The four recorded packages passed their 40-frame checks, and the full
host suite passed 104 tests. These checks do not establish trajectory accuracy.

After the baseline reproduction below and after all competing compute stops,
run the public evaluation serially. The
[pre-public replication addendum](../configs/gyro-bias-v4-public-replication.json)
adds a diagnostic repeat without changing validation selection. The seven roles
are baseline, four cells, selected-current repeat and calibrated-gyro repeat:

```bash
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/klt-imu-repro.wasm --run-dir results/gyro-v4-baseline-repro --verify-baseline
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-current-repro.wasm --run-dir results/gyro-v4-current-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-raw-repro.wasm --run-dir results/gyro-v4-gyro-only-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-cal-residual-repro.wasm --run-dir results/gyro-v4-calibrated-residual-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-cal-gyro-repro.wasm --run-dir results/gyro-v4-calibrated-gyro-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-current-repro.wasm --run-dir results/gyro-v4-selected-repeat-repro
.venv/bin/python scripts/evaluate_public.py --wasm artifacts/gyro-v4-cal-gyro-repro.wasm --run-dir results/gyro-v4-calibrated-gyro-repeat-repro
```

All seven recorded public runs completed; actual full trajectory and per-frame
fuel repeats pass. See the [v4 comparison](../experiments/gyro-bias-v4/comparison.md).
The verified Windows files are `artifacts/AeroVIO-gyro-v4-current.wasm` and
`artifacts/AeroVIO-gyro-v4-cal-gyro.wasm` in the Windows mirror; exact hashes and
roles are in the [copy record](../experiments/gyro-bias-v4/windows-artifacts.json). Keep
the selected-current and unselected diagnostic roles explicit in all reports;
even an exact diagnostic repeat cannot change the frozen validation decision.

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
