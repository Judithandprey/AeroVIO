# Development setup

AeroVIO uses Python and the official **KLT + IMU** example, pinned to
[viobench 2072fa9](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca).
The estimator and generated runtime recipes are preserved without modifications.
See `experiments/baseline-klt-imu/` for the verified execution boundary and results.

## Environment

Use Linux, macOS, or WSL2. On Windows, keep the build checkout in the Linux
filesystem. The verified setup uses Ubuntu in WSL, with a checkout at
`/root/projects/AeroVIO`; the Windows documentation/Git mirror is at
`C:/Users/ROG/Documents/GitHub/AeroVIO`.

```bash
sudo apt update
sudo apt install build-essential cmake curl git python3-venv xz-utils pkg-config
git clone https://github.com/Judithandprey/AeroVIO.git
cd AeroVIO
python3 -m venv .venv
source .venv/bin/activate
mkdir -p external
git clone https://github.com/jonas-eschmann/viobench.git external/viobench
git -C external/viobench checkout --detach 2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca
python -m pip install -r configs/requirements-eval.lock.txt
python -m pip install --no-deps -e ./external/viobench
```

The host dependency snapshot was recorded on Linux with Python 3.14.4. These
pins are evidence of this environment; availability on other Python/platform
combinations is not guaranteed. Guest versions are separately pinned in the
upstream runtime recipes: CPython 3.14.4, NumPy 2.3.4, OpenCV 4.12.0, WASI SDK 34.
The host OpenCV version is not the OpenCV version executing the estimator.

`workspace/` is already tracked. **Do not run `viobench init` over it.** Its
original initialization was `viobench init python workspace`, followed by copying
`external/viobench/viobench/solution/examples/python/klt_imu/solution.py` into
`workspace/solution/solution.py`. Attribution and source SHA-256 are recorded.

## Build and check

```bash
mkdir -p artifacts
python -m viobench bundle workspace/solution -o artifacts/klt_imu.wasm
python -m viobench check artifacts/klt_imu.wasm --frames 20 --logs logs/smoke
```

The first bundle builds the Python/NumPy/OpenCV WASM runtime. Later solution-only
bundles reuse it. The 20-frame check crosses the baseline's initial 0.5-second
calibration period, unlike the five-frame default. It verifies the ABI and basic
execution, not trajectory accuracy.

## Reproduce the public evaluation

```bash
python scripts/evaluate_public.py --run-dir results/baseline-public --verify-baseline
python scripts/evaluate_public.py --run-dir results/baseline-repeat --max-episodes 1 --verify-baseline
```

Choose a new directory name for each run; the helper refuses to overwrite one.
It uses the explicit ten-episode public manifest and saved sandbox limits,
checks the clean pinned upstream checkout, and records source/binary/config
hashes, dependency versions, commands, logs, and runner exit status. The baseline
flag checks the source hash; build the binary from that source first.

Public archives are downloaded once into conta's cache. Videos are disabled to
keep baseline evaluation focused on metrics. `viobench evaluate -o ...` enables
videos; the helper instead runs in its evidence directory and uses the runner's
default output folder.

Results appear in `results/<run>/report-klt_imu/report.json`. Keep generated
trajectories, fuel samples, binaries, data, and logs local. Copy compact reports
and manifests into `experiments/` when preserving a completed experiment.

## Compare an experiment

Build the experiment under a different artifact name, then run the same helper:

```bash
python -m viobench bundle workspace/solution -o artifacts/perspective.wasm
python scripts/evaluate_public.py --wasm artifacts/perspective.wasm --run-dir results/perspective-public
python -m viobench leaderboard results/baseline-public/report-klt_imu results/perspective-public/report-perspective
```

Compare the same episode hashes and limits, all completion/failure counts,
per-episode errors, equal-episode **mean Sim(3) ATE**, and **maximum fuel per
frame**. Also inspect unaligned ATE, rotation and velocity errors: Sim(3)
alignment can hide global scale errors. Do not claim a winning method from
one favorable episode, or treat public-set tuning as held-out generalization.

## Before a course submission

No submission is performed by these commands. Verify active rules and limits at
[the submission server](https://viobench.rl.tools/) before uploading. The course's
November 9 update states **2 submissions/team/day**. Saved local limits have not
been confirmed against the authenticated server configuration. Keep team tokens
and account credentials outside Git.
