# Development setup

This setup is adapted from the [upstream student guide at 2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/docs/STUDENT_GUIDE.md). Commands have been checked against that source but **have not been executed** as part of creating this repository.

## Environment

Upstream supports Linux and macOS. On Windows, use **WSL2**, with the build checkout inside the Linux filesystem (for example `~/projects/AeroVIO`). The initial Windows checkout is suitable for editing these documents; create a separate Linux checkout for builds.

Upstream lists Python 3.10-3.14, git, curl, CMake, a C compiler, make, and venv. On Ubuntu:

```bash
sudo apt install build-essential cmake curl git python3-venv xz-utils
```

## Checkout and dependency

Run in a Linux/macOS shell:

```bash
git clone https://github.com/Judithandprey/AeroVIO.git
cd AeroVIO
python3 -m venv .venv
source .venv/bin/activate
mkdir -p external
git clone https://github.com/jonas-eschmann/viobench.git external/viobench
git -C external/viobench checkout --detach 2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca
python -m pip install -e ./external/viobench
```

The reviewed upstream revision is recorded in [upstream.lock.json](../upstream.lock.json). This pins upstream source only, not every transitive dependency. After establishing a working environment, record exact dependency versions and build-tool versions for reproducibility.

For training on the lab GPUs, the upstream guide uses `python -m pip install -e "./external/viobench[train]"`. Follow the lab CUDA instructions in the course slides after checking the workstation's actual environment.

## Initialize a solution workspace

Choose one implementation language before running initialization. For Python:

```bash
viobench init python workspace
```

For C/C++, the corresponding command is `viobench init native workspace`. Do not run both against the same directory. No choice has been made and no workspace has been generated yet.

The generated workspace contains `solution/`, `toolchain/`, and `runtime/`. Commit source/build recipes; generated build products remain ignored. Upstream alternatives, including `dead_reckoning`, `klt_imu`, and `tiny_cnn`, are in the upstream examples directory.

## Local validation workflow

From the repository root, after workspace initialization:

```bash
mkdir -p artifacts results
viobench bundle workspace/solution -o artifacts/baseline.wasm
viobench check artifacts/baseline.wasm
viobench evaluate artifacts/baseline.wasm -o results/baseline
sha256sum artifacts/baseline.wasm
```

The first Python bundle may build the runtime and take 10-15 minutes. Public episodes are downloaded into conta's cache. The upstream guide says evaluation needs no GPU.

Record results using [the experiment format](../experiments/README.md). A successful `check` is a synthetic smoke check, not evidence of trajectory accuracy or held-out performance.

## Before a course submission

Use the [submission server](https://viobench.rl.tools/) to verify the active rules and remaining quota. The course's November 9 update states **2 submissions/team/day**, overriding the older generic one-per-day default in upstream documentation.

Local evaluation defaults can differ from server settings. The upstream guide describes using the server's `evaluation.limits` with `viobench evaluate ... --limits limits.json` to match limits. Keep tokens out of source, commands stored in Git, logs, and screenshots.

Repository setup does not perform a course submission.
