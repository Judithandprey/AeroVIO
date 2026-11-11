# Tiny CNN: baseline identification and algorithm reproduction

The similar-compute point below our submission is **`tiny_cnn`**. On November 10,
2026 at 22:01 Los Angeles time, the authenticated server's expanded **Exact
values** table showed 0.6684122639764059 m equal-episode mean Sim(3) ATE and
14,684,708 maximum fuel/frame. Our latest submission showed
0.9155300067970294 m and 14,757,611 fuel/frame. Tiny CNN has about 27.00% less
error and 0.49% less fuel there. The server uses **20 episodes**; our local
comparison uses **10 public episodes**. A follow-up Details / Raw JSON read
verified that the server uses exactly the upstream `public+private-10` preset:
our ten public episodes plus ten private episodes. The submitted filename and
SHA-256 identify it as our preserved **v4 calibrated-gyro** artifact; the six
per-init/per-step resource limits match `configs/evaluation-limits.json`. See
[the transcribed observation](server-observation.json).

The local eight-run comparison is complete. Mean Sim(3) ATE is **1.095712 m**
for original training and **1.171630 m** for the matched-budget fit, versus
**0.881818 m** for our calibrated-gyro model at similar fuel. The v6 model is
0.868606 m but uses 21.87 M fuel; the single-frame v5 control is 0.855547 m at
20.14 M. The public portion of the existing server submission matches local
calibrated-gyro ATE and maximum fuel exactly in all ten episodes; see
[server-public-parity.json](server-public-parity.json). This validates the
comparison path, while leaving the official Tiny CNN checkpoint gap unresolved.

## What has and has not been reproduced

The official public source is
[`python/tiny_cnn`](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/viobench/solution/examples/python/tiny_cnn).
Its source is unchanged at upstream main `17303aa14ba14e16cb54d9291d684371e1d3b548`,
checked during this experiment. No trained weights are tracked there and the
repository has zero published releases. The participant UI and documented server
API expose scores, not a baseline model download. We obtained neither the
server's checkpoint nor its training manifest, hyperparameters or runtime hash.
Private evaluation data was not obtained. Therefore this is an **algorithm reproduction with local retraining**, not an
exact reproduction of the server's 0.668412 m result. Similar fuel is supporting
evidence for a similar inference workload, not proof of identical weights/code.

Tiny CNN has **29,078 parameters**:

1. Two 64x48 grayscale frames enter three convolution layers (16/32/64 channels).
2. Global average pooling yields 64 visual features, concatenated with 9
   normalized IMU preintegration values (rotation, delta velocity and position).
3. A 73-to-64-to-6 head predicts camera-local translation and rotation residual.
4. Gyro rotation and the learned residual are composed, then relative transforms
   are chained into a trajectory. Twist comes from motion over the frame interval.

Training uses normalized Smooth L1 relative-motion supervision. Deployment uses
ONNX through OpenCV DNN, with the original NumPy fallback retained. There is no
explicit correlation, warping loss, persistent 3D map, loop closure or startup
gyro calibration. AeroVIO's original CNN backbone was adapted from this example;
this reproduction does not claim the architecture as our invention.

## Fixed comparison

Both new runs reuse the same twelve training flights (2,388 pairs, scenes 2/3/4)
and four validation flights (796 pairs, scenes 5/6) used in AeroVIO. No public
evaluation flight is used for fitting or checkpoint selection. Data archives and
source are hashed; inference Python files are copied byte-for-byte from upstream.

| Run | Training | Selection | Image adaptation |
| --- | --- | --- | --- |
| `upstream` | Original `train.py`, seed 0, 10 epochs, batch 64, Adam 0.001 | Last epoch | Original 640x480 RGB directly to gray/64x48 |
| `matched` | Seed 17, 40 epochs, batch 64, Adam 0.001 with cosine to 0.0001, gradient cap 5 | Validation normalized pose loss; epoch 7 | Same 320x240 RGB adaptation as AeroVIO; exact cache image parity |

The matched recipe retains Tiny CNN's architecture, 9-value features, original
normalization and target construction. It matches v1's data, seed, batch ordering,
budget, optimizer/scheduler and selection policy, rather than replacing those
algorithmic differences with AeroVIO's implementation. Its recorded order hash
also matches the retained v2 same-order training record. This is not a one-factor
ablation. V5/V6 comparison models received additional later fine-tuning.

Two implementation differences remain explicit: upstream preintegration omits
the interval before the first IMU sample of each window, and its target is scaled
by standard deviation without subtracting a mean. AeroVIO includes that first
interval and centers targets; it also adds intrinsics/dt to the network input.
We preserve upstream behavior to reproduce it. This experiment does not isolate
which difference causes any accuracy change.

All eight evaluation runs use the same pinned local runner/runtime, host,
Wasmtime, input resolution, episode ordering and limits. They run serially after
training. Both new bundles pass 40-frame WASM execution checks and all 796
validation pairs pass PyTorch/ONNX parity checks (maximum normalized output
difference below 0.000001). Old artifacts are rerun unchanged; the matched Tiny
CNN also gets a complete repeat. See [comparison.md](comparison.md),
[comparison.json](comparison.json) and [verification.json](verification.json).

## Reproduction commands

From WSL `/root/projects/AeroVIO`, after the original training data exists:

```bash
.train-venv/bin/python -u scripts/reproduce_tiny_cnn.py --variant upstream
.train-venv/bin/python -u scripts/reproduce_tiny_cnn.py --variant matched
.venv/bin/python -u scripts/run_tiny_cnn_evaluation.py
.venv/bin/python scripts/report_tiny_cnn.py
```

The scripts refuse to overwrite existing fits/results. Preserve this record and
choose separate names/protocols for further experiments. The comparison protocol
is [configs/tiny-cnn-v7.json](../../configs/tiny-cnn-v7.json). New WASMs are
`artifacts/tiny-cnn-v7-upstream.wasm` and `artifacts/tiny-cnn-v7-matched.wasm`;
they do not replace any selected/default model or trigger a course upload.
Windows copies are `AeroVIO-tiny-cnn-v7-upstream.wasm` and
`AeroVIO-tiny-cnn-v7-matched.wasm` in the repository's `artifacts/` directory;
[windows-artifacts.json](windows-artifacts.json) records their verified hashes.

The first matched preparation intentionally stopped when image parity failed;
the mismatch revealed the original 640-versus-320 adaptation. A later fit finished
but its post-training lookup expected a hash absent from the old v1 record. That
lookup now checks the retained v2 record; the repeated fit produces the identical
ONNX hash. Both failed-attempt logs and outputs remain under ignored `logs/` and
`models/`, with no public result used in either repair.

## Interpretation boundary

Earlier v1-v6 claims of beating "the baseline" apply to the chosen **KLT + IMU
example on our ten public episodes**, not the complete server baseline frontier.
The server observation proves our existing submission is behind its Tiny CNN
point. Local retraining can compare algorithms under controlled conditions, but
cannot establish why the server's trained model is better or reproduce its score
without the missing checkpoint/training/evaluation provenance. More representative
and longer training flights are a hypothesis to test, not an established cause.
