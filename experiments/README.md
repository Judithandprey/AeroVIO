# Experiment records

| Record | Scope |
| --- | --- |
| [Official KLT + IMU baseline](baseline-klt-imu/README.md) | Frozen source, initial build/public evaluation, source hashes and one-episode repeat |
| [Small motion network and geometry ablations](learned-motion-v1/README.md) | Separate training data, pose/warp models, optional map, model snapshots and provenance |
| [First geometry comparison](learned-motion-v1/comparison.md) | Completed four-way public evaluation with matched runtime, inputs and limits |
| [Rotation and visual-input ablations](motion-ablation-v2/README.md) | Fixed-weight rotation toggle, matched IMU-only training and equivalent compact export |
| [Current ablation comparison](motion-ablation-v2/comparison.md) | Accuracy, compute and completion evidence for the controlled v2 variants |
| [Rotation diagnostics](motion-ablation-v2/rotation-diagnostics.md) | Saved-trajectory evidence and limits on interpreting accumulated orientation errors |

`baseline-klt-imu-v1` preserves the official reference. The current working
solution is the pose-only learned model; do not assume a working-tree filename
identifies an earlier artifact. Evaluation manifests record the source and model
hashes extracted from each actual WASM payload.

Use a directory per experiment. Keep compact reports, configurations, model
snapshots and provenance together; keep raw trajectories, fuel arrays, training
checkpoints, data and large logs in ignored output directories. Retain failed
runs and regressions as evidence. Package each variant with an explicit model
path and a new artifact name.

The [warping and world-model guide](../docs/warping-and-world-models.md) connects
the experiments to the project ideas. The v2 IMU-only control keeps real RGB and
geometry as training supervision while withholding RGB from the prediction
network. Its compact export is an inference-graph transformation of that same
checkpoint, and should be reported separately from the input-information ablation.

## Record template

- **Question / hypothesis:**
- **Implementation commit and source hashes:**
- **Upstream revision, runtime hash and Wasmtime version:**
- **Environment / dependency versions:**
- **Method and configuration:**
- **Training/validation scenes, episode identities and split checks:**
- **Public evaluation episode hashes:**
- **Evaluation limits and scoring definition:**
- **Seeds, initialization, batch order and training schedule:**
- **Checkpoint selection rule and selected epoch:**
- **Exact generation, training, packaging and evaluation commands:**
- **Model and artifact SHA-256:**
- **Raw report location / SHA-256:**
- **Sim(3) ATE, aggregation rule and units:**
- **Maximum fuel per frame and host-specific timing:**
- **Unaligned ATE, fitted scale, rotation and twist errors:**
- **Failures / filled frames and per-episode regressions:**
- **Controlled comparison and repeatability evidence:**
- **Conclusion supported by the data:**
- **Unresolved limitations and next change:**

Compare accuracy and compute together. A public-set gain does not establish
held-out or physical-robot performance. Keep checkpoint selection independent
of public evaluation and isolate one intended change per ablation.
