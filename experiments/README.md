# Experiment records

| Record | Scope |
| --- | --- |
| [Official KLT + IMU baseline](baseline-klt-imu/README.md) | Frozen source, initial build/public evaluation, source hashes and one-episode repeat |
| [Small motion network and geometry ablations](learned-motion-v1/README.md) | Separate training data, pose/warp models, optional map, model snapshots and provenance |
| [First geometry comparison](learned-motion-v1/comparison.md) | Completed four-way public evaluation with matched runtime, inputs and limits |
| [Rotation and visual-input ablations](motion-ablation-v2/README.md) | Fixed-weight rotation toggle, matched IMU-only training and equivalent compact export |
| [Current ablation comparison](motion-ablation-v2/comparison.md) | Accuracy, compute and completion evidence for the controlled v2 variants |
| [Rotation diagnostics](motion-ablation-v2/rotation-diagnostics.md) | Saved-trajectory evidence and limits on interpreting accumulated orientation errors |
| [Rotation-focused training](rotation-focus-v3/README.md) | Calibrated pure-rotation data and matched continuation; six complete public runs show no accepted rotation gain; epoch-zero selection retained |
| [Startup gyro-calibration experiment](gyro-bias-v4/README.md) | Seven complete public runs and exact repeats; diagnostic improves baseline accuracy/compute, while strict validation still retains current |
| [Lightweight feature correlation](feature-correlation-v5/README.md) | Frozen translation and calibrated prior; matched temporal versus single-frame rotation branches, validation decision and six public runs |
| [Explicit correspondence supervision](correspondence-v6/README.md) | Same deployed graph with geometric soft matching labels versus matched continuation; matching EPE and public rotation regress, all seven runs and exact repeat complete |

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

The [rotation-training guide](../docs/rotation-focused-training.md) explains v3's
perspective geometry and coherent virtual IMU. Both v3 fine-tunes retained the
initial checkpoint under the fixed real-validation guards. The separately saved
last-epoch candidates are diagnostics, not selected deployment improvements.
The [pre-public diagnostic addendum](../configs/rotation-focus-v3-public-diagnostics.json)
fixes their inclusion alongside the baseline, starting model and two selected
checkpoints, for six compared variants. It does not revise the selection rule.
The [input examples](rotation-focus-v3/rotation-examples.png) illustrate the
auxiliary task; their [caption](rotation-focus-v3/rotation-examples-caption.md)
records selection and display limits.

The [startup calibration guide](../docs/gyro-bias-calibration.md) explains v4's
documented-hover assumption, causal activation and distinction from the official
KLT initialization. Its [frozen selection](gyro-bias-v4/selection.json) retains
`current`; calibrated-gyro improves validation rotation by 68.77% but raises
aligned ATE by 0.02752%. These are four original simulated validation flights,
not public scores or physical-sensor results. The
[pre-public replication addendum](../configs/gyro-bias-v4-public-replication.json)
adds a diagnostic repeat without changing selection. The seven public runs are
the baseline, four factorial cells, selected-current repeat and calibrated-gyro
diagnostic repeat. All completed; see the [public comparison](gyro-bias-v4/comparison.md)
and [exact output/fuel verification](gyro-bias-v4/output-verification.md).

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
