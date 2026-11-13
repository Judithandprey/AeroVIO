# Joint speed/turn balance V10

Teacher architecture and inference are preserved. This fixed contrast tests joint sampling on six fresh 60-second trajectories. Models and selection rules were frozen before any new-long estimator scores.

Selected reference: **joint-v10**. Selection checks: `{"teacher_distance": true, "teacher_aligned_ate": true, "category-v9_distance": true, "category-v9_aligned_ate": true, "episode_wins": true, "rotations_exact": true, "fuel": true}`.

The aggregate choice includes a material low-net-motion regression, detailed below. Workspace defaults remain unchanged; this is an experimental candidate rather than a universal replacement.

## Pre-training distribution gate

Each of four true-speed bins (<0.1, 0.1–0.25, 0.25–0.5, >=0.5 m/s) contributes exactly 597 examples per epoch. Within each speed bin, turn-rate cells (<5, 5–15, 15–30, >=30 deg/s) are as balanced as available support allows. Per-example exposure is capped at four per epoch. Empty cells remain empty.

Low-speed exposure: original 49.75%; V9 category-balanced 51.51%; V10 joint **25.00% in every epoch**. Across 40 epochs all 2388 original pairs are seen. Training-speed median changes from 0.102 to 0.250 m/s.

| Speed bin (m/s) | Original counts by turn bin | Draws per epoch by turn bin |
| --- | --- | --- |
| <0.1 | [761, 179, 83, 165] | [150, 149, 149, 149] |
| 0.1–0.25 | [277, 81, 16, 20] | [227, 226, 64, 80] |
| 0.25–0.5 | [466, 72, 0, 9] | [281, 280, 0, 36] |
| >=0.5 | [258, 1, 0, 0] | [593, 4, 0, 0] |

Training still has virtually no high-speed sharp turns. Joint sampling corrects the speed marginal but cannot create this missing motion coverage. The rarest cell is capped instead of being repeated hundreds of times.

## Unchanged training and inference

Same seed 17, original cache/normalization, 40 epochs, batch 64, 1,520 optimizer steps, Adam 0.001, cosine schedule and warping weight 2. The original zero-image training graph is folded into the same 1,350-parameter compact network. The original short-validation normalized pose-loss rule chooses epoch 5. New long trajectories never influence training or epoch selection.

A complete same-seed training repeat reproduces the checkpoint, export and validation metrics. Both exports pass full/compact/ONNX checks on 796 validation examples plus 64 synthetic probes. The new WASM uses byte-identical teacher C++ sources; only the compiled weights and build paths change. The 80-frame ABI check passes. Native V8 rotation/confidence changes are not included.

## Fresh long validation

Six flights, 60 seconds and 1,200 frames each: two each in ProcTHOR scenes 5, 6 and 7, using fresh seeds starting at 420000. Training scenes 2/3/4 and public scene 1 are excluded. Scenes 5/6 were used for earlier short validation, so only scene 7 is entirely new to both training and earlier checkpoint validation.

Generation records 6 attempts, accepting only non-crashing flights under the predeclared sequential-seed policy. Estimator accuracy never filters flights. The simulator header and successful-flight capture function are unchanged. Raw RGB/IMU/GT/depth and archives are retained on D: via `data/long-validation-v10`.

Teacher, V9 category balance, V10 joint and a full V10 repeat run on each identical frame, rotating execution order. Generation and training finish before timing. Every model keeps exact teacher orientations/angular twists. The V10 repeat matches all poses, twists and per-frame fuel.

| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Distance ratio median | Rotation RMSE median (deg) | Peak fuel/frame | Paired median step (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher | 0.519165 | 0.186428 | 0.4860 | 13.6685 | 3,124,076 | 0.1610 |
| category-v9 | 0.493918 | 0.184833 | 0.5227 | 13.6685 | 3,124,076 | 0.1604 |
| joint-v10 | 0.480920 | 0.141485 | 0.8306 | 13.6685 | 3,124,076 | 0.1650 |

The table uses the final timing-only three-model follow-up. Each six-frame block contains all six execution orders, shuffled with fixed seed 20261112. Each artifact runs first/second/third exactly 400 times per episode, without a duplicate V10 instance warming its code cache. All 7,200 outputs and per-frame fuel values per model exactly match the accuracy run. Initial four-way timings remain in `long-validation.json`; final timings are in `timing-comparison.json`. Neither timing pass changes model selection.

Against teacher: pooled distance-length RMSE -24.11%, Sim3 ATE -7.37%. Against V9: distance-length RMSE -23.45%, Sim3 ATE -2.63%. Negative error changes mean improvement.

The frozen selection rule uses the mean of per-flight length RMSE (not pooled RMSE), requiring >=2% improvement versus both references, no aligned-ATE regression versus either, at least four distance wins versus teacher, exact rotations and a <=5% peak-fuel guard. The table reports pooled length RMSE to match the previous diagnostic definition.

## Per-flight distance and trajectory errors

| Scene / seed | Teacher length RMSE | V9 length RMSE | V10 length RMSE | Teacher ATE | V9 ATE | V10 ATE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ProcTHOR-Train-5 / 420000 | 0.08089 | 0.08359 | 0.11395 | 0.30383 | 0.29053 | 0.29226 |
| ProcTHOR-Train-5 / 420001 | 0.26037 | 0.26532 | 0.19219 | 0.78014 | 0.69967 | 0.70887 |
| ProcTHOR-Train-6 / 420002 | 0.17715 | 0.17333 | 0.12188 | 0.33625 | 0.30233 | 0.27674 |
| ProcTHOR-Train-6 / 420003 | 0.20164 | 0.19019 | 0.15555 | 0.47720 | 0.45730 | 0.41783 |
| ProcTHOR-Train-7 / 420004 | 0.20270 | 0.19854 | 0.14268 | 0.66376 | 0.67730 | 0.68293 |
| ProcTHOR-Train-7 / 420005 | 0.14513 | 0.14819 | 0.10382 | 0.55381 | 0.53637 | 0.50688 |

## Speed-conditioned length RMSE

| True 1s net speed (m/s) | Teacher | V9 | V10 |
| --- | ---: | ---: | ---: |
| below_0.1 | 0.08748 | 0.09124 | 0.16483 |
| 0.1_to_0.25 | 0.08790 | 0.07520 | 0.07561 |
| 0.25_to_0.5 | 0.19499 | 0.19286 | 0.12378 |
| above_0.5 | 0.39691 | 0.40279 | 0.30850 |

## Material regression despite aggregate acceptance

For windows with true one-second net displacement below 0.1 m, length RMSE rises from 0.08748 m (teacher) and 0.09124 m (V9) to 0.16483 m (V10). This is 88.4% worse than teacher. The larger improvements are in medium/high net-speed bins. These evaluation bins use one-second net displacement, whereas the sampler uses adjacent-frame speeds.

Aggregate acceptance is not a universal improvement. Retain this model as an experimental candidate and investigate low-motion errors before replacing a general deployment default. No new guard was retroactively added and no second sampling sweep was run.

All six first-200-frame ground-truth traces are distinct from each other and from the original 16 training/short-validation flights. Freshness checks are recorded in `new-long-data-distribution.json`.

## Interpretation boundaries

- This is one fixed training seed. Repeating the seed verifies determinism, not across-seed statistical significance.
- The speed marginal was verified before training. Fresh long-flight motion distributions were described only after evaluation; no sampler changes were made in response.
- One-second windows overlap and are not independent experimental trials. Direction/ratio diagnostics omit true net displacement below 2 cm or degenerate predicted direction; length RMSE retains all windows.
- Sim3 alignment removes one global scale. Absolute distance accuracy and aligned trajectory error can disagree; neither should be substituted for the other.
- Interleaved wall time is host-dependent. Use peak and mean fuel for reproducible compute comparisons. No school GPU benchmark or server upload is implied.
- Teacher and V9 artifacts, all earlier experiment records and workspace defaults remain preserved. No additional sampling sweep, larger network, temporal input, map or geometric refinement is included.

Evidence: `sampling-audit.json`, `training-verification.json`, `long-data-manifest.json`, `long-validation.json`, `selection.json`, `paired-timing.json`, `verification.json`. Raw traces and logs: `results/joint-balance-v10/` in WSL.
