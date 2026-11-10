# Rotation-focused training experiment

The auxiliary rotation task became easier for the trained model, but the
predeclared original-flight validation rule accepted **neither fine-tuned checkpoint**.
Both selected models remain the original epoch-0 warp-trained model. This is
not a new deployed improvement.

The experiment tests whether supervision from controlled camera rotation and
its corresponding image changes helps the existing image+IMU estimator.
Settings were fixed in [the training protocol](../../configs/rotation-focus-v3.json)
before fitting. The parent experiment and its rotation/image-input findings are
recorded in [motion-ablation-v2](../motion-ablation-v2/README.md).

## Data and geometric contract

“Original-flight validation” means the original **simulated** HyperDrone flights
with translation and rotation, as opposed to the synthetic pure-rotation
augmentation. It is **not real robot or physical-sensor data**. Code and JSON
retain the field name `real_validation` for this distinction.

The original-flight cache contains 2,388 training pairs from 12 ten-second flights in
ProcTHOR scenes 2/3/4, and 796 validation pairs from four flights in scenes 5/6.
Public scene 1 is excluded. Adjacent frames are 0.05 seconds apart.
Original-flight training rotation has median magnitude **0.0702°**, with 56.8% of pairs
below 0.1°. Its axis-wise 95th percentiles are **[0.1521°, 1.8294°, 0.1026°]**
in camera x/y/z: substantial rotation coverage is concentrated on camera y.
See the [coverage tables](rotation-coverage-with-aux.md) and
[exact statistics and cache hashes](rotation-coverage-with-aux.json).

The auxiliary set reuses the existing HyperDrone RGB observations: 2,400 training
and 800 validation pairs, balanced among camera x, y, z and mixed axes. Angle
magnitudes are sampled uniformly from 0–3° with independent signs. Two virtual
views use symmetric minus/plus half-angle rotations around a fixed camera
centre. A fixed 0.85 crop, updated intrinsics, and source-support checks prevent
angle-dependent black borders. Reprojection occurs at 320×240 before the shared
64×48 grayscale preprocessing. The operation generates no unseen scene content.

Each synthetic pair has ten 200 Hz IMU samples consistent with its constant
angular velocity, gravity, camera/IMU transform, lever-arm acceleration, and
source-episode noise/bias priors. Ground-truth relative rotation maps the current
camera into the previous camera. The network retains its existing convention:
`R_pred = Exp(IMU_rotation) @ Exp(predicted_residual)`. Auxiliary supervision
uses the exact SO(3) rotation; it does **not** supervise zero translation.
Bias is resampled per independent pair, so these examples do not directly teach
persistent bias estimation over long sequences.

[Example inputs and pixel differences](rotation-examples.png) illustrate the
signal. Their [caption](rotation-examples-caption.md) explains sample selection
and display scaling; these are geometric image transformations, not model
predictions or evidence of estimation accuracy.

## Matched training and checkpoint selection

Both runs start from the same v1 warp checkpoint and retain the same
29,398-parameter MotionNet, normalization, inference code, resolution, and
enabled rotation residual. Mapping stays disabled.

- **Control:** continue the original normalized pose loss plus masked
  photometric loss with weight 2.
- **Focus:** the same continuation, adding weight 0.1 times a rotation-only
  Smooth L1 loss on SO(3) geodesic error, normalized by the original rotation
  target scale.

Both use fresh Adam, seed 17, 20 epochs, original-flight batches of 64, and the same
minibatch order. Learning rate decreases from 1e-4 to 1e-5. Each performs 760
optimizer steps and processes 47,760 original-flight examples. Focus also processes 24,320
auxiliary examples in batches of 32, using a separate sampler seed. Its extra
training work is recorded; this is not an equal-training-compute experiment.

Selection minimizes the arithmetic mean of per-episode accumulated orientation
RMSE on **original-flight validation**, with two limits relative to epoch 0: pair translation
RMSE and mean unaligned trajectory position RMSE must each stay within 5%.
Every rollout resets at the episode boundary and includes the initial identity
frame. Epoch 0 remains eligible; ties retain the earlier checkpoint. Auxiliary
validation and image-shuffle results never select checkpoints.

The limits are **11.5865 mm** pair translation RMSE and **0.544635 m** trajectory
position RMSE. The [rollout audit](rollout-audit.json) checks these compositions
against the actual host estimator convention. The [training audit](training-audit.json)
records initialization, normalization, order, source hashes, and selection.

## Validation results and rejection

Here “original” means the v1 warp-trained image+IMU model, not the official KLT
baseline. Original-flight rollout metrics are means across the four ten-second validation
episodes. Auxiliary rotation error is pairwise SO(3) RMSE over 800 synthetic
pairs; it measures a different task and time horizon.

| Checkpoint | Original-flight accumulated rotation RMSE (deg) | Original-flight trajectory position RMSE (m) | Original-flight pair translation RMSE (mm) | Auxiliary pair rotation RMSE (deg) |
| --- | ---: | ---: | ---: | ---: |
| Original / selected epoch 0 | 6.948942 | 0.518700 | 11.034799 | 0.487619 |
| Control, fixed epoch 20, unselected | 6.954437 | 0.559686 | 11.203019 | 0.567904 |
| Focus, fixed epoch 20, unselected | 7.699465 | 0.552149 | 11.243136 | 0.237649 |

Focus reduces the auxiliary task error, but its accumulated original-flight validation
orientation error worsens. Both final candidates pass the pair translation
limit and fail the trajectory position limit. Control briefly reaches a lower
orientation error at epoch 17, but fails the trajectory position limit there
too. No eligible trained checkpoint improves the selection objective, so both
selected ONNX hashes exactly match the original:
`0271431fc621b160cabad4b448cedc91f9ea4e355f080a366823273041afdbb7`.

The first trainer retained only the selected checkpoint. A retention-only
repeat subsequently saved the fixed final epoch for diagnosis. All non-timing
history fields and selected artifacts reproduced the first runs. The
[retention change record](provenance/retention-only-change.json) and
[archived initial trainer](provenance/train_rotation_focus.initial.py) preserve
that distinction. The final candidates remain explicitly unselected.

## What the image probe establishes

The [visual probe](visual-probe.md) uses a predeclared permutation seed, 314162,
to shuffle whole image pairs while holding states and labels fixed. It covers
original-flight and auxiliary validation, with per-axis and angle-bin results in the
[full JSON](visual-probe.json). Original, selected control, and selected focus
produce identical results because their weights are identical. The separately
trained IMU-only sanity model is exactly invariant to image shuffling.

For the fixed epoch-20 focus model:

| Dataset | Correct-pair rotation RMSE (deg) | Shuffled-pair rotation RMSE (deg) | Prediction-change RMS (deg) |
| --- | ---: | ---: | ---: |
| Original-flight validation | 0.114949 | 0.114302 | 0.005138 |
| Auxiliary validation | 0.237649 | 0.237563 | 0.005331 |

The images affect rotation predictions, but correct correspondence does not
lower aggregate error in this probe. Consequently, improved synthetic-task
fitting does not demonstrate that the model learned a useful image-to-rotation
mapping. Sensor-conditioned correction or adaptation to synthetic inputs are
plausible explanations, not established mechanisms.

Shuffling is an artificial, out-of-distribution intervention and can also
mismatch camera intrinsics across episodes. A small or negative error change
does not prove that vision is useless. This probe neither measures long-term
drift nor replaces a matched retrained information ablation.

## Completed public comparison

All six variants completed the same 10 public episodes and 12,000 frames each,
without failures or filled outputs. Runs were serial after training and probes
finished, with identical runner/runtime, environment, limits and episode order.
All learned packages use identical inference Python sources.

| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Median step (ms) | Maximum fuel/frame |
| --- | ---: | ---: | ---: | ---: |
| Official KLT + IMU baseline | 2.041635 | 13.245 | 3.295 | 141,910,846 |
| Original warp-trained model | 0.966622 | 21.593 | 1.075 | 14,078,994 |
| Selected control, epoch 0 | 0.966622 | 21.593 | 1.072 | 14,083,612 |
| Selected focus, epoch 0 | 0.966622 | 21.593 | 1.076 | 14,080,100 |
| Control, fixed epoch 20, unselected | 0.969556 | 21.483 | 1.070 | 14,079,972 |
| Focus, fixed epoch 20, unselected | 0.962323 | 24.084 | 1.077 | 14,081,340 |

The final focus model lowers mean aligned ATE by **0.44%** relative to the
original model, while worsening median rotation RMSE by **11.54%**. Against
the matched final continuation control, it lowers ATE by **0.75%** and worsens
rotation by **12.10%**. This is not a successful rotation improvement. Both final
candidates remain unselected, consistent with the earlier validation decision.
Their deployment architecture remains unchanged; timing differences around
1.1 ms do not establish a meaningful speed improvement.

See [all metrics and episodes](comparison.md) and the original `runs/` reports.
Sim(3) alignment fits scale, rotation and translation; a small aligned ATE gain
does not demonstrate improved metric scale or orientation. Rotation RPE also
worsens for final focus (1.063 degrees versus 0.904 for the original model).

The [public-diagnostic addendum](../../configs/rotation-focus-v3-public-diagnostics.json)
was recorded after original-flight validation and **before any v3 public evaluation**.
It specifies the already fixed final epoch; no candidate is chosen by public
performance. The original checkpoint-selection rule remains unchanged, even if
an unselected candidate later scores better on the public benchmark.

The original/selected learned variants produce byte-identical pose and twist
files and identical accuracy metrics in all ten episodes. The repeated baseline
and original model also reproduce v2 predictions exactly; see
[output verification](output-verification.json). This is a reproducibility check,
not a new improvement. Tiny fuel differences are retained as measured, rather
than rounded into an assertion of identical execution. Public wall time measures
local guest-step computation, not end-to-end sensor latency.

## Evidence and remaining questions

- [Control history and run record](control/run.json), [focus run record](focus/run.json),
  and their `history.json` files retain every epoch and both validation tasks.
- [Control final metadata](control/diagnostic-last/meta.json) and
  [focus final metadata](focus/diagnostic-last/meta.json) mark the fixed diagnostic
  artifacts as unselected; selected weights remain in the parent directories.
- [Training audit](training-audit.json), [rollout audit](rollout-audit.json),
  [test output](tests.txt), coverage, and probe records retain verification evidence.
- **84 tests pass**, and all four newly packaged selected/final artifacts passed
  the upstream 20-frame WASM check. Commands and logs are recorded in
  [pipeline status](pipeline-status.json) and `checks/`.
- [Windows artifact records](windows-artifacts.json) verify all four local WASM
  copies by SHA-256 and distinguish selected fallback from final diagnostics.
- Raw caches, full PyTorch checkpoints, and full trajectory logs remain local.
  No course-server submission, held-out score, or physical-robot result is claimed.

This is one seed, a small scene-disjoint simulated dataset, short original flights, and
an auxiliary distribution made from virtual rotations. It does not establish
broad generalization or the reason for persistent orientation error.

Two next hypotheses are worth separate controlled tests: short-window
accumulated-rotation supervision with an explicit temporal gyro-bias model;
and spatial feature matching or correlation that retains displacement structure
more directly than global pooling. Neither is implemented here, and neither is
guaranteed to improve accuracy. They should be judged on original-flight validation with
the same translation safeguards before another public comparison.
