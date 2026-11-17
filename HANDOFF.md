# AeroVIO - Handoff

## Structural-line relative rotation fallback V13 (November 16, 2026)

Work is isolated on `experiment/structural-lines-v13`, parent `2d09ef5`.
Read `docs/structural-lines-v13.md` and the V13 experiment report. This adds
only a geometry module to the selected rotation-only V8 WASM, not to V10 or
the V11/V12 memory predictors. Original checkpoint and V8 point/IMU sources
remain unchanged. No training, global Manhattan heading, metric wall width,
scale prior, trajectory-straightening rule, main/default or course upload.

Primary arms are frozen V8, detector-only and bounded line correction. The
detector uses exact V8 green160 preprocessing and retains at most 24 lines and
three generic vanishing directions; two independently supported directions
are required. Relative matching uses past-only persistence, ambiguity and
conditioning gates. Point-rotation overlap rejects line updates. Capture phase 0,
solve phase 6, spread over 2 frames; accepted updates invalidate the old anchor
to avoid undoing a delayed correction. Detector-only must match V8 bit for bit.

Actual-WASM feasibility: only 7/100 development images supply two directions;
all occur in one flight, mostly at startup. Isolated detector peak fuel is
16,048,279. LSD sorts all 12,065 gradient records before the retained-line cap,
so reducing output count or frequency does not remove its per-frame cost.
The original probe/source snapshot and contact sheet retain this negative result.

All 143 host tests and 40 native detector/controller geometry checks pass.
The old V8 test helper's 200M whole-batch budget was exhausted by the multi-image
tests; the same test binary passes under a separate 2B batch budget, consuming
409,336,583. Estimator per-frame limits are unchanged. Both attempts/logs remain.
Three complete ABI checks pass. Disabled rebuild matches frozen V8 on the first
development flight; detector-only matches on all four, and correction repeats
match output, fuel and diagnostics exactly.

Development: V8/detect/correct metrics are identical. Mean Sim3 ATE 0.221891 m,
raw ATE 0.429587 m, rotation RMSE 2.003087 deg and 1s rotation RPE 0.629161 deg.
No correction is accepted in 100 decisions: 70 point-overlap, 28 weak and 2 warmup.
Peak fuel rises from 2,589,216 to 16,760,164. Development wall times are excluded.
Rules, model/artifact hashes and sources were frozen after this comparison.

Confirmation is complete on four new 60s flights in rooms 8/9, seeds 450000..450003,
all successful without retries. Baseline / detect / correct mean Sim3 ATE is
0.555830 / 0.555830 / 0.555803 m; raw ATE is 1.601593 / 1.601593 / 1.601620 m.
Mean rotation RMSE is 11.023601 / 11.023601 / 11.019874 deg and 1s rotation RPE
is 0.795742 / 0.795742 / 0.795649 deg. Correction reduces rotation RMSE only
0.0338%; two flights improve slightly and two are unchanged. One corrected
flight's raw ATE and 1s rotation RPE worsen slightly. No practical gain is shown.

Only 96/600 images have two usable directions; 2/600 decisions apply correction,
at frames 902 of seed 450002 and 518 of seed 450003. Four application steps sum
to 0.0007643 rad. Rejections: 396 point-overlap, 169 weak, 29 temporal, 3 unmatched
and 1 residual. Early-gate priority means these counts are not independent
coverage measures. See development-analysis.md and failure-cases.json for cases.

Confirmation peak fuel rises from 3,215,473 to 18,505,582 (5.755x); mean fuel
from 1,234,655 to 2,911,502 (2.358x). Median guest steps are 0.417089 / 0.489067 /
0.482811 ms. The isolated fresh timing follows all generation/build/development
work. Rotation-improvement and peak-fuel guards fail. No reference replacement.
All 24 primary traces/decision logs and eight full correction repeats verify;
detector-only poses/twists match frozen V8 on every flight. Sources stayed frozen.

Windows diagnostic files are artifacts/AeroVIO-native-v13-detect.wasm and
artifacts/AeroVIO-native-v13-correct.wasm. Both are 3,385,914 bytes; identities
are in experiments/structural-lines-v13/windows-artifacts.json. All previous
models, branches, raw data and negative results remain preserved. Main/defaults
and course-server submission are unchanged. The reproducible pipeline and
cost/coverage evidence are useful; do not describe this as solved drift.

## Frozen-predictor memory gate V12 (November 15, 2026)

Work is on `experiment/memory-gate-v12`, based on V11 commit `96e6fcb`.
Read `docs/memory-gate-v12.md` and `experiments/memory-gate-v12/README.md`.
V10 and fixed-final V11 predictors are frozen. Only a 24 -> 8 -> 1 sigmoid gate
is trained (209 parameters); a constant-mixture control trains one scalar bias.
The gate blends translation, and the blended pre-visual velocity becomes the
next causal camera-frame memory. Native gyro and visual fusion are unchanged.
No new visual inputs, stationary detector, ZUPT or training-data changes occur.

Both arms use the exact original target plan, 20-frame contexts, 40 epochs and
1,520 optimizer steps. A complete adaptive repeat reproduces all epoch metrics
and the selected ONNX. Predictor weights remain bit-identical. Candidate
selection was fixed to updated epochs 1..40 before training; both select epoch 1.
Initial 50/50 mixture validation pose loss is 0.278171. Constant/adaptive best
updated losses are 0.278540/0.279215, versus V10 0.277686. Neither beats its
initial mixture or V10 on this criterion. These are experimental candidates.
Adaptive original-validation gate mean/std is 0.526078/0.004975; constant is
0.509373. Do not claim learned turn/stop recognition merely from added capacity.

143 host tests, five 80-frame WASM checks, and C++/ONNX gate/history parity pass.
The gate adds 209 parameters, but preserving exact V10/V11 branches requires
both predictors: 3,165 total network parameters and 3,016 semantic dense MACs.
Runtime memory transport, valid/gap handling and bounds retain the V11 contract.

Old six-flight regression is complete. V10 / V11 / constant / adaptive mean
Sim3 ATE is 0.600712 / 0.692283 / 0.545626 / 0.545311 m. Turning length RMSE
is 0.115551 / 0.130753 / 0.119655 / 0.121176 m. Two stop-event mean errors are
0.095571 / 0.123080 / 0.081228 / 0.081858 m; no variant settles. Cruise length
RMSE is 0.207946 / 0.202840 / 0.201485 / 0.201501 m, with no false-stop runs.
Much of the improvement over V11 is also achieved by the fixed mixture.
Regression wall times are excluded because data generation overlaps them.

Fresh confirmation is complete: six new 60-second flights, seeds 440000..440005,
all successful without retry. V10 / V11 / constant / adaptive mean Sim3 ATE is
0.493972 / 0.652062 / 0.467923 / 0.469452 m. Pooled one-second length RMSE is
0.141926 / 0.168497 / 0.151463 / 0.152184 m. Turning RMSE is 0.093519 /
0.123020 / 0.115388 / 0.116942 m; cruise RMSE is 0.154836 / 0.211336 /
0.175185 / 0.176222 m. Adaptive improves these two regime errors by 4.94% /
16.62% versus V11, but worsens them by 25.05% / 13.81% versus V10. Constant
mixing slightly outperforms the adaptive gate on all four aggregate errors.
This does not establish useful context-dependent memory selection.

Median guest-step times are 0.296443 / 0.302235 / 0.302249 / 0.301182 ms;
adaptive mean fuel is 1,167,016.24 (+0.647% versus V10), peak 3,159,826.
Timing runs after all generation/training/build/regression work has ended;
all 24 execution orders balance each model's position, with repeats excluded.
All 48 primary traces, 12 adaptive repeats and preserved original V10/V11
regression outputs/fuel verify. Five WASM ABI checks and 143 tests pass.

Fresh coverage is 1,920 turning / 1,558 cruise / 247 stationary windows and
zero qualifying stop events. No model has false-stop runs while truly moving.
The stopping guard is inconclusive; turning narrowly misses its predeclared
5% improvement requirement, and cruise/overall distance fail V10 guards.
V10 remains the reference. Reduced Sim3-aligned ATE does not establish improved
metric distance. The old two-stop reduction is tentative and no model settles.
New raw data stays on D: through `data/long-validation-v12`; full traces/logs
are retained in WSL `results/memory-gate-v12`. Missing coverage is not a pass.

Windows candidates: `artifacts/AeroVIO-native-v12-adaptive.wasm` and
`artifacts/AeroVIO-native-v12-constant.wasm`; hashes are retained in
`experiments/memory-gate-v12/windows-artifacts.json`. All earlier artifacts and
branches remain preserved. Main/defaults and the course server are unchanged.

## Explicit velocity memory V11 (November 14, 2026)

Work is on `experiment/velocity-memory-v11`, starting at merged main `d1d0e6b`.
Read `docs/velocity-memory-v11.md` and `experiments/velocity-memory-v11/README.md`.
Both 18-input arms start from V10 with zero
new columns; feedback adds three previous pre-visual velocity components and a
validity flag. Teacher gyro and visual direction fusion are unchanged. Causal
camera-frame transport, 2 m/s component bounds and >0.2 s gap reset are shared
between training and runtime.

Both arms use exactly the V10 target-frame plan (2,388 targets/epoch), up to
19 preceding same-episode context frames, self-predicted history, target-only
pose/warp loss, seed 17, 40 epochs and 1,520 optimizer steps. Both fine-tune the
compact graph, so matched control is essential when attributing memory effects.
The original full-episode validation selector includes epoch zero and selects
it for control, feedback and repeat. Primary reference therefore remains V10.

Before long evaluation, a fixed-final diagnostic replay was added solely to
retain epoch 40. All epoch metrics and selected weights reproduce the original
fit exactly; final-feedback repeat weights also match. This is a diagnostic
endpoint, not a new hyperparameter choice or a long-score-selected checkpoint.
Initial normalized validation loss is 0.277686; fixed-final control/feedback
losses are 0.316048/0.332766. Do not present the final models as accepted updates.

136 host tests and five 80-frame WASM checks pass. A separate logging guest
verifies C++/ONNX motion within 1.12e-8, memory transport within 6.01e-8 and causal
history inputs exactly. The logging guest is excluded from timing. Native model
has 1,606 parameters; semantic dense MACs increase from 1,280 to 1,536.

Long evaluation is complete: six old V10 flights are regression evidence;
six fresh 60-second flights at seeds 430000 through 430005 are confirmation.
All fresh generation attempts succeeded without crashes or retries. Raw data
is on D: through WSL `data/long-validation-v11`. All 48 primary trace hashes,
source/artifact hashes, full feedback output/fuel repeats, selected-zero/V10
equivalence and rotation/twist invariants pass. Each model completes 7,200
frames per set. No default changed and no course-server upload occurred.

On fresh confirmation, teacher / V10 / no-history-final / feedback-final mean
Sim3 ATE is 0.571272 / 0.600712 / 0.579938 / 0.692283 m. Pooled one-second
length RMSE is 0.220737 / 0.167795 / 0.160293 / 0.168791 m. Small-net-motion
RMSE is 0.075371 / 0.157649 / 0.136720 / 0.128124 m. Feedback versus matched
control improves this local metric 6.29%, but worsens overall distance 5.30%
and ATE 19.37%. Versus V10, local error improves 18.73% while ATE worsens 15.24%.
The old regression set also shows worse overall distance and ATE; its local
improvement versus matched control is only 1.28%. This is not an accepted upgrade.

Fresh stationary, slow-straight and decelerating bins improve versus matched
control, while turning worsens. Small net displacement is not the same as
stationary motion. Only two qualifying stopping events occur in fresh data
(none in regression): no variant settles under the predefined rule, and
feedback post-stop displacement error is worse (0.123080 m versus control
0.118099 and V10 0.095571 m). Do not claim successful stop detection or recovery.

Regression wall times are excluded because simulation overlaps them. Fresh
confirmation timing starts after generation/training finish and uses all 24
orders of four distinct artifacts. Median step times in the same order are
0.170562 / 0.171959 / 0.180160 / 0.181052 ms. Feedback is about 5.29% slower
than V10 on this host, while deterministic mean fuel increases only 0.20%.
Peak fuel increases from 3,366,112 to 3,368,499. No speedup is claimed.

Windows diagnostic artifacts are `artifacts/AeroVIO-native-v11-feedback-final.wasm`
and `artifacts/AeroVIO-native-v11-no-history-final.wasm`; hashes are recorded in
`experiments/velocity-memory-v11/windows-artifacts.json`. Feedback adds 805 bytes
to V10's WASM size. Use V10 as the selected reference, not these diagnostic models.

Raw outputs/logs: WSL `results/velocity-memory-v11`. Compact evidence and model
snapshots are archived under `experiments/velocity-memory-v11`. Main, teacher,
V9 and V10 remain preserved. No training or evaluation remains pending.

## Joint speed/turn balance V10 (November 12, 2026)

Read `docs/joint-balance-v10.md` and `experiments/joint-balance-v10/README.md`.
The completed experiment stays on `experiment/native-fusion-v8`. Teacher and
V9 artifacts, earlier records and workspace inference defaults are preserved.

The only training intervention is joint speed/turn sampling. Four speed bins
each draw 597 examples per epoch, then turn cells are balanced subject to a
four-exposures-per-example cap and actual support. Before either fit, all 40
epochs passed exact 25% low-speed exposure, the cap, cache hashes and unchanged
2,388-example epoch size. All original pairs are seen across training. Old
exposure was 49.75% (teacher) and 51.51% (V9). High-speed sharp-turn coverage
is still missing and is not fabricated by oversampling a single example.

The teacher's original training graph, normalization, seed, 40 epochs, 1,520
optimizer steps, loss and original short-validation epoch rule stay unchanged.
Epoch 5 is selected; a complete same-seed repeat reproduces its ONNX and metrics.
The deployed graph has 1,350 parameters. All native C++ source bytes come from
teacher commit `0c4f778`; only the compiled checkpoint changes.

Six new 60-second flights use seeds 420000 through 420005, two in each of scenes
5/6/7. No crashes or retries occurred. Scenes 5/6 appeared in prior short
validation; scene 7 is entirely new. All are excluded from training. The first
200 GT states of every new flight are distinct from the other new flights and
all 16 old training/validation flights. Generation never used estimator scores;
checkpoints were frozen before new-long inference. Raw data is on D: at
`D:/Robotics/我的机器人助手/artifacts/AeroVIO-data/long-validation-v10`, reached
through the WSL `data/long-validation-v10` symlink. No old data was moved.

On this new six-flight set, teacher / V9 / V10 mean per-flight one-second length
RMSE is 0.177983 / 0.176528 / 0.138345 m. V10 improves 22.27% / 21.63% against the
two references. Pooled RMSE is 0.186428 / 0.184833 / 0.141485 m, a different
aggregation (24.11% / 23.45% improvement). Mean Sim3 ATE is 0.519165 / 0.493918 /
0.480920 m (V10 improves 7.37% / 2.63%). Median predicted/true distance rises
from 0.486 / 0.523 to 0.831. V10 wins distance on five of six flights and passes
every predeclared aggregate selection guard.

Material tradeoff: length RMSE for windows with true 1s net displacement <0.1 m
rises from 0.08748 / 0.09124 to 0.16483 m. Do not claim universal improvement or
silently replace a deployment default. The first new flight has worse distance;
one new-scene flight has worse aligned ATE despite better distance. No additional
fit or threshold sweep was performed after observing this tradeoff.

All rotations/angular twists are identical; median rotation RMSE is 13.6685 deg.
Peak fuel is 3,124,076 for all three. V10 mean fuel is only 0.00114% above teacher.
Initial four-way median step times were 0.245 / 0.244 / 0.237 ms; the identical
V10 repeat measured 0.213 ms, exposing cache sensitivity. A final three-artifact
timing pass uses all six execution orders per six-frame block, shuffled with
fixed seed 20261112. Each model occupies each position 400 times per flight.
Final medians are 0.1610 / 0.1604 / 0.1650 ms: V10 is about 2.5% slower than
teacher in this host measurement, with essentially unchanged fuel. Do not claim
a speedup. All 7,200 outputs and fuel values per model match the accuracy run.
The complete inference repeat matches every output and per-frame fuel exactly.
All four runs complete six flights, 7,200 frames each. 128 host tests, 860-pair
full/compact parity checks and the new 80-frame WASM check pass.

Windows test artifact: `artifacts/AeroVIO-native-v10-joint.wasm`, SHA-256
`ce8a2f4dde5108d41ccdd64333ab44906ded11be304d62136af0e03b48a4cd3e`.
It is an aggregate-validation-selected experimental candidate with the low-speed
regression above. Raw runtime logs/traces are in WSL `results/joint-balance-v10/`.
No training/evaluation remains pending and no course-server upload occurred.


## Data balance V9 (November 12, 2026)

Read `docs/data-balance-v9.md` and `experiments/data-balance-v9/README.md` for
the completed sampling-only contrast on `experiment/native-fusion-v8`.
It builds exact teacher `0c4f778` C++ sources, not the V8 rotation/confidence
variants. The control retrains the original IMU-only graph and reproduces both
the teacher compact ONNX and entire Native 160 WASM byte for byte.

Original startup/transition, cruise and turn counts are 1,346 / 749 / 293.
Balanced exposure is 796 per category per epoch, with the original 2,388-example
epoch size, seed 17, normalization, losses, 40 epochs and 1,520 optimizer steps.
The original validation pose-loss criterion selects epochs 14 and 7 for control
and balanced. A full balanced-training repeat reproduces the weights exactly.

Validation retains teacher: mean one-second length RMSE improves only 0.85%,
below the frozen 2% requirement, while mean Sim3 ATE worsens 0.30%. On ten public
flights balancing changes Sim3 ATE from 0.486329 to 0.475930 m (-2.14%), but
one-second displacement-length RMSE worsens from 0.229366 to 0.231712 m (+1.02%).
Median predicted/true distance stays around 0.40. All rotations are identical;
peak fuel is unchanged at 2,232,749. Mean fuel differs by less than 0.0001%.

Category balance did not reduce low-speed exposure: 84.6% of the original turn
examples are below 0.1 m/s, so overall low-speed exposure rises from 49.7% to
51.5%. This does not refute speed-distribution balancing; it exposes a limitation
of this three-category design. No new thresholds or fits were tried after
viewing the results. A speed-by-turn design and fresh long-flight validation
are future discussion, not pending authorized work.

All 123 host tests pass. Both new WASMs pass 80-frame checks. Five full public
runs complete 10/10 with exact teacher/control and both repeat pose/twist/fuel
checks. Serial timing fluctuates even for identical teacher/control binaries,
so a separate 12,000-frame paired timing audit alternates execution order.
Paired medians are 0.1403 ms teacher and 0.1395 ms balanced; treat speed as
essentially unchanged, not a large speedup inferred from serial wall times.
Paired outputs and fuel also reproduce the saved public results.

Windows artifacts: `artifacts/AeroVIO-native-v9-control.wasm` (identical teacher)
and `artifacts/AeroVIO-native-v9-balanced.wasm` (unselected diagnostic).
Hashes are in `experiments/data-balance-v9/windows-artifacts.json`.
Raw logs, traces, build sources and timing plan remain under WSL
`results/data-balance-v9/`; data sampling plans under `data/motion-balance-v9/`.
All earlier models, branch references and workspace defaults remain unchanged.
No training/evaluation is pending, and no course-server upload occurred.

## Native fusion V8 (November 12, 2026)

Read `docs/native-fusion-v8.md` and `experiments/native-fusion-v8/README.md`
for the newest experiment on `experiment/native-fusion-v8`, based directly on
teacher commit `0c4f7781cfc844d7fdc4f7223619a739b9f30cc7`. The old experiment
branch and all earlier models/artifacts remain preserved. The teacher's exact
Native 160 artifact is the reference; a flags-off rebuild has identical
trajectories, twists and metrics, with separately measured runtime overhead.

The fixed 2x2 experiment tests heuristic confidence weighting and bounded
two-view geometric rotation refinement. Original IMU weights, calibration,
tracking resolution and 8/4/8-frame capture/delay/spread schedule stay fixed.
There is no retraining, online bias estimation, 3D map or server submission.
Both CMake options remain OFF by default.

Four original validation flights select `rotation`: mean rotation RMSE falls
from 2.17047 to 2.00309 degrees (-7.71%), with essentially unchanged mean aligned
ATE. Selection is frozen before public evaluation. The confidence and combined
arms fail the validation position guards and remain diagnostic artifacts.

On ten complete public episodes the selected rotation arm achieves 0.479744 m
mean Sim(3) ATE, 11.4083 degrees median rotation RMSE and 2,233,398 peak fuel,
versus teacher Native 160's 0.486329 m, 12.1963 degrees and 2,232,749. Mean fuel
increases 1.39%. Confidence and combined ATE are 0.629407/0.624094 m. Timing is
host-dependent; prefer deterministic fuel when describing compute changes.
The complete repeated-run and ABI/test evidence is in the V8 record. Never
describe these ten public episodes as the course server's private result.

Windows WASMs are under `artifacts/`: `AeroVIO-native-v8-rotation.wasm` is the
validation-selected candidate; `...-confidence.wasm` and `...-combined.wasm`
are rejected diagnostics. Original `AeroVIO-native-160-0c4f778.wasm` stays intact.
Use `experiments/native-fusion-v8/windows-artifacts.json` for hashes.
Raw logs/traces live in WSL `results/native-fusion-v8/`. Reproduce to a fresh
name using `scripts/reproduce_native_fusion.py`; keep both evidence sets.

## Baseline correction (November 10, 2026)

The server's Exact values table identifies the similar-compute frontier point
as `tiny_cnn`, not KLT + IMU. Its current 20-episode score is 0.6684122639764059 m
mean Sim(3) ATE and 14,684,708 max fuel/frame; the user's latest upload
is 0.9155300067970294 m / 14,757,611. All v1-v6 references to "the official
baseline" should be read as the chosen upstream KLT + IMU sample on our local
ten-episode set. Those experiments do not establish superiority over the full
server baseline frontier. See `experiments/tiny-cnn-v7/` and its source/weights
limitations. Preserve the existing defaults and prior acceptance decisions.
Details / Raw JSON subsequently confirmed the upload is v4 calibrated-gyro
(SHA-256 matches the Windows artifact), the server set is `public+private-10`,
and the six saved resource limits match. Private data and official Tiny CNN
weights were not obtained; this is a local algorithm reproduction.

V7 is complete: original trainer (10 epochs/seed 0/final checkpoint) and a
matched-budget fit (40 epochs/seed 17/validation epoch 7) retain byte-identical
upstream inference code. Their local public mean ATE is 1.095712/1.171630 m at
14,680,072/14,685,424 max fuel/frame. All eight serial runs finished 10/10, the
matched full repeat is exact, and all five preserved-reference reruns reproduce
trajectory/twist/fuel/accuracy exactly. Both 40-frame WASM checks and 796-pair
ONNX checks passed. The public part of the user's existing server result also
matches our calibrated-gyro ATE and max fuel exactly for every episode.
Reports, source snapshots, weights, provenance and Windows artifact hashes are
in `experiments/tiny-cnn-v7/`. No experiment or training run is pending.


The v6 explicit-correspondence supervision experiment is complete and rejected
by the frozen acceptance rule. V5 temporal architecture and all guest source are
unchanged; two 40-epoch fine-tunes share initial weights and minibatches. The
supervised arm adds geometric bilinear soft-label matching cross-entropy; the
continuation arm retains rotation-only supervision. Trained epochs 13 and 4,
respectively, are archived as diagnostics. Matching validation EPE worsens from
0.336266 (V5) to 1.067577 image pixels (supervised), versus 0.306630 for continuation
and 0.236517 for zero motion. Cross-entropy decreases but displacement does not
improve. Independent OpenCV projection and all 118 host tests pass.

All seven v6 public runs completed 10/10 episodes. Supervised achieves 0.868606 m
aligned ATE, 10.6784 degrees median rotation error and 1.506 ms median WASM step,
versus V5 temporal's 0.870938 m, 9.7654 degrees and 1.561 ms in this round. Its
rotation median worsens 9.35% while ATE improves only 0.27%; the V5 single-frame
control remains better on both. Max fuel is 21,874,382 (+0.093% versus V5).
Both new 40-frame WASM checks pass. Full supervised repetition and all four
preserved-reference reruns match outputs, accuracy and per-frame fuel exactly.
See `docs/correspondence-supervision.md`, `experiments/correspondence-v6/README.md`,
`selection.json`, `matching-audit.json` and `windows-artifacts.json` in that record.
All previous selections/defaults and artifacts remain unchanged. No new run is
pending; future continuous-displacement supervision is only a hypothesis.

The v5 lightweight feature-correlation experiment is complete. The old translation
network is frozen; a 6,243-parameter local matching branch corrects the actual
causal calibrated gyro prior. A matched current/current-image control isolates
previous-frame information. Both seed-17 fits ran 40 epochs. Validation chose
trained epochs 7 (temporal) and 14 (single-frame) as diagnostics, but neither
qualified: mean rotation RMSE is 2.780753 and 2.453287 degrees versus 2.170470
for calibrated gyro. V5 retains its calibrated-gyro reference, while v4's selected
`current` and the workspace pose-only default remain unchanged.

All six v5 public runs completed 10/10 episodes. Temporal correlation achieves
0.870938 m mean Sim(3) ATE, 9.7654 degrees median rotation RMSE and 1.522 ms median
step time. Against calibrated gyro, these are -1.23%, -19.93% and +26.91%.
The single-frame control is better on the first two metrics (0.855547 m, 9.0634
degrees) and faster (1.440 ms), so this round does not establish an incremental
benefit from temporal matching. 111 tests and both 40-frame WASM checks pass.
Full temporal repetition reproduces outputs, accuracy and frame fuel exactly;
the three preserved references reproduce v4 exactly. See
`docs/feature-correlation.md`, `experiments/feature-correlation-v5/README.md`,
`comparison.md`, `selection.json` and `windows-artifacts.json` in that record.
No new training or default change is pending; proposed next hypotheses are not
authorization to start more experiments.

The v4 startup gyro-calibration experiment has completed original simulated-flight
validation. Selection remains `current`: the unchanged v1 warp model with bias
calibration off and its learned residual on. Calibrated gyro-only rotation lowers
mean orientation RMSE by 68.77% across all four validation flights, but aligned
ATE rises by 0.02752%, violating the frozen nonregression guard. Calibration with
the residual also fails that guard (ATE +0.07454%). No threshold was relaxed.
All 104 host tests and the four new 40-frame WASM checks pass. All seven public
runs completed 10/10 episodes. Calibrated-gyro achieves 0.881818 m mean Sim(3)
ATE, 12.196 degrees median rotation RMSE and 14,757,111 max fuel/frame: 56.81%,
7.92% and 89.60% lower than the official baseline, respectively. Its full repeat
matches poses, twists, accuracy and every frame's fuel exactly. Against current,
it costs 16.69% more median step time and increases median unaligned ATE 1.55%.
The current and candidate Windows WASM copies are SHA-256 verified. Public
results do not override the earlier strict validation decision.
See `docs/gyro-bias-calibration.md`, `experiments/gyro-bias-v4/validation.md`, and
the frozen `selection.json`. No new training or default-model change occurred.

The v3 rotation-focused training run is complete. Both the matched continuation
control and the rotation-auxiliary fine-tune selected epoch zero: neither supplied
a qualifying improvement on original simulated-flight validation within the fixed translation guards.
Their selected model is the original v1 warp checkpoint. Epoch-20 candidates are
retained separately as unselected diagnostics. All six public variants completed
10/10 episodes; 84 tests and all four new WASM checks pass. Final focused training
lowered aligned ATE by 0.44% but worsened rotation RMSE by 11.54% versus the original
model, so it remains unselected. Four Windows WASM copies are hash-verified.
See `experiments/rotation-focus-v3/README.md` and
`docs/rotation-focused-training.md`.

The previous rotation/image ablation remains complete: all five v2 variants
finished 10/10 public episodes (12,000 frames each), 62 tests passed, and all four
new Windows WASM copies were SHA-256 verified. Its findings remain in
`experiments/motion-ablation-v2/README.md`: learned rotation correction helps
overall; visual-input benefits vary by episode; compact IMU-only trades compute.

Read `README.md`, `docs/requirements.md`, `docs/setup.md`,
`docs/learned-motion.md`, `docs/gyro-bias-calibration.md`,
`experiments/gyro-bias-v4/validation.md`, `docs/rotation-focused-training.md`, and
`experiments/rotation-focus-v3/README.md` first.
Use `experiments/motion-ablation-v2/comparison.md` and `rotation-diagnostics.md`
for the earlier input/rotation investigation; preserve `experiments/learned-motion-v1/` as the
completed first experiment. See `docs/warping-and-world-models.md` for the
conceptual distinction between geometric warping and a learned world model.

## Workspace and branch

- Private repository: https://github.com/Judithandprey/AeroVIO.
- Current branch: `experiment/native-fusion-v8`; earlier branch `experiment/perspective-map` is preserved.
- Build checkout: WSL Ubuntu `/root/projects/AeroVIO`.
- Windows Git mirror: `C:/Users/ROG/Documents/GitHub/AeroVIO`.
- WSL owns build products, datasets and caches; source changes must be synchronized
  deliberately with the Windows mirror before publishing.
- Upstream runner/runtime revision: `2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca`.
- Keep the evaluation `.venv` and training `.train-venv` separate. Host PyTorch
  training uses CPU; guest inference uses ONNX through OpenCV DNN in WASM.

## First experiment: preserved implementation and evidence

- B0: original KLT + IMU, frozen at commit
  `8ea22f52b2083ee53d75b9419651c2505f949a67`, tag `baseline-klt-imu-v1`.
  Its 10-episode mean Sim(3) ATE is 2.04163486385224 m and max frame fuel is
  141,910,846. First-episode repeat matched pose/twist files and fuel exactly.
- N1: 29,398-parameter paired-image CNN + IMU, supervised relative motion.
- N2: identical architecture, seed, initialization, batch order, data and schedule,
  with masked training-only photometric warping loss.
- N3: exactly N2's network weights, with bounded sparse-map correction enabled.
- Both learned models trained for 40 epochs, seed 17, on 2,388 pairs from 12
  generated flights; 796 pairs from four scene-disjoint flights select checkpoints.
  Training scenes 2/3/4, validation scenes 5/6; public comparison scene 1 excluded.
- Both saved checkpoints were selected at epoch 4 by validation motion loss.
  Public evaluation scores did not select checkpoints or tune map gates.
- A complete same-seed pose-only training repeat produced an identical ONNX hash
  and best validation metrics; see the archived training-repeatability record.
- Current `workspace/solution/net.onnx` and `meta.json` select N1 (pose-only).
  For N2/N3, package the recorded checkpoint explicitly; see `docs/setup.md`.
- The v1 evidence archive records 39 passing host tests and a 20-frame WASM
  check for each learned variant.
  These checks establish execution/geometry contracts, not accuracy by themselves.
- Completed public-run reports and the accuracy/compute comparison belong in
  `experiments/learned-motion-v1/`; refer there rather than copying provisional
  numbers into new documents. All four comparison runs completed; inspect the
  positional gains together with rotation/scale trade-offs and the map regression.
  Raw trajectories/logs remain in ignored `results/`.

Windows upload-ready copies have been hash-verified in
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/`:
`AeroVIO-motion-pose-v1.wasm`, `AeroVIO-motion-warp-v1.wasm`, and
`AeroVIO-motion-warp-map-v1.wasm`. Packaging/copying is not a server submission.

## Rotation and input ablations

`experiments/motion-ablation-v2/` records the fixed-weight rotation toggle,
matched IMU-only training, and optional algebraically compacted export. Consult
its reports and checks for completed evaluation status and metrics.

- Rotation on/off packages reuse `experiments/learned-motion-v1/warp/net.onnx`.
  `rotation_residual_enabled` defaults to `true`; `false` composes only the IMU
  rotation. The predicted relative translation is unchanged, while subsequent
  world-axis translation can change because the orientation has changed.
  This ablation requires the sparse map to be disabled.
- `input_mode` defaults to `image_imu`. The `imu_only` model is trained with
  zero image inputs and receives zero images at inference. It retains all 14
  state values, including normalized calibration and elapsed time. Real RGB,
  depth and reference motion still supervise its training losses.
- The primary input ablation retains the original 29,398-parameter graph,
  initialization, cache, seed 17, 40 epochs, batch order and warping weight 2.
  Checkpoint selection follows the same validation rule; the IMU-only run
  selected epoch 14.
- `network_inputs` defaults to `images_state`. The optional `state` export
  requires `input_mode=imu_only`; it folds constant zero-image features into
  the first head-layer bias. It has 1,350 parameters and a 5,949-byte ONNX file,
  derived from the same trained checkpoint without retraining. Numerical
  equivalence evidence belongs with `imu-only-compact/run.json`.
- All compared v2 packages use the same guest Python source. The current
  workspace default remains N1, the pose-only image+IMU model.

The v2 Windows artifact filenames under
`C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/` are:
`AeroVIO-motion-current-v2.wasm`, `AeroVIO-motion-no-rotation-v2.wasm`,
`AeroVIO-motion-imu-only-v2.wasm`, and `AeroVIO-motion-imu-compact-v2.wasm`.
Check the v2 experiment record for copy/hash verification. Packaging and copying
remain separate from a course-server submission.

## Startup gyro calibration

`configs/gyro-bias-v4.json` fixes a two-by-two experiment: calibration on/off
and learned rotation residual on/off. All four cells share the original warp
ONNX, raw image/IMU predictor inputs and camera-local translation predictions.
The optional calibrator collects the first complete one-second hover window,
applies the fixed count, span, variation and simulator-prior gates, then either
freezes rejection or applies its IMU-axis bias estimate to future integration.
The closing frame is emitted before activation. There is no reset, replay,
gravity realignment, output hold, map correction or ground-truth input.

Startup evidence comes only from original simulated training/validation flights.
They share a deterministic startup motion; hover does not mean a frozen pose.
The method assumes that documented startup, and low variation cannot distinguish
small constant true rotation from sensor bias. Keep this limitation with claims.

Validation repeats all four complete flights per cell, checks unchanged raw
network inputs/outputs and freezes selection before public evaluation. The
pre-public `configs/gyro-bias-v4-public-replication.json` adds a full repeat of
the promising but unselected calibrated-gyro candidate. Seven public runs cover
the official baseline, four cells, the selected-current repeat and that diagnostic
repeat. Reproducibility cannot retroactively make a candidate eligible.
Four new packages cover `current`, `gyro-only`, `calibrated-residual`, and
`calibrated-gyro`; see `docs/setup.md` for explicit packaging flags.

## Important implementation details

The v3 auxiliary cache is `data/rotation-aux-v3/`: 2,400 training pairs from
scenes 2/3/4 and 800 validation pairs from scenes 5/6. It reprojects existing
320x240 adapted images into symmetric virtual rotations, then uses the shared
64x48 preprocessing. Camera x/y/z/mixed axes are balanced over 0–3 degrees;
the fixed crop and source-support check avoid padded borders. Synthetic IMU
samples include the actual extrinsics, gravity, lever-arm acceleration and
configured independent sensor noise. Only rotation is supervised by this cache.

The fixed protocol is `configs/rotation-focus-v3.json`. Both 20-epoch fine-tunes
start from `models/motion-warp-v1/best.pt`, share 760 optimizer steps and real
minibatches, and keep the original normalization and deployed architecture.
Selection minimizes accumulated real-validation rotation RMSE subject to both
translation guards; synthetic validation and public scores do not select weights.

The retention-only reruns live in `models/rotation-{control|focus}-v3-retained/`,
with compact records archived under `experiments/rotation-focus-v3/control/`
and `focus/`. Their top-level `net.onnx` is the selected epoch-zero model.
`diagnostic-last/net.onnx` is the unselected epoch-20 candidate, explicitly marked
as such in metadata. Keep these roles distinct in packaging, reporting and any
public comparison. The original v3 run records remain preserved as well.
The pre-public addendum `configs/rotation-focus-v3-public-diagnostics.json`
adds the fixed epoch-20 candidates without changing checkpoint selection.
The six compared variants are the original baseline, the starting warp model,
both selected v3 models and both last-epoch diagnostics. Four new v3 packages
cover the selected/control, selected/focus and two diagnostic roles. Public
completion and metrics must come from the finished reports, not package creation.

See `experiments/rotation-focus-v3/rotation-examples.png` for calibrated input
examples and its caption/JSON for exact sample identities and display settings.
The illustration is not evidence that the network estimates those rotations well.

The network receives two 64x48 grayscale frames and 14 values: camera-frame
IMU rotation/delta-velocity/delta-position, normalized intrinsics, and frame dt.
It predicts translation and an IMU rotation residual. `preintegrate.py` is shared
with training and includes the first sample interval. Its 1 microsecond endpoint
tolerance matches the upstream reader; truly future samples are rejected.

Pose maps the current camera into the first-camera frame. Twist is a finite-
interval mean in current camera axes, recomputed after map correction; it is
an approximation to instantaneous twist. All state resets per episode.

Training-only camera-z depth and fixed visibility masks never enter the deployed
network. New/disoccluded pixels do not supervise image reconstruction. Sparse
map points come only from causal observations and estimated poses. Rejected map
updates retain the learned prior. No unseen-surface generation is implemented.

## Next development decisions

Inspect the complete comparison and failure/per-episode evidence before deciding
which variant to extend. This is a small, single-seed pilot. Additional seeds and
scene diversity are needed before making broad superiority/generalization claims.
Keep architecture, data, runtime and limits fixed when isolating a loss or map
change. Do not tune on the public set and then call it independent validation.

## Boundaries

- No course-server upload, held-out evaluation, robot control or WAM training
  has been performed. No team token is stored here.
- November 9 course quota: 2 submissions/team/day. Current authenticated server
  scoring, limits, quota and deadline still require verification before submission.
- First-stage PDF omits a timezone; America/Los_Angeles is the course context.
- Sim(3) alignment removes global scale; inspect unaligned ATE, rotation and
  velocity errors as well. A low aligned error alone does not prove metric maps.
- Course PDFs retain their authors' rights; keep the repository private.
- Keep credentials, generated data, large logs, raw checkpoints and WASM binaries
  outside tracked source; compact experiment model snapshots are intentional.
