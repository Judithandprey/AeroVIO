# Structural-line relative rotation fallback V13

Date: November 16, 2026. Branch: `experiment/structural-lines-v13`.
Parent: `2d09ef5`. This experiment preserves all V8/V10/V11/V12 artifacts and
changes no existing estimator, training weights, main/default or course upload.

## Reference and question

The reference is the **selected V8 rotation-only WASM**, SHA-256
`50fac7f6d17e37f0cc5f473a326bca70d0fc6bc602a502fc0216bb17a5ae19ac`.
It is not the pre-V8 teacher and does not use V10/V11/V12 weights. Its frozen
1,350-parameter checkpoint is `experiments/motion-ablation-v2/imu-only-compact`.
V8's epipolar rotation refinement follows an accepted translation-direction fit,
so it cannot independently trigger in pure rotation or weak parallax. The new
module tests whether structural directions offer useful complementary evidence.

The three primary arms are original V8, detection/diagnostics only, and detection
with a bounded relative rotation correction. A fourth rebuild with line code
disabled verifies integration neutrality. Detection-only must reproduce every
V8 output exactly. No neural training occurs.

## Evidence and geometry

Input remains 320x240 RGB with the original calibration. Line images use exactly
the V8 rounded 2x2 green-channel average, yielding 160x120. Intrinsics use
`fx/2`, `fy/2`, `(cx+.5)/2-.5`, `(cy+.5)/2-.5`.

LSD with no refinement proposes segments. Fixed length, deduplication, spatial
spread and conditioning gates retain at most 24 long lines and three supported
vanishing directions. Each direction needs at least three distinct line supports.
For line endpoints with calibrated rays r0/r1, normalize n=r0 cross r1, then fit
the smallest eigenvector of sum(w*n*n^T). The same expression is K^T*l for an
image line l. Never divide by the vanishing direction's z coordinate: directions
at infinity remain valid. These are structural directions, not semantic walls.

Two nonparallel direction bundles are matched between past and current sampled
images using the gyro prediction, sign/permutation uniqueness and angle
consistency. A proper-rotation fit estimates relative orientation without
forcing the scene directions to be orthogonal. A single direction leaves an
unobservable rotation and is rejected. No room direction is a known global axis;
there is no absolute heading, width, metric scale or straight-trajectory prior.

Only past observations establish persistence. Capture is at frame modulo8=0;
detection/decision is at phase6, after the V8 point fit at phase4. Corrections
are low-gain, limited, spread over two frames, and canceled/rejected on point
rotation overlap or broken history. This delay is part of the estimator: at
20Hz, sampled images are 0.4s apart, processing is delayed 0.3s and application
takes 0.1s. Low per-call CPU time does not remove observation latency.

Raw gyro tracking coordinates remain unchanged. Let G_b be the immutable raw
gyro orientation at the captured current image and E_a the emitted orientation
at the anchor image. If R_line maps current to anchor camera coordinates, the
alignment target is E_a*R_line*G_b^T. Compare this with the alignment at decision
time, not merely raw gyro relative rotation. This avoids reapplying already
emitted alignment changes. Point-overlap gates further restrict this first
implementation to fallback intervals. Recompute twist from the emitted poses.

## Fixed evaluation

`configs/structural-lines-v13.json` fixes reference identities, datasets,
diagnostics, balanced execution order and engineering guards. Four existing
10-second flights in rooms5/6 are development evidence. Detector feasibility
uses only these images; probe fuel is measured inside the actual WASM. Synthetic
tests cover finite/infinite vanishing directions, oblique structures, pure
rotation, translation, sign/permutation ambiguity, missing directions, wrong
structures, resets and bounded correction.

Four new 60-second flights in rooms8/9 use the unchanged generator and
first-success crash policy. Generating them may overlap development, but their
scores are not inspected until rules/artifacts are frozen. No line-coverage or
accuracy filter selects flights. Failed attempts remain recorded. These are
four trajectories in two rooms, not four independent environments.

Fresh wall timing starts after generation and builds end. Six execution orders
balance three primary variants; repeats and diagnostics run outside timed step
calls. Report **maximum fuel per frame**, per-phase costs, and the actual frame
of each peak, as well as median host-specific WASM step time. Detector-only
isolates detector cost; it is not expected to improve accuracy.

Report each trajectory's raw SO(3) rotation RMSE, one-second relative rotation
error, Sim3 position ATE/scale, unaligned position ATE/drift, gate acceptance and
rejection reasons. Sim3 removes overall position scale/orientation and cannot
establish metric or attitude accuracy. Keep negative cases and missing evidence.
Frozen guards are engineering checks, not statistical significance. This first
module has no semantic furniture filter and no guarantee of generalization to
sloped walls, clutter, changing structural frames or real-camera distortion.

## Sources and distinction from full systems

[Kottas and Roumeliotis, RSS 2013](https://www-users.cse.umn.edu/~stergios/papers/RSS_2013_OC_VINS_Manhattan_World_Dimitris.pdf)
provides line-plane constraints and observability discussion; its known-direction
global-heading result does not apply to our self-estimated local scene axes.
[MLINE-VINS](https://arxiv.org/abs/2503.01571) motivates structural tracking and
verification. Its full backend and centered future-frame checks are not used.
This is a small causal inference ablation, not a reproduction of either SLAM
system or evidence that the navigation problem has been solved.

## Reproduction

Use the existing WSL checkout and pinned toolchain. Scripts refuse to overwrite
evaluation evidence; a failed build must use a new attempt name.

```bash
.venv/bin/python scripts/build_structural_lines.py --attempt a1
.venv/bin/python scripts/evaluate_structural_lines.py --phase development
# Complete fresh generation and freeze the development rules before confirmation.
.venv/bin/python scripts/generate_line_validation.py
.venv/bin/python scripts/evaluate_structural_lines.py --phase confirmation
.venv/bin/python scripts/report_structural_lines.py
```

If generation already ran, reuse its manifest; do not regenerate into that path.
To replay completed data without changing the preserved experiment directories:

```bash
.venv/bin/python scripts/reproduce_structural_lines.py --name structural-lines-v13-repro-01
```

The wrapper builds and evaluates in the new namespace and records its own hash.
It requires the original V8 binary and both completed data manifests; stop other
generation/training/build work before comparing wall timing. This wrapper's full
replay is separate from the repeated-output checks in the original experiment.

See `experiments/structural-lines-v13/README.md` for observed outcomes and exact
artifact identities. Raw inputs/logs/traces remain outside Git; compact configs,
versions, hashes, build/probe receipts and reports are preserved in the experiment.
