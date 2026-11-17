# V13 development failure-mechanism audit

Read-only analysis of the four frozen development flights (800 frames, 100 sampled images). No inference rule, threshold, artifact, or frozen source was changed. No new training or replay was run for this audit. Confirmation results are outside this note.

**The correction arm never applied a line update.** All three arms' complete pose/twist arrays are exactly equal on all four flights. Only 7/100 images had two usable directions; five of those seven overlapped actual V8 rotation application. The remaining two images provided one anchor and one valid pair, not the two consecutive valid pairs required by the temporal gate. Thus this set supplies no evidence that line correction improves or worsens accuracy when active.

| Seed | Usable two-direction images | Point-overlap decisions | Weak decisions | Temporal warmup decisions |
| --- | ---: | ---: | ---: | ---: |
| 271840 | 0 | 17 | 8 | 0 |
| 271841 | 0 | 14 | 11 | 0 |
| 271842 | 0 | 18 | 7 | 0 |
| 271843 | 7 | 21 | 2 | 2 |
| Total | 7 | 70 | 28 | 2 |

Rejection counters record the first failing gate. Overlap is checked before weak evidence, so `reject_weak=28` does **not** mean only 28 images lack usable structure: 93/100 lack it, including 65 images classified as overlapping. Of the 30 decisions without overlap, only two have usable directions. Match, ambiguity, shape, residual, and gap rejection counters are all zero; those zeros largely reflect earlier rejection, not demonstrated robustness of the later checks.

## The seven usable images

All occur in ProcTHOR-Train-6, seed 271843. Frame numbers are zero-based. The last column lists actual V8 alignment-change frames after the previous decision and through this decision, recovered from existing saved traces as described below.

| Captured image | Decision frame | Lines / directions | Logged reason | Actual V8 change frames in decision window |
| ---: | ---: | --- | --- | --- |
| 0 | 6 | 9 / 2 | temporal | none |
| 8 | 14 | 9 / 2 | temporal | none |
| 16 | 22 | 9 / 2 | point_overlap | 20-22 |
| 24 | 30 | 9 / 2 | point_overlap | 23-27 |
| 32 | 38 | 9 / 2 | point_overlap | 36-38 |
| 40 | 46 | 10 / 2 | point_overlap | 39-43 |
| 176 | 182 | 9 / 2 | point_overlap | 175-182 |

Image 0 initializes an anchor. The 0-to-8 pair reaches the temporal gate with a residual of **0.01057838325139774 rad (0.606097 degrees)**: the pair passes matching, shape, fit, and magnitude checks, but is only the first valid pair. Before the next decision at frame 22, V8 applies a nonzero rotation at frames 20-22. The line controller therefore clears the anchor and pair history. Later usable observations are also excluded by actual V8 activity, so no uninterrupted qualifying history forms. This is not evidence of residual clipping or temporal-direction disagreement; the only measured pair never reaches a second comparable pair.

The frozen logic is explicit: `structural_line_gate.h:190-195` invalidates history on active V8 rotation or a changed fit counter; `:275-286` prioritizes overlap and initializes missing anchors; `:307-312` requires two consecutive stable pairs. `solution.cpp:215-220` calls the line controller after the current V8 alignment update.

## Actual rotation application, not a fit-counter-only explanation

Current V13 logs do not retain V8's per-frame fit counter or rotation-step vector. We therefore do not invent a counter history. Existing V8 `rotation` and `control` traces cover the same four flights. The old rotation artifact has the same SHA-256 as the frozen V13 baseline, and **all 800 historical rotation outputs and timestamps exactly equal the current baseline**. The old control uses the unchanged gyro orientation with the V8 visual-rotation switch off.

For each saved frame, recover `A_i = R_rotation_i * R_control_i^T`, then `D_i = A_(i-1)^T * A_i`. Measure the change with the stable expression

```text
theta_i = atan2(norm(vee((D_i - D_i^T)/2)), (trace(D_i)-1)/2)
active = theta_i > 1e-8 rad
```

Every one of the **70 overlap decisions** contains at least one recovered active step in its preceding eight-frame decision window. Minimum active step angles for seeds 271840..271843 are respectively `7.377977925463584e-6`, `9.600457531462024e-5`, `1.7721064043755185e-5`, and `4.439763108426826e-5` rad. The largest inactive numerical remainder across all four flights is `5.135292018398109e-16` rad, well below the threshold. Thus no observed overlap rejection needs a fit-counter-only explanation. This does not establish that the counter branch never also fired.

The historical final V8 diagnostics report 11/13/17/17 accepted rotation fits. Their legacy `applied_rotation_norm_sum_rad` field is a scheduled-angle total, not a per-frame application trace; it was not used as proof of when correction occurred.

## Cost and next step

Development peak fuel rises from **2,589,216 to 16,760,164 per frame (6.473x)**. Mean fuel rises from 1,177,757.14 to 2,880,596.92 (2.446x), with no output change. The detector runs before overlap rejection to measure coverage, so even rejected intervals pay its cost.

Proposal only: first reduce extraction cost and establish sustained two-direction coverage specifically during V8-inactive intervals, using measured WASM peak fuel. Skipping already-overlapping work could reduce average cost but cannot by itself fix the peak of a required detection. A cheaper bounded extractor or direction-guided edge search needs a separate controlled feasibility test. Pure-rotation/weak-parallax sequences can test the intended fallback opportunity. Do not infer that loosening the temporal or overlap checks is safe, and do not add training to this geometric module before cost and usable coverage are established. Current rules remain frozen for independent-room confirmation.

## Evidence identity

Paths are relative to `/root/projects/AeroVIO`. Decision logs reside at `results/structural-lines-v13/development/{0,1,2,3}/correct/decisions.jsonl`; historical traces reside at `results/native-fusion-v8/validation/{0,1,2,3}/{rotation,control}/trace.npz`.
Unqualified report names below are in `experiments/structural-lines-v13`; unqualified source filenames are in `workspace/structural_lines_native`. Both historical WASM file hashes were rechecked directly.

| Evidence | SHA-256 |
| --- | --- |
| development.json | `93dc6e7cce2cdc7e42495241ce390b829f963087dad44ff96c23ba7d9c0a4273` |
| frozen-before-confirmation.json | `5aa63ddc39c362ea7efab720e68f5365ec874f7496f9710223828fe69c64f34a` |
| structural_line_gate.h | `c27185745bdb5041946ef0c879492349d1970e3195c3a4c23b74019e577cc28d` |
| solution.cpp | `cb278d0f006b792840c76e8866dac4b4130729ec6a1cb604ec2554dcb31469b4` |
| visual_direction.h | `a45f7dc31e9e3a53ceeafccb8ae8891e8277dfe85af7cf415130b8cb28d5fce7` |
| scripts/evaluate_structural_lines.py | `f4400e1c436a7690add4e16f33a88ef9b2dbe6977ce2d9a49163a1b77841b874` |
| Frozen V8 rotation WASM | `50fac7f6d17e37f0cc5f473a326bca70d0fc6bc602a502fc0216bb17a5ae19ac` |
| Historical V8 control WASM | `88b9d384f4c849b20235ace8ebe740dcb79810762682625279b7819db30d7809` |
| experiments/native-fusion-v8/validation.json | `627e8c1c3f11c5c0d108d0929d5d3feddbd8f3fefd2e49e9a96a762f3b03c391` |
| Episode 0 decisions | `a539b3052d19a563986da2d23d628e0c588bdc649a064f18f874286b5f6e41ee` |
| Episode 1 decisions | `6e98d7a1b320eacc5b16aa741308bedfff9b09db2c419fe25d9f74018103e110` |
| Episode 2 decisions | `68231984f403495282cbf692f5d40b0ed82034e939a77570da68b9640931c6d2` |
| Episode 3 decisions | `17357b23f7dbe23b565ccc60f2768aa0d9c1a1b40a572600bc7c0e5f368888bd` |
| Historical episode 0 rotation trace | `8b735ba6947590eb290c25368858110ca8cea7cb180d506401248b94c49631fc` |
| Historical episode 0 control trace | `1c5bbb3158a433dcd15b536c95365b4168a3925844ddf9b5e8e1370df58024a4` |
| Historical episode 1 rotation trace | `9eb033f818c920393a6b45c07b252cb44b8eaa7034b257154da82803840ad6a8` |
| Historical episode 1 control trace | `c62e507be89f722a47a5a10a80d0a25c673f993f0f0d6e255ace1d166b79cae8` |
| Historical episode 2 rotation trace | `fc459f90487d87e34aaef762913b988af3025787960bc393ce3965340afa811d` |
| Historical episode 2 control trace | `39ca6c8580e45e89c4317eaf829fa20c2a694ce66a80781b7e6d66b468bcb165` |
| Historical episode 3 rotation trace | `84579c99a3bc48b158ff63010b441fa1e17f54f63ab8d85e6c703eb1ad7c629c` |
| Historical episode 3 control trace | `c9c4cdbeac6f348cc8fd575cdd77a59bbb5acbc4cc8ddb324817d47bb6ddaf21` |
| Current episode 3 baseline trace | `1800a8d679ac305c37ab185c0e47517b93422a6317e3102fccf3df96e3a8d342` |
