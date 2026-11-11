# AeroVIO

**Compute-Aware Visual-Inertial Odometry**

Estimate camera pose and twist from RGB + IMU while measuring the
accuracy-compute trade-off in a WebAssembly runtime. Developed for UC Berkeley
EECS 116/216, Fall 2026, Project 3.

**Latest completed experiment:** explicit geometric correspondence supervision
continues the V5 temporal network without changing its deployed architecture.
Matching endpoint error worsens despite lower cross-entropy; public median
rotation error rises from 9.7654 to 10.6784 degrees, with nearly unchanged aligned
ATE and compute. All seven public runs and the full repeat complete. No new model
is accepted; see [v6 evidence](experiments/correspondence-v6/README.md) and the
[supervision guide](docs/correspondence-supervision.md).

**Previous completed experiment:** a lightweight local feature-correlation branch
adds 6,243 trainable rotation parameters while freezing translation. Across ten
public flights it achieves 0.870938 m aligned ATE, 9.7654 degrees median rotation
error and 1.522 ms median WASM step time. However, the matched single-frame
control is more accurate on those aggregate metrics and faster, and both fail
original-flight validation. No replacement is selected. See the
[v5 results](experiments/feature-correlation-v5/README.md) and
[architecture guide](docs/feature-correlation.md).

**Previous investigation:** test a fixed, causal startup gyro calibration with
the existing model, without retraining. On four original simulated validation
flights, calibration without the learned rotation residual reduces mean rotation
RMSE by 68.77%, but raises aligned ATE by 0.02752%. It fails the frozen
nonregression guard, so the selected variant remains `current` (the v1 warp model
with calibration off and its residual on). All seven public runs completed;
the calibrated-gyro diagnostic reduces mean aligned ATE by 56.81% and maximum
fuel/frame by 89.60% versus the official baseline, with exact complete-run
reproduction. Its validation status and tradeoffs remain explicit. See the
[calibration guide](docs/gyro-bias-calibration.md) and
[completed comparison](experiments/gyro-bias-v4/README.md).

The previous rotation-focused experiment used calibrated view changes during
training with a matched continued-training control. Both v3 fine-tunes
retained their starting checkpoint under the fixed original-flight validation
rule; no deployed improvement was established. All six public variants completed.
The final focused candidate slightly improves aligned position error but worsens
rotation error by 11.54%; it remains unselected. See the [rotation-training guide](docs/rotation-focused-training.md)
and [v3 experiment record](experiments/rotation-focus-v3/README.md).

The completed [rotation/input experiment](experiments/motion-ablation-v2/README.md),
[v2 comparison](experiments/motion-ablation-v2/comparison.md), and
[rotation diagnostics](experiments/motion-ablation-v2/rotation-diagnostics.md)
preserve the earlier rotation toggle, visual-input and compact-export evidence.

The [first experiment](experiments/learned-motion-v1/README.md) preserves the
completed CNN, training-only warping and sparse-map comparison, including its
rotation/scale limitations and map regression. The
[warping and world-model guide](docs/warping-and-world-models.md) explains how the
implemented geometric constraint relates to the broader project ideas.

The official KLT + IMU reference remains frozen at tag `baseline-klt-imu-v1`.
It completed 10/10 public episodes with mean Sim(3) ATE **2.041635 m** and maximum
fuel/frame **141,910,846**; see [baseline evidence](experiments/baseline-klt-imu/README.md).
No course-server submission, held-out result, or physical-robot result is claimed.

The working branch is `experiment/perspective-map`. Its default
`workspace/solution/` model is the **pose-only learned variant**, not the tagged
KLT baseline. Package each variant explicitly to avoid mixing model identities.

## Start here

1. Read [project requirements](docs/requirements.md) and [course sources](docs/course/README.md).
2. Follow [setup and artifact reproduction](docs/setup.md).
3. Review the [learned architecture and training protocol](docs/learned-motion.md)
   and [sparse-map geometry](docs/perspective-map.md).
4. Read the [gyro-calibration validation](experiments/gyro-bias-v4/validation.md),
   [rotation-focused training experiment](experiments/rotation-focus-v3/README.md),
   [completed input ablations](experiments/motion-ablation-v2/README.md), and
   [earlier geometry comparison](experiments/learned-motion-v1/comparison.md)
   before drawing conclusions or changing parameters.
5. Use [HANDOFF.md](HANDOFF.md) to resume development.

## First-stage checkpoint

- Due: **November 15, 2026, 11:59 pm**.
- Progress presentation: **November 16, 2026**.
- November 9 course quota: **2 submissions per team per day**; the best is selected automatically.
- Trade-off: **mean Sim(3) ATE vs. maximum fuel per frame** in the pinned local runner.

Dates follow the course's America/Los_Angeles timezone context; the first-stage
PDF does not explicitly give a timezone. The course figure labels ATE as median,
while the pinned runner uses mean. Verify the current server Rules before uploading.

## Layout

```text
workspace/solution/  Learned guest estimator, ONNX weights and optional local map
workspace/runtime/   Pinned upstream WASM runtime recipes
training/            Host-only network and differentiable warping loss
scripts/             Data generation, training, packaging and comparison tools
configs/             Pinned inputs, limits and dependency snapshots
experiments/         Compact reports, model snapshots and provenance
    baseline-klt-imu/
    learned-motion-v1/
    motion-ablation-v2/
    rotation-focus-v3/
    gyro-bias-v4/
    feature-correlation-v5/
    correspondence-v6/
docs/                Course sources, requirements, architecture and setup
reports/             Presentation and final-report sources
HANDOFF.md           Verified state and development boundaries
```

`external/viobench/`, generated data, runtime builds, training checkpoints,
`artifacts/`, and raw `results/` remain ignored. The compact ONNX checkpoints and
metadata under `experiments/` identify the recorded experiments.
The evaluation helper records source/model hashes from each actual WASM payload.

## References

- [Course on bCourses](https://bcourses.berkeley.edu/courses/1557688)
- [Submission server](https://viobench.rl.tools/)
- [Pinned viobench source](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca)
- [Upstream attribution](THIRD_PARTY_NOTICES.md)

Keep team tokens and account credentials outside Git. Course PDFs retain their
authors' rights; this repository is private.
