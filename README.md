# AeroVIO

**Compute-Aware Visual-Inertial Odometry**

Estimate camera pose and twist from RGB + IMU while measuring the
accuracy-compute trade-off in a WebAssembly runtime. Developed for UC Berkeley
EECS 116/216, Fall 2026, Project 3.

**Current experiment:** a trained small motion network, a matched version with
training-only warping supervision, and an optional bounded sparse 3D map.
The three learned variants build and pass 20-frame WASM smoke checks. See the
[experiment record](experiments/learned-motion-v1/README.md) and
[measured comparison](experiments/learned-motion-v1/comparison.md) for completed
public-set results, per-episode regressions, and evidence limits.

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
4. Read the [controlled comparison](experiments/learned-motion-v1/comparison.md)
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
docs/                Course sources, requirements, architecture and setup
reports/             Presentation and final-report sources
HANDOFF.md           Verified state and development boundaries
```

`external/viobench/`, generated data, runtime builds, training checkpoints,
`artifacts/`, and raw `results/` remain ignored. The compact ONNX checkpoints and
metadata in `experiments/learned-motion-v1/` identify the recorded experiments.
The evaluation helper records source/model hashes from each actual WASM payload.

## References

- [Course on bCourses](https://bcourses.berkeley.edu/courses/1557688)
- [Submission server](https://viobench.rl.tools/)
- [Pinned viobench source](https://github.com/jonas-eschmann/viobench/tree/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca)
- [Upstream attribution](THIRD_PARTY_NOTICES.md)

Keep team tokens and account credentials outside Git. Course PDFs retain their
authors' rights; this repository is private.
