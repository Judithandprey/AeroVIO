# AeroVIO

**Compute-Aware Visual-Inertial Odometry**

A project exploring the accuracy-compute trade-off in visual-inertial odometry: estimate camera pose and twist from RGB + IMU, targeting WebAssembly for deterministic compute measurement.

Developed as part of UC Berkeley EECS 116/216, Fall 2026, Project 3.

**Status:** the unchanged official KLT + IMU baseline is built and evaluated locally: **10/10 public episodes**, mean Sim(3) ATE **2.041635 m**, maximum fuel/frame **141,910,846**. See [baseline evidence](experiments/baseline-klt-imu/README.md). No held-out result or course submission is claimed.

## Start here

1. Read [project requirements](docs/requirements.md) and the [original course materials](docs/course/README.md).
2. Follow [development setup](docs/setup.md) on Linux, macOS, or WSL2.
3. Review the [measured baseline](experiments/baseline-klt-imu/README.md) before changing the estimator.
4. Use [HANDOFF.md](HANDOFF.md) to resume work.

## First-stage checkpoint

- Due: **November 15, 2026, 11:59 pm**.
- Progress presentation: **November 16, 2026**.
- Latest course quota: **2 submissions per team per day**; the best is selected automatically.
- Trade-off: **Sim(3) ATE vs. maximum fuel per frame**.

Dates follow the course's America/Los_Angeles timezone context. The first-stage PDF does not explicitly state a timezone. These are the November 9 course rules; verify the submission server's current Rules before uploading.

## Layout

```text
docs/
  course/           Original PDFs, extracted text, and source manifest
  requirements.md   Project contract and dated updates
  setup.md          Upstream-based development instructions
experiments/        Reproducible experiment notes and compact metrics
reports/            Presentation and final-report sources
upstream.lock.json  Reviewed viobench revision and documentation reference
HANDOFF.md          Current state and next steps
```

`workspace/` contains the unchanged upstream estimator and runtime build recipes. `scripts/evaluate_public.py` records reproducible runs using `configs/`. The upstream checkout (`external/viobench/`), build products, `artifacts/`, and raw `results/` remain ignored.

## References

- [Course on bCourses](https://bcourses.berkeley.edu/courses/1557688)
- [Submission server](https://viobench.rl.tools/)
- [viobench upstream](https://github.com/jonas-eschmann/viobench)
- [Reviewed student guide](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/docs/STUDENT_GUIDE.md)

Keep team access tokens and account credentials outside Git. The course PDFs retain their authors' rights; this repository is private.
