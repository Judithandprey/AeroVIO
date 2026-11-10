# EECS 116/216 Project 3: Compute-Aware VIO

Working repository for the Fall 2026 project: estimate camera pose and twist from RGB + IMU while balancing trajectory accuracy and compute cost.

**Status:** project requirements, course sources, and development structure are initialized. No estimator, built WebAssembly module, training run, or evaluation result has been produced yet.

## Start here

1. Read [project requirements](docs/requirements.md) and the [original course materials](docs/course/README.md).
2. Follow [development setup](docs/setup.md) on Linux, macOS, or WSL2.
3. Record the first reproducible baseline in [experiments](experiments/README.md).
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

Setup will create a tracked-code workspace at `workspace/`, an ignored upstream checkout at `external/viobench/`, and ignored output folders such as `artifacts/` and `results/`.

## References

- [Course on bCourses](https://bcourses.berkeley.edu/courses/1557688)
- [Submission server](https://viobench.rl.tools/)
- [viobench upstream](https://github.com/jonas-eschmann/viobench)
- [Reviewed student guide](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/docs/STUDENT_GUIDE.md)

Keep team access tokens and account credentials outside Git. The course PDFs retain their authors' rights; this repository is private.
