# Project requirements

Source snapshot: November 9, 2026 (America/Los_Angeles). Course requirements come from [the original PDFs](course/README.md), not from an implemented solution.

## Task and interface

| Item | Requirement |
| --- | --- |
| Task | Compute-efficient visual-inertial odometry (VIO) |
| RGB input | 320 x 240 or 640 x 480, at 20 Hz |
| IMU input | Acceleration and angular rates, at 200 Hz |
| Pose output | Metric position and orientation relative to the first frame |
| Twist output | Linear velocity and angular velocity |
| Submission | One WebAssembly `.wasm` module |
| Compute measure | Deterministic WebAssembly fuel |
| Methods | Classical, open source, learned, or other methods are allowed |
| Reproducibility | Solution must be reproducible; ML training pipelines must be repeatable |
| Learned methods | Course slides say data generation should be based on HyperDrone |

Coordinate frames, quaternion order, and ABI details must be read from the pinned upstream [TASK](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/docs/TASK.md) and [ABI](https://github.com/jonas-eschmann/viobench/blob/2072fa9b5b4a5eb34160a53f5789a9c4a9d220ca/docs/ABI.md) before implementation. Those interfaces have not yet been audited in this repository.

## Latest first-stage update

The November 9 first-stage PDF supersedes the main slides' one-per-day quota.

- Deadline: Sunday **November 15, 2026, 11:59 pm**.
- Presentations: Monday **November 16, 2026**.
- Submission quota: **2 per team per day**.
- Best submission is selected automatically.
- Pareto axes: **Sim(3) ATE** and **maximum fuel per frame**.
- Evaluation: **10 held-out episodes**, from a distribution very similar to public episodes.
- Announcement: results remain private until the presentation, and this stage's best submissions determine the next frontier.

The course's bCourses timezone is America/Los_Angeles; the PDF itself omits the timezone. This interpretation should be confirmed against the live submission server before relying on the exact cutoff.

The first-stage figure labels its ATE axis "Median", while the reviewed upstream student guide describes a mean across episodes. Preserve this difference and verify the actual server scoring configuration before choosing a local aggregation rule. No live server settings were retrieved during repository initialization.

## Team and compute

- Three students, including at least one EECS 116 undergraduate and one EECS 216 graduate student.
- Two teams share each RTX 5080 lab workstation.
- Alternating days with handover at noon.
- The team token is distributed to registered emails. It must remain local.

## Tentative schedule and attendance

| Week | Milestone |
| --- | --- |
| November 2 | Soft launch |
| November 9 | First-stage metrics and test datasets finalized; examples/baselines define the starting frontier |
| November 16 | First-stage progress presentations; a new frontier is frozen |
| November 23 | Second-stage progress presentations; another frontier is frozen |
| November 30 | Third/final presentations and final frontier |
| TBD | Report submission deadline |

The main slides mark this schedule **tentative**, with possible metric/dataset changes between stages.

One absence is permitted across the first two progress presentations, but at least one team member must present. Final presentation attendance is mandatory for all team members.

## First development milestone

- [ ] Audit the pinned task/ABI and select Python or native implementation.
- [ ] Generate the corresponding upstream workspace.
- [ ] Bundle a baseline to `.wasm`.
- [ ] Run `viobench check`.
- [ ] Evaluate on the public episodes and retain raw reports.
- [ ] Record commit, upstream revision, configuration, metrics, and artifact hash.
- [ ] Compare to baselines under matching evaluation limits.
- [ ] Verify current server rules and remaining quota before a separately authorized submission.

Building, local checking, public-set evaluation, server upload, and held-out scoring are separate milestones.
