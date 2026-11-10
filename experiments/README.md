# Experiment records

The [official KLT + IMU baseline](baseline-klt-imu/README.md) has been built and evaluated on all ten public episodes. Its source, build provenance, reports, and one-episode repeatability check are preserved in `baseline-klt-imu/`.

Use one Markdown file per experiment, for example `001-baseline.md`. Keep large raw reports in ignored `results/`, and record their location and hashes here. Commit compact metrics only after inspecting them for credentials or private data.

## Template

- **Question / hypothesis:**
- **Implementation commit:**
- **viobench commit:**
- **Environment / dependency versions:**
- **Method and configuration:**
- **Dataset / episode hashes:**
- **Evaluation limits and scoring definition:**
- **Random seeds:**
- **Exact build, training, and evaluation commands:**
- **Artifact SHA-256:**
- **Raw report location / SHA-256:**
- **Sim(3) ATE, aggregation rule, and units:**
- **Maximum fuel per frame:**
- **Failures / incomplete episodes:**
- **Comparison baseline:**
- **Conclusion supported by these results:**
- **Next change:**

Compare accuracy and compute together. Separate public-set results from held-out server results. Record one controlled change at a time when evaluating its effect.
